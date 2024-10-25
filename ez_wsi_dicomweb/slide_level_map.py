# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SlideLevelMap defines the relationship between frames and DICOM instances."""

from __future__ import annotations

import base64
import collections
import copy
import dataclasses
import enum
import io
import json
import math
import threading
import typing
from typing import Any, Collection, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple, Union

import cachetools
import dataclasses_json
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import pixel_spacing as pixel_spacing_module
from ez_wsi_dicomweb.ml_toolkit import dicom_json
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
import numpy as np
import pydicom

# Since the level distance is a log function, this max distance value allows
# a level that has a pixel spacing ratio within range of [2^-0.3, 2^0.3] to be
# matched as the closest level by get_level_by_pixel_spacing() function.
MAX_LEVEL_DIST = 0.23
_CONCATENATION_UID = tags.DicomTag(number='00209161', vr='UI')

# Pyramid ICC profiles are optimally serialized in JSON to avoid repeative
# re-initialization. However, some digital pathology DICOM, e.g. Leica, have
# very large ICC profiles, e.g., 12 MB. The default max size of the ICC profile
# controls the maximum size of the ICC profile serialized in JSON.
DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES = 204800

# https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.8
VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.2'
_VL_SLIDE_COORDINATES_MIROSCOPIC_IMAGE_SOP_CLASS_UID = (
    '1.2.840.10008.5.1.4.1.1.77.1.3'
)
VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
UNTILED_IMAGE_SOP_CLASS_UID = frozenset([
    VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_SLIDE_COORDINATES_MIROSCOPIC_IMAGE_SOP_CLASS_UID,
])

_VOLUME = 'VOLUME'
_THUMBNAIL = 'THUMBNAIL'
_LABEL = 'LABEL'
_OVERVIEW = 'OVERVIEW'
_LABEL_OVERVIEW_THUMBNAIL_LEVEL_SET = frozenset([_THUMBNAIL, _LABEL, _OVERVIEW])
LevelIndexType = Union[str, int]


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Instance:
  """Wrapper of a DICOM instance object, with details for frame lookup.

  Used by the Level class to represent an instance in a specific
  magnification level.

  Attributes:
    frame_offset: The frame index offset of the first possible frame from the
      instance on the parent magnification level.
    frame_count: Number of frames contained inside the instance.
    dicom_object: The DicomObject of the instance.
    path: Path to DICOM instance.
    is_tiled_full: Returns true if instance is tiled full.
  """

  frame_offset: int
  frame_count: int
  dicom_object: dicom_web_interface.DicomObject

  @property
  def path(self) -> dicom_path.Path:
    return self.dicom_object.path

  @property
  def is_tiled_full(self) -> bool:
    value = self.dicom_object.get_value(tags.DIMENSION_ORGANIZATION_TYPE)
    if value is None or not isinstance(value, str):
      return False
    return value.upper() == 'TILED_FULL'

  def instance_frame_number_from_wholes_slide_frame_number(
      self, frame_number: int
  ) -> int:
    """Converts a frame_number to a frame_number within this instance.

    Args:
     frame_number: The frame_number to be converted from.

    Returns:
     The frame number within this instance.
    """
    return frame_number - self.frame_offset


class ICCProfileMetadataState(enum.Enum):
  UNINITIALIZED = 0
  INITIALIZED = 1


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ICCProfileBytes:
  icc_profile_bytes_as_b64_encoded_str: str = ''
  metadata_state: ICCProfileMetadataState = (
      ICCProfileMetadataState.UNINITIALIZED
  )


@dataclasses.dataclass(frozen=True)
class ImageDimensions:
  width_px: int
  height_px: int

  def copy(self) -> ImageDimensions:
    return ImageDimensions(int(self.width_px), int(self.height_px))


class ResizedLevel:
  """DICOM pyramid level which is resized from existing level."""

  def __init__(
      self,
      source_image_level: Level,
      resized_level_dim: Union[
          ImageDimensions, pixel_spacing_module.PixelSpacing
      ],
  ):
    width_scale_factor, height_scale_factor = source_image_level.scale_factors(
        resized_level_dim
    )
    if isinstance(resized_level_dim, pixel_spacing_module.PixelSpacing):
      resized_level_dim = ImageDimensions(
          max(1, int(source_image_level.width / width_scale_factor)),
          max(1, int(source_image_level.height / height_scale_factor)),
      )
    self._source_image_level = source_image_level
    self._resized_level_dim = resized_level_dim.copy()
    if width_scale_factor >= 1.0 and height_scale_factor >= 1.0:
      common_scale_factor = max(width_scale_factor, height_scale_factor)
    elif width_scale_factor < 1.0 and height_scale_factor < 1.0:
      common_scale_factor = min(width_scale_factor, height_scale_factor)
    else:
      common_scale_factor = 0
    if (
        common_scale_factor != 0
        and int(source_image_level.height / common_scale_factor)
        == self._resized_level_dim.height_px
        and int(source_image_level.width / common_scale_factor)
        == self._resized_level_dim.width_px
    ):
      # If the larger resizing factor will result in the same dimensions
      # then scale pixel spacing by common factor to keep pixels as a square
      # as possible.
      height_scale_factor = common_scale_factor
      width_scale_factor = common_scale_factor
    source_pixel_spacing = source_image_level.pixel_spacing
    try:
      self._resized_pixel_spacing = pixel_spacing_module.PixelSpacing(
          source_pixel_spacing.column_spacing_mm * width_scale_factor,
          source_pixel_spacing.row_spacing_mm * height_scale_factor,
          source_pixel_spacing.mag_scaling_factor,
          source_pixel_spacing.spacing_diff_tolerance,
      )
    except ez_wsi_errors.UndefinedPixelSpacingError:
      # Source pixel spacing is undefined. Set resized imaging to
      # undefined pixel spacing.
      self._resized_pixel_spacing = source_pixel_spacing
    self._width_scale_factor = width_scale_factor
    self._height_scale_factor = height_scale_factor

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, ResizedLevel):
      return False
    return (
        (
            self._source_image_level is other._source_image_level
            or self._source_image_level == other._source_image_level
        )
        and self._resized_level_dim == other._resized_level_dim
        and self._resized_pixel_spacing == other._resized_pixel_spacing
        and self._width_scale_factor == other._width_scale_factor
        and self._height_scale_factor == other._height_scale_factor
    )

  @property
  def source_level(self) -> Level:
    return self._source_image_level

  @property
  def width(self) -> int:
    return self._resized_level_dim.width_px

  @property
  def height(self) -> int:
    return self._resized_level_dim.height_px

  @property
  def pixel_spacing(self) -> pixel_spacing_module.PixelSpacing:
    return self._resized_pixel_spacing

  def scale_factors(self) -> Tuple[float, float]:
    """Returns horizontal and vertical scale factors."""
    return self._width_scale_factor, self._height_scale_factor


def _get_ps_from_level_or_ds_level(
    level: Union[Level, ResizedLevel],
) -> pixel_spacing_module.PixelSpacing:
  """Returns pixel spacing class stored on Level or ResizedLevel."""
  if isinstance(level, Level):
    return typing.cast(Level, level).pixel_spacing
  else:
    return typing.cast(ResizedLevel, level).pixel_spacing


def _get_id_from_level_or_ds_level(
    level: Union[Level, ResizedLevel],
) -> ImageDimensions:
  """Returns ImageDimensions of Level or ResizedLevel."""
  if isinstance(level, Level):
    level = typing.cast(Level, level)
  else:
    level = typing.cast(ResizedLevel, level)
  return ImageDimensions(int(level.width), int(level.height))


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Level:
  """Represents the dimensions and instances of a specific magnification level.

  Attributes:
    level_index: The index of the level using 1-based indexing. A level with a
      higher index corresponding to a lower magnification level.
    width: The width, in number of pixels, of the entire level.
    height: The height, in number of pixels, of the entire level.
    samples_per_pixel: The number of samples per pixel. i.e. 3 for RGB format.
    bits_allocated: The number of bits allocated for storing one pixel.
    high_bit: The index of the highest bit in each sample of a pixel.
    pixel_spacing: Pixel spacing of the slide level.
    frame_width: The width, in pixels, of all frames on this level.
    frame_height: The width, in pixels, of all frames on this level.
    frame_number_min: The minimum frame number at this level.
    frame_number_max: The maximum frame number at this level.
    number_of_frames: Total number of frames at this level.
    instances: All instances on this level, keyed by the frame_offset of the
      instance. The instances should be ordered in the dictionary by the key.
    transfer_syntax_uid: DICOM transfer syntax instances are encoded as.
    tiled_full: Level is defined having tiled full organization.
  """

  level_index: LevelIndexType
  width: int
  height: int
  samples_per_pixel: int
  bits_allocated: int
  high_bit: int
  pixel_spacing: pixel_spacing_module.PixelSpacing
  frame_width: int
  frame_height: int
  frame_number_min: int
  frame_number_max: int
  instances: Dict[int, Instance]
  transfer_syntax_uid: str
  tiled_full: bool = dataclasses.field(init=False)

  def __post_init__(self):
    self.tiled_full = all([i.is_tiled_full for i in self.instances.values()])

  @property
  def number_of_frames(self) -> int:
    return self.frame_number_max - self.frame_number_min + 1

  @property
  def pixel_format(self) -> np.dtype:
    """Gets the pixel format of the provided level.

    Returns:
      The pixel format of the level as numpy.dtype.

    Raises:
      UnsupportedPixelFormatError if pixel format is not supported.
    """
    bytes_per_sample = math.ceil(self.bits_allocated / 8)
    if bytes_per_sample == 1:
      return np.uint8  # pytype: disable=bad-return-type  # numpy-scalars
    else:
      raise ez_wsi_errors.UnsupportedPixelFormatError(
          f'Pixel format not supported. BITS_ALLOCATED = {self.bits_allocated}'
      )

  def scale_factors(
      self,
      resized_level_dim: Union[
          Level,
          ResizedLevel,
          ImageDimensions,
          pixel_spacing_module.PixelSpacing,
      ],
  ) -> Tuple[float, float]:
    """Returns horizontal and vertical scale factors.

    Args:
      resized_level_dim: Level, ResizedLevel, ImageDimensions, or PixelSpacing
        to calculate scaling factors.

    Returns:
      Tuple(Horizontal, Vertical) scaling factors necessary to convert parameter
      dim to self dimensions.

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: self does not have defined
        pixel spacing and computing scaling factors based on pixel spacing or
        if self has defined pixel spacing and the passed Level or
        ResizedLevel does not.
    """
    if isinstance(resized_level_dim, Level) or isinstance(
        resized_level_dim, ResizedLevel
    ):
      # If passed Level or ResizedLevel prefer to scale with Pixel Spacing
      # scaling by pixel spacing will allow scale factors to be correctly
      # computed between levels from different pyramids. Computing by dimension
      # will not.
      if self.pixel_spacing.is_defined:
        resized_level_dim = _get_ps_from_level_or_ds_level(resized_level_dim)
      else:
        resized_level_dim = _get_id_from_level_or_ds_level(resized_level_dim)
    if isinstance(resized_level_dim, pixel_spacing_module.PixelSpacing):
      # Expected will raise ez_wsi_errors.UndefinedPixelSpacingError if
      # passed pixel spacing and self does not define pixel spacing or
      # Level or ResizedLevel does not contain defined pixel spacing.
      sfx = (
          resized_level_dim.column_spacing_mm
          / self.pixel_spacing.column_spacing_mm
      )
      sfy = resized_level_dim.row_spacing_mm / self.pixel_spacing.row_spacing_mm
    else:
      sfx = float(self.width) / float(resized_level_dim.width_px)
      sfy = float(self.height) / float(resized_level_dim.height_px)
    return sfx, sfy

  def resize(
      self,
      resized_level_dim: Union[
          Level,
          ResizedLevel,
          ImageDimensions,
          pixel_spacing_module.PixelSpacing,
      ],
  ) -> Union[Level, ResizedLevel]:
    """resizes of self (level image) to pixel_spacing or dim of target.

    Args:
      resized_level_dim: Level, ResizedLevel, ImageDimensions, or PixelSpacing
        to calculate scaling factors in relation to.

    Returns:
      Level if unchanged or ResizedLevel of self.

    Raises:
      ez_wsi_errors.UndefinedPixelSpacingError: self does not have defined
        pixel spacing and computing scaling factors based on pixel spacing or
        if self has defined pixel spacing and the passed Level or
        ResizedLevel does not.
    """
    if isinstance(resized_level_dim, Level) or isinstance(
        resized_level_dim, ResizedLevel
    ):
      if self.pixel_spacing.is_defined:
        resized_level_dim = _get_ps_from_level_or_ds_level(resized_level_dim)
      else:
        resized_level_dim = _get_id_from_level_or_ds_level(resized_level_dim)
    ds = ResizedLevel(self, resized_level_dim)
    if ds.width == self.width and ds.height == self.height:
      # if pixel dimensions of resized level are the return
      # source image.
      return self
    return ds

  def icc_profile_bulkdata_uri(self) -> str:
    for instance in self.instances.values():
      bulkdata_uri = instance.dicom_object.icc_profile_bulkdata_url
      if bulkdata_uri:
        return bulkdata_uri
    return ''

  def get_instance_by_frame(self, frame_number: int) -> Optional[Instance]:
    """Gets the instance that contains the requested frame.

    Args:
      frame_number: The frame number of the frame to request. The frame number
        of a frame is stored in the DICOM tag
        CONCATENATION_FRAME_OFFSET_NUMBER('00209228').

    Returns:
      The instance that contains the requested frame, or None if the input frame
      is out of range of any instance.
    """
    for frame_offset, instance in self.instances.items():
      if frame_number <= frame_offset:
        break
      if frame_number <= frame_offset + instance.frame_count:
        return instance
    return None

  def get_frame_number_by_point(self, x: int, y: int) -> int:
    """Gets the frame number corresponding to the input coordinate.

    Args:
      x: The x coordinate at the level the instance belongs to.
      y: The y coordinate at the level the instance belongs to.

    Returns:
      The frame number that contains the input coordinate.

    Raises:
      CoordinateOutofImageDimensionsError if the input coordinate is out of the
      range.
    """
    if x < 0 or y < 0 or x >= self.width or y >= self.height:
      raise ez_wsi_errors.CoordinateOutofImageDimensionsError(
          f'The input coordinate {x, y} is out of the range: '
          f'{0, 0, self.width - 1, self.height - 1}'
      )
    frame_x = int(x / self.frame_width)
    frame_y = int(y / self.frame_height)
    frames_per_row = int(math.ceil(float(self.width) / float(self.frame_width)))
    return frame_y * frames_per_row + frame_x + 1

  def get_frame_position(self, frame_number: int) -> Tuple[int, int]:
    """Gets the coordinate of the upper-left corner of the input frame.

    Args:
      frame_number: The input frame number to get the position of.

    Returns:
      The X and Y coordinates of the upper-left corner of the input frame.

    Raises:
      FrameNumberOutofBounds if the input frame number is out of the
      range.
    """
    if (
        frame_number < self.frame_number_min
        or frame_number > self.frame_number_max
    ):
      raise ez_wsi_errors.FrameNumberOutofBoundsError(
          f'The input frame number ({frame_number}) is out of the range: '
          f'{self.frame_number_min, self.frame_number_max}'
      )
    frame_number = int(frame_number - 1)
    frames_per_row = int(math.ceil(float(self.width) / float(self.frame_width)))
    frame_x = frame_number % frames_per_row
    frame_y = int(frame_number / frames_per_row)
    return frame_x * self.frame_width, frame_y * self.frame_height

  def get_instance_by_point(self, x: int, y: int) -> Optional[Instance]:
    """Gets the instance that contains the input point.

    Args:
      x: The x coordinate at the level the instance belongs to.
      y: The y coordinate at the level the instance belongs to.

    Returns:
      The instance that contains the input point, or None if the point is out of
      range of the level or is not within any instance.
    """
    return self.get_instance_by_frame(self.get_frame_number_by_point(x, y))

  @property
  def instance_iterator(self) -> Iterator[Instance]:
    """Returns iterator of level's DICOM instances."""
    return iter(self.instances.values())

  def get_level_sop_instance_uids(self) -> List[str]:
    return [
        instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
        for instance in self.instances.values()
    ]


def _wsi_dicom_pyramid_level(
    dicom_object: dicom_web_interface.DicomObject,
    level_index: LevelIndexType,
    frame_number_min: int,
    frame_number_max: int,
    spacing_diff_tolerance: float,
    instance: Optional[Instance] = None,
) -> Level:
  instances = {} if instance is None else {instance.frame_offset: instance}
  return Level(
      level_index=level_index,
      width=dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS),
      height=dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_ROWS),
      samples_per_pixel=dicom_object.get_value(tags.SAMPLES_PER_PIXEL),
      bits_allocated=dicom_object.get_value(tags.BITS_ALLOCATED),
      high_bit=dicom_object.get_value(tags.HIGH_BIT),
      pixel_spacing=_get_pixel_spacing(dicom_object, spacing_diff_tolerance),
      frame_width=dicom_object.get_value(tags.COLUMNS),
      frame_height=dicom_object.get_value(tags.ROWS),
      frame_number_min=frame_number_min,
      frame_number_max=frame_number_max,
      instances=instances,
      transfer_syntax_uid=dicom_object.get_value(tags.TRANSFER_SYNTAX_UID),
  )


def _create_wsi_label_thumbnail_or_overview_level(
    level_index: str,
    dicom_object: dicom_web_interface.DicomObject,
    spacing_diff_tolerance: float,
) -> Level:
  """Creates a level to represent a label, thumbnail or overview image."""
  frame_offset = (
      dicom_object.get_value(tags.CONCATENATION_FRAME_OFFSET_NUMBER) or 0
  )
  if frame_offset != 0:
    raise ez_wsi_errors.DicomSlideInitError(
        'Label, thumbnail and overview images should have frame offset 0 or'
        ' have the tag unset.'
    )
  frame_count = dicom_object.get_value(tags.NUMBER_OF_FRAMES) or 1
  if frame_count != 1:
    raise ez_wsi_errors.DicomSlideInitError(
        'Label, thumbnail and overview images should have frame count 1 or'
        ' have the tag unset.'
    )
  return _wsi_dicom_pyramid_level(
      dicom_object,
      level_index=level_index,
      frame_number_min=1,
      frame_number_max=1,
      spacing_diff_tolerance=spacing_diff_tolerance,
      instance=Instance(
          frame_offset=0,
          frame_count=1,
          dicom_object=dicom_object,
      ),
  )


class _LabelOverviewThumbnailImages:
  """Class to hold thumbnail, label and overview images associated with a WSI.

  This class adheres to the DICOM standard's Whole Slide Microscopy Image
  Flavors as define in table C.8.12.4-2.
  """

  def __init__(self):
    self.label: Optional[Level] = None
    self.thumbnail: Optional[Level] = None
    self.overview: Optional[Level] = None

  def has_image(self) -> bool:
    return (
        self.label is not None
        or self.thumbnail is not None
        or self.overview is not None
    )

  def _get_key_val_map(self) -> Mapping[str, Optional[Level]]:
    return {
        _LABEL: self.label,
        _THUMBNAIL: self.thumbnail,
        _OVERVIEW: self.overview,
    }

  @classmethod
  def _add_level(
      cls, inst: _LabelOverviewThumbnailImages, image_type: str, level: Level
  ) -> None:
    """Adds a label, thumbnail or overview image.

    Args:
      inst: Instance of the _LabelOverviewThumbnailImages to add the image to.
      image_type: The type of image being added.
      level: The level(image) to add.

    Raises:
      ez_wsi_errors.DicomSlideInitError: Multiple images of the same type are
      added.
    """
    if image_type == _LABEL:
      if inst.label is not None:
        raise ez_wsi_errors.DicomSlideInitError(
            'Slide contains multiple label images.'
        )
      inst.label = level
    elif image_type == _THUMBNAIL:
      if inst.thumbnail is not None:
        raise ez_wsi_errors.DicomSlideInitError(
            'Slide contains multiple thumbnail images.'
        )
      inst.thumbnail = level
    elif image_type == _OVERVIEW:
      if inst.overview is not None:
        raise ez_wsi_errors.DicomSlideInitError(
            'Slide contains multiple overview images.'
        )
      inst.overview = level
    else:
      raise ez_wsi_errors.DicomSlideInitError(
          f'Unknown image type: {image_type}'
      )

  @classmethod
  def from_dict(
      cls, level_map: Optional[Mapping[str, Any]]
  ) -> _LabelOverviewThumbnailImages:
    """Converts dict representation to _LabelOverviewThumbnailImages."""
    images = _LabelOverviewThumbnailImages()
    if level_map is None:
      return images
    for image_type in _LABEL_OVERVIEW_THUMBNAIL_LEVEL_SET:
      level_dict = level_map.get(image_type)
      if level_dict is None:
        continue
      _LabelOverviewThumbnailImages._add_level(
          images, image_type, Level.from_dict(level_dict)
      )
    return images

  def to_dict(self) -> Mapping[str, Any]:
    """Returns dict representation of _LabelOverviewThumbnailImages."""
    return {
        key: val.to_dict()
        for key, val in self._get_key_val_map().items()
        if val is not None
    }

  @classmethod
  def _get_level_index(cls, image_type: str) -> str:
    if image_type in _LABEL_OVERVIEW_THUMBNAIL_LEVEL_SET:
      return image_type
    raise ez_wsi_errors.DicomSlideInitError(f'Unknown image type: {image_type}')

  def get_level_by_index(self, level_index: str) -> Optional[Level]:
    try:
      return self._get_key_val_map().get(
          _LabelOverviewThumbnailImages._get_level_index(level_index)
      )
    except ez_wsi_errors.DicomSlideInitError:
      return None

  def add_image(
      self,
      image_type: str,
      dicom_object: dicom_web_interface.DicomObject,
      pixel_spacing_diff_tolerance: float,
  ):
    """Adds dicom instance to describing thumbnail, label or overview image.

    Args:
      image_type: The type of image being added.
      dicom_object: The dicom instance to add.
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.

    Raises:
      ez_wsi_errors.DicomSlideInitError: Multiple images of the same type are
      added or imaging being added is not a label, thumbnail or overview image.
    """
    level = _create_wsi_label_thumbnail_or_overview_level(
        _LabelOverviewThumbnailImages._get_level_index(image_type),
        dicom_object,
        pixel_spacing_diff_tolerance,
    )
    _LabelOverviewThumbnailImages._add_level(self, image_type, level)


def _order_untiled_level_map(
    levels: Mapping[str, Level],
) -> Mapping[str, Level]:
  sorted_levels = collections.OrderedDict()
  for key in sorted(list(levels)):
    sorted_levels[key] = levels[key]
  return sorted_levels


def _order_wsi_level_map(levels: Mapping[Any, Level]) -> Mapping[int, Level]:
  """Level map sorted levels based on level index instance frame offset."""
  # Sort levels based on level index.
  sorted_levels = collections.OrderedDict()
  levels_sorted_by_pixel_area = sorted(
      [(level, level.height * level.width) for level in levels.values()],
      key=lambda x: x[1],
      reverse=True,
  )
  for index, sorted_level in enumerate(levels_sorted_by_pixel_area):
    level, _ = sorted_level
    # Sort instances in each level based on frame_offset
    level = dataclasses.replace(level, level_index=index + 1)
    instances = level.instances
    sorted_instances = collections.OrderedDict()
    instance_offsets = sorted(list(instances.keys()))
    for frame_offset in instance_offsets:
      sorted_instances[frame_offset] = instances[frame_offset]
    frame_number_min = instance_offsets[0] + 1
    max_frame_offset = instance_offsets[-1]
    frame_number_max = (
        max_frame_offset + sorted_instances[max_frame_offset].frame_count
    )
    sorted_levels[level.level_index] = dataclasses.replace(
        level,
        frame_number_max=frame_number_max,
        frame_number_min=frame_number_min,
        instances=sorted_instances,
    )

  return sorted_levels


@dataclasses.dataclass(frozen=True)
class PyramidLevelsWithClosestPixelSpacing:
  equal_or_smaller_pixel_spacing: Optional[Level]
  greater_pixel_spacing: Optional[Level]


def _are_pixelspacing_equal(
    level: Level, pixel_spacing_mm: float, rel_tol: float
) -> bool:
  return math.isclose(
      level.pixel_spacing.pixel_spacing_mm, pixel_spacing_mm, rel_tol=rel_tol
  )


def _wsi_pyramid_level_key(
    dicom_object: dicom_web_interface.DicomObject,
) -> str:
  concatenation_uid = dicom_object.get_value(_CONCATENATION_UID)
  if concatenation_uid is not None:
    return concatenation_uid
  return dicom_object.get_value(tags.SOP_INSTANCE_UID)


def _validate_dicom_instance_defines_ez_wsi_required_tags_for_untiled_imaging(
    dicom_object: dicom_web_interface.DicomObject,
):
  _check_for_tag_or_raise(dicom_object, tags.SOP_INSTANCE_UID, True)
  _check_for_tag_or_raise(dicom_object, tags.COLUMNS, True)
  _check_for_tag_or_raise(dicom_object, tags.ROWS, True)
  _check_for_tag_or_raise(dicom_object, tags.SAMPLES_PER_PIXEL, True)
  _check_for_tag_or_raise(dicom_object, tags.BITS_ALLOCATED, True)
  _check_for_tag_or_raise(dicom_object, tags.HIGH_BIT, True)
  _check_for_tag_or_raise(dicom_object, tags.TRANSFER_SYNTAX_UID, True)


def _validate_dicom_instance_defines_ez_wsi_required_tags_for_wsi_imaging(
    dicom_object: dicom_web_interface.DicomObject,
    image_type: List[str],
):
  """Validates that the input DICOM instance defines all required tags."""
  # Makes sure the following tags have defined values in dicom_object.
  _validate_dicom_instance_defines_ez_wsi_required_tags_for_untiled_imaging(
      dicom_object
  )
  _check_for_tag_or_raise(dicom_object, tags.TOTAL_PIXEL_MATRIX_COLUMNS, True)
  _check_for_tag_or_raise(dicom_object, tags.TOTAL_PIXEL_MATRIX_ROWS, True)
  if image_type is not None and _VOLUME in image_type:
    _check_for_tag_or_raise(dicom_object, tags.IMAGE_VOLUME_WIDTH, True)
    _check_for_tag_or_raise(dicom_object, tags.IMAGE_VOLUME_HEIGHT, True)


def _get_pixel_spacing(
    dicom_object: dicom_web_interface.DicomObject, spacing_diff_tolerance: float
) -> pixel_spacing_module.PixelSpacing:
  """Returns pixel spacing from dicom_object.

  Args:
    dicom_object: The dicom object to get pixel spacing from.
    spacing_diff_tolerance: Pixel spacing difference tolerance.

  Returns
    Tuple [column spacing, row spacing]
  """
  try:
    column_spacing = dicom_object.get_value(
        tags.IMAGE_VOLUME_WIDTH
    ) / dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS)
    row_spacing = dicom_object.get_value(
        tags.IMAGE_VOLUME_HEIGHT
    ) / dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_ROWS)
    return pixel_spacing_module.PixelSpacing(
        column_spacing,
        row_spacing,
        spacing_diff_tolerance=spacing_diff_tolerance,
    )
  except TypeError:
    pass
  try:
    shared_functional_groups = dicom_object.get_list_value(
        tags.SHARED_FUNCTIONAL_GROUP_SEQUENCE
    )
    pixel_measure_sequence = dicom_json.GetList(
        shared_functional_groups[0], tags.PIXEL_MEASURES_SEQUENCE
    )
    spacing = dicom_json.GetList(pixel_measure_sequence[0], tags.PIXEL_SPACING)
    row_spacing, column_spacing = spacing
    return pixel_spacing_module.PixelSpacing(
        column_spacing,
        row_spacing,
        spacing_diff_tolerance=spacing_diff_tolerance,
    )
  except (KeyError, ValueError, IndexError, TypeError) as _:
    return pixel_spacing_module.UndefinedPixelSpacing(
        spacing_diff_tolerance=spacing_diff_tolerance,
    )


def _get_ancillary_image_type(image_type: frozenset[str]) -> str:
  for im_type in _LABEL_OVERVIEW_THUMBNAIL_LEVEL_SET:
    if im_type in image_type:
      return im_type
  return ''


def _check_for_tag_or_raise(
    dicom_object: dicom_web_interface.DicomObject,
    tag: tags.DicomTag,
    check_no_zero: bool = False,
):
  """Check for the existence of a DICOM tag in a DICOM object.

  Args:
    dicom_object: The host DICOM object to check against.
    tag: The DICOM tag to check for.
    check_no_zero: If enabled (true), this method will check if the value of the
      tag is non-zero.

  Raises:
    DicomTagNotFoundError: If the tag does not exist in the object.
    InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
    value.
  """
  value = dicom_object.get_value(tag)
  if value is None:
    raise ez_wsi_errors.DicomTagNotFoundError(
        f'DICOM tag {tag.number} is missing from the DICOM object: '
        f'{str(dicom_object.path)}'
    )
  if check_no_zero and value == 0:
    raise ez_wsi_errors.InvalidDicomTagError(
        f'DICOM tag {tag.number} cannot have zero value.'
    )


def _get_dicom_icc_profile_bytes(dcm: pydicom.Dataset) -> bytes:
  if 'OpticalPathSequence' in dcm:
    for dataset in dcm.OpticalPathSequence:
      if 'ICCProfile' in dataset:
        return dataset.ICCProfile
  if 'ICCProfile' in dcm:
    return dcm.ICCProfile
  return b''


def _get_level_icc_profile_bytes(
    path: dicom_path.Path,
    dwi: dicom_web_interface.DicomWebInterface,
) -> bytes:
  """Returns ICC profile bytes for a pyramid level."""
  with io.BytesIO() as dicom_bytes:
    dwi.download_instance_untranscoded(
        path,
        dicom_bytes,
    )
    dicom_bytes.seek(0)
    try:
      with pydicom.dcmread(dicom_bytes) as dcm:
        return _get_dicom_icc_profile_bytes(dcm)
    except pydicom.errors.InvalidDicomError as exp:
      raise ez_wsi_errors.DicomInstanceReadError(
          'Error reading DICOM instance.'
      ) from exp


class SlideLevelMap:
  """A class that builds and maintains the level-instance-frame mapping.

  When converting a WSI slide to DICOM format, a single WSI slide gets mapped to
  one DICOM series, which stores images at multiple levels of magnification with
  multiple DICOM instances.
  At each level of magnification, the image is divided into multiple frames with
  a fixed frame size(i.e. 500).
  DICOM tries to store all frames at a magnification level into one DICOM
  instance. If the number of frames at that level exceeds the limit, which is
  typically 2048, multiple instances are used. In that case, the index of the
  first possible frame within an instance is tagged as the frame_offset of that
  instance.
  """

  def __init__(
      self,
      dicom_objects: Collection[dicom_web_interface.DicomObject],
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
  ):
    """Constructor.

    Args:
      dicom_objects: DICOM objects of all instances within a DICOM series.
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.

    Raises:
      UnexpectedDicomObjectInstanceError: If any of the input objects is not a
      DICOM instance object
      NoDicomLevelsDetectedError: No level was detected in the Dicom object.
      DicomTagNotFoundError: If any of the required tags is missing from any
      DICOM object.
      InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
      value.
      SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide
      level contains instances with different transfer syntaxes.
    """
    self._slide_metadata_lock = threading.RLock()
    self._icc_profile_bytes = ICCProfileBytes()
    self._spacing_diff_tolerance = pixel_spacing_diff_tolerance
    self._level_map, self._label_overview_thumbnail_images = (
        self._build_level_map(dicom_objects)
    )
    self._smallest_level_path = self._get_smallest_level_path()
    level_index_list = list(self._level_map)
    self.level_index_min = level_index_list[0] if level_index_list else None
    self.level_index_max = level_index_list[-1] if level_index_list else None

  @classmethod
  def create_from_json(
      cls,
      json_str: Union[str, Mapping[str, Any]],
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
  ) -> SlideLevelMap:
    """Creates an SlideLevelMap from a JSON string or dict."""
    if not json_str:
      return SlideLevelMap([], pixel_spacing_diff_tolerance)
    instance = SlideLevelMap.__new__(SlideLevelMap)
    instance._slide_metadata_lock = threading.RLock()
    instance._icc_profile_bytes = ICCProfileBytes()
    instance._spacing_diff_tolerance = pixel_spacing_diff_tolerance
    slide_map_metadata = (
        json.loads(json_str) if isinstance(json_str, str) else json_str
    )
    level_map = slide_map_metadata['level_map']
    instance._icc_profile_bytes = ICCProfileBytes.from_json(
        json.dumps(slide_map_metadata['icc_profile'])
    )
    instance._label_overview_thumbnail_images = (
        _LabelOverviewThumbnailImages.from_dict(
            slide_map_metadata['label_overview_thumbnail_images']
        )
    )
    instance._level_map = _order_wsi_level_map(
        {int(key): Level.from_dict(level_map[key]) for key in level_map}
    )
    smallest_level_path = slide_map_metadata['smallest_level_path']
    instance._smallest_level_path = (
        dicom_path.Path.from_dict(smallest_level_path)
        if smallest_level_path
        else None
    )
    level_index_list = list(instance._level_map)
    instance.level_index_min = level_index_list[0] if level_index_list else None
    instance.level_index_max = (
        level_index_list[-1] if level_index_list else None
    )
    return instance

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = copy.copy(self.__dict__)
    del state['_slide_metadata_lock']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._slide_metadata_lock = threading.RLock()

  def _add_wsi_dicom_instance(
      self,
      levels: MutableMapping[str, Level],
      level_instances: MutableMapping[str, MutableMapping[int, Instance]],
      label_overview_thumbnail_images: _LabelOverviewThumbnailImages,
      dicom_object: dicom_web_interface.DicomObject,
  ) -> None:
    """Adds a level to represent a wsi pyramid image."""
    _check_for_tag_or_raise(dicom_object, tags.IMAGE_TYPE)
    image_type = dicom_object.get_list_value(tags.IMAGE_TYPE)
    _validate_dicom_instance_defines_ez_wsi_required_tags_for_wsi_imaging(
        dicom_object, image_type
    )
    if image_type is not None and image_type:
      image_type = _get_ancillary_image_type(frozenset(image_type))
      if image_type:
        label_overview_thumbnail_images.add_image(
            image_type, dicom_object, self._spacing_diff_tolerance
        )
        return

    level_index = _wsi_pyramid_level_key(dicom_object)
    level_index_data = levels.get(level_index)
    if level_index_data is None:
      levels[level_index] = _wsi_dicom_pyramid_level(
          dicom_object,
          level_index='undefined',
          frame_number_min=-1,
          frame_number_max=-1,
          spacing_diff_tolerance=self._spacing_diff_tolerance,
      )
    elif dicom_object.get_value(_CONCATENATION_UID) is None:
      raise ez_wsi_errors.DicomSlideInitError(
          'Series contains multiple instances with the same SOP instance UID.'
      )
    elif level_index_data.transfer_syntax_uid != dicom_object.get_value(
        tags.TRANSFER_SYNTAX_UID
    ):
      raise ez_wsi_errors.SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError()
    frame_offset = (
        dicom_object.get_value(tags.CONCATENATION_FRAME_OFFSET_NUMBER) or 0
    )
    level_instances[level_index][frame_offset] = Instance(
        frame_offset=frame_offset,
        frame_count=dicom_object.get_value(tags.NUMBER_OF_FRAMES) or 1,
        dicom_object=dicom_object,
    )

  def _build_level_map(
      self,
      dicom_objects: Collection[dicom_web_interface.DicomObject],
  ) -> Tuple[Mapping[int, Level], _LabelOverviewThumbnailImages]:
    """Returns a slide level map built from a set of input DICOM instances.

    The algorithm groups instances in multiple sets by level. The result is
    stored
    in a ordered dictionary, using the level index as the key

    Within each level, all instances belonging to that level are stored into a
    dictionary, using the frame offset of the instance as the key:
      Dict[frame_offset, Instance]

    Both the level dictionary and the instance dictionary need to be accessed in
    order by the key values. To avoid sorting the keys on-the-fly, we sort
    both dictionaries beforehand.

    Args:
      dicom_objects: DICOM objects to use to build the SlideLevelMap.

    Returns:
      A sorted dict that contains all pyramid levels of the DICOM object.
      _LabelOverviewThumbnailImages

    Raises:
      UnexpectedDicomObjectInstanceError: If any of the input objects is not a
      DICOM instance object.
      NoDicomLevelsDetectedError: No level was detected in the Dicom object.
      DicomTagNotFoundError: If any of the required tags is missing from any
      DICOM
      object.
      InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
      value.
      SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide
      level
        contains instances with different transfer syntaxes.
    """
    label_overview_thumbnail_images = _LabelOverviewThumbnailImages()
    levels = {}
    level_instances = collections.defaultdict(dict)
    for dicom_object in dicom_objects:
      if dicom_object.type() != dicom_path.Type.INSTANCE:
        raise ez_wsi_errors.UnexpectedDicomObjectInstanceError(
            'SlideLevelMap expects all input DicomObject to have a type of '
            f'INSTANCE. Actual: {dicom_object.type()}'
        )
      _check_for_tag_or_raise(dicom_object, tags.SOP_CLASS_UID)
      sop_class_uid = dicom_object.get_value(tags.SOP_CLASS_UID)
      if sop_class_uid != VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID:
        continue
      self._add_wsi_dicom_instance(
          levels, level_instances, label_overview_thumbnail_images, dicom_object
      )
    if not levels and not label_overview_thumbnail_images.has_image():
      raise ez_wsi_errors.NoDicomLevelsDetectedError(
          f'No level detected from the input DICOM objects {dicom_objects}'
      )
    for level_index in levels:
      levels[level_index] = dataclasses.replace(
          levels[level_index], instances=level_instances[level_index]
      )
    return (_order_wsi_level_map(levels), label_overview_thumbnail_images)

  def set_icc_profile_bytes(self, icc_profile_bytes: bytes) -> None:
    with self._slide_metadata_lock:
      self._icc_profile_bytes.icc_profile_bytes_as_b64_encoded_str = (
          base64.b64encode(icc_profile_bytes).decode('utf-8')
      )
      self._icc_profile_bytes.metadata_state = (
          ICCProfileMetadataState.INITIALIZED
      )

  def is_icc_profile_initialized(self) -> bool:
    with self._slide_metadata_lock:
      return (
          self._icc_profile_bytes.metadata_state
          == ICCProfileMetadataState.INITIALIZED
      )

  def get_icc_profile_bytes(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
  ) -> bytes:
    """Returns ICC profile for bytes for a pyramid level."""
    with self._slide_metadata_lock:
      if self.is_icc_profile_initialized():
        return base64.b64decode(
            self._icc_profile_bytes.icc_profile_bytes_as_b64_encoded_str.encode(
                'utf-8'
            )
        )
      for level in self._level_map.values():
        uri = level.icc_profile_bulkdata_uri()
        if uri:
          icc_profile_bytes = dwi.get_bulkdata(uri)
          self.set_icc_profile_bytes(icc_profile_bytes)
          return icc_profile_bytes
      if self._smallest_level_path is None:
        icc_profile_bytes = b''
      else:
        icc_profile_bytes = _get_level_icc_profile_bytes(
            self._smallest_level_path, dwi
        )
      self.set_icc_profile_bytes(icc_profile_bytes)
      return icc_profile_bytes

  def _get_smallest_level_path(self) -> Optional[dicom_path.Path]:
    """Returns pyramid_level with least number of frames."""
    smallest_level = None
    smallest_level_total_pixels = None
    for slide_level in self.level_map.values():
      frame_count = (
          1 + slide_level.frame_number_max - slide_level.frame_number_min
      )
      frame_pixels = slide_level.frame_width * slide_level.frame_height
      total_pixels = frame_count * frame_pixels
      if total_pixels <= 0:
        continue
      if smallest_level is None or smallest_level_total_pixels > total_pixels:
        smallest_level = slide_level
        smallest_level_total_pixels = total_pixels
    if smallest_level is None:
      return None
    return smallest_level.instances[0].dicom_object.path

  def get_json_encoded_icc_profile_size(self) -> int:
    with self._slide_metadata_lock:
      return len(self._icc_profile_bytes.to_json())

  def to_dict(
      self,
      level_subset: Optional[List[Level]] = None,
      max_json_encoded_icc_profile_size: int = DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> MutableMapping[str, Any]:
    """Converts pyramid level map into Dictionary."""
    with self._slide_metadata_lock:
      level_map = {}
      for key, slide_level in self._level_map.items():
        if level_subset is None or slide_level in level_subset:
          level_map[key] = slide_level.to_dict()
      if level_subset is not None and len(level_subset) != len(level_map):
        raise ez_wsi_errors.LevelNotFoundError('Levels not found in slide.')
      icc_profile_json = self._icc_profile_bytes.to_json()
      if (
          self.get_json_encoded_icc_profile_size()
          > max_json_encoded_icc_profile_size
      ):
        # Do not encode large icc profile that could exceed size limts for
        # VertexAI (1.5 MB payload limit); icc_profile for Leica WSI can
        # exceed 12 MB. If large set value to uninitialized to force decoder to
        # re-init icc profile.
        icc_profile_json = ICCProfileBytes().to_json()
      return {
          'level_map': level_map,
          'label_overview_thumbnail_images': (
              self._label_overview_thumbnail_images.to_dict()
          ),
          'smallest_level_path': (
              self._smallest_level_path.to_dict()
              if self._smallest_level_path is not None
              else {}
          ),
          'icc_profile': json.loads(icc_profile_json),
      }

  def to_json(
      self,
      level_subset: Optional[List[Level]] = None,
      max_json_encoded_icc_profile_size: int = DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> str:
    """Converts pyramid level map into JSON."""
    return json.dumps(
        self.to_dict(level_subset, max_json_encoded_icc_profile_size)
    )

  @property
  def level_map(self) -> Mapping[int, Level]:
    """Returns an ordered level map built from a set of input DICOM instances.

    The algorithm groups instances in multiple sets by level. The result is
    stored in a dictionary, using the level index as the key:
      Dict[level_index, Level]
    """
    return self._level_map

  @property
  def thumbnail(self) -> Optional[Level]:
    return self._label_overview_thumbnail_images.thumbnail

  @property
  def overview(self) -> Optional[Level]:
    return self._label_overview_thumbnail_images.overview

  @property
  def label(self) -> Optional[Level]:
    return self._label_overview_thumbnail_images.label

  def get_level(self, level_index: LevelIndexType) -> Optional[Level]:
    """Returns the level by requested level index.

    Args:
      level_index: The level index to be required. Level index usually starts at
        1.
    """
    level = self._level_map.get(level_index)
    if level is not None:
      return level
    return self._label_overview_thumbnail_images.get_level_by_index(level_index)

  def are_instances_concatenated(self, instance_uids: List[str]) -> bool:
    """Determines whether all instances in the list are concatenated.

    Args:
      instance_uids: A list of SOP Instance UIDs to check

    Returns:
      True if the instances are concatenated or if only one or fewer instance
      uids are provided. Otherwise returns False.
    """
    instance_uids_set = set(instance_uids)
    instance_uid_to_concat_id = {}

    for level in self._level_map.values():
      for instance in level.instances.values():
        if (
            instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
            in instance_uids_set
        ):
          if not instance.dicom_object.get_value(_CONCATENATION_UID):
            return False
          instance_uid_to_concat_id[
              instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
          ] = instance.dicom_object.get_value(_CONCATENATION_UID)

    return (instance_uid_to_concat_id.keys() == instance_uids_set) and (
        len(set(instance_uid_to_concat_id.values())) == 1
    )

  def get_instance_pixel_spacing(
      self, instance_uid: str
  ) -> Optional[pixel_spacing_module.PixelSpacing]:
    """Given an Instance UID retrieves its corresponding PixelSpacing.

    Args:
      instance_uid: A SOP Instance UID to find

    Returns:
      PixelSpacing of the instance. Raises if no match

    Raises:
      PixelSpacingNotFoundForInstanceError if the instance is not in the level
      map.
    """

    for level in self._level_map.values():
      for instance in level.instances.values():
        if (
            instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
            == instance_uid
        ):
          return level.pixel_spacing

    raise ez_wsi_errors.PixelSpacingNotFoundForInstanceError(
        f'Instance UID: {instance_uid} did not match any in the level map.'
    )

  def get_levels_with_closest_pixel_spacing(
      self,
      pixel_spacing_mm: float,
  ) -> PyramidLevelsWithClosestPixelSpacing:
    """Returns the level that has the closest pixel spacing as the input.

    Args:
      pixel_spacing_mm: Pixel spacing of desired imaging.

    Returns:
      The level that most matches with the requested pixel spacing.
    """
    equal_or_smaller_pixel_spacing = None
    greater_pixel_spacing = None
    # Finds the one or two closest levels to the desired level.
    for level in self._level_map.values():
      try:
        level_spacing = level.pixel_spacing.pixel_spacing_mm
        if level_spacing > pixel_spacing_mm:
          if (
              greater_pixel_spacing is None
              or greater_pixel_spacing.pixel_spacing.pixel_spacing_mm
              > level_spacing
          ):
            greater_pixel_spacing = level
        else:
          if (
              equal_or_smaller_pixel_spacing is None
              or equal_or_smaller_pixel_spacing.pixel_spacing.pixel_spacing_mm
              < level_spacing
          ):
            equal_or_smaller_pixel_spacing = level
      except ez_wsi_errors.UndefinedPixelSpacingError:
        continue
    return PyramidLevelsWithClosestPixelSpacing(
        equal_or_smaller_pixel_spacing, greater_pixel_spacing
    )

  def get_level_by_pixel_spacing(
      self,
      pixel_spacing_mm: float,
      relative_pixel_spacing_equality_threshold: float = MAX_LEVEL_DIST,
      maximum_downsample: float = 0.0,
  ) -> Optional[Level]:
    """Returns the level that has the closest pixel spacing as the input.

    Args:
      pixel_spacing_mm: Pixel spacing of desired imaging.
      relative_pixel_spacing_equality_threshold: Maximum relative difference in
        (pyramid / level pixel spacing / desired pixel spacing) at which pixel
        spacing should be considered equilvalent.  E.g., (value of 0.25 =
        pyramid levels with pixels spacings that are 25% smaller - 25% larger
        are considered equlivalent).
      maximum_downsample: Maximum degree to which it is acceptable to downsample
        pixel imaging to acchieve a desired target pixel spacing. Only used when
        source imaging pixel spacing is < and != to desired target pixel
        spacing.

    Returns:
      The level that closely matches with the requestd pixel spacing, or None
      if the requested pixel spacing is out range of any existing levels.
    """
    pyramid_levels = self.get_levels_with_closest_pixel_spacing(
        pixel_spacing_mm
    )
    closest_smaller_level = pyramid_levels.equal_or_smaller_pixel_spacing
    closest_greater_level = pyramid_levels.greater_pixel_spacing
    if closest_smaller_level is not None and _are_pixelspacing_equal(
        closest_smaller_level,
        pixel_spacing_mm,
        relative_pixel_spacing_equality_threshold,
    ):
      return closest_smaller_level
    if closest_greater_level is not None and _are_pixelspacing_equal(
        closest_greater_level,
        pixel_spacing_mm,
        relative_pixel_spacing_equality_threshold,
    ):
      return closest_greater_level
    if (
        closest_smaller_level is None
        or pixel_spacing_mm
        / closest_smaller_level.pixel_spacing.pixel_spacing_mm
        > maximum_downsample
    ):
      return None
    return closest_smaller_level


class UntiledImageMap:
  """A class that builds and maintains the level-instance-frame mapping.

  When converting a WSI slide to DICOM format, a single WSI slide gets mapped to
  one DICOM series, which stores images at multiple levels of magnification with
  multiple DICOM instances.
  At each level of magnification, the image is divided into multiple frames with
  a fixed frame size(i.e. 500).
  DICOM tries to store all frames at a magnification level into one DICOM
  instance. If the number of frames at that level exceeds the limit, which is
  typically 2048, multiple instances are used. In that case, the index of the
  first possible frame within an instance is tagged as the frame_offset of that
  instance.
  """

  def __init__(
      self,
      dicom_objects: Collection[dicom_web_interface.DicomObject],
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
  ):
    """Constructor.

    Args:
      dicom_objects: DICOM objects of all instances within a DICOM series.
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.

    Raises:
      UnexpectedDicomObjectInstanceError: If any of the input objects is not a
      DICOM instance object
      NoDicomLevelsDetectedError: No level was detected in the Dicom object.
      DicomTagNotFoundError: If any of the required tags is missing from any
      DICOM object.
      InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
      value.
      SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide
      level contains instances with different transfer syntaxes.
    """
    self._slide_metadata_lock = threading.Lock()
    self._pixel_spacing_diff_tolerance = pixel_spacing_diff_tolerance
    self._icc_profile_level_bytes_cache = cachetools.LRUCache(maxsize=10)
    self._untiled_image_map = self._build_untiled_image_map(dicom_objects)

  @classmethod
  def create_from_json(
      cls,
      json_str: Union[str, Mapping[str, Any]],
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
  ) -> UntiledImageMap:
    """Creates an UntiledImageMap from a JSON string or dict."""
    if not json_str:
      return UntiledImageMap([], pixel_spacing_diff_tolerance)
    instance = UntiledImageMap.__new__(UntiledImageMap)
    instance._slide_metadata_lock = threading.Lock()
    instance._pixel_spacing_diff_tolerance = pixel_spacing_diff_tolerance
    instance._icc_profile_level_bytes_cache = cachetools.LRUCache(maxsize=10)
    slide_map_metadata = (
        json.loads(json_str) if isinstance(json_str, str) else json_str
    )
    untiled_image_map = slide_map_metadata['untiled_image_map']
    instance._untiled_image_map = _order_untiled_level_map({
        key: Level.from_dict(value) for key, value in untiled_image_map.items()
    })
    return instance

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = copy.copy(self.__dict__)
    del state['_slide_metadata_lock']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._slide_metadata_lock = threading.Lock()

  def _create_untiled_image_level(
      self, dicom_object: dicom_web_interface.DicomObject
  ):
    """Adds a level to represent an untiled image."""
    frame_offset = (
        dicom_object.get_value(tags.CONCATENATION_FRAME_OFFSET_NUMBER) or 0
    )
    if frame_offset != 0:
      raise ez_wsi_errors.DicomSlideInitError(
          'Label, thumbnail and overview images should have frame offset 0 or'
          ' have the tag unset.'
      )
    frame_count = dicom_object.get_value(tags.NUMBER_OF_FRAMES) or 1
    if frame_count != 1:
      raise ez_wsi_errors.DicomSlideInitError(
          'Label, thumbnail and overview images should have frame count 1 or'
          ' have the tag unset.'
      )
    pixel_spacing = dicom_object.get_list_value(tags.PIXEL_SPACING)
    if pixel_spacing is not None and pixel_spacing and len(pixel_spacing) == 2:
      pixel_spacing = pixel_spacing_module.PixelSpacing(
          pixel_spacing[1],
          pixel_spacing[0],
          spacing_diff_tolerance=self._pixel_spacing_diff_tolerance,
      )
    else:
      pixel_spacing = pixel_spacing_module.UndefinedPixelSpacing(
          spacing_diff_tolerance=self._pixel_spacing_diff_tolerance,
      )
    return Level(
        level_index=dicom_object.get_value(tags.SOP_INSTANCE_UID),
        width=dicom_object.get_value(tags.COLUMNS),
        height=dicom_object.get_value(tags.ROWS),
        samples_per_pixel=dicom_object.get_value(tags.SAMPLES_PER_PIXEL),
        bits_allocated=dicom_object.get_value(tags.BITS_ALLOCATED),
        high_bit=dicom_object.get_value(tags.HIGH_BIT),
        pixel_spacing=pixel_spacing,
        frame_width=dicom_object.get_value(tags.COLUMNS),
        frame_height=dicom_object.get_value(tags.ROWS),
        frame_number_min=1,
        frame_number_max=1,
        instances={
            0: Instance(
                frame_offset=0,
                frame_count=1,
                dicom_object=dicom_object,
            )
        },
        transfer_syntax_uid=dicom_object.get_value(tags.TRANSFER_SYNTAX_UID),
    )

  def _add_untiled_image_dicom_instance(
      self,
      untiled_imaging: MutableMapping[str, Level],
      dicom_object: dicom_web_interface.DicomObject,
  ) -> None:
    _validate_dicom_instance_defines_ez_wsi_required_tags_for_untiled_imaging(
        dicom_object
    )
    untiled_image_level = self._create_untiled_image_level(dicom_object)
    untiled_imaging[untiled_image_level.level_index] = untiled_image_level

  def _build_untiled_image_map(
      self,
      dicom_objects: Collection[dicom_web_interface.DicomObject],
  ) -> Mapping[str, Level]:
    """Returns a slide level map built from a set of input DICOM instances.

    The algorithm groups instances in multiple sets by level. The result is
    stored
    in a ordered dictionary, using the level index as the key

    Within each level, all instances belonging to that level are stored into a
    dictionary, using the frame offset of the instance as the key:
      Dict[frame_offset, Instance]

    Both the level dictionary and the instance dictionary need to be accessed in
    order by the key values. To avoid sorting the keys on-the-fly, we sort
    both dictionaries beforehand.

    Args:
      dicom_objects: DICOM objects to use to build the SlideLevelMap.

    Returns:
      A sorted dict that contains all pyramid levels of the DICOM object.

    Raises:
      UnexpectedDicomObjectInstanceError: If any of the input objects is not a
      DICOM instance object.
      NoDicomLevelsDetectedError: No level was detected in the Dicom object.
      DicomTagNotFoundError: If any of the required tags is missing from any
      DICOM
      object.
      InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
      value.
      SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide
      level
        contains instances with different transfer syntaxes.
    """
    images = {}
    for dicom_object in dicom_objects:
      if dicom_object.type() != dicom_path.Type.INSTANCE:
        raise ez_wsi_errors.UnexpectedDicomObjectInstanceError(
            'SlideLevelMap expects all input DicomObject to have a type of '
            f'INSTANCE. Actual: {dicom_object.type()}'
        )
      _check_for_tag_or_raise(dicom_object, tags.SOP_CLASS_UID)
      sop_class_uid = dicom_object.get_value(tags.SOP_CLASS_UID)
      if sop_class_uid not in UNTILED_IMAGE_SOP_CLASS_UID:
        continue
      self._add_untiled_image_dicom_instance(images, dicom_object)
    if not images:
      raise ez_wsi_errors.NoDicomLevelsDetectedError(
          'No non-tiled images detected from the input DICOM objects'
          f' {dicom_objects}'
      )
    return _order_untiled_level_map(images)

  def to_dict(
      self,
      level_subset: Optional[List[Level]] = None,
  ) -> MutableMapping[str, Any]:
    """Converts pyramid level map into Dictionary."""
    level_map = {}
    for key, slide_level in self._untiled_image_map.items():
      if level_subset is None or slide_level in level_subset:
        level_map[key] = slide_level.to_dict()
    if level_subset is not None and len(level_subset) != len(level_map):
      raise ez_wsi_errors.LevelNotFoundError('Levels not found in slide.')
    return {
        'untiled_image_map': level_map,
    }

  def to_json(self, level_subset: Optional[List[Level]] = None) -> str:
    """Converts pyramid level map into JSON."""
    return json.dumps(self.to_dict(level_subset))

  def get_level_icc_profile_bytes(
      self,
      level: Level,
      dwi: dicom_web_interface.DicomWebInterface,
  ) -> bytes:
    """Returns ICC profile for bytes for a pyramid level."""
    with self._slide_metadata_lock:
      instance_path = level.instances[0].dicom_object.path
      key = instance_path.complete_url
      icc_profile = self._icc_profile_level_bytes_cache.get(key)
      if icc_profile is not None:
        return icc_profile
      icc_profile = _get_level_icc_profile_bytes(instance_path, dwi)
      self._icc_profile_level_bytes_cache[key] = icc_profile
      return icc_profile

  @property
  def untiled_image_map(self) -> Mapping[str, Level]:
    return self._untiled_image_map
