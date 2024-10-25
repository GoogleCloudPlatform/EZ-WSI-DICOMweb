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
"""An interface for accessing a Pathology slide stored in a DICOMStore.

Whole-Slide Images (WSI) (https://dicom.nema.org/dicom/dicomwsi) in digital
pathology are scans of glass slides typically created at a magnification of
20x or 40x using microscopes. Pixels typically represent between
0.275 µm - 0.5 µm of a slide but the exact size depends on the scanner camera
resolution. With typical dimensions between of 15mm x 15mm for tissue slides,
WSIs are often about 50,000 x 50,000 pixels for a 20x magnification or
100,000 x 100,000 pixels for a 40x magnification. The images are usually stored
as image pyramid to allow for fast access to different magnification levels.

The class layout/hierarchy is as follows:
DicomSlide: a single (whole) slide image typically represented as
      an image pyramid along with slide level metadata.
   |--> DicomImage: an image at a specific magnification in the image pyramid of
        a slide along with associated metadata.
     |--> DicomImagePatch: a rectangular patch/tile of an Image
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import copy
import dataclasses
import heapq
import importlib.resources
import io
import itertools
import json
import logging
import math
from typing import Any, Collection, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import cv2
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import pixel_spacing as pixel_spacing_module
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
import google.auth
import numpy as np
import PIL
from PIL import ImageCms
import pydicom


# JSON slide metadata keys
_SOP_CLASS_UID = 'sop_class_uid'
LEVEL_MAP = 'level_map'
_SLIDE_PATH = 'slide_path'
UNTILED_MICROSCOPE_IMAGE_MAP = 'untiled_microscope_image_map'

# ICC Profile Color Correction
_RAW = 'raw'
_RGB = 'RGB'
_SRGB = 'sRGB'

# Annotation IOD
# https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_b.5.html
MICROSCOPY_BULK_SIMPLE_ANNOTATIONS_STORAGE = '1.2.840.10008.5.1.4.1.1.91.1'

ImageDimensions = slide_level_map.ImageDimensions
Level = slide_level_map.Level
ResizedLevel = slide_level_map.ResizedLevel


def _read_icc_profile(filename: str) -> bytes:
  # https://setuptools.pypa.io/en/latest/userguide/datafiles.html
  rc_file = importlib.resources.files('third_party')
  return rc_file.joinpath(filename).read_bytes()


def get_srgb_icc_profile_bytes() -> bytes:
  """Returns sRGB ICC Profile bytes."""
  try:
    return _read_icc_profile('sRGB_v4_ICC_preference.icc')
  except FileNotFoundError:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile(_SRGB)).tobytes()


def get_adobergb_icc_profile_bytes() -> bytes:
  """Returns AdobeRGB ICC Profile bytes."""
  return _read_icc_profile('AdobeRGB1998.icc')


def get_rommrgb_icc_profile_bytes() -> bytes:
  """Returns ROMM RGB ICC Profile bytes."""
  return _read_icc_profile('ISO22028-2_ROMM-RGB.icc')


def _get_cmsprofile_from_iccprofile_bytes(b: bytes) -> ImageCms.ImageCmsProfile:
  """Converts ICC Profile bytes to ImageCms.ImageCmsProfile."""
  return ImageCms.getOpenProfile(io.BytesIO(b))


def get_srgb_icc_profile() -> ImageCms.core.CmsProfile:
  """Returns sRGB ICC Profile."""
  return _get_cmsprofile_from_iccprofile_bytes(get_srgb_icc_profile_bytes())


def get_adobergb_icc_profile() -> ImageCms.core.CmsProfile:
  """Returns AdobeRGB ICC Profile."""
  return _get_cmsprofile_from_iccprofile_bytes(get_adobergb_icc_profile_bytes())


def get_rommrgb_icc_profile() -> ImageCms.core.CmsProfile:
  """Returns ROMMRGB ICC Profile."""
  return _get_cmsprofile_from_iccprofile_bytes(get_rommrgb_icc_profile_bytes())


@dataclasses.dataclass(frozen=True)
class _PatchIntersection:
  x_origin: int  # The upper leftmost x coordinate of the patch intersection.
  y_origin: int  # The upper leftmost y coordinate of the patch intersection.
  width: int
  height: int


@dataclasses.dataclass(frozen=True)
class Frame:
  """A frame in a DICOM, in pixel unit."""

  x_origin: int  # The upper leftmost x coordinate of the patch intersection.
  y_origin: int  # The upper leftmost y coordinate of the patch intersection.
  width: int
  height: int
  image_np: np.ndarray


# DICOM Transfer Syntax's which do not require transcoding and are natively
# compatible.
_SUPPORTED_CLIENT_SIDE_DECODING_RAW_TRANSFER_SYNTAXS = (
    '1.2.840.10008.1.2.1',
    '1.2.840.10008.1.2',
)


def is_client_side_pixel_decoding_supported(transfer_syntax: str) -> bool:
  """Returns True if cache supports client side decoding of pixel encoding.

  Args:
    transfer_syntax: DICOM transfer syntax uid.

  Returns:
    True if cache supports operation on instances with encoding.
  """
  return (
      dicom_frame_decoder.can_decompress_dicom_transfer_syntax(transfer_syntax)
      or transfer_syntax in _SUPPORTED_CLIENT_SIDE_DECODING_RAW_TRANSFER_SYNTAXS
  )


def _pixel_spacing_to_level(
    slide: DicomSlide,
    pixel_spacing: pixel_spacing_module.PixelSpacing,
    maximum_downsample: float = 0.0,
) -> slide_level_map.Level:
  """Returns the level that has the lowest level index."""
  logging.info(
      'Pixel spacing parameter will be deprecated in future versions. Modify'
      ' code to define source imaging using slide_level_map.Level.'
  )
  source_image_level = slide.get_level_by_pixel_spacing(
      pixel_spacing, maximum_downsample=maximum_downsample
  )
  if source_image_level is None:
    raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
        'No pyramid level found with pixel spacing:'
        f' ~{pixel_spacing.pixel_spacing_mm}  mm/px'
    )
  return source_image_level


class DicomImage:
  """Represents an image at a specific pixel spacing in a DicomSlide."""

  def __init__(
      self,
      level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
      ],
      source: _DicomSeries,
  ):
    """Constructor for DicomImage.

    Args:
      level: Pyramid level to return.
      source: The series this image belongs to.
    """
    self._output_image_level = level
    self._source = source

  @property
  def pixel_spacing(self) -> pixel_spacing_module.PixelSpacing:
    return self._output_image_level.pixel_spacing

  @property
  def source(self) -> _DicomSeries:
    return self._source

  def get_image_as_patch(
      self, require_fully_in_source_image: bool = False
  ) -> DicomPatch:
    return self._source.get_patch(
        self._output_image_level,
        x=0,
        y=0,
        width=self._output_image_level.width,
        height=self._output_image_level.height,
        require_fully_in_source_image=require_fully_in_source_image,
    )

  @property
  def width(self) -> int:
    return self._output_image_level.width

  @property
  def height(self) -> int:
    return self._output_image_level.height

  def image_bytes(
      self, color_transform: Optional[ImageCms.ImageCmsTransform] = None
  ) -> np.ndarray:
    """Loads the pixel bytes of the DICOM Image.

    Args:
      color_transform: Optional ICC Profile color transformation to perform on
        image.

    Returns:
      Numpy array representing the DICOM Image.
    """
    # Internally reuses the Patch implementation for bytes fetching.
    # An image can be represented as a giant patch starting from (0, 0)
    # and spans the whole slide.
    return self.get_image_as_patch().image_bytes(
        color_transform=color_transform
    )


@dataclasses.dataclass(frozen=True)
class PatchBounds:
  """A bounding rectangle of a patch, in pixel units."""

  x_origin: int  # The upper leftmost x coordinate of the patch intersection.
  y_origin: int  # The upper leftmost y coordinate of the patch intersection.
  width: int
  height: int


class BasePatch:
  """A rectangular patch/tile/view of an Image at a specific pixel spacing."""

  def __init__(
      self,
      x: int,
      y: int,
      width: int,
      height: int,
  ):
    self._x = x
    self._y = y
    self._width = width
    self._height = height

  @property
  def x(self) -> int:
    return self._x

  @property
  def y(self) -> int:
    return self._y

  @property
  def width(self) -> int:
    return self._width

  @property
  def height(self) -> int:
    return self._height

  @property
  def patch_bounds(self) -> PatchBounds:
    return PatchBounds(
        x_origin=self.x, y_origin=self.y, width=self.width, height=self.height
    )

  def is_patch_fully_in_source_image_dim(self, width: int, height: int) -> bool:
    if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
      return False
    return self.x + self.width <= width and self.y + self.height <= height


def get_image_bytes_samples_per_pixel(image_bytes: np.ndarray) -> int:
  """Returns the number of samples per pixel in the image.

  Args:
    image_bytes: Uncompressed image bytes (e.g., 8 bit RGB)

  Raises:
    ez_wsi_errors.GcsImageError: If the image is not 2D or 3D.
  """
  if len(image_bytes.shape) == 2:
    return 1
  elif len(image_bytes.shape) == 3:
    return image_bytes.shape[2]
  raise ez_wsi_errors.GcsImageError(
      f'Invalid image shape: {image_bytes.shape}. Image must be 2D or 3D.'
  )


def transform_image_bytes_color(
    image_bytes: np.ndarray,
    color_transform: Optional[ImageCms.ImageCmsTransform] = None,
) -> np.ndarray:
  """Transforms image bytes color using ICC Profile Transformation."""
  samples_per_pixel = get_image_bytes_samples_per_pixel(image_bytes)
  height, width = image_bytes.shape[0:2]
  if color_transform is None or samples_per_pixel <= 1:
    return image_bytes
  img = PIL.Image.frombuffer(
      _RGB,
      (width, height),
      image_bytes.tobytes(),
      decoder_name=_RAW,
  )
  ImageCms.applyTransform(img, color_transform, inPlace=True)
  return np.asarray(img)


class _SlidePyramidLevelPatch(BasePatch):
  """A rectangular patch/tile/view of an Image at a specific pixel spacing.

  A Patch's data is composed from its overlap with one or more DICOM Frames.
  Patch representation here represents a region of pixels as stored in DICOM.
  """

  def __init__(
      self,
      source: _DicomSeries,
      level: slide_level_map.Level,
      x: int,
      y: int,
      width: int,
      height: int,
  ):
    """Constructor.

    Args:
      source: The source DICOM series this patch belongs to.
      level: Pyramid Level source imaging is derived from.
      x: The X coordinate of the starting point (upper-left corner) of the
        patch.
      y: The Y coordinate of the starting point (upper-left corner) of the
        patch.
      width: The width of the patch.
      height: The height of the patch.
    """
    super().__init__(x, y, width, height)
    self._level = level
    self._source = source
    if not source.has_level(level):
      raise ez_wsi_errors.LevelNotFoundError(
          f'The level {level.level_index} is not found in the slide.'
      )

  def is_patch_fully_in_source_image(self) -> bool:
    return self.is_patch_fully_in_source_image_dim(
        self._level.width, self._level.height
    )

  def get_pyramid_imaging_source_level(self) -> slide_level_map.Level:
    return self._level

  @property
  def pixel_spacing(self) -> pixel_spacing_module.PixelSpacing:
    return self._level.pixel_spacing

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, _SlidePyramidLevelPatch):
      return False
    return (
        self.x == other.x
        and self.y == other.y
        and self.width == other.width
        and self.height == other.height
        and (self._source is other._source or self._source == other._source)
        and (self._level is other._level or self._level == other._level)
    )

  @property
  def source(self) -> _DicomSeries:
    return self._source

  def _get_intersection(
      self,
      frame_x_cord: int,
      frame_y_cord: int,
      frame_width: int,
      frame_height: int,
  ) -> _PatchIntersection:
    """Returns intersection from the source patch and rectangular coordinate.

    Args:
      frame_x_cord: Source frame x coordinate (upper left).
      frame_y_cord: Source frame y coordinate (upper left).
      frame_width: Source frame width
      frame_height: Source frame height

    Returns:
      Intersection between patches and rectangular coordinate.

    Raises:
      PatchIntersectionNotFoundError if there is no overlap between the source
      and the destination patches.
    """
    x = max(frame_x_cord, self.x)
    y = max(frame_y_cord, self.y)
    width = min(frame_x_cord + frame_width - x, self.x + self.width - x)
    height = min(frame_y_cord + frame_height - y, self.y + self.height - y)
    if width <= 0 or height <= 0:
      raise ez_wsi_errors.PatchIntersectionNotFoundError(
          'There is no overlap region between the source and the destination'
          ' patches.'
      )
    return _PatchIntersection(
        x_origin=x, y_origin=y, width=width, height=height
    )

  def _copy_overlapped_region(
      self, src_frame: Frame, dst_image_np: np.ndarray
  ) -> Tuple[int, int]:
    """Copies the overlapped region from the source patch to the dest nd.array.

    Args:
      src_frame: the source frame to be copied from.
      dst_image_np: the destination np to be copied to.

    Returns:
      The width and height of the overlapped region gets copied.

    Raises:
      PatchIntersectionNotFoundError if there is no overlap between the source
      and the destination patches.
    """
    intersection = self._get_intersection(
        src_frame.x_origin,
        src_frame.y_origin,
        src_frame.width,
        src_frame.height,
    )
    x = intersection.x_origin
    y = intersection.y_origin
    width = intersection.width
    height = intersection.height
    # pylint: disable=protected-access
    _copy_ndarray(
        src_frame.image_np,
        x - src_frame.x_origin,
        y - src_frame.y_origin,
        width,
        height,
        dst_image_np,
        x - self.x,
        y - self.y,
    )
    # pylint: enable=protected-access
    return width, height

  def frame_numbers(self) -> Iterator[int]:
    """Generates slide level frame numbers required to render patch.

    Frame numbering starts at 1.

    Yields:
      A generator that produces frame numbers.

    Raises:
      DicomSlideMissingError if slide used to create self is None.
    """
    if self.source is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )
    y = self.y
    x = self.x
    width = self.width
    height = self.height
    cy = y
    frame_width = self._level.frame_width
    frame_height = self._level.frame_height
    cached_instance = None
    last_instance_frame_number = -math.inf
    while cy < y + height and cy < self._level.height:
      cx = x
      region_height = 0
      while cx < x + width and cx < self._level.width:
        try:
          frame_number = self._level.get_frame_number_by_point(cx, cy)
          if (
              cached_instance is None
              or frame_number - cached_instance.frame_offset
              >= cached_instance.frame_count
          ):
            cached_instance = self._level.get_instance_by_frame(frame_number)
            last_instance_frame_number = -math.inf
            if cached_instance is None:
              raise ez_wsi_errors.FrameNumberOutofBoundsError()
          if frame_number > last_instance_frame_number:
            yield frame_number
          last_instance_frame_number = frame_number
          pos_x, pos_y = self._level.get_frame_position(frame_number)
          intersection = self._get_intersection(
              pos_x, pos_y, frame_width, frame_height
          )
          region_width = intersection.width
          region_height = intersection.height
        except ez_wsi_errors.EZWsiError:
          # No frame found at (cx, cy), move 1 pixel in both X, and Y direction.
          region_width = 1
          region_height = max(1, region_height)
        cx += region_width
      cy += region_height

  def image_bytes(
      self, color_transform: Optional[ImageCms.ImageCmsTransform] = None
  ) -> np.ndarray:
    """Returns the patch's image bytes.

    Args:
      color_transform: Optional ICC Profile color transformation to apply to on
        image bytes.

    Returns:
      Numpy array type image.

    Raises:
      DicomSlideMissingError if slide used to create self is None.
      PatchOutOfBoundsError if the patch is not within the bounds of the DICOM
      image.
    """
    if self._source is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )
    image_bytes = np.zeros(
        (self.height, self.width, self._level.samples_per_pixel),
        self._level.pixel_format,
    )
    pixel_copied = False
    # Copies image pixels from all overlapped frames.
    for frame_number in self.frame_numbers():
      frame = self.source.get_frame(self._level, frame_number)
      if frame is None:
        continue
      try:
        self._copy_overlapped_region(frame, image_bytes)
        pixel_copied = True
      except ez_wsi_errors.EZWsiError:
        continue
    if not pixel_copied:
      raise ez_wsi_errors.SectionOutOfImageBoundsError(
          'The requested patch is out of scope of the image.'
      )
    return transform_image_bytes_color(image_bytes, color_transform)

  def get_gcp_data_credential_header(
      self, credential: Optional[google.auth.credentials.Credentials] = None
  ) -> Dict[str, str]:
    """Returns the credential header patch requests."""
    return self.source.get_credential_header(credential)


def _scale_coordinate(pt: int, source_width: int, dest_width: int) -> int:
  """Scale coordinate from source to destination."""
  return int(int(pt) * int(dest_width) / int(source_width))


def _bottom_offset_padding(pt: int, source_width: int, dest_width: int) -> int:
  """Returns starting offset when upsampling.

  Returns 0 if downsampling. Otherwise determines the fraction of the first
  upsampled pixel which falls within the upsampled image.

  Args:
    pt: Coordinate of pixel in source imaging.
    source_width: Width of source imaging.
    dest_width: Width of destination imaging.

  Returns:
    0 if imaging is being downsampled or clipping pixel offset for first
    upsampled pixel.
  """
  if source_width >= dest_width:
    return 0
  sf = int(source_width) / int(dest_width)
  fractional_pt = float(int(pt) * int(source_width) / int(dest_width))
  padding = int((fractional_pt - math.floor(fractional_pt)) / sf)
  return max(padding, 0)


def _top_offset_padding(pt: int, source_width: int, dest_width: int) -> bool:
  """Returns true if last upsampled pixel partially falls on image.

  Returns false if downsampling. Otherwise determines i the last
  upsampled pixel which falls only partially within the upsampled image bounds.

  Args:
    pt: Coordinate of pixel in source imaging.
    source_width: Width of source imaging.
    dest_width: Width of destination imaging.

  Returns:
    False if imaging is being downsampled or True if last pixel only partially
    falls within image patch.
  """
  if source_width >= dest_width:
    return False
  sf = int(source_width) / int(dest_width)
  fractional_pt = float(int(pt) * int(source_width) / int(dest_width))
  padding = int((math.ceil(fractional_pt) - fractional_pt) / sf)
  return max(padding, 0) > 0


class DicomPatch(_SlidePyramidLevelPatch):
  """A rectangular patch/tile/view of an Image at a specific pixel spacing.

  Abstraction over, _SlidePyramidLevelPatch that supports resizing source
  pyramid imaging to target pixel spacing.
  """

  def __init__(
      self,
      source_image_level: slide_level_map.Level,
      x: int,
      y: int,
      width: int,
      height: int,
      source: _DicomSeries,
      destination_image_level: Union[
          slide_level_map.Level, slide_level_map.ResizedLevel, None
      ] = None,
      require_fully_in_source_image: bool = False,
  ):
    """Constructor.

    Args:
      source_image_level: Level that slide source imaging is generated from.
      x: The X coordinate of the starting point (upper-left corner) of the
        generated patch in destination image level coordinates.
      y: The Y coordinate of the starting point (upper-left corner) of the
        generated patch in destination image level coordinates.
      width: The width of the generated patch in destination image.
      height: The height of the generated patch in destination image.
      source: The parent DICOM Slide this patch belongs to.
      destination_image_level: Level that patch represents if undefined defaults
        to source level.
      require_fully_in_source_image: Require patch be fully in image
    """
    if (
        not source_image_level.tiled_full
        and source_image_level.number_of_frames != 1
    ):
      raise ez_wsi_errors.DicomPatchGenerationError(
          'DICOM instance(s) do not have TILED_FULL Dimension Organization'
          ' Type.'
      )
    super().__init__(source, source_image_level, x, y, width, height)
    if destination_image_level is None:
      destination_image_level = source_image_level
    self._destination_image_level = destination_image_level
    source_start_x = _scale_coordinate(
        x, destination_image_level.width, source_image_level.width
    )
    source_start_y = _scale_coordinate(
        y, destination_image_level.height, source_image_level.height
    )
    source_end_x = _scale_coordinate(
        x + width, destination_image_level.width, source_image_level.width
    )
    source_end_y = _scale_coordinate(
        y + height, destination_image_level.height, source_image_level.height
    )
    if (
        source_start_x == x
        and source_start_y == y
        and source_end_x == x + width
        and source_end_y == y + height
    ):
      # patch imaging is not being resized.
      self._resized_patch_source = None
      self._pixel_spacing = source_image_level.pixel_spacing
      self._bottom_x_offset_pad = 0  # Not used in if dimensions unchanged.
      self._bottom_y_offset_pad = 0  # Not used
      self._upsample_target_width = width  # Not used
      self._upsample_target_height = height  # Not used
    else:
      # patch imaging is not being resized.

      # determine offsets for image upsampling.  NOPs for downsampling.
      self._bottom_x_offset_pad = _bottom_offset_padding(
          x, source_image_level.width, destination_image_level.width
      )
      self._bottom_y_offset_pad = _bottom_offset_padding(
          y, source_image_level.height, destination_image_level.height
      )
      # if patch bounds end before end of image and there is an additional
      # pixel in source which fractionally falls on the patch, then include.
      if source_end_x < source_image_level.width and _top_offset_padding(
          x + width, source_image_level.width, destination_image_level.width
      ):
        source_end_x += 1
      # if patch bounds end before end of image and there is an additional
      # pixel in source which fractionally falls on the patch, then include.
      if source_end_y < source_image_level.height and _top_offset_padding(
          y + height,
          source_image_level.height,
          destination_image_level.height,
      ):
        source_end_y += 1

      self._pixel_spacing = destination_image_level.pixel_spacing
      source_width = source_end_x - source_start_x
      source_height = source_end_y - source_start_y
      self._resized_patch_source = _SlidePyramidLevelPatch(
          source,
          source_image_level,
          source_start_x,
          source_start_y,
          source_width,
          source_height,
      )
      # determine dimensions to rescale upsampled image to before clipping
      self._upsample_target_width = _scale_coordinate(
          source_start_x + source_width,
          source_image_level.width,
          destination_image_level.width,
      ) - _scale_coordinate(
          source_start_x,
          source_image_level.width,
          destination_image_level.width,
      )
      self._upsample_target_height = _scale_coordinate(
          source_start_y + source_height,
          source_image_level.height,
          destination_image_level.height,
      ) - _scale_coordinate(
          source_start_y,
          source_image_level.height,
          destination_image_level.height,
      )
    if (
        require_fully_in_source_image
        and not self.is_patch_fully_in_source_image()
    ):
      raise ez_wsi_errors.PatchOutsideOfImageDimensionsError(
          'A portion of the patch does not overlap the image.'
      )
    self._require_fully_in_source_image = require_fully_in_source_image

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, DicomPatch):
      return False
    return (
        self.x == other.x
        and self.y == other.y
        and self.width == other.width
        and self.height == other.height
        and (self.source is other.source or self.source == other.source)
        and (self._level is other._level or self._level == other._level)
        and (
            self._destination_image_level is other._destination_image_level
            or self._destination_image_level == other._destination_image_level
        )
    )

  @property
  def id(self) -> str:
    if not self.source or not self.source.accession_number:
      return (
          f'M_{self.pixel_spacing.as_magnification_string}:'
          f'{self.width:06d}x{self.height:06d}{self.x:+07d}{self.y:+07d}'
      )
    # Be consistent with internal id format.
    # "%s:%06dx%06d%+07d%+07d", image_id, width, height, left, top
    return (
        f'{self.source.accession_number}:M_{self.pixel_spacing.as_magnification_string}:'
        f'{self.width:06d}x{self.height:06d}{self.x:+07d}{self.y:+07d}'
    )

  def is_patch_fully_in_source_image(self) -> bool:
    if self._resized_patch_source is None:
      return super().is_patch_fully_in_source_image()
    return self._resized_patch_source.is_patch_fully_in_source_image()

  @property
  def pixel_spacing(self) -> pixel_spacing_module.PixelSpacing:
    return self._pixel_spacing

  def frame_numbers(self) -> Iterator[int]:
    if self._resized_patch_source is None:
      return super().frame_numbers()
    # returns frame numbers in source image level.
    return self._resized_patch_source.frame_numbers()

  @property
  def is_resized(self) -> bool:
    return self._resized_patch_source is not None

  @property
  def level(
      self,
  ) -> Union[slide_level_map.Level, slide_level_map.ResizedLevel]:
    """Level patch bytes are generated with respect to."""
    return self._destination_image_level

  def image_bytes(
      self, color_transform: Optional[ImageCms.ImageCmsTransform] = None
  ) -> np.ndarray:
    if self._resized_patch_source is None:
      # no change to byte dimensions.
      return super().image_bytes(color_transform=color_transform)
    source_image_bytes = self._resized_patch_source.image_bytes(
        color_transform=color_transform
    )
    if (
        self.width > self._resized_patch_source.width
        or self.height > self._resized_patch_source.height
    ):
      # upsample bytes
      pixels = cv2.resize(
          source_image_bytes,
          (self._upsample_target_width, self._upsample_target_height),
          interpolation=cv2.INTER_CUBIC,
      )
      # clip regions of upsampled imaging not falling in patch.
      return pixels[
          self._bottom_y_offset_pad : (self.height + self._bottom_y_offset_pad),
          self._bottom_x_offset_pad : (self.width + self._bottom_x_offset_pad),
          ...,
      ]
    else:
      # Downsample bytes
      return cv2.resize(
          source_image_bytes,
          (self.width, self.height),
          interpolation=cv2.INTER_AREA,
      )

  def get_patch(
      self,
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: Optional[bool] = None,
  ) -> DicomPatch:
    """Returns a patch based from the existing patch."""
    require_fully_in_source_image = (
        self._require_fully_in_source_image
        if require_fully_in_source_image is None
        else require_fully_in_source_image
    )
    return DicomPatch(
        self._level,
        x,
        y,
        width,
        height,
        self.source,
        self._destination_image_level,
        require_fully_in_source_image=require_fully_in_source_image,
    )


def _get_native_level(
    slm: slide_level_map.SlideLevelMap,
) -> Optional[slide_level_map.Level]:
  """Gets the native level from a SlideLevelMap.

  The native level of a slide has the lowest level index.

  Args:
    slm: The source SlideLevelMap to get the native level from.

  Returns:
    The level that has the lowest index in the slide.

  Raises:
    LevelNotFoundError if the native level does not exist.
  """
  if slm.level_index_min is not None:
    level = slm.get_level(slm.level_index_min)
    if level is not None:
      return level
  return None


def _copy_ndarray(
    src_array: np.ndarray,
    src_x: int,
    src_y: int,
    width: int,
    height: int,
    dst_array: np.ndarray,
    dst_x: int,
    dst_y: int,
) -> None:
  """Copies a sub-array of an ndarray to another ndarray.

  Args:
    src_array: The source ndarray to copy the sub-array from.
    src_x: The X coordinate of the starting point to copy from, in the source
      array.
    src_y: The Y coordinate of the starting point to copy from, in the source
      array.
    width: The width of the sub-array to copy.
    height: The height of the sub-array to copy.
    dst_array: The destination ndarray to copy the sub-array into.
    dst_x: The X coordinate of the starting point to copy to, in the destination
      array.
    dst_y: The Y coordinate of the starting point to copy, in the destination
      array.

  Raises:
    SectionOutOfImageBoundsError if the sub-array to copy is out of scope of the
    source array or the destination array.
  """
  src_height, src_width = src_array.shape[:2]
  dst_height, dst_width = dst_array.shape[:2]
  if (
      src_x < 0
      or src_y < 0
      or src_x + width > src_width
      or src_y + height > src_height
  ):
    raise ez_wsi_errors.SectionOutOfImageBoundsError(
        'The sub-array to copy is out of the scope of the source array.'
    )
  if (
      dst_x < 0
      or dst_y < 0
      or dst_x + width > dst_width
      or dst_y + height > dst_height
  ):
    raise ez_wsi_errors.SectionOutOfImageBoundsError(
        'The sub-array to copy is out of the scope of the destination array.'
    )
  dst_array[dst_y : (dst_y + height), dst_x : (dst_x + width)] = src_array[
      src_y : (src_y + height), src_x : (src_x + width)
  ]


def create_icc_profile_transformation(
    source_image_icc_profile_bytes: bytes,
    dest_icc_profile: Union[bytes, ImageCms.core.CmsProfile],
    rendering_intent: ImageCms.Intent = ImageCms.Intent.PERCEPTUAL,
) -> Optional[ImageCms.ImageCmsTransform]:
  """Returns transformation to from pyramid colorspace to icc_profile.

  Args:
    source_image_icc_profile_bytes: Source image icc profile bytes.
    dest_icc_profile: ICC Profile to DICOM Pyramid imaging to.
    rendering_intent: Rendering intent to use in transformation.

  Returns:
    PIL.ImageCmsTransformation to transform pixel imaging or None.
  """
  if not source_image_icc_profile_bytes or not dest_icc_profile:
    return None
  dicom_input_profile = ImageCms.getOpenProfile(
      io.BytesIO(source_image_icc_profile_bytes)
  )
  if isinstance(dest_icc_profile, bytes):
    dest_icc_profile = ImageCms.getOpenProfile(io.BytesIO(dest_icc_profile))
  return ImageCms.buildTransform(
      dicom_input_profile,
      dest_icc_profile,
      _RGB,
      _RGB,
      renderingIntent=rendering_intent,
  )


def _get_level_series_path(
    level: Union[slide_level_map.Level, slide_level_map.ResizedLevel],
) -> dicom_path.Path:
  """Returns level series path."""
  if isinstance(level, slide_level_map.ResizedLevel):
    level = level.source_level
  return next(iter(level.instances.values())).path.GetSeriesPath()


class _DicomSeries(metaclass=abc.ABCMeta):
  """DICOM images."""

  def __init__(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
      path: dicom_path.Path,
      enable_client_slide_frame_decompression: bool = True,
      accession_number: Optional[str] = None,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ] = None,
  ):
    """Constructor.

    Args:
      dwi: The DicomWebInterface that has been configured to be able to access
        the series referenced by the input path.
      path: The path to a DICOM series object in a DICOMStore.
      enable_client_slide_frame_decompression: Set to True to enable client side
        frame decompression optimization (Recommended value = True); remove
        parameter following GG validation.
      accession_number: The accession_number of the slide.
      logging_factory: The factory that EZ WSI uses to construct a logging
        interface.
      slide_frame_cache: Initialize to use shared slide cache.
    """
    self._logger = None
    if logging_factory is None:
      self._logging_factory = ez_wsi_logging_factory.BasePythonLoggerFactory(
          ez_wsi_logging_factory.DEFAULT_EZ_WSI_PYTHON_LOGGER_NAME
      )
    else:
      self._logging_factory = logging_factory
    self._dwi = dwi
    self.path = path
    self.accession_number = accession_number
    self._slide_frame_cache = slide_frame_cache
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )

  @abc.abstractmethod
  def json_metadata_dict(
      self,
      level_subset: Optional[List[slide_level_map.Level]] = None,
      max_json_encoded_icc_profile_size: int = slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> Mapping[str, Any]:
    """Returns JSON dict metadata for the series."""

  def json_metadata(
      self,
      level_subset: Optional[List[slide_level_map.Level]] = None,
      max_json_encoded_icc_profile_size: int = slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> str:
    """Returns JSON encoded metadata for the series."""
    return json.dumps(
        self.json_metadata_dict(
            level_subset=level_subset,
            max_json_encoded_icc_profile_size=max_json_encoded_icc_profile_size,
        )
    )

  def get_credentials(self) -> google.auth.credentials.Credentials:
    return self._dwi.credentials()

  def get_credential_header(
      self,
      credential: Optional[google.auth.credentials.Credentials] = None,
  ) -> Dict[str, str]:
    """Returns credential header for retrieval of DICOM store."""
    headers = {}
    if credential is None:
      credential = self.get_credentials()
    else:
      credential_factory.refresh_credentials(credential)
    credential.apply(headers)
    return headers

  def get_patch_bounds_dicom_instance_frame_numbers(
      self,
      image_level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
      ],
      patch_bounds_list: List[PatchBounds],
  ) -> Mapping[str, List[int]]:
    """Returns Map[DICOM instances: frame numbers] that fall in patch bounds.

    Args:
      image_level: Level that patch represents.
      patch_bounds_list: List of PatchBounds to return frame indexes for.

    Returns:
      Mapping between DICOM instances, path, and list of frames numbers required
      to render patches.
    """
    if isinstance(image_level, slide_level_map.ResizedLevel):
      source_image_level = image_level.source_level
    else:
      source_image_level = image_level
    slide_instance_frame_map = {}
    indexes_required_for_inference = []
    for patch_bounds in patch_bounds_list:
      patch = DicomPatch(
          source_image_level,
          patch_bounds.x_origin,
          patch_bounds.y_origin,
          patch_bounds.width,
          patch_bounds.height,
          self,
          image_level,
      )
      # Frame indexes returns a list of indexes in the patch. Indexes
      # are returned in sorted order.
      indexes_required_for_inference.append(patch.frame_numbers())
    instance_frame_number_buffer = []
    instance = None
    # Use Heapq to merge pre-sorted lists into single sorted list
    # Result of heapq.merge can have duplicates
    for frame_number in heapq.merge(*indexes_required_for_inference):
      if (
          instance is None
          or instance.instance_frame_number_from_wholes_slide_frame_number(
              frame_number
          )
          > instance.frame_count
      ):
        if instance_frame_number_buffer:
          slide_instance_frame_map[str(instance.dicom_object.path)] = (
              instance_frame_number_buffer
          )
        instance = source_image_level.get_instance_by_frame(frame_number)
        if instance is None:
          instance_frame_number_buffer = []
          continue
        instance_frame_number_buffer = [
            instance.instance_frame_number_from_wholes_slide_frame_number(
                frame_number
            )
        ]
        continue
      instance_frame_number = (
          instance.instance_frame_number_from_wholes_slide_frame_number(
              frame_number
          )
      )
      if (
          instance_frame_number_buffer
          and instance_frame_number_buffer[-1] == instance_frame_number
      ):
        # remove duplicates
        continue
      instance_frame_number_buffer.append(instance_frame_number)
    if instance_frame_number_buffer:
      slide_instance_frame_map[str(instance.dicom_object.path)] = (
          instance_frame_number_buffer
      )
    return slide_instance_frame_map

  def preload_level_in_frame_cache(
      self,
      level: Union[slide_level_map.Level, slide_level_map.ResizedLevel],
      blocking: bool = True,
  ):
    """Preloads entire level into frame cache.

    Args:
      level: Level or resized level to load into frame cache.
      blocking: If method should block until loading is complete.

    Returns:
      None

    Raises:
      ez_wsi_errors.LevelNotFoundError: Level or resized level not found on
        DICOM Slide.
    """
    slide_frame_cache = self.slide_frame_cache
    if slide_frame_cache is None:
      return
    if isinstance(level, slide_level_map.ResizedLevel):
      level = level.source_level
    # get path of instance on the level.
    if _get_level_series_path(level) != self.path.GetSeriesPath():
      raise ez_wsi_errors.LevelNotFoundError(
          'Level path does not match DICOM slide path.'
      )
    slide_frame_cache.cache_whole_instance_in_memory(level, blocking)

  def preload_patches_in_frame_cache(
      self,
      patch_seq: Union[Sequence[DicomPatch], DicomPatch],
      blocking: bool = True,
      copy_from_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ] = None,
  ) -> None:
    """Pre-load sequence of DICOM Patches in frame cache.

    Args:
      patch_seq: patch or list of patches to load in frame cache.
      blocking: If method should block until loading is complete.
      copy_from_cache: Optional cache to copy frames from to avoid
        re-downloading.

    Returns:
      None

    Raises:
      ez_wsi_errors.LevelNotFoundError: Patch not found on DICOM Slide.
    """
    slide_frame_cache = self.slide_frame_cache
    if slide_frame_cache is None or not patch_seq:
      return
    if isinstance(patch_seq, DicomPatch):
      patch_seq = [patch_seq]
    level = None
    patches = []
    index = 0
    seq_len = len(patch_seq)
    slide_series_path = self.path.GetSeriesPath()
    while index < seq_len:
      patch = patch_seq[index]
      if level is None or level == patch.level:
        patches.append(patch)
        level = patch.level
        index += 1
        continue
      # get path of instance on the level.
      if _get_level_series_path(level) != slide_series_path:
        raise ez_wsi_errors.LevelNotFoundError(
            'Patch path does not match DICOM slide path.'
        )
      instance_frame_map = self.get_patch_bounds_dicom_instance_frame_numbers(
          level, [p.patch_bounds for p in patches]
      )
      slide_frame_cache.preload_instance_frame_numbers(
          instance_frame_map, copy_from_cache
      )
      level = None
      patches = []
    if patches:
      if _get_level_series_path(level) != slide_series_path:
        raise ez_wsi_errors.LevelNotFoundError(
            'Patch path does not match DICOM slide path.'
        )
      instance_frame_map = self.get_patch_bounds_dicom_instance_frame_numbers(
          level, [p.patch_bounds for p in patches]
      )
      slide_frame_cache.preload_instance_frame_numbers(
          instance_frame_map, copy_from_cache
      )
    if blocking:
      slide_frame_cache.block_until_frames_are_loaded()

  @property
  def logger(self) -> ez_wsi_logging_factory.AbstractLoggingInterface:
    if self._logger is None:
      self._logger = self._logging_factory.create_logger()
    return self._logger

  @property
  def slide_frame_cache(
      self,
  ) -> Optional[local_dicom_slide_cache.InMemoryDicomSlideCache]:
    """Returns DICOM slide frame cache used by slide."""
    return self._slide_frame_cache

  @slide_frame_cache.setter
  def slide_frame_cache(
      self,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ],
  ) -> None:
    """Sets DICOM slide frame cache used by slide.

    Shared cache's configured using max_cache_frame_memory_lru_cache_size_bytes
    can be used to limit total cache memory utilization across multiple stores.
    It is not recommended to share non-LRU frame caches.

    Args:
      slide_frame_cache: Reference to slide frame cache.
    """
    self._slide_frame_cache = slide_frame_cache

  def init_slide_frame_cache(
      self,
      max_cache_frame_memory_lru_cache_size_bytes: Optional[int] = None,
      number_of_frames_to_read: int = local_dicom_slide_cache.DEFAULT_NUMBER_OF_FRAMES_TO_READ_ON_CACHE_MISS,
      max_instance_number_of_frames_to_prefer_whole_instance_download: int = local_dicom_slide_cache.MAX_INSTANCE_NUMBER_OF_FRAMES_TO_PREFER_WHOLE_INSTANCE_DOWNLOAD,
      optimization_hint: local_dicom_slide_cache_types.CacheConfigOptimizationHint = local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
  ) -> local_dicom_slide_cache.InMemoryDicomSlideCache:
    """Initializes DICOM slide frame cache.

    Args:
      max_cache_frame_memory_lru_cache_size_bytes: Maximum size of cache in
        bytes.  Ideally should be in hundreds of megabyts-to-gigabyte size. If
        None, no limit to size.
      number_of_frames_to_read: Number of frames to read on cache miss.
      max_instance_number_of_frames_to_prefer_whole_instance_download: Max
        number of frames to prefer downloading whole instances over retrieving
        frames in batch (Typically faster for small instances e.g. < 10,0000).
        Optimal threshold will depend on average size of instance frame data and
        the size of non frame instance metadata.
      optimization_hint: Optimize cache to minimize data latency or total
        queries to the DICOM store.

    Returns:
      Instance of frame cache.
    """
    self._slide_frame_cache = local_dicom_slide_cache.InMemoryDicomSlideCache(
        credential_factory=self._dwi.credential_factory,
        max_cache_frame_memory_lru_cache_size_bytes=max_cache_frame_memory_lru_cache_size_bytes,
        number_of_frames_to_read=number_of_frames_to_read,
        max_instance_number_of_frames_to_prefer_whole_instance_download=max_instance_number_of_frames_to_prefer_whole_instance_download,
        optimization_hint=optimization_hint,
        logging_factory=self._logging_factory,
    )
    return self._slide_frame_cache

  def remove_slide_frame_cache(self) -> None:
    """Removes slide frame cache."""
    self._slide_frame_cache = None

  @property
  def dwi(self) -> dicom_web_interface.DicomWebInterface:
    return self._dwi

  def _get_cached_frame_bytes(
      self,
      instance: slide_level_map.Instance,
      frame_number: int,
  ) -> Optional[bytes]:
    """Returns frame bytes from frame cache if possible.

    Args:
      instance: Instance to return frame from.
      frame_number: Instance frame number to return.

    Returns:
      Frame bytes or None.
    """
    if (
        self._slide_frame_cache is None
        or not is_client_side_pixel_decoding_supported(
            instance.dicom_object.get_value(tags.TRANSFER_SYNTAX_UID)
        )
    ):
      return None
    return self._slide_frame_cache.get_frame(
        instance.dicom_object.path,
        instance.frame_count,
        frame_number,
    )

  def _get_frame_bytes_from_server(
      self,
      instance: slide_level_map.Instance,
      instance_frame_index: int,
      transcoding: dicom_web_interface.TranscodeDicomFrame,
  ) -> bytes:
    """Returns frame bytes from server.

    Args:
      instance: DICOM instance.
      instance_frame_index: Frame index to retrieve.
      transcoding: How to transcode DICOM frames.

    Returns:
      Frame bytes.
    """
    instance_path = instance.dicom_object.path
    cache_key = (
        f'i:{str(instance_path)} f:{instance_frame_index} t:{transcoding.value}'
    )
    if self._slide_frame_cache is not None:
      frame_raw_bytes = (
          self._slide_frame_cache.get_cached_externally_acquired_bytes(
              cache_key
          )
      )
      if frame_raw_bytes is not None:
        return frame_raw_bytes
    frame_raw_bytes = self.dwi.get_frame_image(
        instance_path, instance_frame_index, transcoding
    )
    if self._slide_frame_cache is not None:
      self._slide_frame_cache.cache_externally_acquired_bytes(
          cache_key, frame_raw_bytes
      )
    return frame_raw_bytes

  def _get_frame_client_transcoding(
      self,
      instance: slide_level_map.Instance,
      frame_number: int,
  ) -> Optional[np.ndarray]:
    """Returns DICOM Frame using DICOM server, transcodes to raw on server.

    Args:
      instance: DICOM instance within level to return frame from.
      frame_number: Frame number within the instance to return.

    Returns:
      Tuple[numpy array containing frame data, bool indicating if data should
      be cached True in LRU or is already cached within the level]
    """
    compressed_bytes = self._get_cached_frame_bytes(
        instance,
        frame_number,
    )
    if compressed_bytes is None:
      compressed_bytes = self._get_frame_bytes_from_server(
          instance,
          frame_number,
          dicom_web_interface.TranscodeDicomFrame.DO_NOT_TRANSCODE,
      )
    return dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        compressed_bytes,
        instance.dicom_object.get_value(tags.TRANSFER_SYNTAX_UID),
    )

  def _get_frame_server_transcoding(
      self,
      level: slide_level_map.Level,
      instance: slide_level_map.Instance,
      frame_number: int,
  ) -> np.ndarray:
    """Returns DICOM Frame using DICOM server, transcodes to raw on server.

    Args:
      level: WSI pyramid level.
      instance: DICOM instance within level to return frame from.
      frame_number: Frame number within the instance to return.

    Returns:
      Tuple[numpy array containing frame data, bool indicating if data should
      be cached True in LRU or is already cached within the level]
    """
    # Only use cache if frame bytes are stored a uncompressed little endian.
    if (
        level.transfer_syntax_uid
        not in _SUPPORTED_CLIENT_SIDE_DECODING_RAW_TRANSFER_SYNTAXS
    ):
      frame_raw_bytes = None
    else:
      frame_raw_bytes = self._get_cached_frame_bytes(
          instance,
          frame_number,
      )
    if frame_raw_bytes is None:
      frame_raw_bytes = self._get_frame_bytes_from_server(
          instance,
          frame_number,
          dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
      )
    return np.frombuffer(frame_raw_bytes, level.pixel_format).reshape(
        (level.frame_height, level.frame_width, level.samples_per_pixel)
    )

  def get_frame(
      self,
      level: slide_level_map.Level,
      frame_number: int,
  ) -> Optional[Frame]:
    """Gets a frame at a specific pixel_spacing in mm.

    The DICOMWeb API serves image pixels by the unit of frames. Frames have
    fixed size (width and height). Call get_patch() instead if you want to get
    an image patch at a specific loation, or with a specific dimension.

    The function utilizes a LRUCache to cache the most recent used frames.

    Args:
      level: source pyramid level for frame imaging if not defined pyramid level
        is deriveved using pixel_spacing.
      frame_number: The frame number to be fetched. The frames are stored in
        arrays with 1-based indexing.

    Returns:
      Returns the requested frame if exists, None otherwise.

    Raises:
      InputFrameNumberOutOfRangeError if the input frame_number is
      out of range.
    """
    if (
        frame_number < level.frame_number_min
        or frame_number > level.frame_number_max
    ):
      raise ez_wsi_errors.InputFrameNumberOutOfRangeError(
          f'frame_number value [{frame_number}] is out of range: '
          f'[{level.frame_number_min}, {level.frame_number_max}]'
      )
    instance = level.get_instance_by_frame(frame_number)
    if instance is None:
      # A frame may not exist in DICOMStore if it contains no tissue pixels.
      return None
    instance_frame_number = (
        instance.instance_frame_number_from_wholes_slide_frame_number(
            frame_number
        )
    )
    pos_x, pos_y = level.get_frame_position(frame_number)
    frame_ndarray = None
    if (
        self._enable_client_slide_frame_decompression
        and dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
            level.transfer_syntax_uid
        )
    ):
      frame_ndarray = self._get_frame_client_transcoding(
          instance, instance_frame_number
      )
      # if frame_ndarray == None unable to decode bytes, likely cause is actual
      # pixel encoding does not match DICOM transfer syntax. Fail over to server
      # side transcoding.
    if frame_ndarray is None:
      frame_ndarray = self._get_frame_server_transcoding(
          level, instance, instance_frame_number
      )
    frame = Frame(
        x_origin=pos_x,
        y_origin=pos_y,
        width=level.frame_width,
        height=level.frame_height,
        image_np=frame_ndarray,
    )
    return frame

  def get_image(
      self,
      pixel_spacing: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
      ],
  ) -> DicomImage:
    """Gets an image from a specific pixel spacing."""
    return DicomImage(pixel_spacing, self)

  def get_patch(
      self,
      level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
      ],
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: bool = False,
  ) -> DicomPatch:
    """Gets a patch from a specific slide level.

    The area of a patch is defined by its position(x, y) and its dimension
    (width, height). The position of a patch corresponds to the first pixel in
    the patch and has the smallest coordinates. All coordinates(x, y, width and
    height) are defined in the pixel space of the requested pixel spacing.

    This routine creates the requested patch on-the-fly, by sampling image
    pixels from all frames that are overlapped with the patch.

    Args:
      level: Level to generate patch from.
      x: The X coordinate of the patch position in level.
      y: The Y coordinate of the patch position in level.
      width: The width of the patch in level.
      height: The height of the patch in level.
      require_fully_in_source_image: Require patch dimensions fully exist in
        level.

    Returns:
      Returns the requested patch.
    """
    if isinstance(level, slide_level_map.ResizedLevel):
      return DicomPatch(
          level.source_level,
          x,
          y,
          width,
          height,
          self,
          level,
          require_fully_in_source_image=require_fully_in_source_image,
      )
    return DicomPatch(
        level,
        x,
        y,
        width,
        height,
        self,
        level,
        require_fully_in_source_image=require_fully_in_source_image,
    )

  @abc.abstractmethod
  def get_level_by_index(
      self, index: slide_level_map.LevelIndexType
  ) -> Optional[slide_level_map.Level]:
    """Returns the level by requested level index."""

  def has_level(self, level: slide_level_map.Level) -> bool:
    return level is self.get_level_by_index(level.level_index)

  @abc.abstractmethod
  def are_instances_concatenated(self, instance_uids: list[str]) -> bool:
    """Returns True if the instances provided are concatenated."""

  @property
  @abc.abstractmethod
  def all_levels(self) -> Iterator[slide_level_map.Level]:
    """return all levels of series."""

  def get_instance_level(
      self, sop_instance_uid: str
  ) -> Optional[slide_level_map.Level]:
    for level in self.all_levels:
      for instance in level.instances.values():
        if (
            instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
            == sop_instance_uid
        ):
          return level
    return None

  def _filter_dicom_object(
      self,
      instance: dicom_web_interface.DicomObject,
      slide_instances_uid_set: set[str],
      filter_by_annotation_iod: Optional[str],
      filter_by_operator_id: Optional[str],
  ) -> bool:
    """Filters instances by annotation IOD, referenced series UID, and operator ID."""

    ds = pydicom.Dataset.from_json(instance.dicom_tags)

    # Filter by provided IOD.
    if filter_by_annotation_iod and ds.SOPClassUID != filter_by_annotation_iod:
      return False

    # Check if instance is referencing the slide's series or one if its
    # instances.
    reference_found = False

    if ds.SeriesInstanceUID == self.path.series_uid:
      reference_found = True

    try:
      # https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/common-instance-reference/00081115/0020000e
      if (
          not reference_found
          and ds.ReferencedSeriesSequence[0].SeriesInstanceUID
          == self.path.series_uid
      ):
        reference_found = True
    except AttributeError:
      pass
    try:
      # https://dicom.innolitics.com/ciods/microscopy-bulk-simple-annotations/microscopy-bulk-simple-annotations/00081140/00081155
      if (
          not reference_found
          and ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
          in slide_instances_uid_set
      ):
        reference_found = True
    except AttributeError:
      pass

    if not reference_found:
      return False

    # Filter by operator ID.
    try:
      if not filter_by_operator_id:
        return True
      for code in ds.OperatorIdentificationSequence:
        for code in code.PersonIdentificationCodeSequence:
          if code.LongCodeValue == filter_by_operator_id:
            return True
    except AttributeError:
      pass

    return False

  def _find_annotation_instances(
      self,
      levels: Iterator[slide_level_map.Level],
      annotation_dicom_store: Optional[dicom_path.Path] = None,
      filter_by_annotation_iod: Optional[
          str
      ] = MICROSCOPY_BULK_SIMPLE_ANNOTATIONS_STORAGE,
      filter_by_operator_id: Optional[str] = None,
  ) -> Iterator[dicom_path.Path]:
    """Returns iterator that contains all of a slide's annotation instances.

    The annotation instances much either reference the slide's series UID or
    have the same series UID as the slide.

    Args:
      levels: Images to search for annotation instances.
      annotation_dicom_store: The DICOM store to search for annotation
        instances. We assume here that the study UID stays the same between the
        slide and annotation DICOM stores.
      filter_by_annotation_iod: The annotation IOD to filter by. Default is
        Microscopy-Bulk-Simple-Annotations.
        https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_b.5.html
      filter_by_operator_id: The operator ID to filter by. This searches the
        Operator Identification Sequence long code value for the provided ID.
        https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/general-series/00081072/00401101/00080119

    Returns:
      An iterator that contains all of a slide's annotation instances.
    """
    if annotation_dicom_store is None:
      annotation_dicom_store = self.path.GetStorePath()

    if annotation_dicom_store is not None:
      # Update study path to match annotation store.
      study_path = dicom_path.FromPath(
          annotation_dicom_store,
          study_uid=self.path.study_uid,
          series_uid='',
          instance_uid='',
      )
    else:
      # Assume annotations are stored in the same dicom store as the images.
      study_path = self.path.GetStudyPath()

    # Get all instances in the study in the provided DICOM store.
    # Make sure instance is an annotation modality.
    # https://www.dicomlibrary.com/dicom/modality
    dicom_instances = self.dwi.get_instances(study_path, modality='ANN')

    # Get all instances of this slide.
    slide_instances_set_uid = set[str]()
    for level in levels:
      for instance in level.instances.values():
        slide_instances_set_uid.add(
            instance.dicom_object.get_value(tags.SOP_INSTANCE_UID)
        )

    # Keep only relevant instances and transform to path iterator.
    return iter(
        dicom_path.FromPath(
            study_path,
            study_uid=instance.get_value(tags.STUDY_INSTANCE_UID),
            series_uid=instance.get_value(tags.SERIES_INSTANCE_UID),
            instance_uid=instance.get_value(tags.SOP_INSTANCE_UID),
        )
        for instance in dicom_instances
        if self._filter_dicom_object(
            instance,
            slide_instances_set_uid,
            filter_by_annotation_iod,
            filter_by_operator_id,
        )
    )


class DicomSlide(_DicomSeries):
  """Represents a DICOM pathology slide stored in a DICOMStore."""

  def __init__(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
      path: dicom_path.Path,
      enable_client_slide_frame_decompression: bool = True,
      accession_number: Optional[str] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ] = None,
      json_metadata: Union[str, Mapping[str, Any]] = '',
      instances: Optional[Collection[dicom_web_interface.DicomObject]] = None,
  ):
    """Constructor.

    Args:
      dwi: The DicomWebInterface that has been configured to be able to access
        the series referenced by the input path.
      path: The path to a DICOM series object in a DICOMStore.
      enable_client_slide_frame_decompression: Set to True to enable client side
        frame decompression optimization (Recommended value = True); remove
        parameter following GG validation.
      accession_number: The accession_number of the slide.
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.
      logging_factory: The factory that EZ WSI uses to construct a logging
        interface.
      slide_frame_cache: Initialize to use shared slide cache.
      json_metadata: Optional json formatted slide level metadata.
      instances: Optional list of instances to use to initialize the slide.
    """
    path = path.GetSeriesPath()
    super().__init__(
        dwi,
        path,
        enable_client_slide_frame_decompression,
        accession_number,
        logging_factory,
        slide_frame_cache,
    )
    if json_metadata:
      # if metadata is defined convert string to json.
      if isinstance(json_metadata, str):
        try:
          json_metadata = json.loads(json_metadata)
        except json.JSONDecodeError as exp:
          raise ez_wsi_errors.InvalidSlideJsonMetadataError(
              'Error decoding JSON metadata.'
          ) from exp
    if json_metadata:  # Ignore empty JSON metadata.
      try:
        if str(path) != str(
            dicom_path.Path.from_dict(json_metadata[_SLIDE_PATH])
        ):
          raise ez_wsi_errors.SlidePathDoesNotMatchJsonMetadataError(
              'Slide path does not match slide path in json metadata.'
          )
        self._level_map = slide_level_map.SlideLevelMap.create_from_json(
            json_metadata[LEVEL_MAP],
            pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
        )
      except (TypeError, IndexError, KeyError) as exp:
        raise ez_wsi_errors.InvalidSlideJsonMetadataError(
            'Incorrectly formatted JSON metadata.'
        ) from exp
    else:
      self._level_map = slide_level_map.SlideLevelMap(
          dwi.get_instances(path) if instances is None else instances,
          pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
      )
    self._native_level = _get_native_level(self._level_map)

  def __copy__(self) -> DicomSlide:
    instance = DicomSlide.__new__(DicomSlide)
    vars(instance).update(vars(self))
    instance._level_map = copy.copy(self._level_map)
    return instance

  @property
  def native_level(self) -> slide_level_map.Level:
    if self._native_level is None:
      raise ez_wsi_errors.LevelNotFoundError(
          'Slide does not define pyramid imaging.'
      )
    return self._native_level

  @property
  def total_pixel_matrix_columns(self) -> int:
    return self.native_level.width

  @property
  def total_pixel_matrix_rows(self) -> int:
    return self.native_level.height

  @property
  def pixel_format(self) -> np.dtype:
    return self.native_level.pixel_format

  @property
  def native_pixel_spacing(self) -> pixel_spacing_module.PixelSpacing:
    return self.native_level.pixel_spacing

  def set_icc_profile_bytes(self, icc_profile_bytes: bytes) -> None:
    """Sets ICC Profile bytes for pyramid."""
    self._level_map.set_icc_profile_bytes(icc_profile_bytes)

  def get_json_encoded_icc_profile_size(self) -> int:
    return self._level_map.get_json_encoded_icc_profile_size()

  def get_icc_profile_bytes(self) -> bytes:
    """Returns ICC Profile bytes for pyramid."""
    return self._level_map.get_icc_profile_bytes(self._dwi)

  def are_instances_concatenated(self, instance_uids: list[str]) -> bool:
    """Returns True if the instances provided are concatenated.

    This also indicates if the instances are of the same pixel spacing.

    Args:
      instance_uids: A list of SOP Instance UIDs to check

    Returns:
      True if the instances are concatenated or if only one instance uid is
        provided. Otherwise returns False.
    """
    return self._level_map.are_instances_concatenated(instance_uids)

  def get_patch_bounds_dicom_instance_frame_numbers(
      self,
      image_level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
          pixel_spacing_module.PixelSpacing,
      ],
      patch_bounds_list: List[PatchBounds],
  ) -> Mapping[str, List[int]]:
    """Returns Map[DICOM instances: frame numbers] that fall in patch bounds.

    Args:
      image_level: Level that patch represents.
      patch_bounds_list: List of PatchBounds to return frame indexes for.

    Returns:
      Mapping between DICOM instances, path, and list of frames numbers required
      to render patches.
    """
    if isinstance(image_level, pixel_spacing_module.PixelSpacing):
      return super().get_patch_bounds_dicom_instance_frame_numbers(
          _pixel_spacing_to_level(self, image_level), patch_bounds_list
      )
    else:
      return super().get_patch_bounds_dicom_instance_frame_numbers(
          image_level, patch_bounds_list
      )

  def create_icc_profile_transformation(
      self,
      icc_profile: Union[bytes, ImageCms.core.CmsProfile],
      rendering_intent: ImageCms.Intent = ImageCms.Intent.PERCEPTUAL,
  ) -> Optional[ImageCms.ImageCmsTransform]:
    """Returns transformation to from pyramid colorspace to icc_profile.

    Args:
      icc_profile: ICC Profile to DICOM Pyramid imaging to.
      rendering_intent: Rendering intent to use in transformation.

    Returns:
      PIL.ImageCmsTransformation to transform pixel imaging or None.
    """
    return create_icc_profile_transformation(
        self.get_icc_profile_bytes(), icc_profile, rendering_intent
    )

  def get_image(
      self,
      pixel_spacing: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
          pixel_spacing_module.PixelSpacing,
      ],
  ) -> DicomImage:
    """Gets an image from a specific pixel spacing."""
    if isinstance(pixel_spacing, pixel_spacing_module.PixelSpacing):
      return DicomImage(_pixel_spacing_to_level(self, pixel_spacing), self)
    return DicomImage(pixel_spacing, self)

  def get_patch(
      self,
      level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
          pixel_spacing_module.PixelSpacing,
      ],
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: bool = False,
  ) -> DicomPatch:
    if isinstance(level, pixel_spacing_module.PixelSpacing):
      level = _pixel_spacing_to_level(self, level)
    return super().get_patch(
        level,
        x,
        y,
        width,
        height,
        require_fully_in_source_image=require_fully_in_source_image,
    )

  def get_frame(
      self,
      level: Union[slide_level_map.Level, pixel_spacing_module.PixelSpacing],
      frame_number: int,
  ) -> Optional[Frame]:
    """Gets a frame at a specific pixel_spacing in mm.

    The DICOMWeb API serves image pixels by the unit of frames. Frames have
    fixed size (width and height). Call get_patch() instead if you want to get
    an image patch at a specific loation, or with a specific dimension.

    The function utilizes a LRUCache to cache the most recent used frames.

    Args:
      level: source pyramid level for frame imaging if not defined pyramid level
        is deriveved using pixel_spacing.
      frame_number: The frame number to be fetched. The frames are stored in
        arrays with 1-based indexing.

    Returns:
      Returns the requested frame if exists, None otherwise.

    Raises:
      InputFrameNumberOutOfRangeError if the input frame_number is
      out of range.
    """
    if isinstance(level, pixel_spacing_module.PixelSpacing):
      return super().get_frame(
          _pixel_spacing_to_level(self, level), frame_number
      )
    return super().get_frame(level, frame_number)

  def json_metadata_dict(
      self,
      level_subset: Optional[List[slide_level_map.Level]] = None,
      max_json_encoded_icc_profile_size: int = slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> Mapping[str, Any]:
    return {
        _SOP_CLASS_UID: (
            slide_level_map.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
        ),
        _SLIDE_PATH: self.path.to_dict(),
        LEVEL_MAP: self._level_map.to_dict(
            level_subset=level_subset,
            max_json_encoded_icc_profile_size=max_json_encoded_icc_profile_size,
        ),
    }

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, DicomSlide):
      return str(self.path) == str(other.path)
    return False

  @property
  def all_pixel_spacing_mms(self) -> list[float]:
    """Lists all Pixel Spacings in mm in the DicomSlide."""
    return [
        level.pixel_spacing.pixel_spacing_mm
        for level in self._level_map.level_map.values()
        if level.pixel_spacing.is_defined
    ]

  def get_level_by_pixel_spacing(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
      relative_pixel_spacing_equality_threshold: float = slide_level_map.MAX_LEVEL_DIST,
      maximum_downsample: float = 0.0,
  ) -> Optional[slide_level_map.Level]:
    """Gets the level corresponding to the input pixel spacing.

    Args:
      pixel_spacing: The pixel spacing to use for level lookup.
      relative_pixel_spacing_equality_threshold: Maximum relative difference in
        (pyramid / level pixel spacing / desired pixel spacing) at which pixel
        spacing should be considered equilvalent.  E.g., (value of 0.25 =
        pyramid levels with pixels spacings that are 25% smaller - 25% larger
        are considered equlivalent). Pathology imaging is typically represented
        as a pyramid with pyramid levels being 200% or greater relative
        magnifications of each other.
      maximum_downsample: Maximum degree to which it is acceptable to downsample
        pixel imaging to acchieve a desired target pixel spacing. Only used when
        source imaging pixel spacing is < and != to desired target pixel
        spacing.

    Returns:
      The level corresponding to the input pixel spacing. None if the requested
      pixel spacing does not exist.
    """
    # Converts pixel spacing returned by Magnification.NominalPixelSize() from
    # micrometers to millimeters.
    return self._level_map.get_level_by_pixel_spacing(
        pixel_spacing.pixel_spacing_mm,
        relative_pixel_spacing_equality_threshold=relative_pixel_spacing_equality_threshold,
        maximum_downsample=maximum_downsample,
    )

  def get_closest_level_with_pixel_spacing_equal_or_less_than_target(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
  ) -> Optional[slide_level_map.Level]:
    """Gets the level corresponding to the input pixel spacing.

    Args:
      pixel_spacing: The pixel spacing to use for level lookup.

    Returns:
      The level corresponding to the input pixel spacing. None if the requested
      pixel spacing does not exist.
    """
    # Converts pixel spacing returned by Magnification.NominalPixelSize() from
    # micrometers to millimeters.
    return self._level_map.get_levels_with_closest_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    ).equal_or_smaller_pixel_spacing

  def get_closest_level_with_pixel_spacing_greater_than_target(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
  ) -> Optional[slide_level_map.Level]:
    """Gets the level corresponding to the input pixel spacing.

    Args:
      pixel_spacing: The pixel spacing to use for level lookup.

    Returns:
      The level corresponding to the input pixel spacing. None if the requested
      pixel spacing does not exist.
    """
    # Converts pixel spacing returned by Magnification.NominalPixelSize() from
    # micrometers to millimeters.
    return self._level_map.get_levels_with_closest_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    ).greater_pixel_spacing

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
    return self._level_map.get_instance_pixel_spacing(instance_uid)

  @property
  def thumbnail(self) -> Optional[slide_level_map.Level]:
    return self._level_map.thumbnail

  @property
  def label(self) -> Optional[slide_level_map.Level]:
    return self._level_map.label

  @property
  def overview(self) -> Optional[slide_level_map.Level]:
    return self._level_map.overview

  @property
  def levels(self) -> Iterator[slide_level_map.Level]:
    """Returns iterator that contains all of a slide's DICOM Levels."""
    return iter(self._level_map.level_map.values())

  def get_level_by_index(
      self, index: slide_level_map.LevelIndexType
  ) -> Optional[slide_level_map.Level]:
    return self._level_map.get_level(index)

  @property
  def wsi_label_overview_thumbnail_levels(
      self,
  ) -> Iterator[slide_level_map.Level]:
    return iter([
        lvl
        for lvl in [self.thumbnail, self.label, self.overview]
        if lvl is not None
    ])

  @property
  def all_levels(self) -> Iterator[slide_level_map.Level]:
    return itertools.chain(
        self.levels,
        self.wsi_label_overview_thumbnail_levels,
    )

  def _filter_dicom_object(
      self,
      instance: dicom_web_interface.DicomObject,
      slide_instances_uid_set: set[str],
      filter_by_annotation_iod: Optional[str],
      filter_by_operator_id: Optional[str],
  ) -> bool:
    """Filters instances by annotation IOD, referenced series UID, and operator ID."""

    ds = pydicom.Dataset.from_json(instance.dicom_tags)

    # Filter by provided IOD.
    if filter_by_annotation_iod and ds.SOPClassUID != filter_by_annotation_iod:
      return False

    # Check if instance is referencing the slide's series or one if its
    # instances.
    reference_found = False

    if ds.SeriesInstanceUID == self.path.series_uid:
      reference_found = True

    try:
      # https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/common-instance-reference/00081115/0020000e
      if (
          not reference_found
          and ds.ReferencedSeriesSequence[0].SeriesInstanceUID
          == self.path.series_uid
      ):
        reference_found = True
    except AttributeError:
      pass
    try:
      # https://dicom.innolitics.com/ciods/microscopy-bulk-simple-annotations/microscopy-bulk-simple-annotations/00081140/00081155
      if (
          not reference_found
          and ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID
          in slide_instances_uid_set
      ):
        reference_found = True
    except AttributeError:
      pass

    if not reference_found:
      return False

    # Filter by operator ID.
    try:
      if not filter_by_operator_id:
        return True
      for code in ds.OperatorIdentificationSequence:
        for code in code.PersonIdentificationCodeSequence:
          if code.LongCodeValue == filter_by_operator_id:
            return True
    except AttributeError:
      pass

    return False

  def find_annotation_instances(
      self,
      annotation_dicom_store: Optional[dicom_path.Path] = None,
      filter_by_annotation_iod: Optional[
          str
      ] = MICROSCOPY_BULK_SIMPLE_ANNOTATIONS_STORAGE,
      filter_by_operator_id: Optional[str] = None,
  ) -> Iterator[dicom_path.Path]:
    """Returns iterator that contains all of a slide's annotation instances.

    The annotation instances much either reference the slide's series UID or
    have the same series UID as the slide.

    Args:
      annotation_dicom_store: The DICOM store to search for annotation
        instances. We assume here that the study/series UID stays the same
        between the slide and annotation DICOM stores.
      filter_by_annotation_iod: The annotation IOD to filter by. Default is
        Microscopy-Bulk-Simple-Annotations.
        https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_b.5.html
      filter_by_operator_id: The operator ID to filter by. This searches the
        Operator Identification Sequence long code value for the provided ID.
        https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/general-series/00081072/00401101/00080119

    Returns:
      An iterator that contains all of a slide's annotation instances.
    """
    return self._find_annotation_instances(
        self.levels,
        annotation_dicom_store,
        filter_by_annotation_iod,
        filter_by_operator_id,
    )


class DicomMicroscopeImage(_DicomSeries):
  """Represents a Non-tiled DICOM pathology slide stored in a DICOMStore."""

  def __init__(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
      path: dicom_path.Path,
      enable_client_slide_frame_decompression: bool = True,
      accession_number: Optional[str] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ] = None,
      json_metadata: Union[str, Mapping[str, Any]] = '',
      instances: Optional[Collection[dicom_web_interface.DicomObject]] = None,
  ):
    """Constructor.

    Args:
      dwi: The DicomWebInterface that has been configured to be able to access
        the series referenced by the input path.
      path: The path to a DICOM series object in a DICOMStore.
      enable_client_slide_frame_decompression: Set to True to enable client side
        frame decompression optimization (Recommended value = True); remove
        parameter following GG validation.
      accession_number: The accession_number of the slide.
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.
      logging_factory: The factory that EZ WSI uses to construct a logging
        interface.
      slide_frame_cache: Initialize to use shared slide cache.
      json_metadata: Optional json formatted slide level metadata.
      instances: Optional list of instances to use to initialize the slide.
    """
    super().__init__(
        dwi,
        path,
        enable_client_slide_frame_decompression,
        accession_number,
        logging_factory,
        slide_frame_cache,
    )
    if json_metadata:  # Ignore empty JSON metadata.
      # if metadata is defined convert string to json.
      if isinstance(json_metadata, str):
        try:
          json_metadata = json.loads(json_metadata)
        except json.JSONDecodeError as exp:
          raise ez_wsi_errors.InvalidSlideJsonMetadataError(
              'Error decoding JSON metadata.'
          ) from exp
      try:
        if str(path) != str(
            dicom_path.Path.from_dict(json_metadata[_SLIDE_PATH])
        ):
          raise ez_wsi_errors.SlidePathDoesNotMatchJsonMetadataError(
              'Slide path does not match slide path in json metadata.'
          )
        self._non_tiled_levels = (
            slide_level_map.UntiledImageMap.create_from_json(
                json_metadata[UNTILED_MICROSCOPE_IMAGE_MAP],
                pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
            )
        )
      except (TypeError, IndexError, KeyError) as exp:
        raise ez_wsi_errors.InvalidSlideJsonMetadataError(
            'Incorrectly formatted JSON metadata.'
        ) from exp
    else:
      self._non_tiled_levels = slide_level_map.UntiledImageMap(
          dwi.get_instances(path) if instances is None else instances,
          pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
      )

  def __copy__(self) -> DicomMicroscopeImage:
    instance = DicomMicroscopeImage.__new__(DicomMicroscopeImage)
    vars(instance).update(vars(self))
    instance._non_tiled_levels = copy.copy(self._non_tiled_levels)
    return instance

  def json_metadata_dict(
      self,
      level_subset: Optional[List[slide_level_map.Level]] = None,
      max_json_encoded_icc_profile_size: int = slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES,
  ) -> Mapping[str, Any]:
    """Returns json formatted slide level metadata.

    Args:
      level_subset: List of levels defined on the slide to selectively export
        metadata for.
      max_json_encoded_icc_profile_size: Max size of JSON to return; not used.

    Returns
      Json formatted metadata.

    Raises:
      ez_wsi_errors.LevelNotFoundError: Level not defined on Slide.
    """
    del max_json_encoded_icc_profile_size
    return {
        _SOP_CLASS_UID: slide_level_map.VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID,
        _SLIDE_PATH: self.path.to_dict(),
        UNTILED_MICROSCOPE_IMAGE_MAP: self._non_tiled_levels.to_dict(
            level_subset=level_subset,
        ),
    }

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, DicomMicroscopeImage):
      return str(self.path) == str(other.path)
    return False

  def are_instances_concatenated(self, instance_uids: list[str]) -> bool:
    del instance_uids
    return False

  @property
  def levels(self) -> Iterator[slide_level_map.Level]:
    """Returns iterator that contains all of a slide's DICOM Levels."""
    return iter(self._non_tiled_levels.untiled_image_map.values())

  def get_level_by_index(
      self, index: slide_level_map.LevelIndexType
  ) -> Optional[slide_level_map.Level]:
    return self._non_tiled_levels.untiled_image_map.get(index, None)

  @property
  def all_levels(self) -> Iterator[slide_level_map.Level]:
    return self.levels

  def get_level_icc_profile_bytes(
      self, level: Union[slide_level_map.Level, slide_level_map.ResizedLevel]
  ) -> bytes:
    """Returns ICC Profile bytes for pyramid."""
    if isinstance(level, slide_level_map.ResizedLevel):
      level = level.source_level
    return self._non_tiled_levels.get_level_icc_profile_bytes(level, self._dwi)

  def find_annotation_instances(
      self,
      level: slide_level_map.Level,
      annotation_dicom_store: Optional[dicom_path.Path] = None,
      filter_by_annotation_iod: Optional[
          str
      ] = MICROSCOPY_BULK_SIMPLE_ANNOTATIONS_STORAGE,
      filter_by_operator_id: Optional[str] = None,
  ) -> Iterator[dicom_path.Path]:
    """Returns iterator that contains all of a slide's annotation instances.

    The annotation instances much either reference the slide's series UID or
    have the same series UID as the slide.

    Args:
      level: The image to search for annotation instances.
      annotation_dicom_store: The DICOM store to search for annotation
        instances. We assume here that the study/series UID stays the same
        between the slide and annotation DICOM stores.
      filter_by_annotation_iod: The annotation IOD to filter by. Default is
        Microscopy-Bulk-Simple-Annotations.
        https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_b.5.html
      filter_by_operator_id: The operator ID to filter by. This searches the
        Operator Identification Sequence long code value for the provided ID.
        https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/general-series/00081072/00401101/00080119

    Returns:
      An iterator that contains all of a slide's annotation instances.
    """
    return self._find_annotation_instances(
        iter([level]),
        annotation_dicom_store,
        filter_by_annotation_iod,
        filter_by_operator_id,
    )


def _add_sop_class_uid(
    sop_class_uid: str, sop_class_uid_found: Set[str]
) -> None:
  if (
      sop_class_uid
      == slide_level_map.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
      or sop_class_uid in slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID
  ):
    sop_class_uid_found.add(sop_class_uid)


def _load_series(
    dwi: dicom_web_interface.DicomWebInterface,
    path: dicom_path.Path,
    enable_client_slide_frame_decompression: bool,
    json_metadata: Union[str, Mapping[str, Any]],
    instances: Optional[Collection[dicom_web_interface.DicomObject]],
    sop_class_uid_found: Set[str],
    pixel_spacing_diff_tolerance: float,
    logging_factory: Optional[
        ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
    ],
) -> Tuple[Optional[DicomSlide], Optional[DicomMicroscopeImage]]:
  """Loads DicomSlide and/or DicomMicroscopeImage defined on a series."""
  if (
      slide_level_map.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
      in sop_class_uid_found
  ):
    vl_whole_slide_image = DicomSlide(
        dwi,
        path,
        enable_client_slide_frame_decompression,
        json_metadata=json_metadata,
        instances=instances,
        pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
        logging_factory=logging_factory,
    )
  else:
    vl_whole_slide_image = None
  if slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID.intersection(
      sop_class_uid_found
  ):
    microscope_image = DicomMicroscopeImage(
        dwi,
        path,
        enable_client_slide_frame_decompression,
        json_metadata=json_metadata,
        instances=instances,
        pixel_spacing_diff_tolerance=pixel_spacing_diff_tolerance,
        logging_factory=logging_factory,
    )
  else:
    microscope_image = None
  return (vl_whole_slide_image, microscope_image)


class DicomMicroscopeSeries:
  """Returns Microscope Images Defined on a Series."""

  def __init__(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
      path: dicom_path.Path,
      enable_client_slide_frame_decompression: bool = True,
      json_metadata: Union[str, Mapping[str, Any]] = '',
      instance_uids: Optional[list[str]] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
  ):
    if json_metadata:
      sop_class_uid_found = set()
      # if metadata is defined convert string to json.
      try:
        if isinstance(json_metadata, str):
          json_metadata = json.loads(json_metadata)
        sop_class_uid = json_metadata[_SOP_CLASS_UID]
        _add_sop_class_uid(sop_class_uid, sop_class_uid_found)
      except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exp:
        raise ez_wsi_errors.InvalidSlideJsonMetadataError(
            'Error decoding JSON metadata.'
        ) from exp
      self.dicom_slide, self.dicom_microscope_image = _load_series(
          dwi,
          path,
          enable_client_slide_frame_decompression,
          json_metadata,
          None,
          sop_class_uid_found,
          pixel_spacing_diff_tolerance,
          logging_factory,
      )
      if self.dicom_slide is None and self.dicom_microscope_image is None:
        raise ez_wsi_errors.InvalidSlideJsonMetadataError(
            'Error decoding JSON metadata does not encode DICOM imaging.'
        )
      return
    dicom_instances = dwi.get_instances(path)
    json_metadata = {}
    sop_class_uid_found = set()
    for instance in dicom_instances:
      if (
          instance_uids is None
          or instance.get_value(tags.SOP_INSTANCE_UID) in instance_uids
      ):
        _add_sop_class_uid(
            instance.get_value(tags.SOP_CLASS_UID),
            sop_class_uid_found,
        )
    self.dicom_slide, self.dicom_microscope_image = _load_series(
        dwi,
        path,
        enable_client_slide_frame_decompression,
        json_metadata,
        dicom_instances,
        sop_class_uid_found,
        pixel_spacing_diff_tolerance,
        logging_factory,
    )
