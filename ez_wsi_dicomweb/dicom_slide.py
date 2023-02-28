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

import dataclasses
import heapq
import math
from typing import Any, Generator, List, Mapping, Optional, Union, Tuple

import cachetools
from ez_wsi_dicomweb import abstract_slide_frame_cache
from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import magnification as magnification_module
from ez_wsi_dicomweb import slide_level_map
import numpy as np

from hcls_imaging_ml_toolkit import dicom_path


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


class Image:
  """Represents an image at a specific magnification in a DicomSlide."""

  def __init__(
      self, magnification: magnification_module.Magnification, slide: DicomSlide
  ):
    """Constructor for DicomImage.

    Args:
      magnification: The magnification level the image belongs to.
      slide: The parent slide this image belongs to.

    Raises:
      ValueError if the requested magnification is not valid.
    """
    self.magnification = magnification
    self._slide = slide

    level_at_mag = slide.get_level_by_magnification(magnification)
    if not level_at_mag:
      raise ez_wsi_errors.MagnificationLevelNotFoundError(
          f'No level found at magnification {magnification}'
      )

    self.width = level_at_mag.width
    self.height = level_at_mag.height

  @property
  def slide(self):
    return self._slide

  def image_bytes(self) -> np.ndarray:
    """Loads the pixel bytes of the DICOM Image.

    Returns:
      Numpy array representing the DICOM Image.
    """
    # Internally reuses the Patch implementation for bytes fetching.
    # An image can be represented as a giant patch starting from (0, 0)
    # and spans the whole slide.
    return self._slide.get_patch(
        magnification=self.magnification,
        x=0,
        y=0,
        width=self.width,
        height=self.height,
    ).image_bytes()


@dataclasses.dataclass(frozen=True)
class PatchBounds:
  """A bounding rectangle of a patch, in pixel units."""

  x_origin: int  # The upper leftmost x coordinate of the patch intersection.
  y_origin: int  # The upper leftmost y coordinate of the patch intersection.
  width: int
  height: int


class Patch:
  """A rectangular patch/tile/view of an Image at a specific magnification.

  A Patch's data is composed from its overlap with one or more DICOM Frames.
  """

  def __init__(
      self,
      magnification: magnification_module.Magnification,
      x: int,
      y: int,
      width: int,
      height: int,
      slide: DicomSlide = None,
  ):
    """Constructor.

    Args:
      magnification: The magnification level the patch belongs to. It defines
        the pixel space for the coordinates (x, y, width, height).
      x: The X coordinate of the starting point (upper-left corner) of the
        patch.
      y: The Y coordinate of the starting point (upper-left corner) of the
        patch.
      width: The width of the patch.
      height: The height of the patch.
      slide: The parent DICOM Slide this patch belongs to.
    """
    self.magnification = magnification
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self._slide = slide

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Patch):
      return False

    return (
        self.magnification == other.magnification
        and self.x == other.x
        and self.y == other.y
        and self.width == other.width
        and self.height == other.height
        and self._slide == other._slide
    )

  @property
  def id(self) -> str:
    if not self.slide or not self.slide.accession_number:
      return (
          f'M_{self.magnification.as_string}:'
          f'{self.width:06d}x{self.height:06d}{self.x:+07d}{self.y:+07d}'
      )
    # Be consistent with internal id format.
    # "%s:%06dx%06d%+07d%+07d", image_id, width, height, left, top
    return (
        f'{self.slide.accession_number}:M_{self.magnification.as_string}:'
        f'{self.width:06d}x{self.height:06d}{self.x:+07d}{self.y:+07d}'
    )

  @property
  def slide(self):
    return self._slide

  @property
  def patch_bounds(self):
    return PatchBounds(
        x_origin=self.x, y_origin=self.y, width=self.width, height=self.height
    )

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

  def frame_number(self) -> Generator[int, None, None]:
    """Yields slide level frame numbers required to render patch.

    Frame numbering starts at 0.
    """
    if self._slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )
    level = self._slide.get_level_by_magnification(self.magnification)
    if not level:
      raise ez_wsi_errors.MagnificationLevelNotFoundError(
          f'No level found at magnification {self.magnification}'
      )

    y = self.y
    x = self.x
    width = self.width
    height = self.height
    cy = y
    frame_width = level.frame_width
    frame_height = level.frame_height
    cached_instance = None
    while cy < y + height:
      cx = x
      region_height = 0
      while cx < x + width:
        try:
          frame_number = level.get_frame_number_by_point(cx, cy)
          if (
              cached_instance is None
              or frame_number - cached_instance.frame_offset
              >= cached_instance.frame_count
          ):
            cached_instance = level.get_instance_by_frame(frame_number)
            last_instance_frame_index = -1
            if cached_instance is None:
              raise ez_wsi_errors.FrameNumberOutofBoundsError()
          instance_frame_index = cached_instance.frame_index_from_frame_number(
              frame_number
          )
          if instance_frame_index > last_instance_frame_index:
            yield instance_frame_index + cached_instance.frame_offset - 1
          last_instance_frame_index = instance_frame_index
          pos_x, pos_y = level.get_frame_position(frame_number)
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

  def image_bytes(self) -> np.ndarray:
    """Returns the patch's image bytes.

    Returns:
      Numpy array type image.

    Raises:
      DicomSlideMissingError if slide used to create self is None.
      PatchOutOfBoundsError if the patch is not within the bounds of the DICOM
      image.
    """
    if self._slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )
    level = self._slide.get_level_by_magnification(self.magnification)
    if not level:
      raise ez_wsi_errors.MagnificationLevelNotFoundError(
          f'No level found at magnification {self.magnification}'
      )

    image_bytes = np.zeros(
        (self.height, self.width, level.samples_per_pixel),
        self.slide.pixel_format,
    )
    pixel_copied = False
    # Copies image pixels from all overlapped frames.
    for frame_number in self.frame_number():
      frame = self.slide.get_frame(self.magnification, frame_number)
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
    return image_bytes


def _get_native_level(
    slm: slide_level_map.SlideLevelMap,
) -> slide_level_map.Level:
  """Gets the native level from a SlideLevelMap.

  The native level of a slide has the lowest level index.

  Args:
    slm: The source SlideLevelMap to get the native level from.

  Returns:
    The level that has the lowest index in the slide.

  Raises:
    MagnificationLevelNotFoundError if the native level does not exist.
  """
  level = slm.get_level(slm.level_index_min)
  if level is not None:
    return level
  else:
    raise ez_wsi_errors.MagnificationLevelNotFoundError(
        'The native level is missing.'
    )


def _get_magnification(
    level: slide_level_map.Level,
) -> magnification_module.Magnification:
  """Gets the magnification corresponding to the input level.

  Args:
    level: The level to find the magnification of.

  Returns:
    The magnification corresponding to the input level.
  """
  pixel_spacing_um = (
      max(level.pixel_spacing_x_mm, level.pixel_spacing_y_mm) * 1000
  )
  return magnification_module.Magnification.FromPixelSize(pixel_spacing_um)


def _get_pixel_format(level: slide_level_map.Level) -> np.dtype:
  """Gets the pixel format of the provided level.

  Args:
    level: The level to get the pixel format for.

  Returns:
    The pixel format of the level as numpy.dtype.

  Raises:
    UnsupportedPixelFormatError if pixel format is not supported.
  """
  bytes_per_sample = math.ceil(level.bits_allocated / 8)
  if bytes_per_sample == 1:
    return np.uint8
  else:
    raise ez_wsi_errors.UnsupportedPixelFormatError(
        f'Pixel format not supported. BITS_ALLOCATED = {level.bits_allocated}'
    )


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


class DicomSlide:
  """Represents a DICOM pathology slide stored in a DICOMStore."""

  def __init__(
      self,
      dwi: dicom_web_interface.DicomWebInterface,
      path: dicom_path.Path,
      enable_client_slide_frame_decompression: bool,
      accession_number: Optional[str] = None,
      frame_cache_size: int = 128,
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
      frame_cache_size: Size of the LRU cache for retrieving frames.
    """
    self._level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
    self._dwi = dwi
    self.path = path
    self.accession_number = accession_number
    native_level = _get_native_level(self._level_map)
    self.pixel_format = _get_pixel_format(native_level)
    self.native_magnification = _get_magnification(native_level)
    # Native height
    self.total_pixel_matrix_rows = native_level.height
    # Native width
    self.total_pixel_matrix_columns = native_level.width
    self._server_request_frame_lru_cache_size = frame_cache_size
    self._server_request_frame_cache = cachetools.LRUCache(frame_cache_size)
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )
    self._slide_frame_cache: Optional[
        abstract_slide_frame_cache.AbstractSlideFrameCache
    ] = None

  @property
  def slide_frame_cache(
      self,
  ) -> Optional[abstract_slide_frame_cache.AbstractSlideFrameCache]:
    return self._slide_frame_cache

  @slide_frame_cache.setter
  def slide_frame_cache(
      self,
      slide_frame_cache: abstract_slide_frame_cache.AbstractSlideFrameCache,
  ) -> None:
    self._slide_frame_cache = slide_frame_cache

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, DicomSlide):
      return str(self.path) == str(other.path)
    return False

  def __getstate__(self):
    state = self.__dict__.copy()
    del state['_dwi']
    del state['_server_request_frame_cache']
    return state

  def __setstate__(self, dct):
    self.__dict__ = dct
    self._dwi = None
    self._server_request_frame_cache = cachetools.LRUCache(
        self._server_request_frame_lru_cache_size
    )

  @property
  def dwi(self) -> dicom_web_interface.DicomWebInterface:
    return self._dwi

  @dwi.setter
  def dwi(self, val: dicom_web_interface.DicomWebInterface) -> None:
    self._dwi = val

  @property
  def magnifications(self) -> list[magnification_module.Magnification]:
    """Lists all Magnifications in the DicomSlide."""
    return [
        magnification_module.Magnification.FromPixelSize(
            level.pixel_spacing_x_mm * 1000
        )
        for level in self._level_map.level_map.values()
    ]

  def get_level_by_magnification(
      self, magnification: magnification_module.Magnification
  ) -> Optional[slide_level_map.Level]:
    """Gets the level corresponding to the input magnification.

    Args:
      magnification: The magnification to use for level lookup.

    Returns:
      The level corresponding to the input mangnifcation. None if the requested
      magnification level does not exist.
    """
    # Converts pixel spacing returned by Magnification.NominalPixelSize() from
    # micrometers to millimeters.
    pixel_spacing_mm = magnification.nominal_pixel_size / 1000
    return self._level_map.get_level_by_pixel_spacing(pixel_spacing_mm)

  def _get_cached_frame_bytes(
      self,
      level: slide_level_map.Level,
      instance_path: Union[str, slide_level_map.Instance, dicom_path.Path],
      frame_index: int,
  ) -> Optional[bytes]:
    """Returns frame bytes from frame cache if possible.

    Args:
      level: Magnification level of DICOM.
      instance_path: Path to DICOM instance on level.
      frame_index: Instance frame index to return.

    Returns:
      Frame bytes or None.
    """
    if (
        self._slide_frame_cache is None
        or not self._slide_frame_cache.is_supported_transfer_syntax(
            level.transfer_syntax_uid
        )
    ):
      return None
    if isinstance(instance_path, slide_level_map.Instance):
      instance_path = str(instance_path.dicom_object.path)
    elif isinstance(instance_path, dicom_path.Path):
      instance_path = str(instance_path)
    elif not isinstance(instance_path, str):
      return None
    return self._slide_frame_cache.get_frame(instance_path, frame_index)

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
    frame_raw_bytes = self._server_request_frame_cache.get(cache_key)
    if frame_raw_bytes is not None:
      return frame_raw_bytes
    frame_raw_bytes = self.dwi.get_frame_image(
        instance_path, instance_frame_index, transcoding
    )
    self._server_request_frame_cache[cache_key] = frame_raw_bytes
    return frame_raw_bytes

  def _get_frame_client_transcoding(
      self,
      level: slide_level_map.Level,
      instance: slide_level_map.Instance,
      instance_frame_index: int,
  ) -> Optional[np.ndarray]:
    """Returns DICOM Frame using DICOM server, transcodes to raw on server.

    Args:
      level: WSI pyramid level.
      instance: DICOM instance within level to return frame from.
      instance_frame_index: Frame number within the instance to return.

    Returns:
      Tuple[numpy array containing frame data, bool indicating if data should
      be cached True in LRU or is already cached within the level]
    """
    compressed_bytes = self._get_cached_frame_bytes(
        level, instance.dicom_object.path, instance_frame_index
    )
    if compressed_bytes is None:
      compressed_bytes = self._get_frame_bytes_from_server(
          instance,
          instance_frame_index,
          dicom_web_interface.TranscodeDicomFrame.DO_NOT_TRANSCODE,
      )
    return dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        compressed_bytes
    )

  def _get_frame_server_transcoding(
      self,
      level: slide_level_map.Level,
      instance: slide_level_map.Instance,
      instance_frame_index: int,
  ) -> np.ndarray:
    """Returns DICOM Frame using DICOM server, transcodes to raw on server.

    Args:
      level: WSI pyramid level.
      instance: DICOM instance within level to return frame from.
      instance_frame_index: Frame number within the instance to return.

    Returns:
      Tuple[numpy array containing frame data, bool indicating if data should
      be cached True in LRU or is already cached within the level]
    """
    frame_raw_bytes = self._get_cached_frame_bytes(
        level, instance.dicom_object.path, instance_frame_index
    )
    if frame_raw_bytes is None:
      frame_raw_bytes = self._get_frame_bytes_from_server(
          instance,
          instance_frame_index,
          dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
      )
    return np.frombuffer(frame_raw_bytes, self.pixel_format).reshape(
        (level.frame_height, level.frame_width, level.samples_per_pixel)
    )

  def get_frame(
      self, magnification: magnification_module.Magnification, frame_number: int
  ) -> Optional[Frame]:
    """Gets a frame at a specific magnification level.

    The DICOMWeb API serves image pixels by the unit of frames. Frames have
    fixed size (width and height). Call get_patch() instead if you want to get
    an image patch at a specific loation, or with a specific dimension.

    The function utilizes a LRUCache to cache the most recent used frames.

    Args:
      magnification: The magnification level to fetch the frame from.
      frame_number: The frame number to be fetched. The frames are stored in
        arrays with 1-based indexing.

    Returns:
      Returns the requested frame if exists, None otherwise.

    Raises:
      InputFrameNumberOutOfRangeError if the input frame_number is
      out of range.
    """
    level = self.get_level_by_magnification(magnification)
    if level is None:
      return None
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
    instance_frame_number = instance.frame_index_from_frame_number(frame_number)
    pos_x, pos_y = level.get_frame_position(frame_number)
    frame_ndarray = None
    if (
        self._enable_client_slide_frame_decompression
        and dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
            level.transfer_syntax_uid
        )
    ):
      frame_ndarray = self._get_frame_client_transcoding(
          level, instance, instance_frame_number
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
      self, magnification: magnification_module.Magnification
  ) -> Image:
    """Gets an image from a specific magnification."""
    return Image(magnification, self)

  def get_patch(
      self,
      magnification: magnification_module.Magnification,
      x: int,
      y: int,
      width: int,
      height: int,
  ) -> Patch:
    """Gets a patch from a specific magnification level of the slide.

    The area of a patch is defined by its position(x, y) and its dimension
    (width, height). The position of a patch corresponds to the first pixel in
    the patch and has the smallest coordinates. All coordinates(x, y, width and
    height) are defined in the pixel space of the requested magnification level.

    This routine creates the requested patch on-the-fly, by sampling image
    pixels from all frames that are overlapped with the patch.

    Args:
      magnification: The magnification level to fetch the frame from.
      x: The X coordinate of the patch position.
      y: The Y coordinate of the patch position.
      width: The width of the patch.
      height: The height of the patch.

    Returns:
      Returns the requested patch if exists, None otherwise.

    Raises:
      MagnificationLevelNotFoundError if the requested level does not exist, or
      if the requested patch is out of the scope of the image.
    """
    level = self.get_level_by_magnification(magnification)
    if level is None:
      raise ez_wsi_errors.MagnificationLevelNotFoundError(
          f'The requested magnification level {magnification.as_string} '
          'does not exist.'
      )
    patch = Patch(magnification, x, y, width, height, slide=self)
    return patch

  def get_patch_bounds_dicom_instance_frame_indexes(
      self,
      mag: magnification_module.Magnification,
      patch_bounds_list: List[PatchBounds],
  ) -> Mapping[str, List[int]]:
    """Returns Map[DICOM instances: frames indexes] that fall in patch bounds.

    Args:
      mag: Magnification patch bounding list was generated at.
      patch_bounds_list: List of PatchBounds to return frame indexes for.

    Returns:
      Mapping[DICOM instance path, List[FrameIndexes]]
      List of frame indexes is returned in sorted order without duplicates.
    """
    level = self.get_level_by_magnification(mag)
    if level is None:
      return {}
    slide_instance_frame_map = {}
    indexes_required_for_inference = []
    for patch_bounds in patch_bounds_list:
      patch = self.get_patch(
          mag,
          patch_bounds.x_origin,
          patch_bounds.y_origin,
          patch_bounds.width,
          patch_bounds.height,
      )
      # Frame indexes returns a list of indexes in the patch. Indexes
      # are returned in sorted order.
      indexes_required_for_inference.append(patch.frame_number())
    instance_index_buffer = []
    instance = None
    # Use Heapq to merge pre-sorted lists into single sorted list
    # Result of heapq.merge can have duplicates
    for index in heapq.merge(*indexes_required_for_inference):
      if (
          instance is None
          or index - instance.frame_offset >= instance.frame_count
      ):
        if instance_index_buffer:
          slide_instance_frame_map[str(instance.dicom_object.path)] = (
              instance_index_buffer
          )
        instance = level.get_instance_by_frame(index)
        if instance is None:
          instance_index_buffer = []
          continue
        instance_index_buffer = [instance.frame_index_from_frame_number(index)]
        continue

      instance_frame_number = instance.frame_index_from_frame_number(index)

      if (
          instance_index_buffer
          and instance_index_buffer[-1] == instance_frame_number
      ):
        # remove duplicates
        continue
      instance_index_buffer.append(instance_frame_number)
    if instance_index_buffer:
      slide_instance_frame_map[str(instance.dicom_object.path)] = (
          instance_index_buffer
      )
    return slide_instance_frame_map
