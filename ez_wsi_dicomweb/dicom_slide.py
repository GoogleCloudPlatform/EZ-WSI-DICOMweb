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
from typing import Any, Iterator, List, Mapping, Optional, Tuple

from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import pixel_spacing as pixel_spacing_module
from ez_wsi_dicomweb import slide_level_map
from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import tags
import numpy as np


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


class Image:
  """Represents an image at a specific pixel spacing in a DicomSlide."""

  def __init__(
      self, pixel_spacing: pixel_spacing_module.PixelSpacing, slide: DicomSlide
  ):
    """Constructor for DicomImage.

    Args:
      pixel_spacing: The pixel_spacing of the image.
      slide: The parent slide this image belongs to.

    Raises:
      ValueError if the requested pixel spacing is not valid.
      PixelSpacingLevelNotFoundError if the pixel spacing does not exist.
    """
    self.pixel_spacing = pixel_spacing
    self._slide = slide

    level_at_ps = self._slide.get_level_by_pixel_spacing(self.pixel_spacing)
    if not level_at_ps:
      raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
          'No level found at pixel spacing:'
          f' {self.pixel_spacing.pixel_spacing_mm}'
      )

    self.width = level_at_ps.width
    self.height = level_at_ps.height

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
        pixel_spacing=self.pixel_spacing,
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
  """A rectangular patch/tile/view of an Image at a specific pixel spacing.

  A Patch's data is composed from its overlap with one or more DICOM Frames.
  """

  def __init__(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
      x: int,
      y: int,
      width: int,
      height: int,
      slide: DicomSlide = None,
  ):
    """Constructor.

    Args:
      pixel_spacing: The pixel spacing the patch belongs to.
      x: The X coordinate of the starting point (upper-left corner) of the
        patch.
      y: The Y coordinate of the starting point (upper-left corner) of the
        patch.
      width: The width of the patch.
      height: The height of the patch.
      slide: The parent DICOM Slide this patch belongs to.
    """
    self.pixel_spacing = pixel_spacing
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self._slide = slide

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Patch):
      return False

    return (
        self.pixel_spacing == other.pixel_spacing
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
          f'M_{self.pixel_spacing.as_magnification_string}:'
          f'{self.width:06d}x{self.height:06d}{self.x:+07d}{self.y:+07d}'
      )
    # Be consistent with internal id format.
    # "%s:%06dx%06d%+07d%+07d", image_id, width, height, left, top
    return (
        f'{self.slide.accession_number}:M_{self.pixel_spacing.as_magnification_string}:'
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

  def frame_number(self) -> Iterator[int]:
    """Generates slide level frame numbers required to render patch.

    Frame numbering starts at 0.

    Yields:
      A generator that produces frame numbers.

    Raises:
      PixelSpacingNotFoundError if the pixel spacing does not exist.
    """
    if self._slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )

    level_at_ps = self._slide.get_level_by_pixel_spacing(self.pixel_spacing)
    if not level_at_ps:
      raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
          'No level found at pixel spacing:'
          f' {self.pixel_spacing.pixel_spacing_mm}'
      )

    y = self.y
    x = self.x
    width = self.width
    height = self.height
    cy = y
    frame_width = level_at_ps.frame_width
    frame_height = level_at_ps.frame_height
    cached_instance = None
    last_instance_frame_index = -1
    while cy < y + height:
      cx = x
      region_height = 0
      while cx < x + width:
        try:
          frame_number = level_at_ps.get_frame_number_by_point(cx, cy)
          if (
              cached_instance is None
              or frame_number - cached_instance.frame_offset
              >= cached_instance.frame_count
          ):
            cached_instance = level_at_ps.get_instance_by_frame(frame_number)
            last_instance_frame_index = -1
            if cached_instance is None:
              raise ez_wsi_errors.FrameNumberOutofBoundsError()
          instance_frame_index = cached_instance.frame_index_from_frame_number(
              frame_number
          )
          if instance_frame_index > last_instance_frame_index:
            yield instance_frame_index + cached_instance.frame_offset - 1
          last_instance_frame_index = instance_frame_index
          pos_x, pos_y = level_at_ps.get_frame_position(frame_number)
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
      PixelSpacingNotFoundError if the pixel spacing does not exist.
    """
    if self._slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Unable to get image pixels. Parent slide is None.'
      )
    level_at_ps = self._slide.get_level_by_pixel_spacing(self.pixel_spacing)
    if not level_at_ps:
      raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
          'No level found at pixel spacing:'
          f' {self.pixel_spacing.pixel_spacing_mm}'
      )

    image_bytes = np.zeros(
        (self.height, self.width, level_at_ps.samples_per_pixel),
        self.slide.pixel_format,
    )
    pixel_copied = False
    # Copies image pixels from all overlapped frames.
    for frame_number in self.frame_number():
      frame = self.slide.get_frame(self.pixel_spacing, frame_number)
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
    LevelNotFoundError if the native level does not exist.
  """
  level = slm.get_level(slm.level_index_min)
  if level is not None:
    return level
  else:
    raise ez_wsi_errors.LevelNotFoundError('The native level is missing.')


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
    return np.uint8  # pytype: disable=bad-return-type  # numpy-scalars
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
      enable_client_slide_frame_decompression: bool = True,
      accession_number: Optional[str] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
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
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings.
      logging_factory: The factory that EZ WSI uses to construct a logging
        interface.
      slide_frame_cache: Initalize to use shared slide cache.
    """
    self._logger = None
    if logging_factory is None:
      self._logging_factory = ez_wsi_logging_factory.BasePythonLoggerFactory(
          ez_wsi_logging_factory.DEFAULT_EZ_WSI_PYTHON_LOGGER_NAME
      )
    else:
      self._logging_factory = logging_factory
    self._level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
    self._dwi = dwi
    self.path = path
    self.accession_number = accession_number
    native_level = _get_native_level(self._level_map)
    self.pixel_format = _get_pixel_format(native_level)
    self.native_pixel_spacing = pixel_spacing_module.PixelSpacing(
        native_level.pixel_spacing_x_mm,
        native_level.pixel_spacing_y_mm,
        spacing_diff_tolerance=pixel_spacing_diff_tolerance,
    )
    # Native height
    self.total_pixel_matrix_rows = native_level.height
    # Native width
    self.total_pixel_matrix_columns = native_level.width
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )
    self._slide_frame_cache = slide_frame_cache

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
    """Initalizes DICOM slide frame cache.

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
      DICOM slide frame cache initialized on the slide.
    """
    self._slide_frame_cache = local_dicom_slide_cache.InMemoryDicomSlideCache(
        credential_factory=self._dwi.dicomweb_credential_factory,
        number_of_frames_to_read=number_of_frames_to_read,
        max_instance_number_of_frames_to_prefer_whole_instance_download=max_instance_number_of_frames_to_prefer_whole_instance_download,
        max_cache_frame_memory_lru_cache_size_bytes=max_cache_frame_memory_lru_cache_size_bytes,
        optimization_hint=optimization_hint,
        logging_factory=self._logging_factory,
    )
    return self._slide_frame_cache

  def remove_slide_frame_cache(self) -> None:
    """Removes slide frame cache."""
    self._slide_frame_cache = None

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, DicomSlide):
      return str(self.path) == str(other.path)
    return False

  @property
  def dwi(self) -> dicom_web_interface.DicomWebInterface:
    return self._dwi

  @property
  def all_pixel_spacing_mms(self) -> list[float]:
    """Lists all Pixel Spacings in mm in the DicomSlide."""
    return [
        level.pixel_spacing_x_mm for level in self._level_map.level_map.values()
    ]

  def get_level_by_pixel_spacing(
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
    return self._level_map.get_level_by_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    )

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
        str(instance.dicom_object.path), instance.frame_count, frame_number
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
        compressed_bytes
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
    return np.frombuffer(frame_raw_bytes, self.pixel_format).reshape(
        (level.frame_height, level.frame_width, level.samples_per_pixel)
    )

  def get_frame(
      self, pixel_spacing: pixel_spacing_module.PixelSpacing, frame_number: int
  ) -> Optional[Frame]:
    """Gets a frame at a specific pixel_spacing in mm.

    The DICOMWeb API serves image pixels by the unit of frames. Frames have
    fixed size (width and height). Call get_patch() instead if you want to get
    an image patch at a specific loation, or with a specific dimension.

    The function utilizes a LRUCache to cache the most recent used frames.

    Args:
      pixel_spacing: The pixel spacing to fetch the frame from.
      frame_number: The frame number to be fetched. The frames are stored in
        arrays with 1-based indexing.

    Returns:
      Returns the requested frame if exists, None otherwise.

    Raises:
      InputFrameNumberOutOfRangeError if the input frame_number is
      out of range.
    """
    level = self._level_map.get_level_by_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    )
    if not level:
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
      self, pixel_spacing: pixel_spacing_module.PixelSpacing
  ) -> Image:
    """Gets an image from a specific pixel spacing."""
    return Image(pixel_spacing, self)

  def get_patch(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
      x: int,
      y: int,
      width: int,
      height: int,
  ) -> Patch:
    """Gets a patch from a specific pixel spacing of the slide.

    The area of a patch is defined by its position(x, y) and its dimension
    (width, height). The position of a patch corresponds to the first pixel in
    the patch and has the smallest coordinates. All coordinates(x, y, width and
    height) are defined in the pixel space of the requested pixel spacing.

    This routine creates the requested patch on-the-fly, by sampling image
    pixels from all frames that are overlapped with the patch.

    Args:
      pixel_spacing: The pixel spacing to fetch the frame from.
      x: The X coordinate of the patch position.
      y: The Y coordinate of the patch position.
      width: The width of the patch.
      height: The height of the patch.

    Returns:
      Returns the requested patch if exists, None otherwise.

    Raises:
      PixelSpacingNotFoundError if the requested level does not exist, or
      if the requested patch is out of the scope of the image.
    """
    level = self._level_map.get_level_by_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    )
    if not level:
      raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
          f'No level found at pixel spacing: {pixel_spacing.pixel_spacing_mm}'
      )

    patch = Patch(pixel_spacing, x, y, width, height, slide=self)
    return patch

  def get_patch_bounds_dicom_instance_frame_numbers(
      self,
      pixel_spacing: pixel_spacing_module.PixelSpacing,
      patch_bounds_list: List[PatchBounds],
  ) -> Mapping[str, List[int]]:
    """Returns Map[DICOM instances: frame numbers] that fall in patch bounds.

    Args:
      pixel_spacing: pixel spacing patch bounding list was generated at.
      patch_bounds_list: List of PatchBounds to return frame indexes for.

    Returns:
      Mapping between DICOM instances, path, and list of frames numbers required
      to render patches.
    """
    level = self._level_map.get_level_by_pixel_spacing(
        pixel_spacing.pixel_spacing_mm
    )
    if not level:
      return {}

    slide_instance_frame_map = {}
    indexes_required_for_inference = []
    for patch_bounds in patch_bounds_list:
      patch = self.get_patch(
          pixel_spacing,
          patch_bounds.x_origin,
          patch_bounds.y_origin,
          patch_bounds.width,
          patch_bounds.height,
      )
      # Frame indexes returns a list of indexes in the patch. Indexes
      # are returned in sorted order.
      indexes_required_for_inference.append(patch.frame_number())
    instance_frame_number_buffer = []
    instance = None
    # Use Heapq to merge pre-sorted lists into single sorted list
    # Result of heapq.merge can have duplicates
    for index in heapq.merge(*indexes_required_for_inference):
      if (
          instance is None
          or index - instance.frame_offset >= instance.frame_count
      ):
        if instance_frame_number_buffer:
          slide_instance_frame_map[str(instance.dicom_object.path)] = (
              instance_frame_number_buffer
          )
        instance = level.get_instance_by_frame(index)
        if instance is None:
          instance_frame_number_buffer = []
          continue
        instance_frame_number_buffer = [
            instance.frame_index_from_frame_number(index)
        ]
        continue

      instance_frame_number = instance.frame_index_from_frame_number(index)

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

  @property
  def levels(self) -> Iterator[slide_level_map.Level]:
    """Returns iterator that contains all of a slide's DICOM Levels."""
    return iter(self._level_map.level_map.values())
