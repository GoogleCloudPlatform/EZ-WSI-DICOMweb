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
"""An abstraction to sample patches in a slide at a specified pixel spacing."""

import abc
from collections.abc import Sequence
import concurrent.futures
import dataclasses
import functools
import logging
import math
import typing
from typing import Generic, Iterator, Optional, Tuple, TypeVar, Union

import cv2
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
import numpy as np
from PIL import ImageCms


_MAX_GCS_IMAGE_READ_THEADS = 4

TISSUE_MASK_PIXEL_SPACING = pixel_spacing.PixelSpacing.FromMagnificationString(
    '1.250X'
)

# Max number of frames tissue masks can have and be preloaded into cache.
_MAX_TISSUE_MASK_LEVEL_FRAME_COUNT_PRELOAD = 1000


EmbeddingPatch = TypeVar(
    'EmbeddingPatch', dicom_slide.DicomPatch, gcs_image.GcsPatch
)


@dataclasses.dataclass(frozen=True)
class StrideCoordinate:
  """Represents the coordinates of a stride using x, y coordinates.

  See PatchGenerator for usage examples.

  Attributes:
    y_strides: The number of strides along the y axis
    x_strides: The number of strides along the x axis
  """

  y_strides: int
  x_strides: int


def _pixel_spacing_to_level(
    slide: dicom_slide.DicomSlide,
    ps: pixel_spacing.PixelSpacing,
    maximum_downsample: float,
) -> slide_level_map.Level:
  """Returns the level that has the lowest level index."""
  logging.info(
      'Pixel spacing parameter will be deprecated in future versions.'
      ' Modify code to define source imaging using slide_level_map.Level.'
  )
  source_image_level = slide.get_level_by_pixel_spacing(
      ps, maximum_downsample=maximum_downsample
  )
  if source_image_level is None:
    raise ez_wsi_errors.PixelSpacingLevelNotFoundError(
        'No pyramid level found with pixel spacing:'
        f' ~{ps.pixel_spacing_mm}  mm/px'
    )
  return source_image_level


class _BasePatchGenerator(Generic[EmbeddingPatch], Sequence[EmbeddingPatch]):
  """A generator for patch sampling in a slide at specified pixel spacing."""

  def __init__(
      self,
      stride_size: Optional[int],
      patch_size: int,
      min_luminance: Optional[float] = None,
      max_luminance: Optional[float] = None,
      tissue_mask: Optional[np.ndarray] = None,
  ):
    if isinstance(tissue_mask, int):
      # parameter order changed. Raise error if tissue mask incorrectly defined.
      # remove in future versions.
      raise ValueError('Parameter order for patch generator has changed.')
    self._stride_coordinates = None
    self._tissue_mask_cache = None
    self.patch_size = patch_size
    self._user_provided_tissue_mask = tissue_mask
    if self._user_provided_tissue_mask is None:
      if stride_size is None:
        stride_size = patch_size
      self.stride_width = stride_size
      self.stride_height = stride_size
      # default luminance threshold settings expect white background
      self.min_luminance = (
          (1.0 / 255.0) if min_luminance is None else min_luminance
      )
      self.max_luminance = 0.8 if max_luminance is None else max_luminance
    else:
      if not np.issubdtype(self._user_provided_tissue_mask.dtype, np.bool_):
        raise ez_wsi_errors.InvalidTissueMaskError(
            'Tissue must have dtype bool'
        )
      self._user_provided_tissue_mask = (
          self._user_provided_tissue_mask.astype(np.uint8) * 255
      )
      # default luminance threshold settings expect tissue = True (1.0)
      self.min_luminance = 0.5
      self.max_luminance = 1.0
      image_dim = self._image_dimensions()
      height, width = self._user_provided_tissue_mask.shape[:2]
      if (image_dim.width_px < width) or (image_dim.height_px < height):
        raise ez_wsi_errors.InvalidTissueMaskError(
            'Tissue mask dimensions exceed image dimensions.'
        )
      self.stride_width = int(image_dim.width_px // width)
      self.stride_height = int(image_dim.height_px // height)

  def __len__(self) -> int:
    return self.total_num_patches()

  def total_num_patches(self) -> int:
    """Total number of tissue patches in this slide."""
    return len(self._patch_stride_coordinates()[0])

  def total_strides(self) -> StrideCoordinate:
    y, x = self._normalized_tissue_mask().shape
    return StrideCoordinate(y_strides=y, x_strides=x)

  def _stride_to_patch(self, stride_dim: int, stride_count: int) -> int:
    if self.patch_size >= stride_dim:
      return int(stride_dim * stride_count)
    half_stride = stride_dim // 2
    half_patch_size = self.patch_size // 2
    return int(stride_dim * stride_count + half_stride - half_patch_size)

  def strides_to_patch_bounds(
      self, strides: StrideCoordinate
  ) -> dicom_slide.PatchBounds:
    """Converts number of strides to patch coordinates.

    Args:
      strides: Number of strides taken on x and y axis.

    Returns:
      Patch Bounds corresponding to the strides.
    """
    left = self._stride_to_patch(self.stride_width, strides.x_strides)
    top = self._stride_to_patch(self.stride_height, strides.y_strides)
    return dicom_slide.PatchBounds(
        x_origin=left,
        y_origin=top,
        width=self.patch_size,
        height=self.patch_size,
    )

  def _patch_to_stride(self, patch_coordinate: int, stride_dim: int) -> int:
    if self.patch_size >= stride_dim:
      return int(patch_coordinate / stride_dim)
    half_stride = stride_dim // 2
    half_patch_size = self.patch_size // 2
    center_y = patch_coordinate + half_patch_size - half_stride
    return int(center_y // stride_dim)

  def patch_bounds_to_strides(
      self, patch_bounds: dicom_slide.PatchBounds
  ) -> StrideCoordinate:
    """Converts patch bounds to number of strides in tissue mask.

    Args:
      patch_bounds: Describes the patch's coordinates bounding box.

    Returns:
      Corresponding number of strides.
    """
    x = self._patch_to_stride(patch_bounds.x_origin, self.stride_width)
    y = self._patch_to_stride(patch_bounds.y_origin, self.stride_height)
    return StrideCoordinate(y_strides=y, x_strides=x)

  def _normalized_tissue_mask(self) -> np.ndarray:
    """An image mask that is likely to contain tissues.

    It is normalized/down-scaled by the specified stride size, i.e.
    normalized_tissue_mask size = original size / stride.

    Returns:
      A 2D ndarray binary (0, 1) image denoting tissue mask.

    Raises:
      DicomImageMissingRegionError if there is no region with the
      required luminance values.
    """
    if self._tissue_mask_cache is not None:
      return self._tissue_mask_cache
    rgb_at_tissue_ps = self.get_tissue_mask()
    # luminance values
    if np.issubdtype(rgb_at_tissue_ps.dtype, np.floating):
      if rgb_at_tissue_ps.max() <= 1.0:
        rgb_at_tissue_ps *= 255.0
      rgb_at_tissue_ps = rgb_at_tissue_ps.astype(np.uint8)
    if len(rgb_at_tissue_ps.shape) == 2:
      gray_tissue_mask = rgb_at_tissue_ps
    elif len(rgb_at_tissue_ps.shape) == 3 and rgb_at_tissue_ps.shape[2] == 1:
      gray_tissue_mask = np.squeeze(rgb_at_tissue_ps, axis=-1)
    else:
      gray_tissue_mask = cv2.cvtColor(rgb_at_tissue_ps, cv2.COLOR_RGB2GRAY)
    tissue_mask_height, tissue_mask_width = gray_tissue_mask.shape[:2]
    image_dimensions = self._image_dimensions()
    if (
        image_dimensions.height_px < self.patch_size
        or image_dimensions.width_px < self.patch_size
    ):
      raise ez_wsi_errors.InvalidPatchDimensionError(
          'Patch dimensions are exceed image dimensions.'
      )
    # if stride size may be different along x and y axis only when
    # mask is initialized directly by the user.
    if (
        self._user_provided_tissue_mask is None
        and self.stride_width != self.stride_height
    ):
      raise ValueError(
          'Stride dimensions must be equal along both dimensions if not'
          ' initialized from user provided mask.'
      )
    stride_size = self.stride_width
    if self._user_provided_tissue_mask is not None:
      gray_image_at_output_res = rgb_at_tissue_ps
    elif self.patch_size <= stride_size:
      # computes tissue mask when patch size <= stride size.

      # number of patches to sample across the image in the x and y axis.
      width_steps = max(int(image_dimensions.width_px / stride_size), 1)
      height_steps = max(int(image_dimensions.height_px / stride_size), 1)

      # dimensions of the tissue across which patches will be sampled.
      # in most cases does not = whole tissue dimensions due to integer math.
      sampled_image_width = int(width_steps * stride_size)
      sampled_image_height = int(height_steps * stride_size)
      if (
          sampled_image_width < image_dimensions.width_px
          or sampled_image_height < image_dimensions.height_px
      ):
        # if sampled area is less than actual tissue dimensions
        # determine if image representing tissue for tissue mask should be
        # cropped to remove areas which do not correspond to regions patches
        # will be sampled from.

        # determine how much smaller, proportionally, the sampled area is in the
        # the source image.
        width_scale_factor = float(sampled_image_width) / float(
            image_dimensions.width_px
        )
        height_scale_factor = float(sampled_image_height) / float(
            image_dimensions.height_px
        )
        # scale tissue mask by the scale factor.
        tissue_mask_width = int(
            min(
                math.ceil(tissue_mask_width * width_scale_factor),
                tissue_mask_width,
            )
        )
        tissue_mask_height = int(
            min(
                math.ceil(tissue_mask_height * height_scale_factor),
                tissue_mask_height,
            )
        )
        # crop the tissue mask based on the newly computed dimensions.
        # tissue mask mask now corresponds the region being sampled in the
        # the source image.
        gray_tissue_mask = gray_tissue_mask[
            :tissue_mask_height, :tissue_mask_width, ...
        ]
      # resize the tissue mask to number of strides that will be sampled.
      # each pixel in the mask now represents one patch.
      gray_image_at_output_res = cv2.resize(
          gray_tissue_mask,
          (width_steps, height_steps),
          interpolation=cv2.INTER_AREA,
      )
    else:
      # computing tissue mask when patch size > stride size, aka, patches
      # overlap.

      # determine how much larger in px patches are than strides
      small_stride_adjustment = int(max(self.patch_size - stride_size, 0))
      # The number of patches which can be sampled along both
      # dimensions. Due to patches being larger than strides, the
      # adjustment factor calculated previously is subtracted from the dim to
      # guarantee that the patches are always sampled from within the image.
      width_steps = max(
          int(
              (image_dimensions.width_px - small_stride_adjustment)
              / stride_size
          ),
          1,
      )
      height_steps = max(
          int(
              (image_dimensions.height_px - small_stride_adjustment)
              / stride_size
          ),
          1,
      )
      # Allocate a buffer to store the tissue mask results. Each pixel in this
      # buffer will hold the mean value of pixels that fall within a tisse mask
      # within scaled tissue mask patch.
      gray_image_at_output_res = np.zeros(
          (height_steps, width_steps), dtype=np.uint64
      )
      # Scale the dimensions of the origional patch to find the tissue mask
      # patch size. Make sure that at least one pixel will be sampled.
      tisse_mask_patch_width = max(
          int(self.patch_size * tissue_mask_width / image_dimensions.width_px),
          1,
      )
      tisse_mask_patch_height = max(
          int(
              self.patch_size * tissue_mask_height / image_dimensions.height_px
          ),
          1,
      )
      # Allocate temporary buffers to use to speed up calculations.
      column_sum = np.zeros(tissue_mask_width, dtype=np.uint64)
      temp_buffer = np.zeros(tissue_mask_width, dtype=np.uint64)

      # determine patch sampling indices along the horizontal axis.
      # compute once and re-use across all rows.
      x_start_offset = [
          int(x * stride_size * tissue_mask_width / image_dimensions.width_px)
          for x in range(width_steps)
      ]
      last_start = -1
      # compute the mean pixel value for each of the sampling patches in the
      # tissue mask
      for y in range(height_steps):
        # upper left y position
        patch_y_start = int(
            y * stride_size * tissue_mask_height / image_dimensions.height_px
        )
        if last_start == -1:
          # if it's the very first row, being calculated then
          # calculate the sum of the pixels along each column of the patches
          # for the width of the image
          np.sum(
              gray_tissue_mask[
                  patch_y_start : patch_y_start + tisse_mask_patch_height, ...
              ],
              axis=0,
              out=column_sum,
          )
        else:
          # if not first row of the image then
          # compute the sum of the columns which fell in the prior rows patches
          # but not the current and subtract these from the column row totals
          np.sum(
              gray_tissue_mask[last_start:patch_y_start, ...],
              axis=0,
              out=temp_buffer,
          )
          column_sum -= temp_buffer
          # if not first row of the image then
          # compute the sum of the new column area, bottom of prior results to
          # bottom of curren rows patches. Add these to column totals.
          np.sum(
              gray_tissue_mask[
                  last_start
                  + tisse_mask_patch_height : patch_y_start
                  + tisse_mask_patch_height,
                  ...,
              ],
              axis=0,
              out=temp_buffer,
          )
          column_sum += temp_buffer
        last_start = patch_y_start
        for x in range(width_steps):
          # Now step through each patch in the row. Calculate the total sum
          # of the pixel values by summing across the columns which fall in the
          # patch.
          patch_x_start = x_start_offset[x]
          gray_image_at_output_res[y, x] = np.sum(
              column_sum[
                  patch_x_start : patch_x_start + tisse_mask_patch_width
              ],
          )
      # Pixel values in buffer equal total sum of all tissue mask pixel that
      # are sampled for a given tissue mask patch. Divide by number of pixels
      # in patch.
      np.floor_divide(
          gray_image_at_output_res,
          int(tisse_mask_patch_width * tisse_mask_patch_height),
          out=gray_image_at_output_res,
      )
    # Threshold mask buffer to determine if a patch will be generated.
    normalized_tissue_mask = (
        gray_image_at_output_res <= int(255.0 * self.max_luminance)
    ) & (gray_image_at_output_res >= int(255.0 * self.min_luminance))
    if np.all(~normalized_tissue_mask):  # pylint: disable=invalid-unary-operand-type
      raise ez_wsi_errors.DicomImageMissingRegionError(
          'Tissue mask has no regions with luminance value within threshold'
          f' {self.min_luminance} - {self.max_luminance}.'
      )
    self._tissue_mask_cache = normalized_tissue_mask
    return normalized_tissue_mask

  @property
  def normalized_tissue_mask(self) -> np.ndarray:
    return self._normalized_tissue_mask().copy()

  @abc.abstractmethod
  def get_tissue_mask(self) -> np.ndarray:
    """Returns the tissue mask."""

  @abc.abstractmethod
  def patch_at_stride(self, strides: StrideCoordinate) -> EmbeddingPatch:
    """Converts tissue mask coordinates to patch coordinates."""

  @abc.abstractmethod
  def _image_dimensions(self) -> slide_level_map.ImageDimensions:
    """Returns image dimensions."""

  def _patch_stride_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
    if self._stride_coordinates is None:
      self._stride_coordinates = np.where(self._normalized_tissue_mask())
    return self._stride_coordinates

  def __getitem__(self, index: Union[int, slice]):
    y, x = self._patch_stride_coordinates()
    if isinstance(index, int):
      return self.patch_at_stride(
          StrideCoordinate(y_strides=y[index], x_strides=x[index])
      )
    return [
        self.patch_at_stride(StrideCoordinate(y_strides=y, x_strides=x))
        for x, y in zip(x[index], y[index])
    ]

  def __iter__(self) -> Iterator[EmbeddingPatch]:
    """Given tissue mask, emits patches.

    Yields:
      A sampled patch.

    Raises:
      DicomImageMissingRegionError if there is no region with the
      required luminance values.
    """
    y_coords, x_coords = self._patch_stride_coordinates()
    for y, x in zip(y_coords, x_coords):
      yield self.patch_at_stride(StrideCoordinate(y_strides=y, x_strides=x))


class DicomPatchGenerator(_BasePatchGenerator[dicom_slide.DicomPatch]):
  """A generator for patch sampling in a slide at specified pixel spacing.

  Patches are sampled from within the image using areas identified on a tissue
  mask.

  The sampling process moves a sliding window of size patch_size with
  steps of size stride_size. If the size of the patch is <= the size of the
  stride then patching is more efficient. The first sampled patch is centered at
  (-stride_size / 2, -stride_size / 2). We use (y_strides, x_strides) to denote
  how many strides we have taken along the y and x axis. Given a specified
  (y_strides, x_strides), a sampled patch can be uniquely identified and will
  be centered at (-stride_size / 2 + x_strides * stride_size,
  -stride_size / 2 + y_strides * stride_size).

  Patching is less efficient if the patch size > stride side. In this context
  patches overlap. Here the upper left coordinate of the patch is defined by
  x_strides * stride)size, ystrides * stride_size.

  Utility functions are also provided to translate between strides and
  patch coordinates.
  """

  def __init__(
      self,
      dicom_source: Union[
          dicom_slide.DicomSlide, dicom_slide.DicomMicroscopeImage
      ],
      source_image: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
          pixel_spacing.PixelSpacing,
      ],
      patch_size: int,
      mask: Optional[np.ndarray] = None,
      stride_size: Optional[int] = None,
      min_luminance: Optional[float] = None,
      max_luminance: Optional[float] = None,
      mask_level: Union[
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
          pixel_spacing.PixelSpacing,
          None,
      ] = None,
      mask_color_transform: Union[ImageCms.ImageCmsTransform, None] = None,
  ):
    """Constructor.

    Args:
      dicom_source: The slide to generate on.
      source_image: Image pyramid level which should patched.
      patch_size: Patch size at the pixel spacing that we sample patches.
      mask: Binary np.ndarray bool mask, if provided will define image patch
        sampling.
      stride_size: Stride size that patches are sampled at if a mask is not
        provided. Defaults to patch_size.
      min_luminance: Regions with luminance (grayscale) < this threshold are to
        be considered non-tissue background (Range 0.0 - 1.0); not applicable if
        a mask is provided.
      max_luminance: Regions with luminance (grayscale) > this threshold are to
        be considered non-tissue background (Range 0.0 - 1.0); not applicable if
        a mask is provided.
      mask_level: Pyramid level to use to determine where patches should be
        sampled if a mask is not provded; not applicable if a mask is provided.
      mask_color_transform: Color transform to apply when generating the mask;
        not applicable if a mask is provided.
    """
    self.dicom_source = dicom_source
    if isinstance(source_image, pixel_spacing.PixelSpacing):
      if not isinstance(dicom_source, dicom_slide.DicomSlide):
        raise ez_wsi_errors.InvalidTissueMaskError(
            'Can not initialize patch generator for DICOM microscopy images'
            ' using pixel spacing to defined the source imaging.'
        )
      dicom_source = typing.cast(dicom_slide.DicomSlide, dicom_source)
      image_level = _pixel_spacing_to_level(
          dicom_source, source_image, maximum_downsample=8.0
      )
    else:
      image_level = source_image
    if isinstance(dicom_source, dicom_slide.DicomMicroscopeImage):
      mask_level = image_level
    elif isinstance(dicom_source, dicom_slide.DicomSlide):
      if (
          isinstance(source_image, slide_level_map.Level)
          and image_level.number_of_frames == 1
      ):
        mask_level = image_level
      elif (
          isinstance(source_image, slide_level_map.ResizedLevel)
          and image_level.source_level.number_of_frames == 1
      ):
        mask_level = image_level
      elif isinstance(mask_level, pixel_spacing.PixelSpacing):
        mask_level = _pixel_spacing_to_level(
            dicom_source, mask_level, maximum_downsample=8.0
        )
      elif mask_level is None:
        if (
            image_level.pixel_spacing.is_defined
            and image_level.pixel_spacing.pixel_spacing_mm
            >= TISSUE_MASK_PIXEL_SPACING.pixel_spacing_mm
        ):
          mask_level = image_level
        else:
          mask_level = _pixel_spacing_to_level(
              dicom_source, TISSUE_MASK_PIXEL_SPACING, maximum_downsample=8.0
          )
    self._tissue_mask_level = mask_level
    self._image_level = image_level
    self._tissue_mask_color_transform = mask_color_transform
    super().__init__(
        stride_size, patch_size, min_luminance, max_luminance, mask
    )

  def get_tissue_mask(self) -> np.ndarray:
    if self._user_provided_tissue_mask is not None:
      return self._user_provided_tissue_mask.copy()
    if isinstance(self._tissue_mask_level, slide_level_map.ResizedLevel):
      number_of_frames = self._tissue_mask_level.source_level.number_of_frames
    elif isinstance(self._tissue_mask_level, slide_level_map.Level):
      number_of_frames = self._tissue_mask_level.number_of_frames
    else:
      raise ValueError('Unexpected object.')
    if number_of_frames <= _MAX_TISSUE_MASK_LEVEL_FRAME_COUNT_PRELOAD:
      self.dicom_source.preload_level_in_frame_cache(self._tissue_mask_level)
    return self.dicom_source.get_image(self._tissue_mask_level).image_bytes(
        self._tissue_mask_color_transform
    )

  def patch_at_stride(
      self, strides: StrideCoordinate
  ) -> dicom_slide.DicomPatch:
    """Converts tissue mask coordinates to patch coordinates.

    Args:
      strides: Number of strides taken from x and y axis.

    Returns:
      Corresponding Patch.
    """
    patch_bounds = self.strides_to_patch_bounds(strides)
    return self.dicom_source.get_patch(
        self._image_level,
        x=patch_bounds.x_origin,
        y=patch_bounds.y_origin,
        width=patch_bounds.width,
        height=patch_bounds.height,
    )

  def _image_dimensions(self) -> slide_level_map.ImageDimensions:
    """Returns image dimensions."""
    return slide_level_map.ImageDimensions(
        self._image_level.width, self._image_level.height
    )


class GcsImagePatchGenerator(_BasePatchGenerator[gcs_image.GcsPatch]):
  """A generator for patch sampling in a image at specified pixel spacing.

  The sampling process moves a sliding window of size patch_size with
  steps of size stride_size. The first sampled patch is centered at
  (-stride_size / 2, -stride_size / 2). We use (y_strides, x_strides) to denote
  how many strides we have taken along the y and x axis. Given a specified
  (y_strides, x_strides), a sampled patch can be uniquely identified and will
  be centered at (-stride_size / 2 + x_strides * stride_size,
  -stride_size / 2 + y_strides * stride_size).

  Utility functions are also provided to translate between strides and
  patch coordinates.
  color_transform: Color transform to apply when generating tissue mask.
  """

  def __init__(
      self,
      source_image: Union[gcs_image.GcsImage, local_image.LocalImage],
      patch_size: int,
      mask: Optional[np.ndarray] = None,
      stride_size: Optional[int] = None,
      min_luminance: Optional[float] = None,
      max_luminance: Optional[float] = None,
      mask_color_transform: Union[ImageCms.ImageCmsTransform, None] = None,
  ):
    """Constructor.

    Args:
      source_image: The slide to generate on.
      patch_size: Patch size at the pixel spacing that we sample patches.
      mask: Binary np.ndarray bool mask, if provided will define image patch
        sampling.
      stride_size: Stride size that patches are sampled at if a mask is not
        provided. Defaults to patch_size.
      min_luminance: Regions with luminance (grayscale) < this threshold are to
        be considered non-tissue background (Range 0.0 - 1.0); not applicable if
        a mask is provided.
      max_luminance: Regions with luminance (grayscale) > this threshold are to
        be considered non-tissue background (Range 0.0 - 1.0); not applicable if
        a mask is provided.
      mask_color_transform: Color transform to apply when generating the mask;
        not applicable if a mask is provided.
    """
    self.source_image = source_image
    self._tissue_mask_color_transform = mask_color_transform
    super().__init__(
        stride_size, patch_size, min_luminance, max_luminance, mask
    )

  def get_tissue_mask(self) -> np.ndarray:
    if self._user_provided_tissue_mask is not None:
      return self._user_provided_tissue_mask.copy()
    return self.source_image.image_bytes(self._tissue_mask_color_transform)

  def patch_at_stride(self, strides: StrideCoordinate) -> gcs_image.GcsPatch:
    """Converts tissue mask coordinates to patch coordinates.

    Args:
      strides: Number of strides taken from x and y axis.

    Returns:
      Corresponding Patch.
    """
    patch_bounds = self.strides_to_patch_bounds(strides)
    return self.source_image.get_patch(
        x=patch_bounds.x_origin,
        y=patch_bounds.y_origin,
        width=patch_bounds.width,
        height=patch_bounds.height,
    )

  def _image_dimensions(self) -> slide_level_map.ImageDimensions:
    """Returns image dimensions."""
    return slide_level_map.ImageDimensions(
        self.source_image.width, self.source_image.height
    )


GcsImagesToPatchesInputTypes = Union[
    gcs_image.GcsImageSourceTypes,
    gcs_image.GcsImage,
    Iterator[Union[gcs_image.GcsImageSourceTypes, gcs_image.GcsImage]],
    Sequence[Union[gcs_image.GcsImageSourceTypes, gcs_image.GcsImage]],
]


def _get_gcs_image(
    credential_factory: Optional[
        credential_factory_module.AbstractCredentialFactory
    ],
    image_dimensions: Optional[gcs_image.ImageDimensions],
    image: Union[gcs_image.GcsImageSourceTypes, gcs_image.GcsImage],
) -> gcs_image.GcsPatch:
  if isinstance(image, gcs_image.GcsImage):
    return image.get_image_as_patch()
  img = gcs_image.GcsImage(image, credential_factory, image_dimensions)
  return img.get_image_as_patch()


def gcs_images_to_patches(
    images: GcsImagesToPatchesInputTypes,
    credential_factory: Optional[
        credential_factory_module.AbstractCredentialFactory
    ] = None,
    image_dimensions: Optional[gcs_image.ImageDimensions] = None,
    max_thread_count: int = _MAX_GCS_IMAGE_READ_THEADS,
) -> Iterator[gcs_image.GcsPatch]:
  """Converts whole images in GCS into patches."""
  if isinstance(
      images, Union[gcs_image.GcsImageSourceTypes, gcs_image.GcsImage]
  ):
    images = [images]
  get_image = functools.partial(
      _get_gcs_image, credential_factory, image_dimensions
  )
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_thread_count
  ) as executor:
    return executor.map(get_image, images)


LocalImagesToPatchesInputTypes = Union[
    local_image.LocalImageSourceTypes,
    local_image.LocalImage,
    Iterator[Union[local_image.LocalImageSourceTypes, local_image.LocalImage]],
    Sequence[Union[local_image.LocalImageSourceTypes, local_image.LocalImage]],
]


def local_images_to_patches(
    images: LocalImagesToPatchesInputTypes,
    image_dimensions: Optional[local_image.ImageDimensions] = None,
) -> Iterator[gcs_image.GcsPatch]:
  """Converts whole local images in GCS into patches."""
  if isinstance(
      images, Union[local_image.LocalImageSourceTypes, local_image.LocalImage]
  ):
    images = [images]
  for image in images:
    if isinstance(image, local_image.LocalImage):
      li = image
    else:
      li = local_image.LocalImage(image, image_dimensions)
    yield li.get_image_as_patch()
