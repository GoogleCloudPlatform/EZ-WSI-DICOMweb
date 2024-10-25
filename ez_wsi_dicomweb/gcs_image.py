# Copyright 2024 Google LLC
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
"""Creates Images and Patches from tradtional image formats stored on GCS."""

from __future__ import annotations

import base64
import binascii
import copy
import dataclasses
import io
import threading
import typing
from typing import Any, Dict, MutableMapping, Optional, Union

import cv2
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import error_retry_util
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import slide_level_map
import google.api_core.exceptions
import google.auth
import google.cloud.storage
import numpy as np
import PIL
from PIL import ImageCms
import retrying


ImageDimensions = slide_level_map.ImageDimensions
_RGB = 'RGB'


@dataclasses.dataclass(frozen=True)
class _GcsImageState:
  width: int
  height: int
  image_bytes: np.ndarray
  icc_color_profile: Optional[bytes] = None


_CoreImageTypes = Union[str, np.ndarray, bytes]
GcsImageSourceTypes = Union[
    _CoreImageTypes,
    google.cloud.storage.Blob,
    google.cloud.storage.blob.Blob,
]


def _gcs_image_json_metadata(
    image: Union[GcsPatch, GcsImage],
    color_transform: Optional[ImageCms.ImageCmsTransform] = None,
) -> str:
  """Converts uncompressed RGB image to PNG and returns base64 encoded bytes."""
  image_bytes = image.image_bytes(color_transform)
  mode = {1: 'L', 3: _RGB}.get(
      dicom_slide.get_image_bytes_samples_per_pixel(image_bytes)
  )
  if mode is None:
    raise ez_wsi_errors.GcsImageError(
        f'Unsupported image samples per pixel; image shape: {image_bytes.shape}'
    )
  with io.BytesIO() as compressed_bytes:
    with PIL.Image.frombytes(
        mode=mode,
        size=(image.width, image.height),
        data=image_bytes.tobytes(),
        decoder_name='raw',
    ) as pil_image:
      pil_image.save(compressed_bytes, format='PNG')
    return base64.b64encode(compressed_bytes.getvalue()).decode('utf-8')


class GcsPatch(dicom_slide.BasePatch):
  """Represents a patch stored in GCS."""

  def __init__(
      self,
      source: GcsImage,
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: bool = False,
  ):
    super().__init__(x, y, width, height)
    self._source = source
    if (
        require_fully_in_source_image
        and not self.is_patch_fully_in_source_image()
    ):
      raise ez_wsi_errors.PatchOutsideOfImageDimensionsError(
          'A portion of the patch does not overlap the image.'
      )
    self._require_fully_in_source_image = require_fully_in_source_image

  @classmethod
  def create_from_json(
      cls,
      json_metadata: str,
      require_fully_in_source_image: bool = False,
      source_image_dimension: Optional[ImageDimensions] = None,
  ) -> GcsPatch:
    try:
      img = GcsImage(base64.b64decode(json_metadata, validate=True))
    except (binascii.Error, ValueError) as exp:
      raise ez_wsi_errors.GcsImageError('Error decoding image bytes') from exp
    patch = GcsPatch(
        img,
        0,
        0,
        img.width,
        img.height,
        require_fully_in_source_image=require_fully_in_source_image,
    )
    if (
        require_fully_in_source_image
        and source_image_dimension is not None
        and not patch.is_patch_fully_in_source_image_dim(
            source_image_dimension.width_px, source_image_dimension.height_px
        )
    ):
      raise ez_wsi_errors.PatchOutsideOfImageDimensionsError(
          'A portion of the patch does not overlap the image.'
      )
    return patch

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, GcsPatch):
      return False
    return (
        self.x == other.x
        and self.y == other.y
        and self.width == other.width
        and self.height == other.height
        and self._source == other._source
    )

  @property
  def is_resized(self) -> bool:
    return self._source.is_resized

  @property
  def source(self) -> GcsImage:
    return self._source

  def image_bytes(
      self, color_transform: Optional[ImageCms.ImageCmsTransform] = None
  ) -> np.ndarray:
    """Returns the patch's image bytes."""
    image_bytes = self._source.image_bytes()
    if len(image_bytes.shape) == 2:
      image_bytes = np.expand_dims(image_bytes, axis=-1)
    samples_per_pixel = dicom_slide.get_image_bytes_samples_per_pixel(
        image_bytes
    )
    cropped_image = np.zeros(
        (self.height, self.width, samples_per_pixel), dtype=image_bytes.dtype
    )
    y_start = min(max(self.y, 0), self._source.height)
    x_start = min(max(self.x, 0), self._source.width)
    y_end = min(max(self.y + self.height, 0), self._source.height)
    x_end = min(max(self.x + self.width, 0), self._source.width)
    copy_width = x_end - x_start
    copy_height = y_end - y_start

    if copy_width > 0 and copy_height > 0:
      dx = max(0, -self.x)
      dy = max(0, -self.y)
      sx = max(0, self.x)
      sy = max(0, self.y)
      cropped_image[dy : dy + copy_height, dx : dx + copy_width, :] = (
          image_bytes[sy : sy + copy_height, sx : sx + copy_width, :]
      )
    return dicom_slide.transform_image_bytes_color(
        cropped_image, color_transform
    )

  def get_patch(
      self,
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: Optional[bool] = None,
  ) -> GcsPatch:
    """Returns a patch at the specified location and size."""
    require_fully_in_source_image = (
        self._require_fully_in_source_image
        if require_fully_in_source_image is None
        else require_fully_in_source_image
    )
    return GcsPatch(
        self._source,
        x,
        y,
        width,
        height,
        require_fully_in_source_image=require_fully_in_source_image,
    )

  def is_patch_fully_in_source_image(self) -> bool:
    return self.is_patch_fully_in_source_image_dim(
        self._source.width, self._source.height
    )

  def get_gcp_data_credential_header(
      self, credential: Optional[google.auth.credentials.Credentials] = None
  ) -> Dict[str, str]:
    """Returns the credential header patch requests."""
    return self.source.get_credential_header(credential)

  def json_metadata(
      self,
      color_transform: Optional[ImageCms.ImageCmsTransform] = None,
  ) -> str:
    return _gcs_image_json_metadata(self, color_transform)


class GcsImage:
  """Represents an image stored in GCS."""

  def __init__(
      self,
      image_source: GcsImageSourceTypes,
      credential_factory: Optional[
          credential_factory_module.AbstractCredentialFactory
      ] = None,
      image_dimensions: Optional[ImageDimensions] = None,
  ):
    """Initializes the GcsImage.

    GCS image represents a image in tradtional image format (PNG, JPEG, etc)
    that is stored in GCS or passed in directly.

    Args:
      image_source: Image source can be a str representing a gs style path, e.g.
        gs://bucket/path/to/image.png; or numpy array containing uncompressed
          RGB or single channel image; or bytes that contain the compressed
          bytes of a tradtional image format, e.g. PNG or jpeg bytes.
      credential_factory: Credential factory that returns credentials to use
        reading from GCS.
      image_dimensions: Image dimensions to resize images to.

    Raises:
      GcsImageError: If image source is not supported.
    """
    self._gcs_image_lock = threading.RLock()
    self._icc_color_profile = None
    self._credentials = None
    self._source_image_compressed_bytes_size = 0
    self._source_image_compressed_bytes = b''
    self._image_resize_dims = (
        None if image_dimensions is None else image_dimensions.copy()
    )
    self._are_image_bytes_resized = False
    if not isinstance(image_source, _CoreImageTypes):
      # unit testing framwork makes direct testing of blob type difficult,
      # test that assume image is blob if not other expected types.
      image_source = typing.cast(google.cloud.storage.Blob, image_source)
      image_source = f'gs://{image_source.bucket.name}/{image_source.name}'
    if isinstance(image_source, np.ndarray):
      self._gcs_uri = ''
      self._credential_factory = (
          credential_factory_module.NoAuthCredentialsFactory()
      )
      self._image_bytes = image_source.copy()
      try:
        self._height, self._width = self._image_bytes.shape[:2]
      except ValueError as exp:
        raise ez_wsi_errors.GcsImageError(
            f'Unsupported image shape: {self._image_bytes.shape}'
        ) from exp
      if image_source.dtype != np.uint8:
        raise ez_wsi_errors.GcsImageError(
            f'Unsupported image dtype: {image_source.dtype}'
        )
      samples_per_pixel = dicom_slide.get_image_bytes_samples_per_pixel(
          self._image_bytes
      )
      if samples_per_pixel not in (1, 3, 4):
        raise ez_wsi_errors.GcsImageError(
            f'Unsupported image samples per pixel: {samples_per_pixel}'
        )
      if samples_per_pixel == 4:
        # If present Remove alpha channel
        self._image_bytes = self._image_bytes[:, :, :3]
      self._bytes_pre_pixel = samples_per_pixel
      self._resize()
      return
    # uninitialized values
    self._image_bytes = None
    self._width = -1
    self._height = -1
    self._bytes_pre_pixel = -1
    if isinstance(image_source, bytes):
      if not image_source:
        raise ez_wsi_errors.GcsImageError('Image bytes is empty.')
      self._gcs_uri = ''
      self._credential_factory = (
          credential_factory_module.NoAuthCredentialsFactory()
      )
      self._source_image_compressed_bytes = image_source
      self._init_compressed_image_bytes(image_source)
      return
    try:
      # test gcs url formatting looks correct.
      google.cloud.storage.Blob.from_string(image_source)
    except ValueError as exp:
      raise ez_wsi_errors.GcsImagePathFormatError(
          f'Invalid GCS URI: {image_source}'
      ) from exp
    self._gcs_uri = image_source
    self._credential_factory = (
        credential_factory_module.DefaultCredentialFactory()
        if credential_factory is None
        else credential_factory
    )
    if image_dimensions is not None:
      self._width = image_dimensions.width_px
      self._height = image_dimensions.height_px

  @property
  def is_resized(self) -> bool:
    """Returns true if image dimensions have/maybe resized."""
    with self._gcs_image_lock:
      if self.are_image_bytes_loaded:
        # Image dimensions were resized.
        return self._are_image_bytes_resized
    # If image bytes are not loaded then use the definition of the image
    # dimensions as a proxy to avoid loading actual image bytes.
    # return true if image dimensions have been defined. Its clearly possible
    # the defined resize dimensions and the actual image dimensions may
    # be the same.
    return self._image_resize_dims is not None  # safe to access outside of lock

  @property
  def resize_dimensions(self) -> Optional[ImageDimensions]:
    # Never modified outside of constructor. Safe to access outside of lock.
    return self._image_resize_dims

  def _resize(self):
    """Resizes image to resize_dims if provided."""
    resize_dims = self._image_resize_dims
    if resize_dims is None or self._image_bytes is None:
      return
    height, width = self._image_bytes.shape[:2]
    if width == resize_dims.width_px and height == resize_dims.height_px:
      return
    if resize_dims.width_px > width or resize_dims.height_px > height:
      resize_method = cv2.INTER_CUBIC
    else:
      resize_method = cv2.INTER_AREA
    self._are_image_bytes_resized = True
    self._width = resize_dims.width_px
    self._height = resize_dims.height_px
    self._image_bytes = cv2.resize(
        self._image_bytes, (self._width, self._height), resize_method
    )

  @property
  def are_image_bytes_loaded(self) -> bool:
    return self._image_bytes is not None

  def _init_compressed_image_bytes(
      self, source_image_compressed_bytes: bytes
  ) -> None:
    """Initialize image bytes from images compressed bytes."""
    self._source_image_compressed_bytes_size = len(
        source_image_compressed_bytes
    )
    with io.BytesIO(source_image_compressed_bytes) as image_bytes:
      try:
        with PIL.Image.open(image_bytes) as image:
          if image.mode in ('YCbCr', 'CMYK', 'HSV', 'LAB', 'RGBA'):
            image = image.convert('RGB')
          if image.mode == 'L':
            self._bytes_pre_pixel = 1
          elif image.mode == 'RGB':
            self._bytes_pre_pixel = 3
          else:
            raise ez_wsi_errors.GcsImageError(
                f'Unsupported image mode: {image.mode}'
            )
          self._icc_color_profile = image.info.get('icc_profile')
          self._image_bytes = np.asarray(image)
          self._width, self._height = image.size
          self._resize()
      except PIL.UnidentifiedImageError as exp:
        raise ez_wsi_errors.GcsImageError(
            'Error decoding image bytes.'
        ) from exp

  @property
  def credentials(self) -> google.auth.credentials.Credentials:
    with self._gcs_image_lock:
      if self._credentials is None:
        self._credentials = self._credential_factory.get_credentials()
      else:
        credential_factory_module.refresh_credentials(
            self._credentials, self._credential_factory
        )
      return self._credentials

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  @retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
  def _get_gcs_image(self) -> _GcsImageState:
    """Returns image width, height, bytes, and icc profile."""
    with self._gcs_image_lock:
      if self._image_bytes is not None:
        return _GcsImageState(
            self._width,
            self._height,
            self._image_bytes,
            self._icc_color_profile,
        )
      try:
        credentials = self.credentials
        if not credentials.token or isinstance(
            self._credential_factory,
            credential_factory_module.NoAuthCredentialsFactory,
        ):
          client = google.cloud.storage.Client.create_anonymous_client()
        else:
          client = google.cloud.storage.Client(credentials=self.credentials)
        gcs_blob = google.cloud.storage.Blob.from_string(
            self._gcs_uri,
            client=client,
        )
        raw_bytes = gcs_blob.download_as_bytes(raw_download=True)
      except google.api_core.exceptions.GoogleAPICallError as exp:
        raise ez_wsi_errors.raise_ez_wsi_http_exception(exp.message, exp)
      self._init_compressed_image_bytes(raw_bytes)
      return _GcsImageState(
          self._width, self._height, self._image_bytes, self._icc_color_profile
      )

  @classmethod
  def create_from_json(cls, json_metadata: str) -> GcsImage:
    try:
      return GcsImage(base64.b64decode(json_metadata, validate=True))
    except (binascii.Error, ValueError) as exp:
      raise ez_wsi_errors.GcsImageError('Error decoding image bytes') from exp

  @property
  def size_bytes_of_source_image(self) -> Optional[int]:
    with self._gcs_image_lock:
      if self._source_image_compressed_bytes_size == 0:
        return None
      return self._source_image_compressed_bytes_size

  def _get_source_image_bytes_from_file(self) -> bytes:
    return b''

  def source_image_bytes_json_metadata(self) -> str:
    """Returns bytes encoding source image.

    Raises:
      GcsImageError: If source image bytes are not set.
    """
    with self._gcs_image_lock:
      if self._are_image_bytes_resized:
        raise ez_wsi_errors.GcsImageError(
            'Source image bytes have been resized. Source image metadata is not'
            ' available.'
        )
      image_bytes = self._source_image_compressed_bytes
      if not image_bytes:
        image_bytes = self._get_source_image_bytes_from_file()
      if image_bytes:
        return base64.b64encode(image_bytes).decode('utf-8')
      raise ez_wsi_errors.GcsImageError(
          'Source image bytes are not set. Source image metadata is not'
          ' available.'
      )

  def clear_source_image_compressed_bytes(self) -> None:
    """Clears source image compressed bytes."""
    # Source image compressed bytes are used to call embedding api
    # when it is optimal pass a representation of the whole image.
    # This function enables the source bytes to be cleared to save
    # working memory.
    with self._gcs_image_lock:
      self._source_image_compressed_bytes = b''

  def json_metadata(
      self,
      color_transform: Optional[ImageCms.ImageCmsTransform] = None,
  ) -> str:
    return _gcs_image_json_metadata(self, color_transform)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, GcsImage):
      return False
    if self._gcs_uri and other._gcs_uri:
      return (
          self._gcs_uri == other._gcs_uri
          and self._image_resize_dims == other._image_resize_dims
      )
    if self._gcs_uri and not self.are_image_bytes_loaded:
      self._get_gcs_image()  # load images from GCS if required.
    if other._gcs_uri and not other.are_image_bytes_loaded:
      other._get_gcs_image()
    if self.width != other.width or self.height != other.height:
      return False
    # if loaded image bytes never cleared safe to access outside of lock.
    return np.array_equal(self._image_bytes, other._image_bytes)

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = copy.copy(self.__dict__)
    del state['_credentials']
    del state['_gcs_image_lock']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._credentials = None
    self._gcs_image_lock = threading.RLock()

  def get_credential_header(
      self, credentials: Optional[google.auth.credentials.Credentials] = None
  ) -> Dict[str, str]:
    """Returns credential header for retrieval of GCS image.

    Args:
      credentials: Optional credential to use if not provided will use
        credentials provided by credential factory encode the bearer token
        provided to the GcsImage constructor.

    Raises:
      InvalidCredentialsError: If credential factory is not initialized.
    """
    headers = {}
    if credentials is None:
      credentials = self.credentials
    else:
      credential_factory_module.refresh_credentials(credentials)
    credentials.apply(headers)
    return headers

  @property
  def uri(self) -> str:
    # Once initialized never changed. Safe to access outside of lock.
    return self._gcs_uri

  def get_patch(
      self,
      x: int,
      y: int,
      width: int,
      height: int,
      require_fully_in_source_image: bool = False,
  ) -> GcsPatch:
    """Returns a patch of the image."""
    return GcsPatch(
        self,
        x,
        y,
        width,
        height,
        require_fully_in_source_image=require_fully_in_source_image,
    )

  def get_image_as_patch(self) -> GcsPatch:
    return self.get_patch(
        0,
        0,
        self.width,
        self.height,
        require_fully_in_source_image=True,  # patch matches source image dim.
    )

  @property
  def width(self) -> int:
    if self._width == -1:
      self._get_gcs_image()  # initializes self._width
    return self._width

  @property
  def height(self) -> int:
    if self._height == -1:
      self._get_gcs_image()  # initializes self._height
    return self._height

  @property
  def bytes_pre_pixel(self) -> int:
    if self._bytes_pre_pixel == -1:
      self._get_gcs_image()
    return self._bytes_pre_pixel

  def create_icc_profile_transformation(
      self,
      icc_profile: Union[bytes, ImageCms.core.CmsProfile, None],
      rendering_intent: ImageCms.Intent = ImageCms.Intent.PERCEPTUAL,
  ) -> Optional[ImageCms.ImageCmsTransform]:
    """Returns transformation to from pyramid colorspace to icc_profile.

    Args:
      icc_profile: ICC Profile to DICOM Pyramid imaging to.
      rendering_intent: Rendering intent to use in transformation.

    Returns:
      PIL.ImageCmsTransformation to transform pixel imaging or None.
    """
    if icc_profile is None or not icc_profile:
      return None
    return dicom_slide.create_icc_profile_transformation(
        self._get_gcs_image().icc_color_profile, icc_profile, rendering_intent
    )

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
    return dicom_slide.transform_image_bytes_color(
        self._get_gcs_image().image_bytes, color_transform
    )
