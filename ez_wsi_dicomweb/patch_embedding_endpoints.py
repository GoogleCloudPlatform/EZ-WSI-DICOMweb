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
"""Core methods to return endpoint generated embeddings for patch pixels."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import concurrent
import copy
import dataclasses
import enum
import json
import math
import threading
import typing
from typing import Any, Callable, Generic, List, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar, Union

import cachetools
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import error_retry_util
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import patch_embedding_types
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import tags
import google.auth
import numpy as np
from PIL import ImageCms
import requests
import retrying


class EndpointJsonKeys:
  """JSON Keys for pathology v1 and v2 encoder endpoint."""

  # V2 encoder key types
  IMAGE_FILE_URI = 'image_file_uri'
  RAW_IMAGE_BYTES = 'raw_image_bytes'
  DICOM_PATH = 'dicom_path'
  SERIES_PATH = 'series_path'
  INSTANCE_UIDS = 'instance_uids'
  BEARER_TOKEN = 'bearer_token'
  INSTANCES = 'instances'

  PATCH_COORDINATES = 'patch_coordinates'
  X_ORIGIN = 'x_origin'
  Y_ORIGIN = 'y_origin'
  WIDTH = 'width'
  HEIGHT = 'height'

  EXTENSIONS = 'extensions'
  IMAGE_DIMENSIONS = 'image_dimensions'
  TRANSFORM_IMAGING_TO_ICC_PROFILE = 'transform_imaging_to_icc_profile'
  REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE = (
      'require_patches_fully_in_source_image'
  )
  EZ_WSI_STATE = 'ez_wsi_state'

  # key for list of patch bytes, base64 encoded
  PATCHES = 'patches'
  # key whole image bytes, base64 encoded
  IMAGE = 'image'
  # icc profile norm performed on patch imaging.
  ICC_PROFILE_METADATA_NORMALIZATION = 'icc_profile_metadata_normalization'
  # height and width of source image for patch
  SOURCE_IMAGE_WIDTH_PX = 'source_image_width_px'
  SOURCE_IMAGE_HEIGHT_PX = 'source_image_height_px'

  # V1 encoder only
  DICOM_WEB_STORE_URL = 'dicom_web_store_url'
  DICOM_STUDY_UID = 'dicom_study_uid'  # V1 encoder only
  DICOM_SERIES_UID = 'dicom_series_uid'  # V1 encoder only
  GCS_IMAGE_URL = 'gcs_image_url'  # V1 encoder only
  PROJECT_NAME = 'project_name'  # V1 encoder only
  PARAMETERS = 'parameters'  # V1 encoder only
  MODEL_SIZE = 'model_size'  # V1 encoder only
  MODEL_KIND = 'model_kind'  # V1 encoder only

  # embedding encoder response
  PREDICTIONS = 'predictions'
  MODEL_VERSION = 'model_version'
  VERTEXAI_ERROR = 'error'
  ERROR = 'error'
  ERROR_CODE = 'code'
  ERROR_CODE_DESCRIPTION = 'description'
  RESULT = 'result'
  EMBEDDINGS = 'embeddings'  # V1 encoder only
  ERROR_RESPONSE = 'error_response'  # V1 encoder only
  EMBEDDING_RESULT = 'embedding_result'  # V1 encoder only
  EMBEDDING_VECTOR = 'embedding_vector'  # V2 encoder
  PATCH_EMBEDDINGS = 'patch_embeddings'
  PATCH_COORDINATE = 'patch_coordinate'

  # Retryable error codes
  INVALID_CREDENTIALS = 'INVALID_CREDENTIALS_ERROR'


# maximum request size in bytes for endpoint. Less than vertex max to provide
# safety margin.
_MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = 1300000

_DEFAULT_RETRY_COUNT = 5
_DEFAULT_ENDPOINT_THREADS = 5
_DEFAULT_MAX_PATCHES_PER_REQUEST = 100

_MAX_ENDPOINT_THREADS = 10
_ITERATOR_MAX_ENDPOINT_PATCHES_PER_REQUEST = 100
_MAX_V1_ENDPOINT_PATCHES_PER_REQUEST = 3000
_MAX_V2_ENDPOINT_PATCHES_PER_REQUEST = 3000
_DEFAULT_DICOM_INSTANCE_ICC_PROFILE_CACHE_COUNT = 20

# Size safety buffer encode whole images may not exceed.
# Maxsize= _MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES - _WHOLE_IMAGE_SIZE_SAFTY_MARGIN
# Images exceeding this are are encoded as patches which enables them to be
# split across multiple VertexAI requests..
_WHOLE_IMAGE_SIZE_SAFTY_MARGIN = 300000

# Pyramid ICC profiles are optimally serialized in JSON to avoid repeative
# re-initialization. However, some digital pathology DICOM, e.g. Leica, have
# very large ICC profiles, e.g., 12 MB. The default max size of the ICC profile
# controls the maximum size of the ICC profile serialized in JSON.
_MAX_DICOM_SLIDE_ICCPROFILE_METADATA_SIZE = min(
    204800, slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES
)


class IccProfileNormalization(enum.Enum):
  """ICC Profile To Normalize Embedding Patches To."""

  NONE = 'NONE'
  SRGB = 'SRGB'
  ADOBERGB = 'ADOBERGB'
  ROMMRGB = 'ROMMRGB'


_UINT8_MAX_VALUE = 255.0


def _test_patch_coordinates_match(
    pc: Mapping[str, Any], x: int, y: int, width: int, height: int
) -> bool:
  """Test if dict encoded coordinates match expected coordinates."""
  try:
    if pc[EndpointJsonKeys.X_ORIGIN] != x or pc[EndpointJsonKeys.Y_ORIGIN] != y:
      return False
    if (
        pc.get(EndpointJsonKeys.WIDTH, width) != width
        or pc.get(EndpointJsonKeys.HEIGHT, height) != height
    ):
      return False
    return True
  except (IndexError, KeyError, ValueError, TypeError) as _:
    return False


def _get_icc_profile_bytes(
    icc_profile_normalization: IccProfileNormalization,
) -> bytes:
  """Returns ICC Profile bytes for endpoint."""
  if icc_profile_normalization == IccProfileNormalization.NONE:
    return b''
  if icc_profile_normalization == IccProfileNormalization.SRGB:
    return dicom_slide.get_srgb_icc_profile_bytes()
  if icc_profile_normalization == IccProfileNormalization.ADOBERGB:
    return dicom_slide.get_adobergb_icc_profile_bytes()
  if icc_profile_normalization == IccProfileNormalization.ROMMRGB:
    return dicom_slide.get_rommrgb_icc_profile_bytes()
  raise ez_wsi_errors.InternalError('ICC Profile not supported')


RequestResponseType = TypeVar('RequestResponseType')


def normalized_patch_channels(
    width: int, height: int, patch: np.ndarray
) -> np.ndarray:
  """Normalize monochrome and RGBA imaging to RGB."""
  if patch.shape == (height, width, 3):
    return patch
  if patch.shape == (height, width):
    patch = np.expand_dims(patch, axis=-1)
  if patch.shape == (height, width, 1):
    mem = np.zeros((height, width, 3), dtype=patch.dtype)
    mem[..., np.arange(3)] = patch[...]
    return mem
  if patch.shape == (height, width, 4):
    return patch[..., :3]
  raise ez_wsi_errors.PatchEmbeddingDimensionError


class AbstractPreparedEmbeddingRequest(
    Generic[RequestResponseType], metaclass=abc.ABCMeta
):
  """Base class for prepared embedding requests."""

  def __init__(
      self,
      slide_embedding_source: patch_embedding_types.SlideEmbeddingSource,
  ):
    self._slide_embedding_source = slide_embedding_source

  @property
  def slide_embedding_source(
      self,
  ) -> patch_embedding_types.SlideEmbeddingSource:
    return self._slide_embedding_source

  @property
  @abc.abstractmethod
  def json_size_in_bytes(self) -> int:
    """Return size in bytes of json sent to endpoint."""

  @abc.abstractmethod
  def finalize(self) -> None:
    """finalize after this there will be no more changes."""

  @abc.abstractmethod
  def split(
      self,
      endpoint: AbstractPatchEmbeddingEndpoint[RequestResponseType],
      max_size: int,
  ) -> Tuple[
      Optional[AbstractPreparedEmbeddingRequest[RequestResponseType]],
      patch_embedding_types.SlideEmbeddingSource,
  ]:
    """Splits object into parts which meet size and exceed size req."""


def _copy_dict_excluding_keys(
    state: Mapping[str, Any],
    exclude: List[List[str]],
    base_keys: Optional[List[str]] = None,
) -> MutableMapping[str, Any]:
  """Duplicates str keyed dict excluding predfined keys."""
  if base_keys is None:
    base_keys = []
  copy_dict = {}
  for key, value in state.items():
    base_keys.append(key)
    if base_keys in exclude:
      base_keys.pop()
      continue
    if not isinstance(value, Mapping):
      copy_dict[key] = value
    else:
      copy_dict[key] = _copy_dict_excluding_keys(value, exclude, base_keys)
    base_keys.pop()
  return copy_dict


@dataclasses.dataclass
class _VertexModelResult:
  instances: List[Mapping[str, Any]]


class PreparedVertexEmbeddingRequest(
    AbstractPreparedEmbeddingRequest[_VertexModelResult]
):
  """Internral respresentation of embedding json embedding requests."""

  def __init__(
      self,
      prepared_request: Mapping[str, Any],
      slide_embedding_source: patch_embedding_types.SlideEmbeddingSource,
  ):
    super().__init__(slide_embedding_source)
    self._embedding_request: Mapping[str, Any] = prepared_request
    self._embedding_json: Optional[str] = None

  @classmethod
  def init_from_json_finalized(
      cls,
      json_str: str,
      slide_embedding_source: patch_embedding_types.SlideEmbeddingSource,
  ) -> PreparedVertexEmbeddingRequest:
    """create finalized vertex embedding request from json and source."""
    instance = PreparedVertexEmbeddingRequest.__new__(
        PreparedVertexEmbeddingRequest
    )
    super(PreparedVertexEmbeddingRequest, instance).__init__(
        slide_embedding_source
    )
    instance._embedding_json = json_str
    instance._embedding_request = None
    return instance

  @property
  def embedding_request(self) -> Mapping[str, Any]:
    if self._embedding_request is None:
      raise ez_wsi_errors.InternalError('Request has been finalized.')
    return self._embedding_request

  @property
  def json(self) -> str:
    if self._embedding_json is None:
      self._embedding_json = json.dumps(self._embedding_request)
    return self._embedding_json

  @property
  def json_size_in_bytes(self) -> int:
    return len(self.json)

  def finalize(self) -> None:
    if self._embedding_json is None:
      self._embedding_json = json.dumps(self._embedding_request)
    self._embedding_request = None

  def _non_splitable(
      self,
  ) -> Tuple[None, patch_embedding_types.SlideEmbeddingSource]:
    return None, self.slide_embedding_source

  def _split_results(self, split_request: str, end_split_index: int) -> Tuple[
      Optional[PreparedVertexEmbeddingRequest],
      patch_embedding_types.SlideEmbeddingSource,
  ]:
    """Returns split prepared vertex embedding result."""
    split_half_slide_embedding_source = (
        patch_embedding_types.SlideEmbeddingSource(
            self.slide_embedding_source.patches[:end_split_index]
        )
    )
    finalized_split_half = (
        PreparedVertexEmbeddingRequest.init_from_json_finalized(
            split_request, split_half_slide_embedding_source
        )
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        self.slide_embedding_source.patches[end_split_index:]
    )
    return (
        finalized_split_half,
        slide_embedding_source,
    )

  def _split_on_patch_coordinates_only(
      self,
      endpoint: AbstractPatchEmbeddingEndpoint[_VertexModelResult],
      max_size: int,
  ) -> Tuple[
      Optional[PreparedVertexEmbeddingRequest],
      patch_embedding_types.SlideEmbeddingSource,
  ]:
    """If possilbe splits prepared request to meet size req."""
    if self.json_size_in_bytes < endpoint.max_request_size_bytes():
      # it is much less desirable to split DicomPatches or GcsImages into
      # multiple requests. The majority of the metadata could be in state
      # which cannot be split and would be sent in duplicate across multiple
      # requests. Test if entire message could be encoded by itself. If it can
      # defer sending.
      return self._non_splitable()
    try:
      coordinates = self.embedding_request[EndpointJsonKeys.PATCH_COORDINATES]
    except KeyError:
      return self._non_splitable()
    # make copy of whats in patch exclude coordinates and cached patches.
    base_request = _copy_dict_excluding_keys(
        self.embedding_request,
        [[EndpointJsonKeys.PATCH_COORDINATES]],
    )
    request_size = len(json.dumps(base_request))
    end_split_index = 0
    for coordinate in coordinates:
      patch_md_size = len(json.dumps(coordinate))
      if request_size + patch_md_size >= max_size:
        break
      request_size += patch_md_size
      end_split_index += 1
    while True:
      if end_split_index <= 0:
        return self._non_splitable()
      split_request = copy.copy(base_request)
      split_request[EndpointJsonKeys.PATCH_COORDINATES] = coordinates[
          :end_split_index
      ]
      split_request = json.dumps(split_request)
      if len(split_request) <= max_size:
        break
      end_split_index -= 1
    return self._split_results(split_request, end_split_index)

  def split(
      self,
      endpoint: AbstractPatchEmbeddingEndpoint[_VertexModelResult],
      max_size: int,
  ) -> Tuple[
      Optional[PreparedVertexEmbeddingRequest],
      patch_embedding_types.SlideEmbeddingSource,
  ]:
    """If possible splits object into parts which meet size and exceed size req."""
    if EndpointJsonKeys.DICOM_PATH in self.embedding_request:
      return self._split_on_patch_coordinates_only(endpoint, max_size)
    elif EndpointJsonKeys.IMAGE_FILE_URI in self.embedding_request:
      try:
        ez_wsi_state = self.embedding_request[EndpointJsonKeys.EXTENSIONS][
            EndpointJsonKeys.EZ_WSI_STATE
        ]
        patches = ez_wsi_state[EndpointJsonKeys.PATCHES]
        coordinates = self.embedding_request[EndpointJsonKeys.PATCH_COORDINATES]
      except KeyError:
        return self._split_on_patch_coordinates_only(endpoint, max_size)
      if len(patches) <= 1:
        return self._non_splitable()
      if len(coordinates) != len(patches):
        raise ez_wsi_errors.InternalError(
            'Patch state and coordinate counts do not match.'
        )
      # make copy of whats in patch exclude coordinates and cached patches.
      base_request = _copy_dict_excluding_keys(
          self.embedding_request,
          [
              [EndpointJsonKeys.PATCH_COORDINATES],
              [
                  EndpointJsonKeys.EXTENSIONS,
                  EndpointJsonKeys.EZ_WSI_STATE,
                  EndpointJsonKeys.PATCHES,
              ],
          ],
      )
      request_size = len(json.dumps(base_request))
      end_split_index = 0
      for patch_metadata in patches:
        patch_md_size = len(patch_metadata)
        if request_size + patch_md_size >= max_size:
          break
        request_size += patch_md_size
        end_split_index += 1
      while True:
        if end_split_index <= 0:
          return self._non_splitable()
        patch_coordinate_size = len(json.dumps(coordinates[:end_split_index]))
        if patch_coordinate_size + request_size < max_size:
          break
        end_split_index -= 1
        if end_split_index > 0:
          request_size -= len(patches[end_split_index])
      while True:
        if end_split_index <= 0:
          return self._non_splitable()
        split_request = copy.copy(base_request)
        split_request[EndpointJsonKeys.PATCH_COORDINATES] = coordinates[
            :end_split_index
        ]
        split_request[EndpointJsonKeys.EXTENSIONS][
            EndpointJsonKeys.EZ_WSI_STATE
        ][EndpointJsonKeys.PATCHES] = patches[:end_split_index]
        split_request = json.dumps(split_request)
        if len(split_request) <= max_size:
          break
        end_split_index -= 1
      return self._split_results(split_request, end_split_index)
    raise ez_wsi_errors.InternalError('unidentified JSON')


class AbstractPatchEmbeddingEndpoint(
    Generic[RequestResponseType], metaclass=abc.ABCMeta
):
  """Abstract class for embedding endpoint."""

  def __init__(
      self,
      icc_profile_normalization: IccProfileNormalization,
  ):
    self._icc_profile_normalization = icc_profile_normalization
    self._icc_profile_bytes = None

  @abc.abstractmethod
  def max_request_size_bytes(self) -> int:
    """Maximum size in bytes that can be sent in single request."""

  @abc.abstractmethod
  def max_threads(self) -> int:
    """Returns maximum number of threads to spawn."""

  @abc.abstractmethod
  def patch_width(self) -> int:
    """Returns embedding endpoint input size width in pixels."""

  @abc.abstractmethod
  def patch_height(self) -> int:
    """Returns embedding endpoint input size height in pixels."""

  @abc.abstractmethod
  def max_number_of_patches_per_request(self) -> int:
    """Maximum number of patches to send endpoint in a request."""

  @abc.abstractmethod
  def endpoint_max_number_of_patches_per_request(self) -> int:
    """Maximum number of patches that the endpoint supports."""

  @abc.abstractmethod
  def retry_count(self) -> int:
    """Maximum number of get_embedding attempts before endpoint raises.."""

  @abc.abstractmethod
  def prepare_embedding_request(
      self,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> AbstractPreparedEmbeddingRequest[RequestResponseType]:
    """Converts slide embedding source to request to JSON formatted request."""

  @abc.abstractmethod
  def request_embeddings(
      self,
      embedding_inputs: Sequence[
          AbstractPreparedEmbeddingRequest[RequestResponseType]
      ],
  ) -> RequestResponseType:
    """Returns raw embedding result."""

  @property
  def request_embedding_return_type(self) -> Type[RequestResponseType]:
    return RequestResponseType

  @abc.abstractmethod
  def process_response(
      self,
      embedding_inputs: Sequence[patch_embedding_types.SlideEmbeddingSource],
      msg: RequestResponseType,
  ) -> List[patch_embedding_types.PatchEmbeddingEnsembleResult]:
    """Converts raw embedding response to list of embedding results."""

  @property
  def icc_profile_normalization(self) -> IccProfileNormalization:
    """Returns ICC Profile bytes for endpoint will transform imaging to."""
    return self._icc_profile_normalization

  def icc_profile_bytes(self) -> bytes:
    """Returns ICC Profile bytes for endpoint will transform imaging to."""
    if self._icc_profile_bytes is None:
      self._icc_profile_bytes = _get_icc_profile_bytes(
          self._icc_profile_normalization
      )
    return self._icc_profile_bytes


def _get_gcs_image_md_size(
    json_metadata: Mapping[str, Union[int, str, List[str]]],
) -> int:
  """returns size in bytes of data encoded in metadata."""
  size = 0
  for value in json_metadata.values():
    if isinstance(value, int):
      size += len(str(value))
    elif isinstance(value, str):
      size += len(value)
    elif isinstance(value, list):
      size += sum(len(md) for md in value)
    else:
      raise ez_wsi_errors.InternalError(
          f'Unsupported metadata value type: {value}'
      )
  return size


def _patch_pixel_area(
    slide_embedding: patch_embedding_types.SlideEmbeddingSource,
) -> int:
  patch_pixel_area = 0
  for ip in slide_embedding.patches:
    patch_pixel_area += ip.patch.width * ip.patch.height
  return patch_pixel_area


def _get_gcs_whole_image_metadata(
    source_image: gcs_image.GcsImage,
    icc_profile_normalization: IccProfileNormalization,
    c_transform: Optional[ImageCms.ImageCmsTransform],
    max_image_size_bytes: int,
) -> Tuple[Mapping[str, Any], int]:
  """Return whole image metadata and metadata size in bytes."""
  if source_image.size_bytes_of_source_image is not None:
    # image bytes are base64 encoded estimate actual size as 4x original.
    # plus 100 bytes of padding.
    estimated_size = int(
        math.ceil(source_image.size_bytes_of_source_image * 8 / 6)
        + len(IccProfileNormalization.NONE.value)
    )
    if estimated_size >= max_image_size_bytes:
      return {}, estimated_size
  try:
    source_image_metadata = {
        EndpointJsonKeys.IMAGE: source_image.source_image_bytes_json_metadata(),
        EndpointJsonKeys.ICC_PROFILE_METADATA_NORMALIZATION: (
            IccProfileNormalization.NONE.value
        ),
    }
  except ez_wsi_errors.GcsImageError:
    # image bytes initialized from in memory representation.
    source_image_md_size = int(
        math.ceil(
            source_image.width
            * source_image.height
            * source_image.bytes_pre_pixel
            * 8  # bits per channel
            / 12  # base 64 encoding + est 2x reduction due to PNG compression
        )
        + len(icc_profile_normalization.value)
    )
    if source_image_md_size >= max_image_size_bytes:
      return {}, source_image_md_size
    source_image_metadata = {
        EndpointJsonKeys.IMAGE: source_image.json_metadata(c_transform),
        EndpointJsonKeys.ICC_PROFILE_METADATA_NORMALIZATION: (
            icc_profile_normalization.value
        ),
    }
  source_image_md_size = _get_gcs_image_md_size(source_image_metadata)
  if source_image_md_size >= max_image_size_bytes:
    return {}, source_image_md_size
  return source_image_metadata, source_image_md_size


def _gcs_image_json_metadata(
    slide_embedding: patch_embedding_types.SlideEmbeddingSource,
    icc_profile_normalization: IccProfileNormalization,
    c_transform: Optional[ImageCms.ImageCmsTransform],
    max_endpoint_request_size_bytes: int,
) -> Mapping[str, Any]:
  """Returns metadata for GCS images."""
  patch = typing.cast(gcs_image.GcsPatch, slide_embedding.patches[0].patch)
  source_image = patch.source
  max_image_size_bytes = max(
      0, max_endpoint_request_size_bytes - _WHOLE_IMAGE_SIZE_SAFTY_MARGIN
  )
  source_image_metadata, source_image_md_size = _get_gcs_whole_image_metadata(
      source_image,
      icc_profile_normalization,
      c_transform,
      max_image_size_bytes,
  )
  if not source_image_metadata or _patch_pixel_area(slide_embedding) < int(
      0.95 * float(source_image.width * source_image.height)
  ):
    # If patch area is smaller than 95% source image area, compute patch
    # metadata. (5% factor to account for over head associated defining
    # multiple patches instead of single image.)
    patch_metadata = {
        EndpointJsonKeys.SOURCE_IMAGE_WIDTH_PX: int(patch.source.width),
        EndpointJsonKeys.SOURCE_IMAGE_HEIGHT_PX: int(patch.source.height),
        EndpointJsonKeys.ICC_PROFILE_METADATA_NORMALIZATION: (
            icc_profile_normalization.value
        ),
        EndpointJsonKeys.PATCHES: [
            typing.cast(gcs_image.GcsPatch, ip.patch).json_metadata(c_transform)
            for ip in slide_embedding.patches
        ],
    }
    if (
        not source_image_metadata
        or _get_gcs_image_md_size(patch_metadata) < source_image_md_size
    ):
      return patch_metadata
  return source_image_metadata


def _get_gcs_image_metadata(
    max_endpoint_request_size_bytes: int,
    encode_patch_data_in_request: bool,
    icc_profile_normalization: IccProfileNormalization,
    icc_profile_bytes: bytes,
    bearer_token: str,
    slide_embedding: patch_embedding_types.SlideEmbeddingSource,
) -> Mapping[str, Any]:
  """Returns metadata for GCS images."""
  source_image = typing.cast(
      gcs_image.GcsPatch, slide_embedding.patches[0].patch
  ).source
  c_transform = source_image.create_icc_profile_transformation(
      icc_profile_bytes
  )
  gcs_image_url = source_image.uri
  if not bearer_token or not gcs_image_url:
    # always send image data if no bearer token or gcs image url or project.
    return _gcs_image_json_metadata(
        slide_embedding,
        icc_profile_normalization,
        c_transform,
        max_endpoint_request_size_bytes,
    )
  if encode_patch_data_in_request and source_image.are_image_bytes_loaded:
    json_metadata = _gcs_image_json_metadata(
        slide_embedding,
        icc_profile_normalization,
        c_transform,
        max_endpoint_request_size_bytes,
    )
    if (
        source_image.size_bytes_of_source_image is None
        or _get_gcs_image_md_size(json_metadata)
        <= source_image.size_bytes_of_source_image
    ):
      # Send JSON metadata if the image was initialized from raw bytes or
      # the size of the json is smaller than source image.
      return json_metadata
  return {}


def _get_dicom_instance_uids_and_required_levels(
    slide_embedding: patch_embedding_types.SlideEmbeddingSource,
) -> Tuple[List[str], List[str]]:
  """Returns list of DICOM instances and leveles required for WSI patches."""
  instance_uids = set()
  required_levels = []
  for patch in slide_embedding.patches:
    level = typing.cast(
        dicom_slide.DicomPatch, patch.patch
    ).get_pyramid_imaging_source_level()
    if level in required_levels:
      continue
    required_levels.append(level)
    for instance in level.instances.values():
      instance_uids.add(instance.dicom_object.get_value(tags.SOP_INSTANCE_UID))
  return list(instance_uids), required_levels


class _PatchEmbeddingEndpointBase(
    AbstractPatchEmbeddingEndpoint[_VertexModelResult]
):
  """Shared implementation of patch embedding endpoint for V1 and V2 endpoints."""

  def __init__(
      self,
      patch_width: int,
      patch_height: int,
      icc_profile_normalization: IccProfileNormalization,
      send_gcs_patch_bytes_from_client_to_server: bool,
      end_point_url: str,
      max_threads: int,
      max_patches_per_request: int,
      endpoint_max_patches_per_request: int,
      retry_count: int,
      credential_factory: Optional[
          credential_factory_module.AbstractCredentialFactory
      ],
      expected_model_version: str,
  ):
    super().__init__(icc_profile_normalization)
    self._patch_width = patch_width
    self._patch_height = patch_height
    self._expected_model_version = expected_model_version
    self._credentials = None
    self._credentials_factory = (
        credential_factory
        if credential_factory is not None
        else credential_factory_module.DefaultCredentialFactory()
    )
    self._send_gcs_patch_bytes_from_client_to_server = (
        send_gcs_patch_bytes_from_client_to_server
    )
    self._end_point_url = end_point_url
    self._max_threads = max(1, min(max_threads, _MAX_ENDPOINT_THREADS))
    self._endpoint_max_patches_per_request = int(
        max(1, endpoint_max_patches_per_request)
    )
    self._max_patches_per_request = int(
        max(
            1,
            min(
                max_patches_per_request, self._endpoint_max_patches_per_request
            ),
        )
    )
    self._retry_count = max(0, retry_count)

  @property
  def end_point_url(self) -> str:
    return self._end_point_url

  @property
  def expected_model_version(self) -> str:
    return self._expected_model_version

  def max_request_size_bytes(self) -> int:
    """Maximum size in bytes that can be sent in single request."""
    return _MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES

  @property
  def credentials(self) -> google.auth.credentials.Credentials:
    if self._credentials is None:
      self._credentials = self._credentials_factory.get_credentials()
    else:
      self._credentials = credential_factory_module.refresh_credentials(
          self._credentials, self._credentials_factory
      )
    return self._credentials

  def vertex_endpoint_authentication_header(self) -> MutableMapping[str, str]:
    headers = {}
    self.credentials.apply(headers)
    return headers

  def retry_count(self) -> int:
    return self._retry_count

  def max_threads(self) -> int:
    return self._max_threads

  def patch_width(self) -> int:
    return self._patch_width

  def patch_height(self) -> int:
    return self._patch_height

  def max_number_of_patches_per_request(self) -> int:
    return self._max_patches_per_request

  def endpoint_max_number_of_patches_per_request(self) -> int:
    """Maximum number of patches that can be sent to the endpoint at once."""
    return self._endpoint_max_patches_per_request

  @abc.abstractmethod
  def _dicom_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Returns DICOM patch embedding request.

    Args:
      bearer_token: Bearer for store requests.
      slide_embedding: DICOM embedding inputs.

    Returns:
      JSON formatted embedding request.
    """

  @abc.abstractmethod
  def _get_embedding_request(
      self, embedding_inputs: Sequence[PreparedVertexEmbeddingRequest]
  ) -> str:
    """Returns patch embedding request.

    Args:
      embedding_inputs: Embedding inputs.

    Returns:
      JSON formatted embedding request.
    """

  @abc.abstractmethod
  def _gcs_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Returns GCS patch embedding request.

    Args:
      bearer_token: Bearer for store requests.
      slide_embedding: GCS embedding inputs.

    Returns:
      JSON formatted embedding request.
    """

  def prepare_embedding_request(
      self,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> PreparedVertexEmbeddingRequest:
    first_patch = slide_embedding.patches[0].patch
    if len(slide_embedding.patches) > 1:
      first_patch_type = type(first_patch)
      for p in slide_embedding.patches:
        if not isinstance(p.patch, first_patch_type):
          raise ez_wsi_errors.InternalError(
              'Patch in request are not all the same type.'
          )
    if isinstance(first_patch, dicom_slide.DicomPatch):
      bearer_token = slide_embedding.get_bearer_token()
      return PreparedVertexEmbeddingRequest(
          self._dicom_patch_embedding_request(
              bearer_token,
              slide_embedding,
          ),
          slide_embedding,
      )
    elif isinstance(first_patch, gcs_image.GcsPatch):
      bearer_token = slide_embedding.get_bearer_token()
      return PreparedVertexEmbeddingRequest(
          self._gcs_patch_embedding_request(
              bearer_token,
              slide_embedding,
          ),
          slide_embedding,
      )
    raise ez_wsi_errors.InternalError(
        'Patch is not a dicom_slide.DicomPatch or gcs_image.GcsPatch.'
    )

  @property
  def send_gcs_patch_bytes_from_client_to_server(self) -> bool:
    return self._send_gcs_patch_bytes_from_client_to_server

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  @retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
  def _request_embeddings(self, json_msg: str) -> str:
    """Sends json request to Vertex AI endpoint."""
    try:
      headers = self.vertex_endpoint_authentication_header()
      headers['Content-Length'] = f'{len(json_msg)}'
      headers['Content-Type'] = 'application/json'
      response = requests.post(
          self._end_point_url,
          headers=headers,
          data=json_msg,
      )
      # Raises a HTTPError if the response code was not 200
      response.raise_for_status()
      return response.text
    except requests.exceptions.HTTPError as exp:
      ez_wsi_errors.raise_ez_wsi_http_exception(exp.response.reason, exp)

  @abc.abstractmethod
  def _is_request_error_retryable(
      self, json_response: Mapping[str, Any]
  ) -> bool:
    """Returns true if error at request level is retryable."""

  @abc.abstractmethod
  def _decode_response(
      self,
      embedding_inputs: Sequence[PreparedVertexEmbeddingRequest],
      json_response: Mapping[str, Any],
  ) -> _VertexModelResult:
    """Decodes json_response response from Vertex AI endpoint into _VertexModelResult."""

  @abc.abstractmethod
  def _instance_has_retryable_error(self, json_dict: Mapping[str, Any]) -> bool:
    """Decodes response from Vertex AI endpoint into _VertexModelResult."""

  def _regenerate_instance_with_new_auth_token(
      self, prepared_request: PreparedVertexEmbeddingRequest
  ) -> PreparedVertexEmbeddingRequest:
    prepared_request = self.prepare_embedding_request(
        prepared_request.slide_embedding_source
    )
    prepared_request.finalize()
    return prepared_request

  def _merge_embedding_input_results(
      self, model_result: _VertexModelResult, partial_result: _VertexModelResult
  ) -> _VertexModelResult:
    partial_result_counter = 0
    for index in range(len(model_result.instances)):
      if self._instance_has_retryable_error(model_result.instances[index]):
        model_result.instances[index] = partial_result.instances[
            partial_result_counter
        ]
        partial_result_counter += 1
    return model_result

  def _regenerate_embedding_input_requests_with_new_auth_token(
      self, embedding_inputs: Sequence[PreparedVertexEmbeddingRequest]
  ) -> Tuple[str, Sequence[PreparedVertexEmbeddingRequest]]:
    embedding_inputs = [
        self._regenerate_instance_with_new_auth_token(ei)
        for ei in embedding_inputs
    ]
    return self._get_embedding_request(embedding_inputs), embedding_inputs

  def _retry_failed_embedding_input_requests(
      self,
      model_result: _VertexModelResult,
      embedding_inputs: Sequence[PreparedVertexEmbeddingRequest],
  ):
    retry_list = []
    for index, embedding_input in enumerate(embedding_inputs):
      if self._instance_has_retryable_error(model_result.instances[index]):
        retry_list.append(embedding_input)
    if not retry_list:
      return '', retry_list
    return self._regenerate_embedding_input_requests_with_new_auth_token(
        retry_list
    )

  def request_embeddings(
      self,
      embedding_inputs: Sequence[
          AbstractPreparedEmbeddingRequest[_VertexModelResult]
      ],
  ) -> _VertexModelResult:
    """Requests embeddings from Vertex AI endpoint.

    Method is roboust to expiration of authentication tokens in either instance
    or endpoint. Will retry three times to attempt to correct authentication
    issues.

    Args:
      embedding_inputs: Prepared list of instances (images) with patches to get
        embeddings.

    Returns:
      VertexModelResult, (List of JSON) results for each instance.
    """
    if not embedding_inputs:
      return _VertexModelResult([])
    embedding_inputs = typing.cast(
        Sequence[PreparedVertexEmbeddingRequest], embedding_inputs
    )
    attempts = 0
    # generates initial embedding request
    json_msg = self._get_embedding_request(embedding_inputs)
    if not json_msg and not embedding_inputs:
      return _VertexModelResult([])
    request_embedding_inputs = embedding_inputs
    model_result = None
    # This retry loop handles authentication errors encountered by
    # the endpoint attempting to connect to data sources for which
    # expired tokens have been provided. In this case we are connecting
    # to the endpoint, but the endpoint is unable to fulfill the request
    # completely and is returning errors in one or more of its responses.
    # More general authentication and retry logic for connecting to the endpoint
    # is handled by the decorators attached to _request_embeddings.
    while True:
      # Get get JSON response from Vertex AI endpoint.
      json_response = self._request_embeddings(json_msg)
      attempts += 1
      # Decode JSON response.
      try:
        json_response = json.loads(json_response)
      except json.JSONDecodeError as exp:
        raise ez_wsi_errors.PatchEmbeddingEndpointError(
            'Endpoint returned invalid JSON.'
        ) from exp
      # If error at response level is found and retryable, regenerate request
      # whole request with new authentication token and retry.
      if self._is_request_error_retryable(json_response):
        json_msg, request_embedding_inputs = (
            self._regenerate_embedding_input_requests_with_new_auth_token(
                request_embedding_inputs
            )
        )
        continue
      # Decode json response into _VertexModelResult.
      result = self._decode_response(request_embedding_inputs, json_response)
      if model_result is None:
        model_result = result
      else:
        # if results already processed merge with prexisiting results.
        self._merge_embedding_input_results(model_result, result)
      if attempts >= 3:
        # if on third attempt, then just return result.
        # should never take more than two attempts.
        return model_result
      # Retry to retrieve embeddings only for the instances that failed.
      # with retriable errors.
      json_msg, request_embedding_inputs = (
          self._retry_failed_embedding_input_requests(
              model_result, embedding_inputs
          )
      )
      if not json_msg:
        # if json message is empty then nothing to do but return.
        if model_result is None:
          # return empty result should never occure.
          return _VertexModelResult([])
        else:
          return model_result


class V1PatchEmbeddingEndpoint(_PatchEmbeddingEndpointBase):
  """Implements Patch embedding V1 API."""

  def __init__(
      self,
      patch_width: int = 224,
      patch_height: int = 224,
      endpoint_api_version: str = 'v1',  # Vertex API version
      project_id: str = 'hai-cd3-foundations',
      endpoint_location: str = 'us-central1',
      endpoint_id: str = '160',
      max_threads: int = _DEFAULT_ENDPOINT_THREADS,
      max_patches_per_request: int = _DEFAULT_MAX_PATCHES_PER_REQUEST,
      retry_count: int = _DEFAULT_RETRY_COUNT,
      send_gcs_patch_bytes_from_client_to_server: bool = False,
      credential_factory: Optional[
          credential_factory_module.AbstractCredentialFactory
      ] = None,
      expected_model_version: str = '',
  ):
    end_point: List[str] = [
        f'https://{endpoint_location}-aiplatform.googleapis.com',
        endpoint_api_version,
        'projects',
        project_id,
        'locations',
        endpoint_location,
        'endpoints',
        f'{endpoint_id}:predict',
    ]
    end_point_url = '/'.join([ep.strip('/') for ep in end_point])
    super().__init__(
        patch_width,
        patch_height,
        IccProfileNormalization.NONE,
        send_gcs_patch_bytes_from_client_to_server,
        end_point_url,
        max_threads,
        max_patches_per_request,
        _MAX_V1_ENDPOINT_PATCHES_PER_REQUEST,
        retry_count,
        credential_factory,
        expected_model_version,
    )
    self._model_size = 'MEDIUM'
    self._model_kind = 'LOW_PIXEL_SPACING'

  def _dicom_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Generates Embedding request for image stored in DICOM Store."""
    patch = typing.cast(
        dicom_slide.DicomPatch, slide_embedding.patches[0].patch
    )
    source_series = patch.source
    path = source_series.path
    instance_uids, required_levels = (
        _get_dicom_instance_uids_and_required_levels(slide_embedding)
    )
    if patch.is_resized:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'V1 encoder does not support image level resizing.'
      )
    if not bearer_token:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'V1 encoder does not support empty bearer tokens.'
      )
    return {
        EndpointJsonKeys.DICOM_WEB_STORE_URL: path.GetStorePath().complete_url,
        EndpointJsonKeys.DICOM_STUDY_UID: path.study_uid,
        EndpointJsonKeys.DICOM_SERIES_UID: path.series_uid,
        EndpointJsonKeys.BEARER_TOKEN: bearer_token,
        EndpointJsonKeys.EZ_WSI_STATE: source_series.json_metadata_dict(
            level_subset=required_levels,
            max_json_encoded_icc_profile_size=_MAX_DICOM_SLIDE_ICCPROFILE_METADATA_SIZE,
        ),
        EndpointJsonKeys.INSTANCE_UIDS: instance_uids,
        EndpointJsonKeys.PATCH_COORDINATES: [
            {
                EndpointJsonKeys.X_ORIGIN: int(input.patch.x),
                EndpointJsonKeys.Y_ORIGIN: int(input.patch.y),
                EndpointJsonKeys.WIDTH: int(input.patch.width),
                EndpointJsonKeys.HEIGHT: int(input.patch.height),
            }
            for input in slide_embedding.patches
        ],
    }

  def _gcs_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Generates Embedding request for image stored in DICOM Store."""
    json_metadata = _get_gcs_image_metadata(
        self.max_request_size_bytes(),
        self.send_gcs_patch_bytes_from_client_to_server,
        self.icc_profile_normalization,
        self.icc_profile_bytes(),
        bearer_token,
        slide_embedding,
    )
    gcs_patch = typing.cast(
        gcs_image.GcsPatch, slide_embedding.patches[0].patch
    )
    uri = gcs_patch.source.uri
    if gcs_patch.is_resized:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'V1 encoder does not support image image resizing.'
      )
    if not bearer_token and gcs_patch.source.uri:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'V1 encoder does not support empty bearer tokens.'
      )
    return {
        EndpointJsonKeys.PROJECT_NAME: 'placeholder',
        EndpointJsonKeys.GCS_IMAGE_URL: uri,
        EndpointJsonKeys.BEARER_TOKEN: bearer_token,
        EndpointJsonKeys.EZ_WSI_STATE: json_metadata,
        EndpointJsonKeys.PATCH_COORDINATES: [
            {
                EndpointJsonKeys.X_ORIGIN: int(input.patch.x),
                EndpointJsonKeys.Y_ORIGIN: int(input.patch.y),
                EndpointJsonKeys.WIDTH: int(input.patch.width),
                EndpointJsonKeys.HEIGHT: int(input.patch.height),
            }
            for input in slide_embedding.patches
        ],
    }

  def _validate_embedding_response(
      self,
      embedding_source: patch_embedding_types.SlideEmbeddingSource,
      embedding_response: Mapping[str, Any],
  ):
    """Validate embedding DICOM UID match patch request."""
    if embedding_source.patches and isinstance(
        embedding_source.patches[0].patch, dicom_slide.DicomPatch
    ):
      # Test StudyInstanceUID and SeriesInstanceUID from request and response
      # match if returning result for DICOM slide.
      patch = typing.cast(
          dicom_slide.DicomPatch, embedding_source.patches[0].patch
      )
      path = patch.source.path
      if (
          embedding_response[EndpointJsonKeys.DICOM_STUDY_UID] != path.study_uid
          or embedding_response[EndpointJsonKeys.DICOM_SERIES_UID]
          != path.series_uid
      ):
        raise ez_wsi_errors.PatchEmbeddingEndpointError(
            'Study or Series UID of embedding does not match request.'
        )

  def _is_request_error_retryable(
      self, json_response: Mapping[str, Any]
  ) -> bool:
    """Returns true if error at request level is retryable."""
    try:
      predictions = json_response[EndpointJsonKeys.PREDICTIONS]
      if isinstance(predictions, Mapping):
        # This is raw response vertex endpoint translates response
        # to alternative format
        error = predictions[EndpointJsonKeys.ERROR_RESPONSE]
      else:
        # This is response as translasted by vertex endpoint
        returned_slide_embeddings, error, ml_version = predictions
        del returned_slide_embeddings, ml_version
      return error == EndpointJsonKeys.INVALID_CREDENTIALS
    except (KeyError, ValueError, TypeError) as _:
      return False

  def _decode_response(
      self,
      embedding_inputs: Sequence[PreparedVertexEmbeddingRequest],
      json_response: Mapping[str, Any],
  ) -> _VertexModelResult:
    """Decodes response from Vertex AI endpoint into _VertexModelResult."""
    try:
      predictions = json_response[EndpointJsonKeys.PREDICTIONS]
      if isinstance(predictions, Mapping):
        # This is raw response vertex endpoint translates response
        # to alternative format
        returned_slide_embeddings = predictions[
            EndpointJsonKeys.EMBEDDING_RESULT
        ]
        error = predictions[EndpointJsonKeys.ERROR_RESPONSE]
        ml_version = predictions[EndpointJsonKeys.MODEL_VERSION]
      else:
        # This is response as translasted by vertex endpoint
        returned_slide_embeddings, error, ml_version = predictions
    except (KeyError, ValueError, TypeError) as exp:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'Endpoint returned incorrectly formatted JSON.'
      ) from exp
    if error is not None and error:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          f'Endpoint Version({ml_version}) returned error: {error}'
      )
    if (
        self.expected_model_version
        and ml_version != self.expected_model_version
    ):
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          f'Model version {ml_version} does not match expected version'
          f' {self.expected_model_version}'
      )
    # Test the number of slide embedding responses matches the request.
    if len(embedding_inputs) != len(returned_slide_embeddings):
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'Number of embedding responses received does not match number of'
          f' embedding requests; expected: {len(embedding_inputs)}; received:'
          f' {len(returned_slide_embeddings)}.'
      )
    return _VertexModelResult(returned_slide_embeddings)

  def _instance_has_retryable_error(self, json_dict: Mapping[str, Any]) -> bool:
    return False

  def process_response(
      self,
      embedding_inputs: Sequence[patch_embedding_types.SlideEmbeddingSource],
      msg: _VertexModelResult,
  ) -> List[patch_embedding_types.PatchEmbeddingEnsembleResult]:
    """Returns patch embedding results for input and returned embeddings."""
    result = []
    endpoint_patch_width = self.patch_width()
    endpoint_patch_height = self.patch_height()
    for patch_embeddings, instance_input in zip(
        msg.instances, embedding_inputs
    ):
      self._validate_embedding_response(instance_input, patch_embeddings)
      patch_embeddings = patch_embeddings[EndpointJsonKeys.PATCH_EMBEDDINGS]
      # Test the number of patches received for the slide matches the request.
      if len(patch_embeddings) != len(instance_input.patches):
        raise ez_wsi_errors.PatchEmbeddingEndpointError(
            'Number of patches in embedding response does not match request;'
            f' expected: {len(instance_input.patches)}; received:'
            f' {len(patch_embeddings)}.'
        )
      for patch_embedding, source in zip(
          patch_embeddings, instance_input.patches
      ):
        pc = patch_embedding[EndpointJsonKeys.PATCH_COORDINATE]
        # Test the coodinates of the patch matches the request.
        if not _test_patch_coordinates_match(
            pc,
            source.patch.x,
            source.patch.y,
            endpoint_patch_width,
            endpoint_patch_height,
        ):
          raise ez_wsi_errors.PatchEmbeddingEndpointError(
              'Embedding patch coordinates or dimensions do not match request.'
          )
        embedding_value = np.asarray(
            patch_embedding[EndpointJsonKeys.EMBEDDINGS]
        )
        result.append(
            patch_embedding_types.PatchEmbeddingEnsembleResult(
                source, embedding_value, None
            )
        )
    return result

  def _get_embedding_request(
      self, embedding_inputs: Sequence[PreparedVertexEmbeddingRequest]
  ) -> str:
    """Returns JSON formmatted embedding request."""
    instances = ','.join(i.json for i in embedding_inputs)
    return f'{{"{EndpointJsonKeys.PARAMETERS}":{{"{EndpointJsonKeys.MODEL_SIZE}":"{self._model_size}","{EndpointJsonKeys.MODEL_KIND}":"{self._model_kind}"}},"{EndpointJsonKeys.INSTANCES}":[{instances}]}}'


def _gen_v2_extensions(
    require_fully_in_source_image: bool,
    image_dimensions: Optional[dicom_slide.ImageDimensions],
    icc_profile: IccProfileNormalization,
    ez_wsi_state: Mapping[str, Any],
) -> Mapping[str, Any]:
  """Returns extensions for pathology embeddings."""
  extension = {
      EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: str(
          require_fully_in_source_image
      )
  }
  if image_dimensions is not None:
    extension[EndpointJsonKeys.IMAGE_DIMENSIONS] = dataclasses.asdict(
        image_dimensions
    )
  if icc_profile:
    extension[EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE] = str(
        icc_profile.value
    )
  if ez_wsi_state:
    extension[EndpointJsonKeys.EZ_WSI_STATE] = ez_wsi_state
  return extension


def _format_error_message(error_code: str, error_description: str) -> str:
  if not error_description:
    return f'Error code: {error_code}'
  return f'Error code: {error_code}; {error_description}'


class V2PatchEmbeddingEndpoint(_PatchEmbeddingEndpointBase):
  """Implements Patch embedding V2 API."""

  def __init__(
      self,
      patch_width: int = 224,
      patch_height: int = 224,
      endpoint_api_version: str = 'v1',  # Vertex API version
      project_id: str = 'hai-cd3-foundations',
      endpoint_location: str = 'us-central1',
      endpoint_id: str = '162',
      max_threads: int = _DEFAULT_ENDPOINT_THREADS,
      max_patches_per_request: int = _DEFAULT_MAX_PATCHES_PER_REQUEST,
      retry_count: int = _DEFAULT_RETRY_COUNT,
      icc_profile_normalization: IccProfileNormalization = (
          IccProfileNormalization.NONE
      ),
      send_gcs_patch_bytes_from_client_to_server: bool = False,
      require_fully_in_source_image: bool = True,
      credential_factory: Optional[
          credential_factory_module.AbstractCredentialFactory
      ] = None,
      expected_model_version: str = '',
  ):
    end_point: List[str] = [
        f'https://{endpoint_location}-aiplatform.googleapis.com',
        endpoint_api_version,
        'projects',
        project_id,
        'locations',
        endpoint_location,
        'endpoints',
        f'{endpoint_id}:predict',
    ]
    end_point_url = '/'.join([ep.strip('/') for ep in end_point])
    super().__init__(
        patch_width,
        patch_height,
        icc_profile_normalization,
        send_gcs_patch_bytes_from_client_to_server,
        end_point_url,
        max_threads,
        max_patches_per_request,
        _MAX_V2_ENDPOINT_PATCHES_PER_REQUEST,
        retry_count,
        credential_factory,
        expected_model_version,
    )
    self._require_fully_in_source_image = require_fully_in_source_image

  def _dicom_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Generates Embedding request for image stored in DICOM Store."""
    patch = typing.cast(
        dicom_slide.DicomPatch, slide_embedding.patches[0].patch
    )
    instance_uids, required_levels = (
        _get_dicom_instance_uids_and_required_levels(slide_embedding)
    )
    if patch.is_resized:
      image_dimensions = dicom_slide.ImageDimensions(
          patch.level.width, patch.level.height
      )
    else:
      image_dimensions = None
    request = {
        EndpointJsonKeys.DICOM_PATH: {
            EndpointJsonKeys.SERIES_PATH: str(
                patch.source.path.GetSeriesPath()
            ),
            EndpointJsonKeys.INSTANCE_UIDS: instance_uids,
        },
        EndpointJsonKeys.EXTENSIONS: _gen_v2_extensions(
            self._require_fully_in_source_image,
            image_dimensions,
            self._icc_profile_normalization,
            patch.source.json_metadata_dict(
                level_subset=required_levels,
                max_json_encoded_icc_profile_size=_MAX_DICOM_SLIDE_ICCPROFILE_METADATA_SIZE,
            ),
        ),
        EndpointJsonKeys.PATCH_COORDINATES: [
            {
                EndpointJsonKeys.X_ORIGIN: int(input.patch.x),
                EndpointJsonKeys.Y_ORIGIN: int(input.patch.y),
                EndpointJsonKeys.WIDTH: int(input.patch.width),
                EndpointJsonKeys.HEIGHT: int(input.patch.height),
            }
            for input in slide_embedding.patches
        ],
    }
    if bearer_token:
      request[EndpointJsonKeys.BEARER_TOKEN] = bearer_token
    return request

  def _gcs_patch_embedding_request(
      self,
      bearer_token: str,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> Mapping[str, Any]:
    """Generates Embedding request for image stored in DICOM Store."""
    json_metadata = _get_gcs_image_metadata(
        self.max_request_size_bytes(),
        self.send_gcs_patch_bytes_from_client_to_server,
        self.icc_profile_normalization,
        self.icc_profile_bytes(),
        bearer_token,
        slide_embedding,
    )
    gcs_patch = typing.cast(
        gcs_image.GcsPatch, slide_embedding.patches[0].patch
    )
    uri = gcs_patch.source.uri
    request = {
        EndpointJsonKeys.IMAGE_FILE_URI: (
            uri if uri else 'gs:///placeholder.png'
        ),
        EndpointJsonKeys.EXTENSIONS: _gen_v2_extensions(
            self._require_fully_in_source_image,
            gcs_patch.source.resize_dimensions,
            self._icc_profile_normalization,
            json_metadata,
        ),
        EndpointJsonKeys.PATCH_COORDINATES: [
            {
                EndpointJsonKeys.X_ORIGIN: int(input.patch.x),
                EndpointJsonKeys.Y_ORIGIN: int(input.patch.y),
                EndpointJsonKeys.WIDTH: int(input.patch.width),
                EndpointJsonKeys.HEIGHT: int(input.patch.height),
            }
            for input in slide_embedding.patches
        ],
    }
    if bearer_token:
      request[EndpointJsonKeys.BEARER_TOKEN] = bearer_token
    return request

  def _get_embedding_request(
      self, embedding_inputs: Sequence[PreparedVertexEmbeddingRequest]
  ) -> str:
    """Returns JSON formmatted embedding request."""
    instances = ','.join(i.json for i in embedding_inputs)
    return f'{{"{EndpointJsonKeys.INSTANCES}":[{instances}]}}'

  def _is_request_error_retryable(
      self, json_response: Mapping[str, Any]
  ) -> bool:
    """Returns true if error at request level is retryable."""
    try:
      error_code = json_response[EndpointJsonKeys.VERTEXAI_ERROR]
      if isinstance(error_code, dict):
        error_code = error_code[EndpointJsonKeys.ERROR_CODE]
      return error_code == EndpointJsonKeys.INVALID_CREDENTIALS
    except (KeyError, ValueError, TypeError):
      return False

  def _decode_response(
      self,
      embedding_inputs: Sequence[PreparedVertexEmbeddingRequest],
      json_response: Mapping[str, Any],
  ) -> _VertexModelResult:
    """Decodes response from Vertex AI endpoint into _VertexModelResult."""
    try:
      returned_slide_embeddings = json_response[EndpointJsonKeys.PREDICTIONS]
    except (KeyError, ValueError, TypeError):
      try:
        error_code = json_response[EndpointJsonKeys.VERTEXAI_ERROR]
        error_description = ''
        if isinstance(error_code, dict):
          error_description = error_code.get(
              EndpointJsonKeys.ERROR_CODE_DESCRIPTION, ''
          )
          error_code = error_code[EndpointJsonKeys.ERROR_CODE]
        if isinstance(error_code, str):
          msg = _format_error_message(error_code, error_description)
          raise ez_wsi_errors.PatchEmbeddingEndpointError(  # pylint: disable=raise-missing-from
              f'Endpoint error; {msg}'
          )
      except (KeyError, ValueError, TypeError):
        pass
      raise ez_wsi_errors.PatchEmbeddingEndpointError(  # pylint: disable=raise-missing-from
          'Endpoint did not return a valid JSON response.'
      )
    # Test the number of slide embedding responses matches the request.
    if len(embedding_inputs) != len(returned_slide_embeddings):
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'Number of embedding responses received does not match number of'
          f' embedding requests; expected: {len(embedding_inputs)}; received:'
          f' {len(returned_slide_embeddings)}.'
      )
    return _VertexModelResult(returned_slide_embeddings)

  def _instance_has_retryable_error(self, json_dict: Mapping[str, Any]) -> bool:
    """Decodes response from Vertex AI endpoint into _VertexModelResult."""
    error = json_dict.get(EndpointJsonKeys.ERROR)
    if error is None:
      return False
    try:
      return (
          error[EndpointJsonKeys.ERROR_CODE]
          == EndpointJsonKeys.INVALID_CREDENTIALS
      )
    except (KeyError, TypeError, IndexError) as _:
      return False

  def process_response(
      self,
      embedding_inputs: Sequence[patch_embedding_types.SlideEmbeddingSource],
      msg: _VertexModelResult,
  ) -> List[patch_embedding_types.PatchEmbeddingEnsembleResult]:
    """Returns patch embedding results for input and returned embeddings."""
    result = []
    endpoint_patch_width = self.patch_width()
    endpoint_patch_height = self.patch_height()
    for returned_instance, instance_input in zip(
        msg.instances, embedding_inputs
    ):
      try:
        model_version = returned_instance[EndpointJsonKeys.MODEL_VERSION]
        if (
            self.expected_model_version
            and model_version != self.expected_model_version
        ):
          raise ez_wsi_errors.PatchEmbeddingEndpointError(
              f'Model version {model_version} does not match expected version'
              f' {self.expected_model_version}'
          )
        error = returned_instance.get(EndpointJsonKeys.ERROR)
        if error is not None:
          error_code = error[EndpointJsonKeys.ERROR_CODE]
          error_description = error.get(
              EndpointJsonKeys.ERROR_CODE_DESCRIPTION, ''
          )
          error_message = '\n'.join([
              'Endpoint error generating instance embeddings.',
              f'Endpoint: {self.end_point_url}; Model: {model_version}',
              f'{_format_error_message(error_code, error_description)}',
          ])
          error = patch_embedding_types.PatchEmbeddingError(
              error_code, error_message
          )
          # Return PatchEmbeddingEnsembleResult with an error
          # for each expected patch. Errors will be raised when
          # embedding values from the patches are accessed. Typically this will
          # occure almost immediately after during ensemble reduction. However,
          # this will also enable callers using PatchEmbeddingSequence
          # (indexed based) to access data for instances which may succeed
          # after a instance fails.
          for patch_source in instance_input.patches:
            result.append(
                patch_embedding_types.PatchEmbeddingEnsembleResult(
                    patch_source, None, error
                )
            )
          continue
        patch_embeddings = returned_instance[EndpointJsonKeys.RESULT][
            EndpointJsonKeys.PATCH_EMBEDDINGS
        ]
        # Test the number of patches received for the slide matches the request.
        if len(patch_embeddings) != len(instance_input.patches):
          raise ez_wsi_errors.PatchEmbeddingEndpointError(
              'Number of patches in embedding response does not match request;'
              f' expected: {len(instance_input.patches)}; received:'
              f' {len(patch_embeddings)}.'
          )
        for patch_embedding, patch_source in zip(
            patch_embeddings, instance_input.patches
        ):
          pc = patch_embedding[EndpointJsonKeys.PATCH_COORDINATE]
          # Test the coodinates of the patch matches the request.
          if not _test_patch_coordinates_match(
              pc,
              patch_source.patch.x,
              patch_source.patch.y,
              endpoint_patch_width,
              endpoint_patch_height,
          ):
            raise ez_wsi_errors.PatchEmbeddingEndpointError(
                'Embedding patch coordinates or dimensions do not match'
                ' request.'
            )
          embedding_value = np.asarray(
              patch_embedding[EndpointJsonKeys.EMBEDDING_VECTOR]
          )
          result.append(
              patch_embedding_types.PatchEmbeddingEnsembleResult(
                  patch_source, embedding_value, None
              )
          )
      except (KeyError, IndexError, TypeError, ValueError) as exp:
        raise ez_wsi_errors.PatchEmbeddingEndpointError(
            'Endpoint returned an unexpected response.'
        ) from exp
    return result


class PreparedLocalEmbeddingRequest(
    AbstractPreparedEmbeddingRequest[np.ndarray]
):
  """Base class for prepared embedding requests."""

  def __init__(
      self,
      slide_embedding_source: patch_embedding_types.SlideEmbeddingSource,
      endpoint_thread_pool: concurrent.futures.ThreadPoolExecutor,
      icc_profile_bytes: bytes,
      icc_profile_cache: cachetools.LRUCache,
      icc_profile_cache_lock: threading.Lock,
      require_fully_in_source_image: bool,
  ):
    super().__init__(slide_embedding_source)
    self._input_patch_bytes = []
    self._input_patch_bytes_future = None
    self._thread_pool = endpoint_thread_pool
    self._target_icc_profile_bytes = icc_profile_bytes
    self._icc_profile_cache = icc_profile_cache
    self._icc_profile_cache_lock = icc_profile_cache_lock
    self._require_fully_in_source_image = require_fully_in_source_image

  @property
  def json_size_in_bytes(self) -> int:
    return 0

  def _load_patch_bytes(self) -> List[np.ndarray]:
    """Loads patch bytes for each request."""
    if (
        self._slide_embedding_source is None
        or not self._slide_embedding_source.patches
    ):
      return []
    first_patch = self._slide_embedding_source.patches[0].patch
    if isinstance(first_patch, gcs_image.GcsPatch):
      # Make a shallow copy of source patch Source GcsImage to ensure
      # if image bytes are are not retained after the request is processed.
      source_copy = copy.copy(first_patch.source)
      image_bytes = []
      icc_profile_normalization = source_copy.create_icc_profile_transformation(
          self._target_icc_profile_bytes
      )
      for patch in self._slide_embedding_source.patches:
        p = patch.patch
        if (
            self._require_fully_in_source_image
            and not p.is_patch_fully_in_source_image()
        ):
          raise ez_wsi_errors.PatchOutsideOfImageDimensionsError(
              'A portion of the patch does not overlap the image.'
          )
        # create a patch with same coordinates a the temporary source.
        # require in source image is not relevant to the temp patch. Setting to
        # false.
        temp_patch = gcs_image.GcsPatch(
            source_copy, p.x, p.y, p.width, p.height, False
        )
        image_bytes.append(temp_patch.image_bytes(icc_profile_normalization))
      return image_bytes
    if isinstance(first_patch, dicom_slide.DicomPatch):
      # Load in slide DICOM slide imaging using frame cache that is not shared
      # to scope imageing bytes to the embedding request and also avoid.
      # possible LRU cache eviction across parallel reads.

      # Make a shallow copy of source patch Source DicomSlide or
      # or Dicom Microscopy image
      source_copy = copy.copy(first_patch.source)

      # Init new frame cache on the shallow copy source.
      # Source and copy will no longer share the cache.
      fc = source_copy.init_slide_frame_cache()

      # Construct list of patches to return embedding for.
      patch_list = typing.cast(
          List[dicom_slide.DicomPatch],
          [patch.patch for patch in self._slide_embedding_source.patches],
      )
      # Preload list of patches into frame cache. Copy across any loaded
      # imaging that was loaded in the original cache.
      source_copy.preload_patches_in_frame_cache(
          patch_list, False, first_patch.source.slide_frame_cache
      )

      if (
          self._target_icc_profile_bytes is None
          or not self._target_icc_profile_bytes
      ):
        icc_profile_normalization = None
      else:
        dicom_path = str(source_copy.path)
        with self._icc_profile_cache_lock:
          source_icc_profile_bytes = self._icc_profile_cache.get(dicom_path)
        if source_icc_profile_bytes is None:
          if isinstance(source_copy, dicom_slide.DicomSlide):
            source_icc_profile_bytes = source_copy.get_icc_profile_bytes()
          elif isinstance(source_copy, dicom_slide.DicomMicroscopeImage):
            source_icc_profile_bytes = source_copy.get_level_icc_profile_bytes(
                first_patch.level
            )
          else:
            raise ValueError('Unexpected object')
        with self._icc_profile_cache_lock:
          self._icc_profile_cache[dicom_path] = source_icc_profile_bytes
        icc_profile_normalization = (
            dicom_slide.create_icc_profile_transformation(
                source_icc_profile_bytes, self._target_icc_profile_bytes
            )
        )
      fc.block_until_frames_are_loaded()
      # Generate image bytes for each patch.
      image_bytes = []
      for p in patch_list:
        if (
            self._require_fully_in_source_image
            and not p.is_patch_fully_in_source_image()
        ):
          raise ez_wsi_errors.PatchOutsideOfImageDimensionsError(
              'A portion of the patch does not overlap the image.'
          )
        # create  copy of the patch.
        # set the patch to point to the copied source to make patch image
        # retrieval read from the copied sources frame cache.
        temp_patch = dicom_slide.DicomPatch(
            p.get_pyramid_imaging_source_level(),
            p.x,
            p.y,
            p.width,
            p.height,
            source_copy,
            p.level,
            # ensuring patch falls inside image dimensions is not relevant
            # to the temp patch.
            False,
        )
        image_bytes.append(temp_patch.image_bytes(icc_profile_normalization))
      return image_bytes
    raise ValueError('Unexpected object')

  @property
  def input_patch_bytes(self) -> List[np.ndarray]:
    if self._input_patch_bytes_future is not None:
      self._input_patch_bytes = self._input_patch_bytes_future.result()
      self._input_patch_bytes_future = None
    return self._input_patch_bytes

  def finalize(self) -> None:
    self._input_patch_bytes_future = self._thread_pool.submit(
        self._load_patch_bytes
    )

  def split(
      self, endpoint: AbstractPatchEmbeddingEndpoint[np.ndarray], max_size: int
  ) -> Tuple[
      Optional[AbstractPreparedEmbeddingRequest[np.ndarray]],
      patch_embedding_types.SlideEmbeddingSource,
  ]:
    # splitting is not relevant for local endpoints.
    # return an unsplit response.
    if self._slide_embedding_source is None:
      raise ValueError('Slide embedding source is None.')
    return None, self._slide_embedding_source


class LocalEndpoint(AbstractPatchEmbeddingEndpoint[np.ndarray]):
  """Endpoint for generating embeddings with a locally loaded model."""

  def __init__(
      self,
      model: Callable[[np.ndarray], np.ndarray],
      icc_profile_normalization: IccProfileNormalization = (
          IccProfileNormalization.NONE
      ),
      patch_width: int = 224,
      patch_height: int = 224,
      require_fully_in_source_image: bool = True,
      max_threads: int = _DEFAULT_ENDPOINT_THREADS,
      retry_count: int = _DEFAULT_RETRY_COUNT,
      max_patches_per_request: int = _DEFAULT_MAX_PATCHES_PER_REQUEST,
      dicom_instance_icc_profile_cache_count: int = _DEFAULT_DICOM_INSTANCE_ICC_PROFILE_CACHE_COUNT,
  ):
    super().__init__(icc_profile_normalization)
    self._require_fully_in_source_image = require_fully_in_source_image
    self._patch_width = patch_width
    self._patch_height = patch_height
    self._max_threads = max(1, max_threads)
    self._max_patches_per_request = int(max(1, max_patches_per_request))
    self._retry_count = max(0, retry_count)
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_threads
    )
    self._dicom_instance_icc_profile_cache_count = (
        dicom_instance_icc_profile_cache_count
    )
    self._model = model
    self._icc_profile_cache = cachetools.LRUCache(
        self._dicom_instance_icc_profile_cache_count
    )
    self._icc_profile_cache_lock = threading.Lock()

  def __del__(self):
    self._thread_pool.shutdown(wait=False, cancel_futures=True)  # pylint: disable=attribute-error

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = copy.copy(self.__dict__)
    del state['_thread_pool']
    del state['_icc_profile_cache']
    del state['_icc_profile_cache_lock']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._max_threads
    )
    self._icc_profile_cache = cachetools.LRUCache(
        self._dicom_instance_icc_profile_cache_count
    )
    self._icc_profile_cache_lock = threading.Lock()

  def max_request_size_bytes(self) -> int:
    """Maximum size in bytes that can be sent in single request."""
    return 0xFFFFFFFFFFFFFFFF  # max size uint64

  def retry_count(self) -> int:
    return self._retry_count

  def max_threads(self) -> int:
    return self._max_threads

  def patch_width(self) -> int:
    return self._patch_width

  def patch_height(self) -> int:
    return self._patch_height

  def max_number_of_patches_per_request(self) -> int:
    return self._max_patches_per_request

  def endpoint_max_number_of_patches_per_request(self) -> int:
    """Maximum number of patches that can be sent to the endpoint at once."""
    return self._max_patches_per_request

  def prepare_embedding_request(
      self,
      slide_embedding: patch_embedding_types.SlideEmbeddingSource,
  ) -> PreparedLocalEmbeddingRequest:
    return PreparedLocalEmbeddingRequest(
        slide_embedding,
        self._thread_pool,
        self.icc_profile_bytes(),
        self._icc_profile_cache,
        self._icc_profile_cache_lock,
        self._require_fully_in_source_image,
    )

  def normalize_imaging(self, input_patch_bytes: np.ndarray) -> np.ndarray:
    """Normalizes input patch bytes to float32 in range [0, 1]."""
    return input_patch_bytes.astype(np.float32) / _UINT8_MAX_VALUE

  def request_embeddings(
      self,
      embedding_inputs: Sequence[AbstractPreparedEmbeddingRequest[np.ndarray]],
  ) -> np.ndarray:
    if not embedding_inputs:
      return np.zeros((), dtype=np.float32)
    normalized_imaging_list = []
    patch_width = self.patch_width()
    patch_height = self.patch_height()
    for e in embedding_inputs:
      e = typing.cast(PreparedLocalEmbeddingRequest, e)
      for single_patch in e.input_patch_bytes:
        normalized_imaging_list.append(
            np.expand_dims(
                normalized_patch_channels(
                    patch_width,
                    patch_height,
                    self.normalize_imaging(single_patch),
                ),
                axis=0,
            )
        )
    ml_input = np.concatenate(normalized_imaging_list, axis=0)
    return self._model(ml_input)

  def process_response(
      self,
      embedding_inputs: Sequence[patch_embedding_types.SlideEmbeddingSource],
      msg: np.ndarray,
  ) -> List[patch_embedding_types.PatchEmbeddingEnsembleResult]:
    """Converts raw embedding response to list of embedding results."""
    if not bool(msg.shape):
      generated_embedding_count = 0
    else:
      generated_embedding_count = msg.shape[0]
    total_patches = sum(len(i.patches) for i in embedding_inputs)
    if total_patches != generated_embedding_count:
      raise ez_wsi_errors.PatchEmbeddingEndpointError(
          'Number of patches in embedding response does not match request.'
      )
    results = []
    index = 0
    for instance_input in embedding_inputs:
      for patch_source in instance_input.patches:
        results.append(
            patch_embedding_types.PatchEmbeddingEnsembleResult(
                patch_source, msg[index, ...], None
            )
        )
        index += 1
    return results
