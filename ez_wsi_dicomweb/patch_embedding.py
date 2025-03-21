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
"""Main functions to return embeddings for Patches or Images."""

import collections.abc
from concurrent import futures
import dataclasses
import functools
import os
import threading
import time
import typing
from typing import Any, Iterator, List, Optional, Sequence, Union

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import error_retry_util
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import patch_embedding_ensemble_methods
from ez_wsi_dicomweb import patch_embedding_types
from ez_wsi_dicomweb import patch_generator
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
import numpy as np
import retrying


@dataclasses.dataclass(frozen=True)
class BatchEmbeddingRequest:
  """Batch Embedding request."""

  json_request: str
  prepared_request: Sequence[
      patch_embedding_endpoints.AbstractPreparedEmbeddingRequest
  ]


# by default embedding request throttling is disabled.
_max_requests_per_minute: Optional[int] = None
_last_request_time = 0.0
_request_lock = threading.Lock()


def _init_request_throttle() -> None:
  global _request_lock
  _request_lock = threading.Lock()


def disable_embedding_request_throttling():
  """Disables embedding request throttling."""
  global _max_requests_per_minute
  with _request_lock:
    _max_requests_per_minute = None


def set_max_embedding_requests_per_min(max_requests_per_minute: int) -> None:
  """Sets maximum number of requests per minute which can occure in process."""
  global _max_requests_per_minute
  with _request_lock:
    _max_requests_per_minute = max_requests_per_minute


def _get_embedding_thread(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    slide_embeddings: Sequence[
        patch_embedding_endpoints.AbstractPreparedEmbeddingRequest
    ],
) -> List[patch_embedding_types.PatchEmbeddingEnsembleResult]:
  """Returns endpoint embedding for list of slide patches."""

  @retrying.retry(
      **error_retry_util.other_http_exception_retry_config(
          endpoint.retry_count()
      )
  )
  def _inner_func() -> str:
    if _max_requests_per_minute is None:
      # embedding request throttling is disabled.
      return endpoint.request_embeddings(slide_embeddings)
    # embedding request throttling is enabled.
    while True:
      global _last_request_time
      min_sec_between_requests = 60.0 / _max_requests_per_minute
      with _request_lock:
        current_time = time.time()
        # min average time between requests
        delta = min_sec_between_requests - (current_time - _last_request_time)
      if delta <= 0:
        _last_request_time = current_time
        return endpoint.request_embeddings(slide_embeddings)
      # sleep until delta predicted to expire.
      time.sleep(delta)

  response = _inner_func()
  return endpoint.process_response(
      [s.slide_embedding_source for s in slide_embeddings], response
  )


@dataclasses.dataclass(frozen=True)
class _GeneratedPreparedRequest:
  prepared_requests: List[
      patch_embedding_endpoints.AbstractPreparedEmbeddingRequest
  ]
  source_overflow: List[patch_embedding_types.SlideEmbeddingSource]
  overflow_size: int


class _EmbeddingAPIRequest:
  """Collects patch embedding api requests."""

  def __init__(self):
    self._slide_processing: Union[
        gcs_image.GcsImage,
        slide_level_map.Level,
        slide_level_map.ResizedLevel,
        None,
    ] = None
    # list of unique slides, gcsimages, etc, each has a list of one or more
    # patches.
    self._queued_embedding_image_requests: List[
        patch_embedding_types.SlideEmbeddingSource
    ] = []
    self._mag_scaled_patch_count = 0
    self._patch_count = 0

  @property
  def has_queued_embedding_requests(self) -> bool:
    # is there a image (slides, gcsimage) queued.
    return bool(self._queued_embedding_image_requests)

  def __len__(self) -> int:
    """Number of image sources, slides, gcsimages that have patch requests."""
    return len(self._queued_embedding_image_requests)

  def _recalulate_patch_counts_for_queued_requests(self) -> None:
    self._patch_count = 0
    self._mag_scaled_patch_count = 0
    for image_request in self._queued_embedding_image_requests:
      self._mag_scaled_patch_count += (
          image_request.mag_scaled_embedding_patch_count
      )
      self._patch_count += len(image_request.patches)

  def _generate_request(
      self, endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint
  ) -> _GeneratedPreparedRequest:
    """Transforms list of embedding request into a request and overflow.

    Vertex Endpoints have size limits that restrict the total number of bytes
    that can be sent in a single request and the pathology embedding endpoints
    restrict the total number of patch embeddings that can be requested at once.
    This function transforms a list of embedding requests into a valid
    request and a overflow list that will be processed in a different request.

    Args:
      endpoint: Patch embedding endpoint to use.

    Returns:
      _GeneratedPreparedRequest(
          prepared list of embeddings that can be processed,
          overflow list of embeddings that could not be processed,
          size of the overflow in bytes.
      )

    Raises:
      ez_wsi_errors.PatchEmbeddingEndpointError: If the first embedding request
      exceeds the size limit of the endpoint.
    """
    pending_request_size_in_bytes = 0
    prepared_request_list: List[
        patch_embedding_endpoints.AbstractPreparedEmbeddingRequest
    ] = []
    request_overflow = []
    overflow_size = 0
    for index, embedding_request in enumerate(
        self._queued_embedding_image_requests
    ):
      prepared_request = endpoint.prepare_embedding_request(embedding_request)
      prepared_request_size = prepared_request.json_size_in_bytes
      max_request_size = endpoint.max_request_size_bytes()
      if (
          pending_request_size_in_bytes + prepared_request_size
          <= max_request_size
      ):
        pending_request_size_in_bytes += prepared_request_size
        prepared_request_list.append(prepared_request)
        prepared_request.finalize()
      else:
        split_prepared_request, overflow_embedding_source = (
            prepared_request.split(
                endpoint, max_request_size - pending_request_size_in_bytes
            )
        )
        if split_prepared_request is not None:
          split_prepared_request.finalize()
          prepared_request_list.append(split_prepared_request)
          # slightly under estimates size of overflow, doesn't count duplicate
          # state.
          overflow_size = (
              prepared_request.json_size_in_bytes
              - split_prepared_request.json_size_in_bytes
          )
        elif index == 0:
          raise ez_wsi_errors.PatchEmbeddingEndpointError(
              'Embedding request size,'
              f' {prepared_request_size} (bytes), exceeds endpoint'
              f' size limit, {max_request_size} (bytes).'
          )
        else:
          overflow_size = prepared_request.json_size_in_bytes
        request_overflow = [overflow_embedding_source]
        request_overflow.extend(
            self._queued_embedding_image_requests[index + 1 :]
        )
        break
    return _GeneratedPreparedRequest(
        prepared_request_list, request_overflow, overflow_size
    )

  def generate_prepared_embedding_request(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
  ) -> Iterator[_GeneratedPreparedRequest]:
    """returns prepared embedding requests."""
    run_batch_request_loop = self.has_queued_embedding_requests
    max_request_size = endpoint.max_request_size_bytes()
    while run_batch_request_loop:
      gen_request = self._generate_request(endpoint)
      run_batch_request_loop = (
          len(gen_request.source_overflow) > 1
          or gen_request.overflow_size > max_request_size
      )
      yield gen_request
      self._queued_embedding_image_requests = gen_request.source_overflow
      if not self._queued_embedding_image_requests:
        self._slide_processing = None
      self._recalulate_patch_counts_for_queued_requests()

  def add_new_slide(
      self,
      slide_key: Union[
          gcs_image.GcsImage,
          slide_level_map.Level,
          slide_level_map.ResizedLevel,
      ],
  ):
    """Adds new slide to the embedding request."""
    self._queued_embedding_image_requests.append(
        patch_embedding_types.SlideEmbeddingSource([])
    )
    self._slide_processing = slide_key

  @property
  def slide_processing(self) -> Union[
      gcs_image.GcsImage,
      slide_level_map.Level,
      slide_level_map.ResizedLevel,
      None,
  ]:
    """Returns key for the slide currently being processed."""
    return self._slide_processing

  @property
  def mag_scaled_patch_count(self) -> int:
    """Returns total number of embeddings requested scaled by magnification."""
    return self._mag_scaled_patch_count

  @property
  def patch_count(self) -> int:
    """Returns total number of embeddings requested."""
    return self._patch_count

  def add_patch(
      self,
      embedding_request: patch_embedding_types.PatchEmbeddingSource,
      mag_scaled_patch_count: int,
  ) -> None:
    """Adds an embedding request for current slide."""
    self._queued_embedding_image_requests[-1].patches.append(embedding_request)
    self._mag_scaled_patch_count += mag_scaled_patch_count
    self._patch_count += 1


def _generate_prepared_embedding_requests(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    patch_embedding_sources: Union[
        Iterator[patch_embedding_types.PatchEmbeddingSource],
        Sequence[patch_embedding_types.PatchEmbeddingSource],
    ],
) -> Iterator[List[patch_embedding_endpoints.AbstractPreparedEmbeddingRequest]]:
  """Yields embedding requests to be processed on an endpoint.

  Args:
    endpoint: Patch embedding endpoint to use.
    patch_embedding_sources: Iterator of embedding requests.

  Yields:
     embedding requests to perform in batch
  """
  api_request = _EmbeddingAPIRequest()
  max_number_of_patches_per_request = (
      endpoint.max_number_of_patches_per_request()
  )
  endpoint_max_mag_scaled_patch_count = (
      endpoint.endpoint_max_number_of_patches_per_request()
  )
  for patch_embedding_source in patch_embedding_sources:
    patch = patch_embedding_source.patch
    if isinstance(patch, dicom_slide.DicomPatch):
      slide_key = patch.level
    elif isinstance(patch, gcs_image.GcsPatch):
      slide_key = patch.source
    else:
      raise ez_wsi_errors.InternalError(
          'Patch is not a dicom_slide.DicomPatch or gcs_image.GcsPatch.'
      )
    patch_count = patch_embedding_source.mag_scaled_embedding_patch_count
    if api_request.mag_scaled_patch_count > 0 and (
        api_request.mag_scaled_patch_count + patch_count
        > endpoint_max_mag_scaled_patch_count
        or api_request.patch_count + 1 > max_number_of_patches_per_request
    ):
      for br in api_request.generate_prepared_embedding_request(endpoint):
        yield br.prepared_requests
    if api_request.slide_processing != slide_key:
      api_request.add_new_slide(slide_key)
    api_request.add_patch(patch_embedding_source, patch_count)
  while api_request.has_queued_embedding_requests:
    yield_result = False
    for br in api_request.generate_prepared_embedding_request(endpoint):
      yield br.prepared_requests
      yield_result = True
    if not yield_result:
      raise ez_wsi_errors.InternalError(
          'Error request queue is not processing.'
      )


def _embedding_api_call(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    patch_embedding_sources: Union[
        Iterator[patch_embedding_types.PatchEmbeddingSource],
        Sequence[patch_embedding_types.PatchEmbeddingSource],
    ],
) -> Iterator[List[patch_embedding_types.PatchEmbeddingEnsembleResult]]:
  """Yields an embedding results.

  Args:
    endpoint: Patch embedding endpoint to use.
    patch_embedding_sources: Iterator of embedding requests.

  Yields:
    Embedding results.
  """
  max_threads = endpoint.max_threads()
  map_func = functools.partial(_get_embedding_thread, endpoint)
  prepared_embedding_requests = _generate_prepared_embedding_requests(
      endpoint, patch_embedding_sources
  )
  if max_threads < 2:
    for response in map(map_func, prepared_embedding_requests):
      yield response
  else:
    try:
      with futures.ThreadPoolExecutor(max_workers=max_threads) as pool:
        for response in pool.map(
            map_func,
            prepared_embedding_requests,
            # scale endpoint timeout to allow for internal retry.
            timeout=None if endpoint.timeout is None else endpoint.timeout * 4,
        ):
          yield response
    except TimeoutError as exp:
      raise ez_wsi_errors.ThreadPoolTimeoutError(
          'Timeout while waiting for embedding results.'
      ) from exp


def _generate_ensemble_for_patches(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    ensemble_method: patch_embedding_ensemble_methods.PatchEnsembleMethod,
    patches: Union[
        Sequence[patch_embedding_types.EmbeddingPatch],
        Iterator[patch_embedding_types.EmbeddingPatch],
    ],
) -> Iterator[patch_embedding_types.PatchEmbeddingSource]:
  """Yields embedding api requests for user defined patches.

  Args:
    endpoint: Patch embedding endpoint to use.
    ensemble_method: Method to use to genenerate embedding requests for user
      defined patch.
    patches: Iterator of user defined patches.

  Yields:
    Embedding api requests.
  """
  for patch in patches:
    for ag_patch in ensemble_method.generate_ensemble(endpoint, patch):
      yield ag_patch


def _reduce_embedding_ensemble(
    ensemble_method: patch_embedding_ensemble_methods.PatchEnsembleMethod,
    result_lists: Iterator[
        Sequence[patch_embedding_types.PatchEmbeddingEnsembleResult]
    ],
) -> Iterator[patch_embedding_types.EmbeddingResult]:
  """Yields embedding results for user defined patches.

  Args:
    ensemble_method: Method to use to genenerate embedding requests for user
      defined patch.
    result_lists: Iterator of List embedding results.

  Yields:
    Embedding results for use defined patches.
  """
  ensemble_id = ''
  ensemble_list: List[patch_embedding_types.PatchEmbeddingEnsembleResult] = []
  for result_list in result_lists:
    for result in result_list:
      if result.input_patch.ensemble_id == ensemble_id:
        ensemble_list.append(result)
        continue
      if ensemble_list:
        yield ensemble_method.reduce_ensemble(
            ensemble_list[0].input_patch.ensemble_source_patch, ensemble_list
        )
      ensemble_id = result.input_patch.ensemble_id
      ensemble_list = [result]
  if ensemble_list:
    yield ensemble_method.reduce_ensemble(
        ensemble_list[0].input_patch.ensemble_source_patch, ensemble_list
    )


def _create_patch_embedding_batch_request(
    endpoint: patch_embedding_endpoints.AbstractVertexPatchEmbeddingEndpointBase,
    patches: Union[
        Sequence[patch_embedding_types.EmbeddingPatch],
        Iterator[patch_embedding_types.EmbeddingPatch],
    ],
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> Sequence[BatchEmbeddingRequest]:
  """Returns Sequence of embedding requests to be processed in batch.

  Args:
    endpoint: Patch embedding endpoint to use.
    patches: Iterator of user defined patches.
    ensemble_method: Method to use to genenerate embedding requests for user
      defined patch.

  Returns:
    Sequence of embedding requests to be processed in batch.
  """
  if ensemble_method is None:
    ensemble_method = (
        patch_embedding_ensemble_methods.DefaultSinglePatchEnsemble()
    )
  # force embedding requests will be created with credentials that
  # have been acquired at the time the batch request was initialized.
  credential_factory_module.clear_credential_cache()
  embedding_request = []
  for prepared_requests in _generate_prepared_embedding_requests(
      endpoint,
      _generate_ensemble_for_patches(endpoint, ensemble_method, patches),
  ):
    embedding_request.append(
        BatchEmbeddingRequest(
            endpoint.get_embedding_request(
                typing.cast(
                    Sequence[
                        patch_embedding_endpoints.PreparedVertexEmbeddingRequest
                    ],
                    prepared_requests,
                ),
            ),
            prepared_requests,
        )
    )
  return embedding_request


def generate_patch_embeddings(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    patches: Union[
        Sequence[patch_embedding_types.EmbeddingPatch],
        Iterator[patch_embedding_types.EmbeddingPatch],
    ],
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> Iterator[patch_embedding_types.EmbeddingResult]:
  """Returns Iterator embedding results for user defined patches.

  Args:
    endpoint: Patch embedding endpoint to use.
    patches: Iterator of user defined patches.
    ensemble_method: Method to use to genenerate embedding requests for user
      defined patch.

  Returns:
    Iterator embedding results for user defined patches.
  """
  if ensemble_method is None:
    ensemble_method = (
        patch_embedding_ensemble_methods.DefaultSinglePatchEnsemble()
    )
  return _reduce_embedding_ensemble(
      ensemble_method,
      _embedding_api_call(
          endpoint,
          _generate_ensemble_for_patches(endpoint, ensemble_method, patches),
      ),
  )


def get_patch_embedding(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    patch: patch_embedding_types.EmbeddingPatch,
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> np.ndarray:
  """Returns embedding for a user defined patch.

  Args:
    endpoint: Patch embedding endpoint to use.
    patch: user defined patch.
    ensemble_method: Method to use to genenerate embedding requests for user
      defined patch.

  Returns:
    Returns embedding (numpy array) for a user defined patch.
  """
  return next(
      generate_patch_embeddings(endpoint, [patch], ensemble_method)
  ).embedding


class PatchEmbeddingSequence(
    collections.abc.Sequence[patch_embedding_types.EmbeddingResult]
):
  """Sequence of patches to return embeddings by index.

  If all embeddings in the sequence are going to be iteratated across accessing
  the embeddings via the iterator will provide higher performance by enabling
  multiple patches to be requested concurrently.
  """

  def __init__(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      patches: Sequence[patch_embedding_types.EmbeddingPatch],
      ensemble_method: Optional[
          patch_embedding_ensemble_methods.PatchEnsembleMethod
      ] = None,
  ):
    """Constructor for PatchEmbeddingSequence.

    Args:
      endpoint: Is the an abstraction interface through which EZ-WSI
        communicates with the various embedding model VertexAI endpoints and or
        local execution.
      patches: A sequence of patches. Patches should be clustered in the input
        sequence such that patch from the same data source are fall sequentially
        in the sequence.
      ensemble_method: Ensemble methods are optional and enable EZ-WSI to
        generate embeddings for patches which exceed the embedding dimensions of
        the endpoint. If not provided, input patches must match the input width
        and height dimensions of the endpoint.
    """
    super().__init__()
    self._endpoint = endpoint
    self._patches = patches
    self._ensemble_method = ensemble_method

  def __eq__(self, value: Any) -> bool:
    if not isinstance(value, PatchEmbeddingSequence):
      return False
    return self._patches == value._patches

  def __contains__(self, value: Any) -> bool:
    if not isinstance(value, patch_embedding_types.EmbeddingPatch):
      return False
    return value in self._patches

  def __getitem__(self, index: Union[int, slice]):
    if isinstance(index, int):
      return next(
          generate_patch_embeddings(
              self._endpoint, [self._patches[index]], self._ensemble_method
          )
      )
    return list(
        generate_patch_embeddings(
            self._endpoint, self._patches[index], self._ensemble_method
        )
    )

  def get_patch(self, index: int) -> patch_embedding_types.EmbeddingPatch:
    return self._patches[index]

  def get_embedding(self, index: int) -> np.ndarray:
    return self.__getitem__(index).embedding  # pytype: disable=attribute-error

  def __iter__(self) -> Iterator[patch_embedding_types.EmbeddingResult]:
    return generate_patch_embeddings(
        self._endpoint, self._patches, self._ensemble_method
    )

  def __len__(self) -> int:
    return len(self._patches)


def get_dicom_image_embeddings(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    slide: dicom_slide.DicomSlide,
    ps: Union[
        slide_level_map.Level,
        slide_level_map.ResizedLevel,
        pixel_spacing.PixelSpacing,
    ],
    patch_size: Optional[int] = None,
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
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> PatchEmbeddingSequence:
  """Returns Itertor of embeddings for a level of whole slide pyramid.

  Args:
    endpoint: Patch embedding endpoint to use.
    slide: DICOM Slide to extract patches from.
    ps: Pixel spacing of the slide pyramid level to extract patches from.
    patch_size: Size of the patch to extract defaults to endpoint patch size.
    mask: If provided, will be used as the embedding patch sampling mask.
    stride_size: Stride size to use when extracting patches defaults to patch
      size.
    min_luminance: Regions with luminance (grayscale) < this threshold are to be
      considered non-tissue background, and will be discarded in the patch
      sampling.
    max_luminance: Regions with luminance (grayscale) > this threshold are to be
      considered non-tissue background, and will be discarded in the patch
      sampling.
    mask_level: Pyramid level to use to determine where tissue is present if a
      tissue mask is not provded.
    ensemble_method: Method to use to genenerate embedding patches; required
      only patch dimensions != endpoint patch dimensions.

  Returns:
    Sequence of embedding results
  """
  if patch_size is None:
    patch_size = endpoint.patch_width()
  if stride_size is None:
    stride_size = patch_size
  if mask is None and mask_level is None:
    mask_level = patch_generator.TISSUE_MASK_PIXEL_SPACING
  if isinstance(mask_level, pixel_spacing.PixelSpacing):
    if (
        slide.get_level_by_pixel_spacing(mask_level, maximum_downsample=8.0)
        is None
    ):
      mask_level = ps
  target_icc_profile_bytes = endpoint.icc_profile_bytes()
  color_transform = (
      slide.create_icc_profile_transformation(target_icc_profile_bytes)
      if target_icc_profile_bytes
      else None
  )
  return PatchEmbeddingSequence(
      endpoint,
      patch_generator.DicomPatchGenerator(
          slide,
          ps,
          patch_size=patch_size,
          mask=mask,
          stride_size=stride_size,
          min_luminance=min_luminance,
          max_luminance=max_luminance,
          mask_level=mask_level,
          mask_color_transform=color_transform,
      ),
      ensemble_method,
  )


def get_gcs_image_embeddings(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    image: Union[gcs_image.GcsImage, local_image.LocalImage],
    patch_size: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    stride_size: Optional[int] = None,
    min_luminance: Optional[float] = None,
    max_luminance: Optional[float] = None,
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> PatchEmbeddingSequence:
  """Returns Itertor of embeddings for a level of whole slide pyramid.

  Args:
    endpoint: Patch embedding endpoint to use.
    image: Image to extract patches from.
    patch_size: Size of the patch to extract defaults to endpoint patch size.
    mask: If provided, will be used as the embedding patch sampling mask.
    stride_size: Stride size to use when extracting patches defaults to patch
      size.
    min_luminance: Regions with luminance (grayscale) < this threshold are to be
      considered non-tissue background, and will be discarded in the patch
      sampling.
    max_luminance: Regions with luminance (grayscale) > this threshold are to be
      considered non-tissue background, and will be discarded in the patch
      sampling.
    ensemble_method: Method to use to genenerate embedding patches; required
      only patch dimensions != endpoint patch dimensions.

  Returns:
    Iterator embedding results
  """
  if patch_size is None:
    patch_size = endpoint.patch_width()
  if stride_size is None:
    stride_size = patch_size
  target_icc_profile_bytes = endpoint.icc_profile_bytes()
  color_transform = (
      image.create_icc_profile_transformation(target_icc_profile_bytes)
      if target_icc_profile_bytes
      else None
  )
  return PatchEmbeddingSequence(
      endpoint,
      patch_generator.GcsImagePatchGenerator(
          image,
          patch_size=patch_size,
          mask=mask,
          stride_size=stride_size,
          min_luminance=min_luminance,
          max_luminance=max_luminance,
          mask_color_transform=color_transform,
      ),
      ensemble_method,
  )


def gcs_images_to_embeddings(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    images: patch_generator.GcsImagesToPatchesInputTypes,
    credential_factory: Optional[
        credential_factory_module.AbstractCredentialFactory
    ] = None,
    image_dimensions: Optional[gcs_image.ImageDimensions] = None,
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> Iterator[patch_embedding_types.EmbeddingResult]:
  """Converts whole images in GCS into embeddings."""
  return generate_patch_embeddings(
      endpoint,
      patch_generator.gcs_images_to_patches(
          images, credential_factory, image_dimensions, endpoint.max_threads()
      ),
      ensemble_method,
  )


def local_images_to_embeddings(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    images: patch_generator.LocalImagesToPatchesInputTypes,
    image_dimensions: Optional[gcs_image.ImageDimensions] = None,
    ensemble_method: Optional[
        patch_embedding_ensemble_methods.PatchEnsembleMethod
    ] = None,
) -> Iterator[patch_embedding_types.EmbeddingResult]:
  """Converts whole local images into embeddings."""
  return generate_patch_embeddings(
      endpoint,
      patch_generator.local_images_to_patches(images, image_dimensions),
      ensemble_method,
  )


# init class module variables if forked.
os.register_at_fork(after_in_child=_init_request_throttle)  # pylint: disable=protected-access
