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
import os
import threading
import time
from typing import Any, Generic, Iterator, List, Optional, Sequence, TypeVar, Union

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
) -> Any:
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

  return _inner_func()


FutureReturnType = TypeVar('FutureReturnType')


@dataclasses.dataclass(frozen=True)
class _FuturePatchRequest(Generic[FutureReturnType]):
  requested_slide_embedding_sources: Sequence[
      patch_embedding_types.SlideEmbeddingSource
  ]
  future: futures.Future[FutureReturnType]


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

  def request_future_embeddings(
      self,
      endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
      executor: futures.ThreadPoolExecutor,
  ) -> List[_FuturePatchRequest]:
    """Executes threaded request for patch embeddings."""
    if not self.has_queued_embedding_requests:
      return []
    future_list = []
    run_request_futures_loop = True
    while run_request_futures_loop:
      run_request_futures_loop = False

      pending_request_size_in_bytes = 0
      prepared_request_list: List[
          patch_embedding_endpoints.AbstractPreparedEmbeddingRequest
      ] = []
      request_overflow = []
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
          # continue to processes if more than one element remaining in the
          run_request_futures_loop = (
              len(request_overflow) > 1 or overflow_size > max_request_size
          )
          break
      future_list.append(
          _FuturePatchRequest[endpoint.request_embedding_return_type](
              [pr.slide_embedding_source for pr in prepared_request_list],
              executor.submit(
                  _get_embedding_thread, endpoint, prepared_request_list
              ),
          )
      )
      self._queued_embedding_image_requests = request_overflow
      if not self._queued_embedding_image_requests:
        self._slide_processing = None
      self._recalulate_patch_counts_for_queued_requests()
    return future_list

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
  api_request = _EmbeddingAPIRequest()
  max_number_of_patches_per_request = (
      endpoint.max_number_of_patches_per_request()
  )
  endpoint_max_mag_scaled_patch_count = (
      endpoint.endpoint_max_number_of_patches_per_request()
  )
  future_patch_embedding_sources = []
  with futures.ThreadPoolExecutor(
      max_workers=endpoint.max_threads()
  ) as executor:
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
        future_patch_embedding_sources.extend(
            api_request.request_future_embeddings(endpoint, executor)
        )
      if api_request.slide_processing != slide_key:
        api_request.add_new_slide(slide_key)
      api_request.add_patch(patch_embedding_source, patch_count)
    request_queue_length = len(future_patch_embedding_sources)
    while api_request.has_queued_embedding_requests:
      future_patch_embedding_sources.extend(
          api_request.request_future_embeddings(endpoint, executor)
      )
      post = len(future_patch_embedding_sources)
      if post == request_queue_length:
        raise ez_wsi_errors.InternalError(
            'Error request queue is not processing.'
        )
      request_queue_length = post
    for fc in future_patch_embedding_sources:
      yield endpoint.process_response(
          fc.requested_slide_embedding_sources, fc.future.result()
      )


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
  color_transform = slide.create_icc_profile_transformation(
      endpoint.icc_profile_bytes()
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
  color_transform = image.create_icc_profile_transformation(
      endpoint.icc_profile_bytes()
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
