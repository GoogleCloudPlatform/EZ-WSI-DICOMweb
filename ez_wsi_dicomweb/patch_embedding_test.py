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
"""Patch embedding unit tests."""

from concurrent import futures
import os
import time
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_embedding
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import patch_embedding_ensemble_methods
from ez_wsi_dicomweb import patch_embedding_types
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
from ez_wsi_dicomweb.test_utils import embedding_endpoint_mock
import numpy as np
import PIL
import pydicom
import requests_mock

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


_ERROR_MESSAGE = None
_VERSION = 'MOCK_EMBEDDINGS_VERSION'
# Standardize size of patches returned in metadata
_TEST_LOCAL_IMAGE_PATCH_SIZE_BYTES = 108


def mock_v2_embedding_response(
    patch_count: int,
) -> patch_embedding_endpoints._VertexModelResult:
  embedding_results = [{
      'model_version': _VERSION,
      'result': {
          'patch_embeddings': [
              {
                  'embedding_vector': [1.1, 2.1, 3.1],
                  'patch_coordinate': {
                      'x_origin': 0,
                      'y_origin': 0,
                      'width': 224,
                      'height': 224,
                  },
              },
          ] * patch_count,
      },
  }]
  return patch_embedding_endpoints._VertexModelResult(embedding_results)


class DicomPatchEmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    test_instance_2 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2.SeriesInstanceUID = '1.4'
    series_path_2 = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_2.StudyInstanceUID}/series/{test_instance_2.SeriesInstanceUID}'
    mock_store = self.enter_context(
        dicom_store_mock.MockDicomStores(dicom_store_path)
    )
    mock_store[dicom_store_path].add_instance(test_instance)
    mock_store[dicom_store_path].add_instance(test_instance_2)
    embedding_endpoint_mock.V1DicomEmbeddingEndpointMock(
        mock_store[dicom_store_path].mock_request,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/hai-cd3-foundations/locations/us-central1/endpoints/160:predict',
    )
    self.slide = dicom_slide.DicomSlide(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path),
        enable_client_slide_frame_decompression=True,
    )
    self.slide_2 = dicom_slide.DicomSlide(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path_2),
        enable_client_slide_frame_decompression=True,
    )
    self.ps = pixel_spacing.PixelSpacing.FromDicomPixelSpacingTag(
        test_instance.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )

  def test_patch_embedding(self):
    patch = self.slide.get_patch(
        self.slide.get_level_by_pixel_spacing(self.ps), 0, 0, 224, 224
    )
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    embedding = patch_embedding.get_patch_embedding(endpoint, patch)
    self.assertEqual(
        embedding.tolist(),
        [197.64901546556123, 182.4219347895408, 210.72072305484693],
    )

  def test_patch_embeddings(self):
    patch_1 = self.slide.get_patch(self.ps, 0, 0, 224, 224)
    patch_2 = self.slide.get_patch(self.ps, 224, 0, 224, 224)
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    embeddings = list(
        patch_embedding.generate_patch_embeddings(endpoint, [patch_1, patch_2])
    )
    self.assertEqual(
        [embedding.embedding.tolist() for embedding in embeddings],
        [
            [197.64901546556123, 182.4219347895408, 210.72072305484693],
            [224.57752710459184, 216.72927295918367, 225.5372090242347],
        ],
    )

  def test_image_embeddings(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    embeddings = patch_embedding.get_dicom_image_embeddings(
        endpoint, self.slide, self.ps
    )
    self.assertEqual(
        [embedding.embedding.tolist() for embedding in embeddings],
        [
            [197.64901546556123, 182.4219347895408, 210.72072305484693],
            [202.0694355867347, 186.63464604591837, 212.57577327806123],
            [174.03579400510205, 144.70822704081633, 193.98284040178572],
            [198.14895567602042, 175.61427774234693, 207.2947225765306],
            [166.46960698341837, 131.35658482142858, 189.21237244897958],
            [201.8584582270408, 182.96193399234693, 211.23140545280611],
            [203.85528938137756, 187.38020169005102, 212.38071986607142],
            [200.1580038265306, 179.3311144770408, 209.46047911352042],
        ],
    )

  @mock.patch.object(
      patch_embedding,
      '_get_embedding_thread',
      autospec=True,
      return_value=mock_v2_embedding_response(2),
  )
  def test_embedding_api_call_scales_patch_count_by_mag_single_request(
      self, mock_get_embedding_thread
  ):
    patch = self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)
    sources = [
        patch_embedding_types.PatchEmbeddingSource(patch, patch, '1'),
        patch_embedding_types.PatchEmbeddingSource(patch, patch, '1'),
    ]
    list(
        patch_embedding._embedding_api_call(
            patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
                max_patches_per_request=8
            ),
            sources,
        )
    )
    mock_get_embedding_thread.assert_called_once()

  @mock.patch.object(
      patch_embedding,
      '_get_embedding_thread',
      autospec=True,
      return_value=mock_v2_embedding_response(1),
  )
  def test_embedding_api_call_scales_patch_count_by_mag_two_request(
      self, mock_get_embedding_thread
  ):
    resize_level = self.slide.native_level.resize(
        dicom_slide.ImageDimensions(
            int(self.slide.native_level.width // 8),
            int(self.slide.native_level.height // 8),
        )
    )
    patch = self.slide.get_patch(resize_level, 0, 0, 224, 224)
    sources = [
        patch_embedding_types.PatchEmbeddingSource(patch, patch, '1'),
        patch_embedding_types.PatchEmbeddingSource(patch, patch, '1'),
    ]
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    endpoint._endpoint_max_patches_per_request = 8
    list(
        patch_embedding._embedding_api_call(
            endpoint,
            sources,
        )
    )
    self.assertEqual(mock_get_embedding_thread.call_count, 2)

  @mock.patch.object(
      patch_embedding,
      '_get_embedding_thread',
      autospec=True,
      return_value=mock_v2_embedding_response(10),
  )
  def test_patches_from_resize_levels_with_same_mag_and_source_are_processed_together(
      self, mock_get_embedding_thread
  ):
    sources = []
    for _ in range(10):
      resize_level = self.slide.native_level.resize(
          dicom_slide.ImageDimensions(
              int(self.slide.native_level.width // 2),
              int(self.slide.native_level.height // 2),
          )
      )
      patch = self.slide.get_patch(resize_level, 0, 0, 224, 224)
      sources.append(
          patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
      )
    list(
        patch_embedding._embedding_api_call(
            patch_embedding_endpoints.V2PatchEmbeddingEndpoint(),
            sources,
        )
    )
    mock_get_embedding_thread.assert_called_once()
    self.assertLen(mock_get_embedding_thread.call_args[0][1], 1)
    coords = ', '.join(
        ['{"x_origin": 0, "y_origin": 0, "width": 224, "height": 224}'] * 10
    )
    self.assertIn(
        f'"patch_coordinates": [{coords}]',
        mock_get_embedding_thread.call_args[0][1][0].json,
    )

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_len(self, count):
    sq = patch_embedding.PatchEmbeddingSequence(
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint(),
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)] * count,
    )
    self.assertLen(sq, count)

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_equal_true(self, count):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)] * count,
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)] * count,
    )
    self.assertEqual(sq_1, sq_2)

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_equal_false_different_lengths(self, count):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)] * count,
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)]
        * (count + 1),
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_coord(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 10, 10, 224, 224)],
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_dim(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 225, 223)],
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_source_levels_resize(
      self,
  ):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [
            self.slide.get_patch(
                self.slide.native_level.resize(
                    dicom_slide.ImageDimensions(10, 10)
                ),
                0,
                0,
                224,
                224,
            )
        ],
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_slides(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide_2.get_patch(self.slide_2.native_level, 0, 0, 224, 224)],
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_contains_true(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    patch = self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    self.assertIn(patch, sq)

  def test_patch_embedding_sequence_contains_false_not_same_patch(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    patch = self.slide.get_patch(self.slide.native_level, 10, 0, 224, 224)
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    self.assertNotIn(patch, sq)

  def test_patch_embedding_sequence_contains_false_not_patch(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    self.assertNotIn('A', sq)

  def test_patch_embedding_sequence_get_embedding(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    result = [round(r, 1) for r in sq.get_embedding(0).tolist()]
    self.assertEqual(result, [197.6, 182.4, 210.7])

  def test_patch_embedding_sequence_out_of_bounds_raises_index_error(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    with self.assertRaises(IndexError):
      _ = sq[1]

  def test_patch_embedding_sequence_index_returns_embedding_result(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    self.assertIsInstance(sq[0], patch_embedding_types.EmbeddingResult)

  def test_patch_embedding_sequence_sliceindex_returns_list_of_embeding_result(
      self,
  ):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    self.assertIsInstance(sq[:2], list)
    self.assertLen(sq[:2], 1)
    self.assertIsInstance(sq[0], patch_embedding_types.EmbeddingResult)

  def test_patch_get_embedding_result(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint,
        [self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)],
    )
    result = sq[0]
    embedding = [round(r, 1) for r in result.embedding.tolist()]  # pytype: disable=attribute-error
    self.assertEqual(embedding, [197.6, 182.4, 210.7])
    self.assertEqual(
        result.patch,  # pytype: disable=attribute-error
        self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224),
    )

  @parameterized.named_parameters([
      dict(testcase_name='min_size_one_patch', max_size=6280),
      dict(testcase_name='min_size_one_patch_plus_coord', max_size=6340),
  ])
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_future_embeddings_request_resize_dicom_patch_source(
      self, mock_thread_request, mock_endpoint_max_size, max_size
  ):
    mock_endpoint_max_size.return_value = max_size
    source_level = self.slide.native_level.resize(
        dicom_slide.ImageDimensions(
            self.slide.native_level.width // 2,
            self.slide.native_level.height // 2,
        )
    )
    patch = self.slide.get_patch(source_level, 0, 0, 224, 224)
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(source_level)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # two patches mag 2x 4x multiplier on request count
    self.assertEqual(embedding_request.mag_scaled_patch_count, 8)
    self.assertEqual(embedding_request.patch_count, 2)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 1
      )
    mock_thread_request.assert_called_once()
    # patches remain in queue
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    self.assertLen(embedding_request, 1)  # one slide in queue
    # one patch in queue with 4x magnification multiplier
    self.assertEqual(embedding_request.mag_scaled_patch_count, 4)
    self.assertEqual(embedding_request.patch_count, 1)

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
      return_value=6264,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_future_embeddings_request_dicom_patch_source(
      self, mock_thread_request, unused_mock
  ):
    source_level = self.slide.native_level
    patch = self.slide.get_patch(source_level, 0, 0, 224, 224)
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(source_level)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # two patches mag 1x multiplier on request count
    self.assertEqual(embedding_request.mag_scaled_patch_count, 2)
    self.assertEqual(embedding_request.patch_count, 2)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 1
      )
    mock_thread_request.assert_called_once()
    # patches remain in queue
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    self.assertLen(embedding_request, 1)  # one slide in queue
    # one patch in queue with 1x magnification multiplier
    self.assertEqual(embedding_request.mag_scaled_patch_count, 1)
    self.assertEqual(embedding_request.patch_count, 1)

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
      return_value=6179,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_split_dicom_patch_source_greater_than_min_endpoint_sizes_raises(
      self, mock_thread_request, unused_mock_endpoint_max_size
  ):
    source_level = self.slide.native_level
    patch = self.slide.get_patch(source_level, 0, 0, 224, 224)
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(source_level)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # two patches mag 1x multiplier on request count
    self.assertEqual(embedding_request.mag_scaled_patch_count, 2)
    self.assertEqual(embedding_request.patch_count, 2)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchEmbeddingEndpointError,
          'Embedding request size,.*, exceeds endpoint size limit,.*',
      ):
        embedding_request.request_future_embeddings(endpoint, executor)
    mock_thread_request.assert_not_called()


class GcsPatchEmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    with PIL.Image.open(dicom_test_utils.testdata_path('dcm_frame_1.jpg')) as i:
      i.resize((512, 512)).save(os.path.join(self.temp_dir, 'test_image.jpg'))
    self.enter_context(gcs_mock.GcsMock({'test_bucket': self.temp_dir}))
    mock_request = self.enter_context(requests_mock.Mocker())
    embedding_endpoint_mock.V1GcsEmbeddingEndpointMock(
        mock_request,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/hai-cd3-foundations/locations/us-central1/endpoints/160:predict',
    )
    self.enter_context(
        mock.patch.object(
            credential_factory,
            'get_default_gcp_project',
            autospec=True,
            return_value='MOCK_PROJECT',
        )
    )
    self.enter_context(
        mock.patch.object(
            credential_factory.CredentialFactory,
            'get_credentials',
            autospec=True,
        )
    )

  def test_get_gcs_image_embeddings(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    embeddings = patch_embedding.get_gcs_image_embeddings(endpoint, image)
    self.assertEqual(
        [embedding.embedding.tolist() for embedding in embeddings],
        [
            [201.29547991071428, 189.19955755739795, 213.2258450255102],
            [143.90625, 106.48393654336735, 177.84875239158163],
        ],
    )

  def test_get_gcs_image_embeddings_throttled_initalization(self):
    self.assertIsNone(patch_embedding._max_requests_per_minute)
    try:
      patch_embedding.set_max_embedding_requests_per_min(15)
      self.assertEqual(patch_embedding._max_requests_per_minute, 15)
    finally:
      patch_embedding.disable_embedding_request_throttling()
      self.assertIsNone(patch_embedding._max_requests_per_minute)

  def test_get_gcs_image_embeddings_throttled(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        max_patches_per_request=1
    )
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    try:
      patch_embedding.set_max_embedding_requests_per_min(15)
      start_time = time.time()
      list(
          patch_embedding.generate_patch_embeddings(
              endpoint, [image.get_patch(0, 0, 224, 224)] * 2
          )
      )
      total_time = time.time() - start_time
      self.assertGreater(total_time, 4)
    finally:
      patch_embedding.disable_embedding_request_throttling()

  def test_get_gcs_image_embeddings_unthrottled(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        max_patches_per_request=1
    )
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    start_time = time.time()
    list(
        patch_embedding.generate_patch_embeddings(
            endpoint, [image.get_patch(0, 0, 224, 224)] * 2
        )
    )
    total_time = time.time() - start_time
    self.assertLess(total_time, 4)

  def test_init_request_throttle(self) -> None:
    patch_embedding._request_lock = None
    patch_embedding._init_request_throttle()
    self.assertIsNotNone(patch_embedding._request_lock)

  def test_gcs_images_to_embeddings(self):
    path = os.path.join(self.temp_dir, 'test_image.jpg')
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(path, img)
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    result = patch_embedding.gcs_images_to_embeddings(
        endpoint,
        ['gs://test_bucket/test_image.jpg'] * 2,
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    self.assertLen(list(result), 2)

  def test_local_images_to_embeddings(self):
    path = os.path.join(self.temp_dir, 'test_image.jpg')
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    result = patch_embedding.local_images_to_embeddings(
        endpoint,
        [img] * 2,
        image_dimensions=gcs_image.ImageDimensions(224, 224),
    )
    self.assertLen(list(result), 2)

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_len(self, count):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq = patch_embedding.PatchEmbeddingSequence(
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint(),
        [image.get_patch(0, 0, 224, 224)] * count,
    )
    self.assertLen(sq, count)

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_equal_true(self, count):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)] * count
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)] * count
    )
    self.assertEqual(sq_1, sq_2)

  @parameterized.parameters(list(range(3)))
  def test_patch_embedding_sequence_equal_false_different_lengths(self, count):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)] * count
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)] * (count + 1)
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_coord(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)]
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(10, 10, 224, 224)]
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_dim(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)]
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 225, 223)]
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_different_source(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image_1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    image_2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_2.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image_1.get_patch(0, 0, 224, 224)]
    )
    sq_2 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image_2.get_patch(0, 0, 224, 224)]
    )
    self.assertNotEqual(sq_1, sq_2)

  def test_patch_embedding_sequence_equal_false_not_embedding_seq(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image_1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image_1.get_patch(0, 0, 224, 224)]
    )
    self.assertNotEqual(sq_1, [])

  def test_patch_embedding_sequence_get_patch(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image_1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    patch = image_1.get_patch(0, 0, 224, 224)
    sq_1 = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image_1.get_patch(0, 0, 224, 224)]
    )
    self.assertEqual(sq_1.get_patch(0), patch)

  def test_patch_embedding_sequence_contains_true(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    patch = image.get_patch(0, 0, 224, 224)
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)]
    )
    self.assertIn(patch, sq)

  def test_patch_embedding_sequence_contains_false_not_same_patch(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    patch = image.get_patch(10, 0, 224, 224)
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)]
    )
    self.assertNotIn(patch, sq)

  def test_patch_embedding_sequence_contains_false_not_patch(self):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    sq = patch_embedding.PatchEmbeddingSequence(
        endpoint, [image.get_patch(0, 0, 224, 224)]
    )
    self.assertNotIn('A', sq)

  def test_embedding_api_request_len_counts_number_of_slides(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    self.assertEmpty(embedding_request)
    self.assertFalse(embedding_request.has_queued_embedding_requests)
    embedding_request.add_new_slide(image)
    self.assertLen(embedding_request, 1)
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    embedding_request.add_new_slide(image)
    self.assertLen(embedding_request, 2)
    self.assertTrue(embedding_request.has_queued_embedding_requests)

  @parameterized.named_parameters([
      dict(
          testcase_name='min_size_for_one_patch_no_metadata',
          max_size=274,
          patch_count=1,
      ),
      dict(
          testcase_name='min_size_for_two_patches_no_metadata',
          max_size=333,
          patch_count=2,
      ),
  ])
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_future_embeddings_request_within_single_small_request(
      self, mock_thread_request, mock_endpoint_max_size, max_size, patch_count
  ):
    mock_endpoint_max_size.return_value = max_size

    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            'MOCK_TOKEN'
        ),
    )
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(image)
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    for _ in range(patch_count):
      embedding_request.add_patch(
          source, source.mag_scaled_embedding_patch_count
      )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # one embedding requests in queue
    self.assertEqual(embedding_request.mag_scaled_patch_count, patch_count)
    self.assertEqual(embedding_request.patch_count, patch_count)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 1
      )
    # residual size of last embedding is less than max embedding_count == 0
    self.assertFalse(embedding_request.has_queued_embedding_requests)
    # sent one endpoint request
    mock_thread_request.assert_called_once()
    self.assertEqual(embedding_request.mag_scaled_patch_count, 0)
    self.assertEqual(embedding_request.patch_count, 0)
    self.assertEmpty(embedding_request)  # no slides in queue

  @parameterized.named_parameters([
      dict(testcase_name='max_size_one_patch_with_metadata', max_size=484),
      dict(
          testcase_name='max_size_one_patch_and_metadata_for_next', max_size=588
      ),
      dict(
          testcase_name='max_size_one_patch_and_md_and_cord_for_next',
          max_size=647,
      ),
  ])
  @mock.patch.object(
      gcs_image,
      '_gcs_image_json_metadata',
      autospec=True,
      return_value='*' * _TEST_LOCAL_IMAGE_PATCH_SIZE_BYTES,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_split_across_multiple_requests_small_residual(
      self, mock_thread_request, mock_endpoint_max_size, _, max_size
  ):
    mock_endpoint_max_size.return_value = max_size
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(image)
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # two embedding requests in queue
    self.assertEqual(embedding_request.mag_scaled_patch_count, 2)
    self.assertEqual(embedding_request.patch_count, 2)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 1
      )
    # residual size of last embedding is less than max embedding_count > 0
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    # sent one endpoint request
    mock_thread_request.assert_called_once()
    # one embeddings in queue residual size for embeddings less than endpoint
    # max size. Note residual size under estimates the actual size of the
    # embedding, it does not account for the non-optional json metadata.
    self.assertEqual(embedding_request.mag_scaled_patch_count, 1)
    self.assertEqual(embedding_request.patch_count, 1)
    self.assertLen(embedding_request, 1)  # one slide in queue

  @mock.patch.object(
      gcs_image,
      '_gcs_image_json_metadata',
      autospec=True,
      return_value='*' * _TEST_LOCAL_IMAGE_PATCH_SIZE_BYTES,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
      return_value=484,  # Min size to return one patch
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_split_across_multiple_requests_remaining_request(
      self, mock_thread_request, *unused_mocks
  ):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(image)
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # four embedding requests in queue
    self.assertEqual(embedding_request.mag_scaled_patch_count, 4)
    self.assertEqual(embedding_request.patch_count, 4)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 2
      )
    # residual size of last embedding is less than max embedding_count > 0
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    # sent two endpoint requests, after first request total size of was >
    # endpoint max
    self.assertEqual(mock_thread_request.call_count, 2)
    # two embeddings in queue residual size for embeddings less than endpoint
    # max size. Note residual size under estimates the actual size of the
    # embedding, it does not account for the non-optional json metadata.
    self.assertEqual(embedding_request.mag_scaled_patch_count, 2)
    self.assertEqual(embedding_request.patch_count, 2)
    self.assertLen(embedding_request, 1)  # one slide in queue

  @mock.patch.object(
      gcs_image,
      '_gcs_image_json_metadata',
      autospec=True,
      return_value='*' * _TEST_LOCAL_IMAGE_PATCH_SIZE_BYTES,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
      return_value=484,  # Min size to return one patch
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_split_across_multiple_requests_two_source_images(
      self, mock_thread_request, *unused_mocks
  ):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(image)
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    embedding_request.add_new_slide(image)
    embedding_request.add_patch(source, source.mag_scaled_embedding_patch_count)
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # two slide in queue
    self.assertLen(embedding_request, 2)
    # four embedding requests in queue
    self.assertEqual(embedding_request.mag_scaled_patch_count, 4)
    self.assertEqual(embedding_request.patch_count, 4)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertLen(
          embedding_request.request_future_embeddings(endpoint, executor), 3
      )
    # residual size of last embedding is less than max embedding_count > 0
    self.assertTrue(embedding_request.has_queued_embedding_requests)
    # sent all requests for first slide split across 3 requests
    # expected to fully process first slide to reduce queued slides to at most
    # one slide.
    self.assertEqual(mock_thread_request.call_count, 3)
    # one embeddings in queue residual size for embeddings less than endpoint
    # max size. Note residual size under estimates the actual size of the
    # embedding, it does not account for the non-optional json metadata.
    self.assertEqual(embedding_request.mag_scaled_patch_count, 1)
    self.assertEqual(embedding_request.patch_count, 1)
    self.assertLen(embedding_request, 1)  # one slide in queue

  @parameterized.named_parameters(
      [
          dict(
              testcase_name='cannot_split_if_only_one_patch',
              patch_count=1,
              max_size=483,
          ),
          dict(
              testcase_name='insufficient_space_for_metadata',
              patch_count=2,
              max_size=483,
          ),
          dict(
              testcase_name='insufficient_space_for_coordinates',
              patch_count=2,
              max_size=385,
          ),
      ],
  )
  @mock.patch.object(
      gcs_image,
      '_gcs_image_json_metadata',
      autospec=True,
      return_value='*' * _TEST_LOCAL_IMAGE_PATCH_SIZE_BYTES,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'max_request_size_bytes',
      autospec=True,
  )
  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_future_embeddings_request_endpoint_max_less_than_min_raises(
      self,
      mock_thread_request,
      mock_endpoint_max_size,
      _,
      patch_count,
      max_size,
  ):
    mock_endpoint_max_size.return_value = max_size
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    embedding_request.add_new_slide(image)
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    for _ in range(patch_count):
      embedding_request.add_patch(
          source, source.mag_scaled_embedding_patch_count
      )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    # one slide in queue
    self.assertLen(embedding_request, 1)
    # one embedding requests in queue
    self.assertEqual(embedding_request.mag_scaled_patch_count, patch_count)
    self.assertEqual(embedding_request.patch_count, patch_count)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchEmbeddingEndpointError,
          'Embedding request size,.*, exceeds endpoint size limit,.*',
      ):
        embedding_request.request_future_embeddings(endpoint, executor)
    mock_thread_request.assert_not_called()

  @mock.patch.object(patch_embedding, '_get_embedding_thread', autospec=True)
  def test_request_future_embeddings_empty_request(self, mock_thread_request):
    embedding_request = patch_embedding._EmbeddingAPIRequest()
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
      self.assertEqual(
          embedding_request.request_future_embeddings(endpoint, executor), []
      )
    mock_thread_request.assert_not_called()

  def test_get_embedding_thread_retry(self):
    mock_endpoint = mock.create_autospec(
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
        instance=True,
    )
    mock_endpoint.retry_count.return_value = 4
    mock_request = [
        mock.create_autospec(
            patch_embedding_endpoints.PreparedVertexEmbeddingRequest,
            instance=True,
        )
    ] * 4
    mock_endpoint.request_embeddings.side_effect = ez_wsi_errors.HttpError
    with self.assertRaises(ez_wsi_errors.HttpError):
      start = time.time()
      patch_embedding._get_embedding_thread(mock_endpoint, mock_request)
    elapsed = time.time() - start
    self.assertGreater(elapsed, 1.0)
    self.assertEqual(mock_endpoint.request_embeddings.call_count, 4)

  @parameterized.named_parameters([
      dict(
          testcase_name='no_results',
          results=[],
          expected_result=[],
      ),
      dict(
          testcase_name='one_result',
          results=[['a']],
          expected_result=[1],
      ),
      dict(
          testcase_name='multiple_in_one',
          results=[['a', 'b', 'c']],
          expected_result=[1, 1, 1],
      ),
      dict(
          testcase_name='multiple_combined_in_one',
          results=[['a', 'a', 'a']],
          expected_result=[3],
      ),
      dict(
          testcase_name='multiple_combined_in_multiple_in_one',
          results=[['a', 'a', 'a', 'b', 'c', 'c', 'd']],
          expected_result=[3, 1, 2, 1],
      ),
      dict(
          testcase_name='single_split_over_multi_response',
          results=[['a'], ['a'], ['a']],
          expected_result=[3],
      ),
      dict(
          testcase_name='multiple_split_over_multi_response',
          results=[['a', 'a', 'a', 'b'], ['b', 'b'], ['b', 'c']],
          expected_result=[3, 4, 1],
      ),
  ])
  def test_reduce_embedding_ensemble(self, results, expected_result):
    # tests the ensemble reducer by mocking patch ensembling results.
    # test pass in list is of ensemble ids then uses a mocked ensemble reducer
    # that counts and returns the number of ids that were grouped. Ids are
    # expected to be contiguous, e.g., 'a', 'a' would be grouped but 'a', 'b',
    # 'a' is is not valid. The expected response for something like this would
    # be 1, 1, 1
    class MockEnsembleReducer:

      def reduce_ensemble(self, unused_patch, ensemble_list):
        return len(ensemble_list)

    mock_ensable_method = typing.cast(
        patch_embedding_ensemble_methods.PatchEnsembleMethod,
        MockEnsembleReducer(),
    )
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    mock_embedding_result = np.zeros((1), dtype=np.uint8)
    split_responses = []
    for result_id_list in results:
      split_responses.append([
          patch_embedding_types.PatchEmbeddingEnsembleResult(
              patch_embedding_types.PatchEmbeddingSource(
                  mock_patch, mock_patch, id
              ),
              mock_embedding_result,
              None,
          )
          for id in result_id_list
      ])
    self.assertEqual(
        list(
            patch_embedding._reduce_embedding_ensemble(
                mock_ensable_method, iter(split_responses)
            )
        ),
        expected_result,
    )


if __name__ == '__main__':
  absltest.main()
