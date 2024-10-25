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
"""Tests patch embedding endpoints."""

import base64
import concurrent.futures
import functools
import hashlib
import http
import io
import json
import os
import shutil
import threading
import typing
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cachetools
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb import patch_embedding
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import patch_embedding_types
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import google.auth
import numpy as np
import PIL.Image
import pydicom
import requests
import requests_mock

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


_DEFAULT_ENDPOINT_URL = 'https://us-central1-aiplatform.googleapis.com/v1/projects/hai-cd3-foundations/locations/us-central1/endpoints/160:predict'
_ERROR_MESSAGE = None
_VERSION = 'MOCK_EMBEDDINGS_VERSION'
_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys


def _mock_request_response(msg: str) -> requests.Response:
  response = mock.create_autospec(requests.Response, instance=True)
  type(response).text = mock.PropertyMock(return_value=msg)
  return response


def _mock_apply_credentials(
    default_token: str,
    headers: MutableMapping[str, str],
    token: Optional[str] = None,
) -> None:
  headers['authorization'] = 'Bearer {}'.format(token or default_token)


def _credential_mock(token: str) -> google.auth.credentials.Credentials:
  credentials_mock = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  type(credentials_mock).token = mock.PropertyMock(return_value=token)
  type(credentials_mock).valid = mock.PropertyMock(return_value='True')
  type(credentials_mock).expired = mock.PropertyMock(return_value='False')
  credentials_mock.apply.side_effect = functools.partial(
      _mock_apply_credentials, token
  )
  return credentials_mock


def _mock_model(image_patches: np.ndarray) -> np.ndarray:
  return np.mean(image_patches, axis=(1, 2))


def _predictions(
    pred: List[Any],
) -> Dict[str, Tuple[Any, Optional[str], str]]:
  return {_EndpointJsonKeys.PREDICTIONS: (pred, _ERROR_MESSAGE, _VERSION)}


def _load_img_bytes(img: str) -> bytes:
  """Decode base64 encoded image to uncompressed bytes."""
  with PIL.Image.open(io.BytesIO(base64.b64decode(img))) as img:
    return np.asarray(img).tobytes()


def _unencoded_result_images(results: Mapping[str, Any]) -> Dict[str, Any]:
  """Decodes png images that are in gcs metadata and replaces with raw bytes."""
  results = dict(results)
  for instance_index in range(len(results[_EndpointJsonKeys.INSTANCES])):
    instance = results[_EndpointJsonKeys.INSTANCES][instance_index]
    instance[_EndpointJsonKeys.EXTENSIONS][_EndpointJsonKeys.EZ_WSI_STATE] = (
        instance[_EndpointJsonKeys.EXTENSIONS][_EndpointJsonKeys.EZ_WSI_STATE]
    )
    ez_wsi_state = instance[_EndpointJsonKeys.EXTENSIONS][
        _EndpointJsonKeys.EZ_WSI_STATE
    ]
    for index, patch in enumerate(ez_wsi_state.get('patches', [])):
      ez_wsi_state['patches'][index] = _load_img_bytes(patch)
    image_bytes = ez_wsi_state.get(_EndpointJsonKeys.IMAGE, '')
    if image_bytes:
      ez_wsi_state[_EndpointJsonKeys.IMAGE] = _load_img_bytes(image_bytes)
  return results


class PatchEmbeddingEndpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    mock_store = self.enter_context(
        dicom_store_mock.MockDicomStores(dicom_store_path)
    )
    mock_store[dicom_store_path].add_instance(test_instance)
    self.mock_store_instance = mock_store[dicom_store_path]
    self.slide = dicom_slide.DicomSlide(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(str(series_path)),
        enable_client_slide_frame_decompression=True,
    )
    self.ps = pixel_spacing.PixelSpacing.FromDicomPixelSpacingTag(
        test_instance.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )

  def test_v1_patch_dimensions(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    self.assertEqual(endpoint.patch_width(), 224)
    self.assertEqual(endpoint.patch_height(), 224)

  def test_v1_patch_defaults(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    self.assertEqual(endpoint._end_point_url, _DEFAULT_ENDPOINT_URL)
    self.assertEqual(endpoint._model_size, 'MEDIUM')
    self.assertEqual(endpoint._model_kind, 'LOW_PIXEL_SPACING')
    self.assertEqual(endpoint.max_threads(), 5)
    self.assertEqual(endpoint.max_number_of_patches_per_request(), 100)
    self.assertEqual(
        endpoint.endpoint_max_number_of_patches_per_request(), 3000
    )

  def test_v1_patch_param(self):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        endpoint_api_version='v2',
        project_id='foo-project',
        endpoint_location='us-west1',
        endpoint_id='123',
        max_threads=1,
        max_patches_per_request=20,
    )
    self.assertEqual(
        endpoint._end_point_url,
        'https://us-west1-aiplatform.googleapis.com/v2/projects/foo-project/locations/us-west1/endpoints/123:predict',
    )
    self.assertEqual(endpoint.max_threads(), 1)
    self.assertEqual(endpoint.max_number_of_patches_per_request(), 20)

  @parameterized.named_parameters([
      dict(testcase_name='min_threads', max_threads=-1, expected_threads=1),
      dict(
          testcase_name='max_threads',
          max_threads=99999,
          expected_threads=10,
      ),
  ])
  def test_v1_min_max_threads(self, max_threads, expected_threads):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        max_threads=max_threads
    )
    self.assertEqual(endpoint.max_threads(), expected_threads)

  @parameterized.named_parameters([
      dict(
          testcase_name='min_patches_per_request',
          max_patches_per_request=-1,
          expected_max_patches_per_requests=1,
      ),
      dict(
          testcase_name='max_patches_per_request',
          max_patches_per_request=99999,
          expected_max_patches_per_requests=3000,
      ),
  ])
  def test_v1_min_max_patches_per_request(
      self, max_patches_per_request, expected_max_patches_per_requests
  ):
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        max_patches_per_request=max_patches_per_request
    )
    self.assertEqual(
        endpoint.max_number_of_patches_per_request(),
        expected_max_patches_per_requests,
    )

  def test_v1_patch_empty_request(self):
    result = (
        patch_embedding_endpoints.V1PatchEmbeddingEndpoint().request_embeddings(
            []
        )
    )
    self.assertEmpty(result.instances)

  def _request_call_back(self, request, context):
    del context
    # test expected api request is received
    try:
      test_json = json.loads(request.text)
    except json.decoder.JSONDecodeError as exp:
      raise ValueError(f'Error decoding: {request.text}') from exp
    with open(
        dicom_test_utils.testdata_path('v1_endpoint_request.json'), 'rt'
    ) as infile:
      expected_json = json.load(infile)
    self.assertEqual(test_json, expected_json)
    embedding_results = []
    embedding_results.append({
        _EndpointJsonKeys.DICOM_STUDY_UID: (
            '1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315'
        ),
        _EndpointJsonKeys.DICOM_SERIES_UID: (
            '1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634'
        ),
        'patch_embeddings': [
            {
                _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                _EndpointJsonKeys.PATCH_COORDINATE: {
                    _EndpointJsonKeys.X_ORIGIN: 10,
                    _EndpointJsonKeys.Y_ORIGIN: 0,
                    _EndpointJsonKeys.WIDTH: 224,
                    _EndpointJsonKeys.HEIGHT: 224,
                },
            },
            {
                _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                _EndpointJsonKeys.PATCH_COORDINATE: {
                    _EndpointJsonKeys.X_ORIGIN: 10,
                    _EndpointJsonKeys.Y_ORIGIN: 0,
                    _EndpointJsonKeys.WIDTH: 224,
                    _EndpointJsonKeys.HEIGHT: 224,
                },
            },
        ],
    })
    embedding_results.append(
        {
            _EndpointJsonKeys.DICOM_STUDY_UID: '1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315',
            _EndpointJsonKeys.DICOM_SERIES_UID: '1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634',
            'patch_embeddings': [{
                _EndpointJsonKeys.EMBEDDINGS: [4.1, 5.1, 6.1],
                _EndpointJsonKeys.PATCH_COORDINATE: {
                    _EndpointJsonKeys.X_ORIGIN: 210,
                    _EndpointJsonKeys.Y_ORIGIN: 0,
                    _EndpointJsonKeys.WIDTH: 224,
                    _EndpointJsonKeys.HEIGHT: 224,
                },
            }],
        },
    )
    response = _predictions(embedding_results)
    return json.dumps(response)

  def test_v1_patch_request_success(self):
    self.mock_store_instance.mock_request.post(
        _DEFAULT_ENDPOINT_URL,
        text=self._request_call_back,
        status_code=http.HTTPStatus.OK,
    )
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    mock_embedding_patch_1 = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=endpoint.patch_width(),
        height=endpoint.patch_height(),
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    mock_embedding_patch_2 = dicom_slide.DicomPatch(
        source=self.slide,
        x=210,
        y=0,
        width=endpoint.patch_width(),
        height=endpoint.patch_height(),
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    mock_source_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=1024,
        height=2048,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    slide_embedding_source_1 = patch_embedding_types.SlideEmbeddingSource(
        [
            patch_embedding_types.PatchEmbeddingSource(
                mock_embedding_patch_1, mock_source_patch, '1'
            ),
            patch_embedding_types.PatchEmbeddingSource(
                mock_embedding_patch_1, mock_source_patch, '2'
            ),
        ],
    )
    slide_embedding_source_2 = patch_embedding_types.SlideEmbeddingSource(
        [
            patch_embedding_types.PatchEmbeddingSource(
                mock_embedding_patch_2, mock_source_patch, '3'
            )
        ],
    )
    # test mock embedding request, requesting three embeddings.
    source_list = [slide_embedding_source_1, slide_embedding_source_2]
    prep_list = [
        endpoint.prepare_embedding_request(slide_embedding_source_1),
        endpoint.prepare_embedding_request(slide_embedding_source_2),
    ]
    msg = endpoint.request_embeddings(prep_list)
    results = endpoint.process_response(source_list, msg)
    self.assertLen(results, 3)
    self.assertEqual(results[0].embedding.tolist(), [1.1, 2.1, 3.1])
    self.assertEqual(results[1].embedding.tolist(), [1.1, 2.1, 3.1])
    self.assertEqual(results[2].embedding.tolist(), [4.1, 5.1, 6.1])

  @parameterized.named_parameters([
      dict(
          testcase_name='to_many_prediction_responses',
          resp=_predictions([{}, {}]),
      ),
      dict(
          testcase_name='to_few_prediction_responses',
          resp=_predictions([]),
      ),
      dict(
          testcase_name='study_instance_uid_does_not_match_request',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
          }]),
      ),
      dict(
          testcase_name='series_instance_uid_does_not_match_request',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5',
          }]),
      ),
      dict(
          testcase_name='to_many_patch_embedding_responses',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 10,
                          _EndpointJsonKeys.Y_ORIGIN: 0,
                          _EndpointJsonKeys.WIDTH: 224,
                          _EndpointJsonKeys.HEIGHT: 224,
                      },
                  },
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 10,
                          _EndpointJsonKeys.Y_ORIGIN: 0,
                          _EndpointJsonKeys.WIDTH: 224,
                          _EndpointJsonKeys.HEIGHT: 224,
                      },
                  },
              ],
          }]),
      ),
      dict(
          testcase_name='to_few_patch_embedding_responses',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [],
          }]),
      ),
      dict(
          testcase_name='invalid_patch_x_coordinate',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 99,
                          _EndpointJsonKeys.Y_ORIGIN: 0,
                          _EndpointJsonKeys.WIDTH: 224,
                          _EndpointJsonKeys.HEIGHT: 224,
                      },
                  },
              ],
          }]),
      ),
      dict(
          testcase_name='invalid_patch_y_coordinate',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 10,
                          _EndpointJsonKeys.Y_ORIGIN: 99,
                          _EndpointJsonKeys.WIDTH: 224,
                          _EndpointJsonKeys.HEIGHT: 224,
                      },
                  },
              ],
          }]),
      ),
      dict(
          testcase_name='invalid_patch_width',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 10,
                          _EndpointJsonKeys.Y_ORIGIN: 0,
                          _EndpointJsonKeys.WIDTH: 9,
                          _EndpointJsonKeys.HEIGHT: 224,
                      },
                  },
              ],
          }]),
      ),
      dict(
          testcase_name='invalid_patch_height',
          resp=_predictions([{
              _EndpointJsonKeys.DICOM_STUDY_UID: '1.2.3',
              _EndpointJsonKeys.DICOM_SERIES_UID: '4.5.6',
              'patch_embeddings': [
                  {
                      _EndpointJsonKeys.EMBEDDINGS: [1.1, 2.1, 3.1],
                      _EndpointJsonKeys.PATCH_COORDINATE: {
                          _EndpointJsonKeys.X_ORIGIN: 10,
                          _EndpointJsonKeys.Y_ORIGIN: 0,
                          _EndpointJsonKeys.WIDTH: 224,
                          _EndpointJsonKeys.HEIGHT: 9,
                      },
                  },
              ],
          }]),
      ),
  ])
  def test_v1_patch_request_raises_invalid_response(self, resp):
    self.mock_store_instance.mock_request.post(
        _DEFAULT_ENDPOINT_URL,
        text=json.dumps(resp),
        status_code=http.HTTPStatus.OK,
    )
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    mock_embedding_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=endpoint.patch_width(),
        height=endpoint.patch_height(),
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    mock_source_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=1024,
        height=2048,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [
            patch_embedding_types.PatchEmbeddingSource(
                mock_embedding_patch, mock_source_patch, '1'
            ),
        ],
    )
    # test mock embedding request, requesting three embeddings.
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      source_list = [slide_embedding_source]
      msg = endpoint.request_embeddings(
          [endpoint.prepare_embedding_request(source_list[0])]
      )
      endpoint.process_response(source_list, msg)

  @parameterized.named_parameters([
      dict(
          testcase_name='icc_profile_none',
          norm=patch_embedding_endpoints.IccProfileNormalization.NONE,
          expected=b'',
      ),
      dict(
          testcase_name='icc_profile_srgb',
          norm=patch_embedding_endpoints.IccProfileNormalization.SRGB,
          expected=dicom_slide.get_srgb_icc_profile_bytes(),
      ),
      dict(
          testcase_name='icc_profile_adobergb',
          norm=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
          expected=dicom_slide.get_adobergb_icc_profile_bytes(),
      ),
      dict(
          testcase_name='icc_profile_rommrgb',
          norm=patch_embedding_endpoints.IccProfileNormalization.ROMMRGB,
          expected=dicom_slide.get_rommrgb_icc_profile_bytes(),
      ),
  ])
  def test_get_icc_profile_bytes(self, norm, expected):
    self.assertEqual(
        patch_embedding_endpoints._get_icc_profile_bytes(norm), expected
    )

  def test_get_invalid_icc_profile_bytes_raises(self):
    with self.assertRaises(ez_wsi_errors.InternalError):
      patch_embedding_endpoints._get_icc_profile_bytes(
          typing.cast(patch_embedding_endpoints.IccProfileNormalization, 'abc')
      )

  @parameterized.named_parameters([
      dict(testcase_name='list', test={'a': ['abc', '1234']}, expected=7),
      dict(testcase_name='str', test={'a': 'abc'}, expected=3),
      dict(
          testcase_name='list_str',
          test={'a': 'abc', 'b': ['abc', '1234']},
          expected=10,
      ),
  ])
  def test_get_gcs_image_md_size(self, test, expected):
    self.assertEqual(
        patch_embedding_endpoints._get_gcs_image_md_size(test), expected
    )

  def test_invalid_dict_value_raises(self):
    with self.assertRaises(ez_wsi_errors.InternalError):
      patch_embedding_endpoints._get_gcs_image_md_size(
          typing.cast(Mapping[str, str], {'a': {}})
      )

  def test_gcs_image_json_metadata_sends_patchs_if_image_metadata_large(
      self,
  ) -> None:
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    image_patch = image.get_image_as_patch()
    source_patch = patch_embedding_types.PatchEmbeddingSource(
        image_patch, image_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource([source_patch] * 20)
    old_val = patch_embedding_endpoints._WHOLE_IMAGE_SIZE_SAFTY_MARGIN
    try:
      patch_embedding_endpoints._WHOLE_IMAGE_SIZE_SAFTY_MARGIN = (
          patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES - 1
      )
      result = patch_embedding_endpoints._gcs_image_json_metadata(
          source,
          patch_embedding_endpoints.IccProfileNormalization.NONE,
          None,
          patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES,
      )
    finally:
      patch_embedding_endpoints._WHOLE_IMAGE_SIZE_SAFTY_MARGIN = old_val
    patches = result.get(_EndpointJsonKeys.PATCHES)
    self.assertLen(patches, 20)

  def test_gcs_image_json_metadata_patch_embedding(self) -> None:
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    image_patch = image.get_image_as_patch()
    patch = image_patch.get_patch(0, 0, 10, 10)
    source_patch = patch_embedding_types.PatchEmbeddingSource(
        patch, image_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource([source_patch])
    result = patch_embedding_endpoints._gcs_image_json_metadata(
        source,
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES,
    )
    patches = result.get(_EndpointJsonKeys.PATCHES)
    self.assertLen(patches, 1)
    metadata_patch = gcs_image.GcsPatch.create_from_json(patches[0])
    np.testing.assert_array_equal(
        metadata_patch.image_bytes(), patch.image_bytes()
    )

  def test_gcs_image_json_metadata_patch_embedding_two_patches(self) -> None:
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    image_patch = image.get_image_as_patch()
    patch_1 = image_patch.get_patch(0, 0, 10, 10)
    source_patch1 = patch_embedding_types.PatchEmbeddingSource(
        patch_1, image_patch, '1'
    )
    patch_2 = image_patch.get_patch(10, 10, 10, 10)
    source_patch2 = patch_embedding_types.PatchEmbeddingSource(
        patch_2, image_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource(
        [source_patch1, source_patch2]
    )
    result = patch_embedding_endpoints._gcs_image_json_metadata(
        source,
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES,
    )
    patches = result.get(_EndpointJsonKeys.PATCHES)
    self.assertLen(patches, 2)
    metadata_patch = gcs_image.GcsPatch.create_from_json(patches[0])
    np.testing.assert_array_equal(
        metadata_patch.image_bytes(), patch_1.image_bytes()
    )
    metadata_patch = gcs_image.GcsPatch.create_from_json(patches[1])
    np.testing.assert_array_equal(
        metadata_patch.image_bytes(), patch_2.image_bytes()
    )

  def test_gcs_image_json_metadata_transfers_same_whole_for_both_encodings(
      self,
  ) -> None:
    # Initialize image from file enables origional image to be read.
    # Image will have reference to original image bytes.
    source = dicom_test_utils.test_jpeg_path()
    image_from_path = local_image.LocalImage(source)

    # Create another image, remove the reference to the original image bytes.
    with open(source, 'rb') as f:
      image_from_bytes = local_image.LocalImage(f.read())
    image_from_bytes.clear_source_image_compressed_bytes()

    # Create a request against the image with reference to original image bytes.
    image_patch_1 = image_from_path.get_image_as_patch()
    # request 10 patches to create big request which will trigger
    # returning the whole image.
    result = patch_embedding_endpoints._gcs_image_json_metadata(
        patch_embedding_types.SlideEmbeddingSource([
            patch_embedding_types.PatchEmbeddingSource(
                image_patch_1, image_patch_1, '1'
            )
            for _ in range(10)
        ]),
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES,
    )
    image_1 = result.get(_EndpointJsonKeys.IMAGE)
    # Size in bytes of the returned image as a base64 encoded string.
    self.assertLen(image_1, 84156)

    # Create a request against the image with no ref to original image bytes.
    image_patch_2 = image_from_bytes.get_image_as_patch()
    # request 10 patches to create big request which will trigger
    # returning the whole image.
    result = patch_embedding_endpoints._gcs_image_json_metadata(
        patch_embedding_types.SlideEmbeddingSource([
            patch_embedding_types.PatchEmbeddingSource(
                image_patch_2, image_patch_2, '1'
            )
            for _ in range(10)
        ]),
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES,
    )
    image_2 = result.get(_EndpointJsonKeys.IMAGE)
    # Assert that the image bytes are the same even though the encodings of the
    # images are different (different sizes).
    self.assertNotEqual(
        hashlib.sha256(base64.b64decode(image_1)).hexdigest(),
        hashlib.sha256(base64.b64decode(image_2)).hexdigest(),
    )
    np.testing.assert_array_equal(
        gcs_image.GcsImage.create_from_json(image_1).image_bytes(),
        gcs_image.GcsImage.create_from_json(image_2).image_bytes(),
    )

  def test_gcs_image_json_metadata_transfers_for_large_gcs_does_not_include_md(
      self,
  ) -> None:
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg'),
        os.path.join(temp_dir, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')

      # Create a request against the image with reference to original image
      # bytes.
      image_patch_1 = image.get_image_as_patch()
      source = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
          for _ in range(10)
      ])
      embedding_request = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()._gcs_patch_embedding_request(
          'mock_token', source
      )
    self.assertEqual(
        embedding_request[_EndpointJsonKeys.GCS_IMAGE_URL],
        'gs://test_bucket/test_image.jpg',
    )
    self.assertEqual(
        embedding_request[_EndpointJsonKeys.BEARER_TOKEN], 'mock_token'
    )
    self.assertEmpty(embedding_request[_EndpointJsonKeys.EZ_WSI_STATE])
    self.assertLen(embedding_request[_EndpointJsonKeys.PATCH_COORDINATES], 10)

  @parameterized.named_parameters([
      dict(
          testcase_name='load_image_bytes_and_copy_md',
          copy_image_from_client_to_server=True,
          load_image_bytes_from_gcs=True,
      ),
      dict(
          testcase_name='do_not_load_bytes_and_do_not_copy_md',
          copy_image_from_client_to_server=False,
          load_image_bytes_from_gcs=False,
      ),
      dict(
          testcase_name='load_bytes_and_do_not_copy_md',
          copy_image_from_client_to_server=False,
          load_image_bytes_from_gcs=True,
      ),
      dict(
          testcase_name='do_not_load_bytes_and_copy_md',
          copy_image_from_client_to_server=True,
          load_image_bytes_from_gcs=False,
      ),
  ])
  def test_gcs_image_json_metadata_transfers_for_small_gcs_includes_md(
      self, copy_image_from_client_to_server, load_image_bytes_from_gcs
  ) -> None:
    md_expected = copy_image_from_client_to_server and load_image_bytes_from_gcs
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg'),
        os.path.join(temp_dir, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')

      # Create a request against the image with reference to original image
      # bytes.
      image_patch_1 = image.get_patch(0, 0, 10, 10)
      source = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      # Get image bytes to load image from GCS
      if load_image_bytes_from_gcs:
        image.image_bytes()
      embedding_request = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
          send_gcs_patch_bytes_from_client_to_server=copy_image_from_client_to_server
      )._gcs_patch_embedding_request(
          'mock_token',
          source,
      )
    self.assertEqual(
        embedding_request[_EndpointJsonKeys.GCS_IMAGE_URL],
        'gs://test_bucket/test_image.jpg',
    )
    self.assertEqual(
        embedding_request[_EndpointJsonKeys.BEARER_TOKEN], 'mock_token'
    )
    self.assertEqual(
        bool(embedding_request[_EndpointJsonKeys.EZ_WSI_STATE]),
        md_expected,
    )
    self.assertLen(embedding_request['patch_coordinates'], 1)

  def test_process_response_empty_response_v1(self):
    self.assertEqual(
        patch_embedding_endpoints.V1PatchEmbeddingEndpoint().process_response(
            [], patch_embedding_endpoints._VertexModelResult([])
        ),
        [],
    )

  def test_process_response_empty_response_v2(self):
    self.assertEqual(
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint().process_response(
            [], patch_embedding_endpoints._VertexModelResult([])
        ),
        [],
    )

  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
      return_value='1,2',
  )
  def test_v1_response_returns_bad_json_raises(self, *unused_mocks):
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint().request_embeddings(r)

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
      return_value='1,2',
  )
  def test_v2_response_returns_bad_json_raises(self, *unused_mocks):
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().request_embeddings(r)

  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
  )
  def test_v1_response_returns_error_raises(self, mock_response, _):
    mock_response.return_value = json.dumps(
        {_EndpointJsonKeys.PREDICTIONS: ([1, 2], 'mock_error', _VERSION)}
    )
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint().request_embeddings(r)

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
  )
  def test_v2_response_returns_error_raises(self, mock_response, _):
    mock_response.return_value = json.dumps(
        {patch_embedding_endpoints.EndpointJsonKeys.VERTEXAI_ERROR: 'bad'}
    )
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().request_embeddings(r)

  @parameterized.named_parameters([
      dict(
          testcase_name='patch_coordinates_match_no_dim',
          patch_coordinate=dict(
              x_origin=4,
              y_origin=5,
          ),
          exp=True,
      ),
      dict(
          testcase_name='patch_coordinates_match_width',
          patch_coordinate=dict(
              x_origin=4,
              y_origin=5,
              width=10,
          ),
          exp=True,
      ),
      dict(
          testcase_name='patch_coordinates_match_full_dim',
          patch_coordinate=dict(
              x_origin=4,
              y_origin=5,
              width=10,
              height=12,
          ),
          exp=True,
      ),
      dict(
          testcase_name='patch_coordinates_match_height',
          patch_coordinate=dict(
              x_origin=4,
              y_origin=5,
              height=12,
          ),
          exp=True,
      ),
      dict(
          testcase_name='patch_coordinates_fail_missing_coordinates',
          patch_coordinate=dict(width=10, height=12),
          exp=False,
      ),
      dict(
          testcase_name='empty',
          patch_coordinate=dict(),
          exp=False,
      ),
      dict(
          testcase_name='x_does_not_match',
          patch_coordinate=dict(x_origin=10, y_origin=5),
          exp=False,
      ),
      dict(
          testcase_name='y_does_not_match',
          patch_coordinate=dict(x_origin=4, y_origin=10),
          exp=False,
      ),
      dict(
          testcase_name='width_does_not_match',
          patch_coordinate=dict(x_origin=4, y_origin=5, width=11, height=12),
          exp=False,
      ),
      dict(
          testcase_name='height_does_not_match',
          patch_coordinate=dict(x_origin=4, y_origin=5, width=10, height=11),
          exp=False,
      ),
  ])
  def test_patch_coordinates_match(self, patch_coordinate, exp):
    self.assertEqual(
        patch_embedding_endpoints._test_patch_coordinates_match(
            patch_coordinate, 4, 5, 10, 12
        ),
        exp,
    )

  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
  )
  def test_v1_model_version_does_not_match_expectation_raise(
      self, mock_response, _
  ):
    pred_list = [{'patch_embeddings': [1]}]
    error = None
    ml_version = '1.2.3'
    mock_response.return_value = json.dumps(
        {_EndpointJsonKeys.PREDICTIONS: (pred_list, error, ml_version)}
    )
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Model version 1.2.3 does not match expected version abc',
    ):
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
          expected_model_version='abc'
      ).request_embeddings(r)

  def test_v1_number_of_embeddings_in_request_and_response_not_match_raise(
      self,
  ):
    source = patch_embedding_types.SlideEmbeddingSource([])
    pred_list = [{'patch_embeddings': [1]}]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Number of patches in embedding response does not match request;'
        ' expected: 0; received: 1.',
    ):
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint().process_response(
          [source], patch_embedding_endpoints._VertexModelResult(pred_list)
      )

  def test_v2_number_of_embeddings_in_request_and_response_not_match_raise(
      self,
  ):
    source = patch_embedding_types.SlideEmbeddingSource([])
    pred_list = [{
        'model_version': '1234',
        'result': {'patch_embeddings': {'patch_embeddings': [1]}},
    }]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Number of patches in embedding response does not match request;'
        ' expected: 0; received: 1.',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().process_response(
          [source], patch_embedding_endpoints._VertexModelResult(pred_list)
      )

  def test_prediction_v1_and_embedding_coordinates_do_not_match_raise(self):
    mk_dicom_slide = mock.create_autospec(dicom_slide.DicomSlide, instance=True)
    type(mk_dicom_slide).path = dicom_path.FromString(
        '/projects/prj/locations/loc/datasets/ds/dicomStores/dicomStore/dicomWeb/studies/1.2/series/1.2.3'
    )
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    type(mock_patch).x = mock.PropertyMock(return_value=10)
    type(mock_patch).y = mock.PropertyMock(return_value=10)
    type(mock_patch).source = mk_dicom_slide
    source = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            mock_patch, mock_patch, 'mock_id'
        )
    ])
    pred_list = [{
        _EndpointJsonKeys.DICOM_STUDY_UID: '1.2',
        _EndpointJsonKeys.DICOM_SERIES_UID: '1.2.3',
        _EndpointJsonKeys.PATCH_EMBEDDINGS: [
            {_EndpointJsonKeys.PATCH_COORDINATE: dict(x_origin=10, y_origin=0)}
        ],
    }]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Embedding patch coordinates or dimensions do not match request.',
    ):
      patch_embedding_endpoints.V1PatchEmbeddingEndpoint().process_response(
          [source], patch_embedding_endpoints._VertexModelResult(pred_list)
      )

  def test_prediction_v2_and_embedding_coordinates_do_not_match_raise(self):
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    type(mock_patch).x = mock.PropertyMock(return_value=10)
    type(mock_patch).y = mock.PropertyMock(return_value=10)
    source = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            mock_patch, mock_patch, 'mock_id'
        )
    ])
    pred_list = [{
        _EndpointJsonKeys.MODEL_VERSION: '1234',
        _EndpointJsonKeys.RESULT: {
            _EndpointJsonKeys.PATCH_EMBEDDINGS: [{
                _EndpointJsonKeys.PATCH_COORDINATE: dict(
                    x_origin=10, y_origin=0
                )
            }]
        },
    }]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Embedding patch coordinates or dimensions do not match request.',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().process_response(
          [source], patch_embedding_endpoints._VertexModelResult(pred_list)
      )

  @parameterized.parameters([True, False])
  def test_gen_v2_extensions_require_patches_fully_in_source_image(
      self, require_fully_in_source_image
  ):
    self.assertEqual(
        patch_embedding_endpoints._gen_v2_extensions(
            require_fully_in_source_image=require_fully_in_source_image,
            image_dimensions=None,
            icc_profile=patch_embedding_endpoints.IccProfileNormalization.NONE,
            ez_wsi_state={},
        ),
        {
            'require_patches_fully_in_source_image': str(
                require_fully_in_source_image
            ),
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE',
        },
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='none',
          icc_profile_norm=patch_embedding_endpoints.IccProfileNormalization.NONE,
          expected='NONE',
      ),
      dict(
          testcase_name='adobergb',
          icc_profile_norm=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
          expected='ADOBERGB',
      ),
      dict(
          testcase_name='srgb',
          icc_profile_norm=patch_embedding_endpoints.IccProfileNormalization.SRGB,
          expected='SRGB',
      ),
      dict(
          testcase_name='rommrgb',
          icc_profile_norm=patch_embedding_endpoints.IccProfileNormalization.ROMMRGB,
          expected='ROMMRGB',
      ),
  ])
  def test_gen_v2_extensions_icc_profile(self, icc_profile_norm, expected):
    self.assertEqual(
        patch_embedding_endpoints._gen_v2_extensions(
            require_fully_in_source_image=True,
            image_dimensions=None,
            icc_profile=icc_profile_norm,
            ez_wsi_state={},
        ),
        {
            'require_patches_fully_in_source_image': 'True',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: expected,
        },
    )

  def test_gen_v2_extensions_ez_wsi_state(self):
    self.assertEqual(
        patch_embedding_endpoints._gen_v2_extensions(
            require_fully_in_source_image=False,
            image_dimensions=None,
            icc_profile=patch_embedding_endpoints.IccProfileNormalization.NONE,
            ez_wsi_state={'test': 'state'},
        ),
        {
            'require_patches_fully_in_source_image': 'False',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE',
            _EndpointJsonKeys.EZ_WSI_STATE: {'test': 'state'},
        },
    )

  def test_gen_v2_extensions_image_dimensions(self):
    self.assertEqual(
        patch_embedding_endpoints._gen_v2_extensions(
            require_fully_in_source_image=False,
            image_dimensions=dicom_slide.ImageDimensions(10, 20),
            icc_profile=patch_embedding_endpoints.IccProfileNormalization.NONE,
            ez_wsi_state={},
        ),
        {
            'require_patches_fully_in_source_image': 'False',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE',
            _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                'width_px': 10,
                'height_px': 20,
            },
        },
    )

  def test_gen_v2_extensions_coombined(self):
    self.assertEqual(
        patch_embedding_endpoints._gen_v2_extensions(
            require_fully_in_source_image=True,
            image_dimensions=dicom_slide.ImageDimensions(10, 20),
            icc_profile=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
            ez_wsi_state={'test': 'state'},
        ),
        {
            'require_patches_fully_in_source_image': 'True',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ADOBERGB',
            _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                'width_px': 10,
                'height_px': 20,
            },
            _EndpointJsonKeys.EZ_WSI_STATE: {'test': 'state'},
        },
    )

  def test_v2_dicom_native_level_embedding_request(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            dicom_patch, dicom_patch, 'mock_id'
        )
    ])
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    results = endpoint._get_embedding_request(
        [endpoint.prepare_embedding_request(embedding_inputs)]
    )
    with open(
        dicom_test_utils.testdata_path(
            'v2_dicom_endpoint_native_level_request.json'
        ),
        'rt',
    ) as infile:
      expected = json.load(infile)
    self.assertEqual(json.loads(results), expected)

  def test_v2_dicom_native_level_embedding_request_no_bearer_token(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    self.slide._dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            dicom_patch, dicom_patch, 'mock_id'
        )
    ])
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    results = endpoint._get_embedding_request(
        [endpoint.prepare_embedding_request(embedding_inputs)]
    )
    with open(
        dicom_test_utils.testdata_path(
            'v2_dicom_endpoint_native_level_request_no_bearer_token.json'
        ),
        'rt',
    ) as infile:
      expected = json.load(infile)
    self.assertEqual(json.loads(results), expected)

  def test_v2_dicom_resized_level_embedding_request(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level.resize(dicom_slide.ImageDimensions(500, 500)),
        10,
        10,
        224,
        224,
    )
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            dicom_patch, dicom_patch, 'mock_id'
        )
    ])
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    results = endpoint._get_embedding_request(
        [endpoint.prepare_embedding_request(embedding_inputs)]
    )
    with open(
        dicom_test_utils.testdata_path(
            'v2_dicom_endpoint_resized_level_request.json'
        ),
        'rt',
    ) as infile:
      expected = json.load(infile)
    self.assertEqual(json.loads(results), expected)

  def test_v1_dicom_resized_level_embedding_raises(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level.resize(dicom_slide.ImageDimensions(500, 500)),
        10,
        10,
        224,
        224,
    )
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            dicom_patch, dicom_patch, 'mock_id'
        )
    ])
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'V1 encoder does not support image level resizing.',
    ):
      endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
      endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )

  def test_v1_dicom_no_bearer_token_embedding_raises(self):
    self.slide._dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            dicom_patch, dicom_patch, 'mock_id'
        )
    ])
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'V1 encoder does not support empty bearer tokens.',
    ):
      endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
      endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )

  @parameterized.parameters([True, False])
  def test_v2_gcs_image_embedding_request_no_state(self, include_state):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        os.path.join(temp_dir, 'test_image.png'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage('gs://test_bucket/test_image.png')
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      # image bytes only sent to server if loaded. image bytes not loaded.
      self.assertFalse(image.are_image_bytes_loaded)
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          send_gcs_patch_bytes_from_client_to_server=include_state
      )
      results = endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )
      with open(
          dicom_test_utils.testdata_path(
              'v2_gcs_endpoint_no_state_request.json'
          ),
          'rt',
      ) as infile:
        self.assertEqual(json.loads(results), json.load(infile))

  def test_v2_gcs_image_embedding_request_ez_wsi_state(self):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        os.path.join(temp_dir, 'test_image.png'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage('gs://test_bucket/test_image.png')
      _ = image.image_bytes()
      # image bytes are now loaded.
      self.assertTrue(image.are_image_bytes_loaded)
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          send_gcs_patch_bytes_from_client_to_server=True
      )
      results = endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )
      with open(
          dicom_test_utils.testdata_path(
              'v2_gcs_endpoint_ez_wsi_state_request.json'
          ),
          'rt',
      ) as infile:
        # decodes png in results ez-wsi GCS metadata and replaces with
        # uncompressed bytes. PNG cannot be compared directly due to including
        # time of creation metadata in bytes.
        self.assertEqual(
            _unencoded_result_images(json.loads(results)),
            _unencoded_result_images(json.load(infile)),
        )

  def test_v2_gcs_image_request_ez_wsi_state_not_sent_if_image_is_smaller(self):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg'),
        os.path.join(temp_dir, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
      _ = image.image_bytes()
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          send_gcs_patch_bytes_from_client_to_server=True
      )
      results = endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )
      with open(
          dicom_test_utils.testdata_path(
              'v2_gcs_endpoint_image_bytes_smaller_than_state_request.json'
          ),
          'rt',
      ) as infile:
        self.assertEqual(json.loads(results), json.load(infile))

  def test_v2_resize_gcs_image(self):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        os.path.join(temp_dir, 'test_image.png'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.png',
          image_dimensions=gcs_image.ImageDimensions(1000, 1000),
      )
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      # image bytes only sent to server if loaded. image bytes not loaded.
      self.assertFalse(image.are_image_bytes_loaded)
      endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          send_gcs_patch_bytes_from_client_to_server=False,
          icc_profile_normalization=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
      )
      results = endpoint._get_embedding_request(
          [endpoint.prepare_embedding_request(embedding_inputs)]
      )
      with open(
          dicom_test_utils.testdata_path('v2_resize_gcs_image.json'),
          'rt',
      ) as infile:
        self.assertEqual(json.loads(results), json.load(infile))

  def test_v1_resize_gcs_image_raises(self):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        os.path.join(temp_dir, 'test_image.png'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.png',
          image_dimensions=gcs_image.ImageDimensions(1000, 1000),
      )
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchEmbeddingEndpointError,
          'V1 encoder does not support image image resizing.',
      ):
        endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
            send_gcs_patch_bytes_from_client_to_server=False,
        )
        endpoint._get_embedding_request(
            [endpoint.prepare_embedding_request(embedding_inputs)]
        )

  def test_v1_no_bearer_token_gcs_image_raises(self):
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        os.path.join(temp_dir, 'test_image.png'),
    )
    with gcs_mock.GcsMock({'test_bucket': temp_dir}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.png',
          credential_factory.NoAuthCredentialsFactory(),
      )
      image_patch_1 = image.get_patch(0, 0, 224, 224)
      embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
          patch_embedding_types.PatchEmbeddingSource(
              image_patch_1, image_patch_1, '1'
          )
      ])
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchEmbeddingEndpointError,
          'V1 encoder does not support empty bearer tokens.',
      ):
        endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
            send_gcs_patch_bytes_from_client_to_server=False,
        )
        endpoint._get_embedding_request(
            [endpoint.prepare_embedding_request(embedding_inputs)]
        )

  def test_v2_local_image(self):
    image = local_image.LocalImage(
        dicom_test_utils.testdata_path('low_res_slide_img.png')
    )
    # image bytes loaded at constructor
    self.assertTrue(image.are_image_bytes_loaded)
    image_patch_1 = image.get_patch(0, 0, 224, 224)
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            image_patch_1, image_patch_1, '1'
        )
    ])
    # important.
    # local files sent with no bearer token.
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
        send_gcs_patch_bytes_from_client_to_server=False,
        icc_profile_normalization=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
    )
    results = endpoint._get_embedding_request(
        [endpoint.prepare_embedding_request(embedding_inputs)]
    )
    with open(
        dicom_test_utils.testdata_path('local_file.json'),
        'rt',
    ) as infile:
      self.assertEqual(
          _unencoded_result_images(json.loads(results)),
          _unencoded_result_images(json.load(infile)),
      )

  def test_v2_resize_local_file_with_many_patches(self):
    image = local_image.LocalImage(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        local_image.ImageDimensions(300, 300),
    )
    # image bytes loaded at constructor
    self.assertTrue(image.are_image_bytes_loaded)
    image_patch_1 = image.get_patch(10, 10, 224, 224)
    # size of encoding each bytes for each patch > then just sending image.
    # expect single encode image will be sent.
    embedding_inputs = patch_embedding_types.SlideEmbeddingSource(
        [
            patch_embedding_types.PatchEmbeddingSource(
                image_patch_1, image_patch_1, '1'
            )
        ]
        * 100
    )
    # important.
    # local files sent with no bearer token.
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
        send_gcs_patch_bytes_from_client_to_server=False,
        icc_profile_normalization=patch_embedding_endpoints.IccProfileNormalization.ADOBERGB,
    )
    results = endpoint._get_embedding_request(
        [endpoint.prepare_embedding_request(embedding_inputs)]
    )
    with open(
        dicom_test_utils.testdata_path(
            'resize_local_file_with_many_patches.json'
        ),
        'rt',
    ) as infile:
      self.assertEqual(
          _unencoded_result_images(json.loads(results)),
          _unencoded_result_images(json.load(infile)),
      )

  def test_get_auth_headers_and_bearer_token(self):
    patch = self.slide.get_patch(self.slide.native_level, 10, 10, 224, 224)
    slide_embedding_source_1 = patch_embedding_types.SlideEmbeddingSource(
        [patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')]
    )
    self.assertEqual(
        slide_embedding_source_1.get_bearer_token(), 'MOCK_BEARER_TOKEN'
    )

  def test_get_auth_headers_and_bearer_local_image_no_auth(self):
    image = local_image.LocalImage(
        dicom_test_utils.testdata_path('low_res_slide_img.png'),
        local_image.ImageDimensions(300, 300),
    )
    patch = image.get_patch(10, 10, 224, 224)
    slide_embedding_source_1 = patch_embedding_types.SlideEmbeddingSource(
        [patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')]
    )
    self.assertEqual(slide_embedding_source_1.get_bearer_token(), '')

  def test_validate_accessing_embedding_request_after_final_raises(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    self.assertNotEmpty(request.embedding_request)
    request.finalize()
    with self.assertRaisesRegex(
        ez_wsi_errors.InternalError, 'Request has been finalized.'
    ):
      _ = request.embedding_request

  def test_validate_finalize_generates_json(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    self.assertIs(request.slide_embedding_source, slide_embedding_source)
    request.finalize()
    self.assertNotEmpty(request.json)

  def test_split_on_patch_coord_returns_unsplit_result_if_less_than_endpoint_max(
      self,
  ):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    result = request._split_on_patch_coordinates_only(
        endpoint, request.json_size_in_bytes - 1
    )
    self.assertEqual(result, (None, slide_embedding_source))

  def test_split_on_patch_coord_returns_unsplit_result_if_no_coord(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    del request._embedding_request['patch_coordinates']  # pytype: disable=unsupported-operands
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      result = request._split_on_patch_coordinates_only(
          endpoint, request.json_size_in_bytes - 1
      )
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val
    self.assertEqual(result, (None, slide_embedding_source))

  @mock.patch.object(
      gcs_image,
      '_gcs_image_json_metadata',
      autospec=True,
      return_value='*',
  )
  def test_split_on_patch_coord_returns_split_result(self, _):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      result = request._split_on_patch_coordinates_only(
          endpoint, request.json_size_in_bytes - 1
      )
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val
    self.assertEqual(
        result[0].json,  # pytype: disable=attribute-error
        '{"image_file_uri": "gs:///placeholder.png", "extensions":'
        ' {"require_patches_fully_in_source_image": "True",'
        ' "transform_imaging_to_icc_profile": "NONE", "ez_wsi_state":'
        ' {"source_image_width_px": 454, "source_image_height_px": 156,'
        ' "icc_profile_metadata_normalization": "NONE", "patches": ["*",'
        ' "*"]}}, "patch_coordinates": [{"x_origin": 0, "y_origin": 0, "width":'
        ' 10, "height": 10}]}',
    )
    self.assertEqual(result[0].slide_embedding_source.patches, [source])  # pytype: disable=attribute-error
    self.assertEqual(result[1].patches, [source])

  def test_split_on_patch_returns_unsplit_result_if_no_coord(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    del request._embedding_request['patch_coordinates']  # pytype: disable=unsupported-operands
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      result = request.split(endpoint, request.json_size_in_bytes - 1)
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val
    self.assertEqual(result, (None, slide_embedding_source))

  def test_split_on_patch_returns_coord_split_if_no_state(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    del request._embedding_request[_EndpointJsonKeys.EXTENSIONS][
        _EndpointJsonKeys.EZ_WSI_STATE
    ]  # pytype: disable=unsupported-operands
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      result = request.split(endpoint, request.json_size_in_bytes - 1)
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val
    self.assertEqual(
        result[0].json,  # pytype: disable=attribute-error
        '{"image_file_uri": "gs:///placeholder.png", "extensions":'
        ' {"require_patches_fully_in_source_image": "True",'
        ' "transform_imaging_to_icc_profile": "NONE"}, "patch_coordinates":'
        ' [{"x_origin": 0, "y_origin": 0, "width": 10, "height": 10}]}',
    )
    self.assertEqual(result[0].slide_embedding_source.patches, [source])  # pytype: disable=attribute-error
    self.assertEqual(result[1].patches, [source])

  def test_split_on_patch_raises_if_coord_and_patch_metadata_dont_match(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    del request._embedding_request['patch_coordinates'][-1]
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.InternalError,
          'Patch state and coordinate counts do not match.',
      ):
        request.split(endpoint, request.json_size_in_bytes - 1)
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val

  def test_split_on_patch_raises_if_unexpected_object(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    patch = image.get_patch(0, 0, 10, 10)
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source, source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    request = endpoint.prepare_embedding_request(slide_embedding_source)
    del request._embedding_request[_EndpointJsonKeys.IMAGE_FILE_URI]  # pytype: disable=unsupported-operands
    old_val = patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES
    try:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = (
          request.json_size_in_bytes - 1
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.InternalError, 'unidentified JSON'
      ):
        request.split(endpoint, request.json_size_in_bytes - 1)
    finally:
      patch_embedding_endpoints._MAX_VERTEX_AI_V1_REQUEST_SIZE_BYTES = old_val

  def test_prepare_embedding_request_raises_if_source_not_known_patch(self):
    patch = typing.cast(dicom_slide.DicomPatch, 'A')
    source = patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    with self.assertRaisesRegex(
        ez_wsi_errors.InternalError,
        'Patch is not a dicom_slide.DicomPatch or gcs_image.GcsPatch.',
    ):
      endpoint.prepare_embedding_request(slide_embedding_source)

  def test_prepare_embedding_request_raises_if_patches_not_all_same_type(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    image_patch = image.get_patch(0, 0, 10, 10)
    image_source = patch_embedding_types.PatchEmbeddingSource(
        image_patch, image_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source, image_source]
    )
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    with self.assertRaisesRegex(
        ez_wsi_errors.InternalError,
        'Patch in request are not all the same type.',
    ):
      endpoint.prepare_embedding_request(slide_embedding_source)

  def test_v1_endpoint_url_property(self):
    self.assertEqual(
        patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
            project_id='test_project'
        ).end_point_url,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/test_project/locations/us-central1/endpoints/160:predict',
    )

  def test_v2_endpoint_url_property(self):
    self.assertEqual(
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
            project_id='test_project'
        ).end_point_url,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/test_project/locations/us-central1/endpoints/162:predict',
    )

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
      return_value='{}',
  )
  def test_v2_endpoint_bad_cannot_parse_prediction_response_raises(
      self, *unused_mocks
  ):
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Endpoint did not return a valid JSON response.',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().request_embeddings(r)

  @parameterized.named_parameters(
      dict(
          testcase_name='error', error={_EndpointJsonKeys.ERROR: 'test_error'}
      ),
      dict(
          testcase_name='empty',
          error={
              _EndpointJsonKeys.ERROR: {
                  _EndpointJsonKeys.ERROR_CODE: 'test_error'
              }
          },
      ),
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autospec=True,
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_request_embeddings',
      autospec=True,
  )
  def test_v2_endpoint_returns_error_raises(
      self, mock_response, unused_mocks, error
  ):
    mock_response.return_value = json.dumps(error)
    r = [
        mock.create_autospec(
            patch_embedding_endpoints.AbstractPreparedEmbeddingRequest[
                patch_embedding_endpoints._VertexModelResult
            ],
            instance=True,
        )
    ]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Endpoint error; Error code: test_error',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().request_embeddings(r)

  def test_v2_endpoint_instance_ml_model_does_not_match_expectation_raises(
      self,
  ):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource([dicom_source])
    error = [{'model_version': 'a', 'results': {}}]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Model version a does not match expected version abc',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
          expected_model_version='abc'
      ).process_response(
          [source], patch_embedding_endpoints._VertexModelResult(error)
      )

  def test_v2_endpoint_instance_ml_model_returns_unexpected_response(
      self,
  ):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource([dicom_source])
    error = [{'results': {}}]
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Endpoint returned an unexpected response.',
    ):
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint().process_response(
          [source], patch_embedding_endpoints._VertexModelResult(error)
      )

  @parameterized.parameters(range(3))
  def test_v2_endpoint_instance_returns_endpoint_error(self, patch_count):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source] * patch_count
    )
    error = [{
        _EndpointJsonKeys.MODEL_VERSION: '1.2.3',
        _EndpointJsonKeys.ERROR: {_EndpointJsonKeys.ERROR_CODE: 'test_error'},
    }]
    results = (
        patch_embedding_endpoints.V2PatchEmbeddingEndpoint().process_response(
            [source], patch_embedding_endpoints._VertexModelResult(error)
        )
    )
    self.assertLen(results, patch_count)
    for result in results:
      with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
        _ = result.embedding
      self.assertEqual(result.error.error_code, 'test_error')  # pytype: disable=attribute-error

  @parameterized.named_parameters([
      dict(
          testcase_name='error_code_only',
          description='',
          expected='Error code: foo',
      ),
      dict(
          testcase_name='error_code_and_description',
          description='bar',
          expected='Error code: foo; bar',
      ),
  ])
  def test_format_error_message(self, description, expected):
    self.assertEqual(
        patch_embedding_endpoints._format_error_message('foo', description),
        expected,
    )

  def test_local_embedding_endpoint_local_image(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    li = local_image.LocalImage(
        dicom_test_utils.test_jpeg_path(), local_image.ImageDimensions(224, 224)
    )
    results = list(
        patch_embedding.generate_patch_embeddings(
            endpoint, [li.get_patch(0, 0, 224, 224)] * 1000
        )
    )
    self.assertLen(results, 1000)
    for result in results:
      self.assertEqual(result.embedding.shape, (3,))
      self.assertEqual(
          result.embedding.tolist(),
          [0.8814465403556824, 0.8670073747634888, 0.8791013956069946],
      )

  def test_local_embedding_endpoint_dicom_image(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    patch = self.slide.get_patch(self.slide.native_level, 10, 10, 224, 224)
    results = list(
        patch_embedding.generate_patch_embeddings(endpoint, [patch] * 1000)
    )
    self.assertLen(results, 1000)
    for result in results:
      self.assertEqual(result.embedding.shape, (3,))
      self.assertEqual(
          result.embedding.tolist(),
          [0.7646538615226746, 0.6989836692810059, 0.8194547295570374],
      )

  def test_empty_local_embedding_endpoint_prediction(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    self.assertEmpty(
        list(patch_embedding.generate_patch_embeddings(endpoint, []))
    )

  def test_empty_local_embedding_endpoint_set_get_state(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    state = endpoint.__getstate__()
    th_pool = endpoint._thread_pool
    icc_profile_cache = endpoint._icc_profile_cache
    cache_lock = endpoint._icc_profile_cache_lock
    self.assertIsNotNone(th_pool)
    self.assertIsNotNone(icc_profile_cache)
    self.assertIsNotNone(cache_lock)
    self.assertNotIn('_thread_pool', state)
    self.assertNotIn('_icc_profile_cache', state)
    self.assertNotIn('_icc_profile_cache_lock', state)
    endpoint.__setstate__(state)
    self.assertIsInstance(
        endpoint._thread_pool, concurrent.futures.ThreadPoolExecutor
    )
    self.assertIsInstance(endpoint._icc_profile_cache, cachetools.LRUCache)
    self.assertIsNotNone(endpoint._icc_profile_cache_lock)
    self.assertIsNot(endpoint._thread_pool, th_pool)
    self.assertIsNot(endpoint._icc_profile_cache, icc_profile_cache)
    self.assertIsNot(endpoint._icc_profile_cache_lock, cache_lock)

  def test_local_endpoint_request_empty_embeddings(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    self.assertEqual(endpoint.request_embeddings([]).shape, tuple())

  def test_local_endpoint_process_request_empty_success(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    response = endpoint.process_response(
        [],
        np.zeros(
            (),
        ),
    )
    self.assertEqual(response, [])

  def test_local_endpoint_process_request_returned_result_size_and_request_not_match(
      self,
  ):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    with self.assertRaisesRegex(
        ez_wsi_errors.PatchEmbeddingEndpointError,
        'Number of patches in embedding response does not match request.',
    ):
      endpoint.process_response(
          [],
          np.zeros(
              (2, 10),
          ),
      )

  def test_prepared_local_embedding_request_split_none(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source]
    )
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      request._slide_embedding_source = None
      with self.assertRaisesRegex(ValueError, 'Slide embedding source is None'):
        request.split(endpoint, 1000)

  def test_prepared_local_embedding_request(self):
    endpoint = patch_embedding_endpoints.LocalEndpoint(_mock_model)
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, 10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source]
    )
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      self.assertEqual(
          request.split(endpoint, 1000), (None, slide_embedding_source)
      )

  def test_load_patch_empty_prepared_request_bytes(self):
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource([])
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      self.assertEqual(request._load_patch_bytes(), [])

  def test_load_patch_invalid_prepared_request_object(self):
    bad_patch_data = typing.cast(dicom_slide.DicomPatch, 'A')
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource([
        patch_embedding_types.PatchEmbeddingSource(
            bad_patch_data, bad_patch_data, '1'
        )
    ])
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      with self.assertRaisesRegex(ValueError, 'Unexpected object'):
        request._load_patch_bytes()

  def test_load_dicom_patch_does_not_overlap_source_raises(self):
    dicom_patch = self.slide.get_patch(
        self.slide.native_level, -10, 10, 224, 224
    )
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        dicom_patch, dicom_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source]
    )
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchOutsideOfImageDimensionsError,
          'A portion of the patch does not overlap the image.',
      ):
        request._load_patch_bytes()

  def test_load_gcs_patch_does_not_overlap_source_raises(self):
    image = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    image_patch = image.get_patch(-10, 0, 10, 10)
    image_source = patch_embedding_types.PatchEmbeddingSource(
        image_patch, image_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [image_source]
    )
    icc_profile_bytes = b''
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      with self.assertRaisesRegex(
          ez_wsi_errors.PatchOutsideOfImageDimensionsError,
          'A portion of the patch does not overlap the image.',
      ):
        request._load_patch_bytes()

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_load_dicom_patch_with_slide_microscopy_image_icc_profile(
      self, sop_class_uid
  ):
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    test_instance.Columns = 512
    test_instance.Rows = 512
    test_instance.PixelData = np.zeros((512, 512, 1), dtype=np.uint8).tobytes()
    self.mock_store_instance.add_instance(test_instance)
    image = dicom_slide.DicomMicroscopeImage(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        path=dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
        ),
    )
    level = next(image.levels)
    image_patch = image.get_patch(level, 0, 0, 224, 224)
    dicom_source = patch_embedding_types.PatchEmbeddingSource(
        image_patch, image_patch, '1'
    )
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [dicom_source]
    )
    icc_profile_bytes = dicom_slide.get_srgb_icc_profile_bytes()
    cache = cachetools.LRUCache(10)
    cache_lock = threading.Lock()
    require_fully_in_source_image = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as thread_pool:
      request = patch_embedding_endpoints.PreparedLocalEmbeddingRequest(
          slide_embedding_source,
          thread_pool,
          icc_profile_bytes,
          cache,
          cache_lock,
          require_fully_in_source_image,
      )
      result = request._load_patch_bytes()
    self.assertLen(result, 1)
    np.testing.assert_array_equal(
        result[0], np.zeros((224, 224, 1), dtype=np.uint8)
    )

  def test_get_gcs_whole_image_md_returns_empty_if_image_est_size_to_large(
      self,
  ):
    image = local_image.LocalImage(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg')
    )
    metadata, size = patch_embedding_endpoints._get_gcs_whole_image_metadata(
        image, patch_embedding_endpoints.IccProfileNormalization.NONE, None, 100
    )
    self.assertEqual(metadata, {})
    self.assertEqual(size, 11228)

  def test_get_gcs_whole_image_md_returns_metadatae_if_smaller_than_max(
      self,
  ):
    image = local_image.LocalImage(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg')
    )
    metadata, size = patch_embedding_endpoints._get_gcs_whole_image_metadata(
        image,
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        11229,
    )
    self.assertNotEmpty(metadata)
    self.assertEqual(size, 11228)

  def test_get_gcs_whole_image_md_returns_empty_if_raw_bytes_est_to_large(self):
    with PIL.Image.open(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg')
    ) as raw_bytes:
      image = gcs_image.GcsImage(np.asarray(raw_bytes))
    metadata, size = patch_embedding_endpoints._get_gcs_whole_image_metadata(
        image, patch_embedding_endpoints.IccProfileNormalization.NONE, None, 100
    )
    self.assertEqual(metadata, {})
    self.assertEqual(size, 131076)

  def test_get_gcs_whole_image_md_returns_metadata_if_raw_bytes_less_than_max(
      self,
  ):
    with PIL.Image.open(
        dicom_test_utils.testdata_path('dcm_frame_1.jpg')
    ) as raw_bytes:
      image = gcs_image.GcsImage(np.asarray(raw_bytes))
    metadata, size = patch_embedding_endpoints._get_gcs_whole_image_metadata(
        image,
        patch_embedding_endpoints.IccProfileNormalization.NONE,
        None,
        393221,
    )
    self.assertNotEmpty(metadata)
    self.assertLess(size, 131076)

  def test_normalized_patch_channels_nop_if_expected(self):
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    self.assertIs(
        patch_embedding_endpoints.normalized_patch_channels(10, 10, patch),
        patch,
    )

  def test_normalized_patch_channels_1(self):
    patch = np.zeros((10, 10, 1), dtype=np.uint8)
    for i in range(100):
      patch[int(i / 10), int(i % 10), 0] = i
    norm_patch = patch_embedding_endpoints.normalized_patch_channels(
        10, 10, patch
    )
    self.assertIsNot(
        norm_patch,
        patch,
    )
    self.assertEqual(norm_patch.shape, (10, 10, 3))
    for i in range(3):
      np.testing.assert_array_equal(norm_patch[..., i], patch[..., 0])

  def test_normalized_patch_channels_0(self):
    patch = np.zeros((10, 10), dtype=np.uint8)
    for i in range(100):
      patch[int(i / 10), int(i % 10)] = i
    norm_patch = patch_embedding_endpoints.normalized_patch_channels(
        10, 10, patch
    )
    self.assertIsNot(
        norm_patch,
        patch,
    )
    self.assertEqual(norm_patch.shape, (10, 10, 3))
    for i in range(3):
      np.testing.assert_array_equal(norm_patch[..., i], patch)

  def test_normalized_patch_channels_4(self):
    patch = np.zeros((10, 10, 4), dtype=np.uint8)
    for i in range(100):
      patch[int(i / 10), int(i % 10), :] = i
    norm_patch = patch_embedding_endpoints.normalized_patch_channels(
        10, 10, patch
    )
    self.assertIsNot(
        norm_patch,
        patch,
    )
    self.assertEqual(norm_patch.shape, (10, 10, 3))
    for i in range(3):
      np.testing.assert_array_equal(norm_patch[..., i], patch[..., 0])

  @parameterized.named_parameters([
      dict(testcase_name='different_width', width=11, height=10),
      dict(testcase_name='different_height', width=10, height=11),
  ])
  def test_normalized_patch_channels_raises_if_dim_different(
      self, height, width
  ):
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingDimensionError):
      patch_embedding_endpoints.normalized_patch_channels(width, height, patch)

  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      '_get_embedding_request',
      autpspec=True,
      return_value='"MockJSON"',
  )
  @mock.patch.object(
      patch_embedding_endpoints.V2PatchEmbeddingEndpoint,
      'vertex_endpoint_authentication_header',
      autpspec=True,
  )
  def test_request_embeddings_retry(self, mock_get_vertex_auth, _):
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    url = endpoint.end_point_url
    with requests_mock.Mocker() as mocker:
      mocker.register_uri(
          'POST',
          url,
          [
              {
                  'status_code': http.client.UNAUTHORIZED,
                  'text': 'bad',
              },
              {
                  'status_code': http.client.TOO_MANY_REQUESTS,
                  'text': 'bad',
              },
              {
                  'status_code': http.client.OK,
                  'text': '{"predictions": [{"results": "Good"}]}',
              },
          ],
      )
      results = endpoint.request_embeddings([
          mock.create_autospec(
              patch_embedding_endpoints.PreparedVertexEmbeddingRequest,
              instance=True,
          )
      ])
    self.assertEqual(results.instances, [{'results': 'Good'}])
    self.assertEqual(mock_get_vertex_auth.call_count, 3)

  @mock.patch.object(
      requests,
      'post',
      autospec=True,
  )
  @mock.patch.object(credential_factory, 'refresh_credentials', autospec=True)
  def test_authentication_retry_v1(
      self, refresh_credentials_mock, mock_request_embeddings
  ):
    mock_request_embeddings.side_effect = [
        _mock_request_response(
            json.dumps({
                _EndpointJsonKeys.PREDICTIONS: (
                    [],
                    patch_embedding_endpoints.EndpointJsonKeys.INVALID_CREDENTIALS,
                    'MOCK_MODEL_VERSION',
                )
            })
        ),
        _mock_request_response(
            json.dumps({
                _EndpointJsonKeys.PREDICTIONS: (
                    [],
                    patch_embedding_endpoints.EndpointJsonKeys.INVALID_CREDENTIALS,
                    'MOCK_MODEL_VERSION',
                )
            })
        ),
        _mock_request_response(
            json.dumps({
                _EndpointJsonKeys.PREDICTIONS: (
                    [{'MOCK_RESULT': 'MOCK'}],
                    None,
                    'MOCK_MODEL_VERSION',
                )
            })
        ),
    ]
    refresh_credentials_mock.side_effect = [
        _credential_mock('TOKEN_1'),
        _credential_mock('TOKEN_2'),
        _credential_mock('TOKEN_3'),
        _credential_mock('TOKEN_4'),
        _credential_mock('TOKEN_5'),
        _credential_mock('TOKEN_6'),
    ]
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    patch = self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')]
    )
    prepared_request = endpoint.prepare_embedding_request(
        slide_embedding_source
    )
    result = endpoint.request_embeddings([prepared_request])
    # test request resolved.
    self.assertEqual(result.instances, [{'MOCK_RESULT': 'MOCK'}])
    # number of calls to refresh credentials.
    self.assertEqual(refresh_credentials_mock.call_count, 6)
    self.assertEqual(mock_request_embeddings.call_count, 3)

  @mock.patch.object(
      requests,
      'post',
      autospec=True,
  )
  @mock.patch.object(credential_factory, 'refresh_credentials', autospec=True)
  def test_authentication_retry_v2(
      self, refresh_credentials_mock, mock_request_embeddings
  ):
    mock_request_embeddings.side_effect = [
        _mock_request_response(
            json.dumps({
                patch_embedding_endpoints.EndpointJsonKeys.MODEL_VERSION: (
                    'MOCK_MODEL_VERSION'
                ),
                patch_embedding_endpoints.EndpointJsonKeys.ERROR: (
                    patch_embedding_endpoints.EndpointJsonKeys.INVALID_CREDENTIALS
                ),
            })
        ),
        _mock_request_response(
            json.dumps({
                patch_embedding_endpoints.EndpointJsonKeys.MODEL_VERSION: (
                    'MOCK_MODEL_VERSION'
                ),
                patch_embedding_endpoints.EndpointJsonKeys.PREDICTIONS: [{
                    patch_embedding_endpoints.EndpointJsonKeys.ERROR: {
                        patch_embedding_endpoints.EndpointJsonKeys.ERROR_CODE: (
                            patch_embedding_endpoints.EndpointJsonKeys.INVALID_CREDENTIALS
                        )
                    },
                }],
            })
        ),
        _mock_request_response(
            json.dumps({
                patch_embedding_endpoints.EndpointJsonKeys.MODEL_VERSION: (
                    'MOCK_MODEL_VERSION'
                ),
                patch_embedding_endpoints.EndpointJsonKeys.PREDICTIONS: [{
                    patch_embedding_endpoints.EndpointJsonKeys.RESULT: {
                        patch_embedding_endpoints.EndpointJsonKeys.PATCH_EMBEDDINGS: [
                            'MockPatchEmbedding'
                        ]
                    }
                }],
            })
        ),
    ]
    refresh_credentials_mock.side_effect = [
        _credential_mock('TOKEN_1'),
        _credential_mock('TOKEN_2'),
        _credential_mock('TOKEN_3'),
        _credential_mock('TOKEN_4'),
        _credential_mock('TOKEN_5'),
        _credential_mock('TOKEN_6'),
    ]
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint()
    patch = self.slide.get_patch(self.slide.native_level, 0, 0, 224, 224)
    slide_embedding_source = patch_embedding_types.SlideEmbeddingSource(
        [patch_embedding_types.PatchEmbeddingSource(patch, patch, '1')]
    )
    prepared_request = endpoint.prepare_embedding_request(
        slide_embedding_source
    )
    result = endpoint.request_embeddings([prepared_request])
    # test request resolved.
    self.assertEqual(
        result.instances,
        [{'result': {'patch_embeddings': ['MockPatchEmbedding']}}],
    )
    # number of calls to refresh credentials.
    self.assertEqual(refresh_credentials_mock.call_count, 6)
    self.assertEqual(mock_request_embeddings.call_count, 3)


if __name__ == '__main__':
  absltest.main()
