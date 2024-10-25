# Copyright 2023 Google LLC`
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
"""Tests for dicom web interface."""

import http.client
import io
import json
from typing import Any, List, Mapping, MutableMapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_json
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import google.auth
import pydicom
import requests
import requests_mock

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock

_URI = 'http://healthcareapi.com/test'

_AUTH = (mock.create_autospec(google.auth.credentials.Credentials), None)


def _create_mock_instance_metadata() -> MutableMapping[str, Any]:
  instance_metadata = {}
  dicom_json.Insert(instance_metadata, tags.STUDY_INSTANCE_UID, 1)
  dicom_json.Insert(instance_metadata, tags.SERIES_INSTANCE_UID, 2)
  dicom_json.Insert(instance_metadata, tags.SOP_INSTANCE_UID, 3)
  return instance_metadata


def _create_test_dwi() -> dicom_web_interface.DicomWebInterface:
  dwi = dicom_web_interface.DicomWebInterface(
      credential_factory.CredentialFactory()
  )
  return dwi


class DicomWebInterfaceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._dicom_store_path = dicom_path.FromString(
        dicom_test_utils.TEST_STORE_PATH
    )
    # Mock the first set of DICOM paths:
    # studies/{_TEST_STUDY_1}
    self._study_path_1 = dicom_path.FromPath(
        self._dicom_store_path, study_uid=dicom_test_utils.TEST_STUDY_UID_1
    )
    # studies/{_TEST_STUDY_1}/series/{_TEST_SERIES_1}
    self._series_path_1 = dicom_path.FromPath(
        self._study_path_1, series_uid=dicom_test_utils.TEST_SERIES_UID_1
    )
    # studies/{_TEST_STUDY_1}/series/{_TEST_SERIES_1}/instances/{_TEST_INSTANCE_1}
    self._instance_path_1 = dicom_path.FromPath(
        self._series_path_1, instance_uid=dicom_test_utils.TEST_INSTANCE_UID_1
    )

    # Mock the second set of dicom paths:
    # studies/{_TEST_STUDY_2}
    self._study_path_2 = dicom_path.FromPath(
        self._dicom_store_path, study_uid=dicom_test_utils.TEST_STUDY_UID_2
    )
    # studies/{_TEST_STUDY_2}/series/{_TEST_SERIES_2}
    self._series_path_2 = dicom_path.FromPath(
        self._study_path_2, series_uid=dicom_test_utils.TEST_SERIES_UID_2
    )
    # studies/{_TEST_STUDY_2}/series/{_TEST_SERIES_2}/instances/{_TEST_INSTANCE_2}
    self._instance_path_2 = dicom_path.FromPath(
        self._series_path_2, instance_uid=dicom_test_utils.TEST_INSTANCE_UID_2
    )

  def _create_test_dwi_wadors(
      self, all_studies_json: bytes
  ) -> dicom_web_interface.DicomWebInterface:
    self.enter_context(
        mock.patch.object(
            dicom_web_interface,
            '_wado_rs',
            autospec=True,
            return_value=all_studies_json,
        )
    )
    return _create_test_dwi()

  def _create_test_dwi_qidors(
      self,
      all_studies_json: List[Mapping[str, Any]],
  ) -> dicom_web_interface.DicomWebInterface:
    self.enter_context(
        mock.patch.object(
            dicom_web_interface,
            '_qido_rs',
            autospec=True,
            return_value=all_studies_json,
        )
    )
    return _create_test_dwi()

  def test_dicom_credential_factory_property(self):
    cf = credential_factory.CredentialFactory()
    dwi = dicom_web_interface.DicomWebInterface(cf)
    self.assertIs(dwi.credential_factory, cf)

  def test_dicom_object_constructor_with_invalid_input_raise_error(self):
    with self.assertRaises(ez_wsi_errors.DicomPathError):
      dicom_web_interface.DicomObject(self._dicom_store_path, {}, '')

  def test_get_studies_with_invalid_input_raise_error(self):
    dwi = _create_test_dwi()
    with self.subTest(name='Study'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._study_path_1)
    with self.subTest(name='Series'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._series_path_1)
    with self.subTest(name='Instance'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._instance_path_1)

  @requests_mock.Mocker()
  def testInvokeHttpRequest(self, mock_request):
    body = 'body'
    mock_request.register_uri('GET', _URI, text=body)
    resp = dicom_web_interface._invoke_http_request(
        'mock',
        mock.create_autospec(
            google.auth.credentials.Credentials, instance=True
        ),
        _URI,
        {},
        3600,
    )
    self.assertEqual(resp.status_code, http.client.OK)
    self.assertEqual(resp.text, body)

  @parameterized.parameters(
      http.client.TOO_MANY_REQUESTS,
      http.client.REQUEST_TIMEOUT,
      http.client.SERVICE_UNAVAILABLE,
      http.client.GATEWAY_TIMEOUT,
  )
  @requests_mock.Mocker()
  def testInvokeHttpRequestWithRetriedErrors(self, error_code, mock_request):
    body = 'body'
    mock_request.register_uri(
        'GET',
        _URI,
        [
            {
                'status_code': error_code,
                'text': 'mock error',
            },
            {'status_code': http.client.OK, 'text': body},
        ],
    )
    resp = dicom_web_interface._invoke_http_request(
        'mock',
        mock.create_autospec(
            google.auth.credentials.Credentials, instance=True
        ),
        _URI,
        {},
        3600,
    )
    self.assertEqual(resp.status_code, http.client.OK)
    self.assertEqual(resp.text, body)

  @requests_mock.Mocker()
  def testQidoSuccess(self, mock_request):
    mock_instance_metadata = _create_mock_instance_metadata()
    mock_request.register_uri(
        'GET',
        _URI,
        [
            {
                'status_code': http.client.OK,
                'text': json.dumps([mock_instance_metadata]),
            },
            {'status_code': http.client.NO_CONTENT, 'text': ''},
        ],
    )
    resp = dicom_web_interface._qido_rs(
        mock.create_autospec(
            google.auth.credentials.Credentials, instance=True
        ),
        _URI,
    )
    self.assertEqual(resp, [mock_instance_metadata])

    resp = dicom_web_interface._qido_rs(
        mock.create_autospec(
            google.auth.credentials.Credentials, instance=True
        ),
        _URI,
    )
    self.assertEqual(resp, [])

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_studies_with_valid_input(self, unused_mock_credientals):
    all_studies_json = [
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_1],
                'vr': 'UI',
            }
        },
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_2],
                'vr': 'UI',
            }
        },
    ]
    dwi = self._create_test_dwi_qidors(all_studies_json)
    studies = dwi.get_studies(self._dicom_store_path)
    self.assertCountEqual(
        studies,
        [
            dicom_web_interface.DicomObject(
                self._study_path_1, all_studies_json[0], ''
            ),
            dicom_web_interface.DicomObject(
                self._study_path_2, all_studies_json[1], ''
            ),
        ],
        (
            'All dicom objects returned should have a type of STUDY, with '
            'corresponding dicom path and original dicom tags attached.'
        ),
    )

  def test_get_series_with_invalid_input_raise_error(self):
    dwi = _create_test_dwi()
    invalid_path = [
        self._series_path_1,
        self._instance_path_1,
    ]
    for path in invalid_path:
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_series(path)

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_series_with_store_level_path(self, unused_mock_credientals):
    all_series_json = [
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_1],
                'vr': 'UI',
            },
            '0020000E': {
                'Value': [dicom_test_utils.TEST_SERIES_UID_1],
                'vr': 'UI',
            },
        },
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_2],
                'vr': 'UI',
            },
            '0020000E': {
                'Value': [dicom_test_utils.TEST_SERIES_UID_2],
                'vr': 'UI',
            },
        },
    ]
    dwi = self._create_test_dwi_qidors(all_series_json)
    series = dwi.get_series(self._dicom_store_path)
    self.assertCountEqual(
        series,
        [
            dicom_web_interface.DicomObject(
                self._series_path_1, all_series_json[0], ''
            ),
            dicom_web_interface.DicomObject(
                self._series_path_2, all_series_json[1], ''
            ),
        ],
        (
            'All DICOM objects returned should have a type of SERIES, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_series_with_study_level_path(self, unused_mock_credientals):
    all_series_json = [{
        '0020000E': {
            'Value': [dicom_test_utils.TEST_SERIES_UID_1],
            'vr': 'UI',
        }
    }]
    dwi = self._create_test_dwi_qidors(all_series_json)
    instances = dwi.get_series(self._study_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._series_path_1, all_series_json[0], ''
        ),
        (
            'The returned DICOM object should have a type of SERIES, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  def test_get_instances_with_invalid_input_raise_error(self):
    dwi = _create_test_dwi()
    with self.assertRaises(ez_wsi_errors.DicomPathError):
      dwi.get_instances(self._instance_path_1)

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_instances_with_store_level_path(self, unused_mock_credientals):
    all_instances_json = [
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_1],
                'vr': 'UI',
            },
            '0020000E': {
                'Value': [dicom_test_utils.TEST_SERIES_UID_1],
                'vr': 'UI',
            },
            '00080018': {
                'Value': [dicom_test_utils.TEST_INSTANCE_UID_1],
                'vr': 'UI',
            },
        },
        {
            '0020000D': {
                'Value': [dicom_test_utils.TEST_STUDY_UID_2],
                'vr': 'UI',
            },
            '0020000E': {
                'Value': [dicom_test_utils.TEST_SERIES_UID_2],
                'vr': 'UI',
            },
            '00080018': {
                'Value': [dicom_test_utils.TEST_INSTANCE_UID_2],
                'vr': 'UI',
            },
        },
    ]
    dwi = self._create_test_dwi_qidors(all_instances_json)
    instances = dwi.get_instances(self._dicom_store_path)
    self.assertCountEqual(
        instances,
        [
            dicom_web_interface.DicomObject(
                self._instance_path_1, all_instances_json[0], ''
            ),
            dicom_web_interface.DicomObject(
                self._instance_path_2, all_instances_json[1], ''
            ),
        ],
        (
            'All DICOM objects returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_instances_with_study_level_path(self, unused_mock_credientals):
    all_instances_json = [{
        '0020000E': {'Value': [dicom_test_utils.TEST_SERIES_UID_1], 'vr': 'UI'},
        '00080018': {
            'Value': [dicom_test_utils.TEST_INSTANCE_UID_1],
            'vr': 'UI',
        },
    }]
    dwi = self._create_test_dwi_qidors(all_instances_json)
    instances = dwi.get_instances(self._study_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._instance_path_1, all_instances_json[0], ''
        ),
        (
            'All DICOM objects returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_instances_with_series_level_path(self, unused_mock_credientals):
    all_instances_json = [{
        '00080018': {
            'Value': [dicom_test_utils.TEST_INSTANCE_UID_1],
            'vr': 'UI',
        },
        '00280010': {'Value': [500], 'vr': 'US'},
        '00280002': {'Value': [3], 'vr': 'US'},
        '00280100': {'Value': [8], 'vr': 'US'},
        '00280102': {'Value': [7], 'vr': 'US'},
        '00280011': {'Value': [500], 'vr': 'US'},
        '00480006': {'Value': [98816], 'vr': 'UL'},
        '00480007': {'Value': [199168], 'vr': 'UL'},
    }]
    dwi = self._create_test_dwi_qidors(all_instances_json)
    instances = dwi.get_instances(self._series_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._instance_path_1, all_instances_json[0], ''
        ),
        (
            'The DICOM object returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )
    self.assertEqual(500, instances[0].get_value(tags.ROWS))
    self.assertEqual(500, instances[0].get_value(tags.COLUMNS))
    self.assertEqual(3, instances[0].get_value(tags.SAMPLES_PER_PIXEL))
    self.assertEqual(8, instances[0].get_value(tags.BITS_ALLOCATED))
    self.assertEqual(7, instances[0].get_value(tags.HIGH_BIT))
    self.assertEqual(
        98816, instances[0].get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS)
    )
    self.assertEqual(
        199168, instances[0].get_value(tags.TOTAL_PIXEL_MATRIX_ROWS)
    )
    self.assertEmpty(instances[0].icc_profile_bulkdata_url)

  @parameterized.named_parameters(
      dict(
          testcase_name='root_icc_profile',
          icc_profile_metadata={
              '00282000': {'vr': 'OB', 'BulkDataURI': 'fake_bulkdata_uri'}
          },
      ),
      dict(
          testcase_name='first_optical_path_sq',
          icc_profile_metadata={
              '00480105': {
                  'vr': 'SQ',
                  'Value': [{
                      '00282000': {
                          'vr': 'OB',
                          'BulkDataURI': 'fake_bulkdata_uri',
                      }
                  }],
              }
          },
      ),
      dict(
          testcase_name='second_optical_path_sq',
          icc_profile_metadata={
              '00480105': {
                  'vr': 'SQ',
                  'Value': [
                      {},
                      {
                          '00282000': {
                              'vr': 'OB',
                              'BulkDataURI': 'fake_bulkdata_uri',
                          }
                      },
                  ],
              }
          },
      ),
  )
  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_instances_with_series_level_path_with_icc_profile_bulkdata_uri(
      self, unused_mock_credientals, icc_profile_metadata
  ):
    instance_metadata = {
        '00080018': {
            'Value': [dicom_test_utils.TEST_INSTANCE_UID_1],
            'vr': 'UI',
        },
        '00280010': {'Value': [500], 'vr': 'US'},
        '00280002': {'Value': [3], 'vr': 'US'},
        '00280100': {'Value': [8], 'vr': 'US'},
        '00280102': {'Value': [7], 'vr': 'US'},
        '00280011': {'Value': [500], 'vr': 'US'},
        '00480006': {'Value': [98816], 'vr': 'UL'},
        '00480007': {'Value': [199168], 'vr': 'UL'},
    }
    instance_metadata.update(icc_profile_metadata)
    all_instances_json = [instance_metadata]
    dwi = self._create_test_dwi_qidors(all_instances_json)
    instances = dwi.get_instances(self._series_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._instance_path_1, all_instances_json[0], 'fake_bulkdata_uri'
        ),
        (
            'The DICOM object returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )
    self.assertEqual(500, instances[0].get_value(tags.ROWS))
    self.assertEqual(500, instances[0].get_value(tags.COLUMNS))
    self.assertEqual(3, instances[0].get_value(tags.SAMPLES_PER_PIXEL))
    self.assertEqual(8, instances[0].get_value(tags.BITS_ALLOCATED))
    self.assertEqual(7, instances[0].get_value(tags.HIGH_BIT))
    self.assertEqual(
        98816, instances[0].get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS)
    )
    self.assertEqual(
        199168, instances[0].get_value(tags.TOTAL_PIXEL_MATRIX_ROWS)
    )
    self.assertEqual(instances[0].icc_profile_bulkdata_url, 'fake_bulkdata_uri')
    self.assertNotIn(tags.ICC_PROFILE.number, instances[0].dicom_tags)
    self.assertNotIn(tags.OPTICAL_PATH_SEQUENCE.number, instances[0].dicom_tags)

  def test_get_frame_image_with_invalid_input_raise_error(self):
    dwi = _create_test_dwi()
    with self.subTest(name='Store'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_frame_image(
            self._dicom_store_path,
            1,
            dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
        )
    with self.subTest(name='Study'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_frame_image(
            self._study_path_1,
            1,
            dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
        )
    with self.subTest(name='Series'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_frame_image(
            self._series_path_1,
            1,
            dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
        )

  @mock.patch.object(google.auth, 'default', autospec=True, return_value=_AUTH)
  def test_get_frame_image_with_valid_input(self, unused_mock_credientals):
    expected_image = b'\x01\x02\x03\x04'
    dwi = self._create_test_dwi_wadors(expected_image)
    self.assertEqual(
        expected_image,
        dwi.get_frame_image(
            self._instance_path_1,
            1,
            dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
        ),
    )

  def test_get_set_state(self):
    dwi = self._create_test_dwi_wadors(b'\x01\x02\x03\x04')
    self.assertIsNotNone(dwi._interface_lock)
    state = dwi.__getstate__()
    for val in ('_credentials', '_dicom_web_client', '_interface_lock'):
      self.assertNotIn(val, state)
    dwi.__setstate__(state)
    self.assertIsNone(dwi._credentials)
    self.assertIsNotNone(dwi._interface_lock)

  def test_download_instance_untranscoded_succeeds(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      with io.BytesIO() as downloaded_instance:
        dwi.download_instance_untranscoded(instance_path, downloaded_instance)
        downloaded_instance.seek(0)
        self.assertEqual(
            pydicom.dcmread(downloaded_instance).to_json_dict(),
            test_instance.to_json_dict(),
        )

  def test_download_instance_untranscoded_fails(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path):
      with io.BytesIO() as downloaded_instance:
        with self.assertRaises(ez_wsi_errors.HttpNotFoundError):
          dwi.download_instance_untranscoded(instance_path, downloaded_instance)

  @mock.patch.object(requests.Session, 'get', side_effect=ValueError)
  def test_request_get_called_with_expected_accept_header(self, mock_get):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with io.BytesIO() as downloaded_instance:
      try:
        dwi.download_instance_untranscoded(
            instance_path,
            downloaded_instance,
        )
      except ValueError:
        # Throwing Value Error purposefully when mocked function is called.
        # To avoid further function execution. Test checks that request.get
        # is called with expected parameters.
        pass

    mock_get.assert_called_once_with(
        instance_path.complete_url,
        headers={'Accept': 'application/dicom; transfer-syntax=*'},
        stream=True,
    )

  def test_download_instance_frame_list_untranscoded(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    number_of_frames = test_instance.NumberOfFrames
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      frames = dwi.download_instance_frame_list_untranscoded(
          instance_path,
          range(1, number_of_frames + 1, 2),
      )
    self.assertLen(frames, int(number_of_frames / 2) + 1)
    for index, frame in enumerate(frames):
      self.assertEqual(
          dicom_test_utils.test_dicom_instance_frame_bytes((index * 2) + 1),
          frame,
      )

  def test_download_instance_frames_untranscoded(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    number_of_frames = test_instance.NumberOfFrames
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      frames = dwi.download_instance_frames_untranscoded(
          instance_path,
          1,
          number_of_frames,
      )
    self.assertLen(frames, number_of_frames)
    for index, frame in enumerate(frames):
      self.assertEqual(
          dicom_test_utils.test_dicom_instance_frame_bytes(index + 1), frame
      )

  def test_get_icc_profile_bulkdata(self):
    mock_icc_profile = b'MOCK_ICC_PROFILE'
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance.ICCProfile = mock_icc_profile
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      self.assertEqual(
          dwi.get_bulkdata(
              '/'.join([
                  dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL,
                  dicom_store_path_instance_path,
                  'studies',
                  test_instance.StudyInstanceUID,
                  'series',
                  test_instance.SeriesInstanceUID,
                  'instances',
                  test_instance.SOPInstanceUID,
                  'bulkdata',
                  '00282000',
              ]),
          ),
          mock_icc_profile,
      )

  def test_get_icc_profile_bulkdata_instance_not_found_returns_none(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path):
      with self.assertRaises(ez_wsi_errors.HttpBadRequestError):
        dwi.get_bulkdata(
            '/'.join([
                dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL,
                dicom_store_path_instance_path,
                'studies',
                test_instance.StudyInstanceUID,
                'series',
                test_instance.SeriesInstanceUID,
                'instances',
                test_instance.SOPInstanceUID,
                'bulkdata',
                '00282000',
            ]),
        )

  def test_get_icc_profile_bulkdata_missing_bulkdata_returns_empty(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory.NoAuthCredentialsFactory()
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      self.assertEqual(
          dwi.get_bulkdata(
              '/'.join([
                  dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL,
                  dicom_store_path_instance_path,
                  'studies',
                  test_instance.StudyInstanceUID,
                  'series',
                  test_instance.SeriesInstanceUID,
                  'instances',
                  test_instance.SOPInstanceUID,
                  'bulkdata',
                  '00282000',
              ]),
          ),
          b'',
      )


if __name__ == '__main__':
  absltest.main()
