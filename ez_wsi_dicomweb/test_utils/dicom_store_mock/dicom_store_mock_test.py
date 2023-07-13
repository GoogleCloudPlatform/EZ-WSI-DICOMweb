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
"""Tests dicom_store_mock."""
import http.client
import io
import json
import os
import tempfile
from typing import Mapping

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb.test_utils import dicom_test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock_types
import google_auth_httplib2
import pydicom
import requests
import requests_toolbelt

_MOCK_STORE_URL = 'http://mock.dicom.store1.com/dicomweb'


class _MockGetDicomUidTripleInterface(
    dicom_store_mock_types.AbstractGetDicomUidTripleInterface
):

  def __init__(self, study: str, series: str, instance: str):
    self._study, self._series, self._instance = study, series, instance

  def get_dicom_uid_triple(self) -> dicom_store_mock_types.DicomUidTriple:
    return dicom_store_mock_types.DicomUidTriple(
        self._study, self._series, self._instance
    )


class DicomStoreMockTest(parameterized.TestCase):

  @parameterized.parameters(
      [('foo', 'bar'), ('foo/', 'bar'), ('foo', '/bar'), ('foo/', '/bar')]
  )
  def test_join_url_parts(self, p1: str, p2: str):
    self.assertEqual(dicom_store_mock._join_url_parts(p1, p2), 'foo/bar')

  def test_convert_instance_from_store_json(self):
    dcm = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    dcm_json = dicom_store_mock._pydicom_file_dataset_to_json(dcm)
    dcm2 = dicom_store_mock._convert_to_pydicom_file_dataset(dcm_json)
    self.assertEqual(dcm2.to_json_dict(), dcm.to_json_dict())
    self.assertEqual(
        dcm2.file_meta.to_json_dict(), dcm.file_meta.to_json_dict()
    )

  def test_get_dicom_uid_from_dicom_uid_triple(self):
    study = '1.2.3'
    series = '1.2.3.4'
    instance = '1.2.3.4.5'

    result = dicom_store_mock._get_dicom_uid(
        dicom_store_mock_types.DicomUidTriple(study, series, instance)
    )

    self.assertEqual(result.study_instance_uid, study)
    self.assertEqual(result.series_instance_uid, series)
    self.assertEqual(result.sop_instance_uid, instance)

  def test_get_dicom_uid_from_dicom_json(self):
    dcm = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())

    result = dicom_store_mock._get_dicom_uid(dcm.to_json_dict())

    self.assertEqual(result.study_instance_uid, dcm.StudyInstanceUID)
    self.assertEqual(result.series_instance_uid, dcm.SeriesInstanceUID)
    self.assertEqual(result.sop_instance_uid, dcm.SOPInstanceUID)

  def test_get_dicom_abstract_get_dicom_uid_triple_interface(self):
    result = dicom_store_mock._get_dicom_uid(
        _MockGetDicomUidTripleInterface('1', '2', '3')
    )
    self.assertEqual(result.study_instance_uid, '1')
    self.assertEqual(result.series_instance_uid, '2')
    self.assertEqual(result.sop_instance_uid, '3')

  def test_assert_empty_store(self):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].assert_empty(self)

  def test_assert_not_empty_store(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      mock_ds[_MOCK_STORE_URL].assert_not_empty(self)

  def test_multiple_stores(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(
        'store1', 'store2', 'store1'
    ) as mock_ds:
      self.assertLen(mock_ds, 2)
      mock_ds['store1'].add_instance(test_dicom_path)
      mock_ds['store1'].assert_not_empty(self)
      mock_ds['store2'].assert_empty(self)

  def test_assert_uid_not_in_store(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, test_dicom_path)

  def test_assert_uid_in_store(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_upload_dicom_to_store(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      headers = {'Content-Type': 'application/dicom'}
      with requests.Session() as session:
        with open(test_dicom_path, 'rb') as infile:
          response = session.post(
              f'{_MOCK_STORE_URL}/studies', data=infile, headers=headers
          )
          response.raise_for_status()

      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_upload_invalid_dicom_to_store_raises_error(self):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL):
      headers = {'Content-Type': 'application/dicom'}
      with requests.Session() as session:
        response = session.post(
            f'{_MOCK_STORE_URL}/studies', data=io.BytesIO(), headers=headers
        )
        with self.assertRaises(requests.exceptions.HTTPError):
          response.raise_for_status()

  def test_upload_duplicate_dicom_to_store_succeeds(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'Content-Type': 'application/dicom'}
      with requests.Session() as session:
        with open(test_dicom_path, 'rb') as infile:
          response = session.post(
              f'{_MOCK_STORE_URL}/studies', data=infile, headers=headers
          )
          response.raise_for_status()

      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_delete_study_instance_uid_from_store(self):
    dcm = dicom_test_utils.test_dicominstance_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    dcm2.StudyInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      with requests.Session() as session:
        response = session.delete(
            f'{_MOCK_STORE_URL}/studies/{studyuid}', headers=headers
        )

        response.raise_for_status()
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, dcm2)
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, dcm)

  def test_delete_series_instance_uid_from_store(self):
    dcm = dicom_test_utils.test_dicominstance_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    seriesuid = dcm2.SeriesInstanceUID
    dcm2.SeriesInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      with requests.Session() as session:
        response = session.delete(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}',
            headers=headers,
        )

        response.raise_for_status()
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, dcm2)
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, dcm)

  def test_delete_sop_instance_uid_from_store(self):
    dcm = dicom_test_utils.test_dicominstance_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    seriesuid = dcm2.SeriesInstanceUID
    instanceuid = dcm2.SOPInstanceUID
    dcm2.SOPInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      with requests.Session() as session:
        response = session.delete(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}',
            headers=headers,
        )

        response.raise_for_status()
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, dcm2)
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, dcm)

  @parameterized.parameters([
      ('123.456.789', '', ''),
      ('', '123.456.789', ''),
      ('', '', '123.456.789'),
  ])
  def test_delete_for_instance_that_does_not_exist_raises_error(
      self, study_instance_uid, series_instance_uid, sop_instance_uid
  ):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    study_instance_uid = (
        dcm.StudyInstanceUID if not study_instance_uid else study_instance_uid
    )
    series_instance_uid = (
        dcm.SeriesInstanceUID
        if not series_instance_uid
        else series_instance_uid
    )
    sop_instance_uid = (
        dcm.SOPInstanceUID if not sop_instance_uid else sop_instance_uid
    )

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.delete(
            (
                f'{_MOCK_STORE_URL}/studies/{study_instance_uid}/series/'
                f'{series_instance_uid}/instances/{sop_instance_uid}'
            ),
            headers=headers,
        )
        with self.assertRaises(requests.exceptions.HTTPError):
          response.raise_for_status()

  def test_download_instance_from_store(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom; transfer-syntax=*'}
      with requests.Session() as session:
        response = session.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}',
            headers=headers,
        )
        with tempfile.TemporaryDirectory() as tempdir:
          tempath = os.path.join(tempdir, 'temp.dcm')
          with open(tempath, 'wb') as f:
            f.write(response.raw.getvalue())
          dcm2 = pydicom.dcmread(tempath)
    self.assertEqual(dcm.to_json_dict(), dcm2.to_json_dict())

  @parameterized.parameters([
      ('123.456.789', '', ''),
      ('', '123.456.789', ''),
      ('', '', '123.456.789'),
  ])
  def test_download_instance_that_does_not_exist_raises_error(
      self, study_instance_uid, series_instance_uid, sop_instance_uid
  ):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    study_instance_uid = (
        dcm.StudyInstanceUID if not study_instance_uid else study_instance_uid
    )
    series_instance_uid = (
        dcm.SeriesInstanceUID
        if not series_instance_uid
        else series_instance_uid
    )
    sop_instance_uid = (
        dcm.SOPInstanceUID if not sop_instance_uid else sop_instance_uid
    )

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{study_instance_uid}/series/'
                f'{series_instance_uid}/instances/{sop_instance_uid}'
            ),
            headers=headers,
        )
        with self.assertRaises(requests.exceptions.HTTPError):
          response.raise_for_status()

  @parameterized.parameters(['', '?', '/?'])
  def test_study_metadata_request(self, url_suffix: str):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/instances{url_suffix}',
            headers=headers,
        )
        metadata_response = json.loads(response.text)

        self.assertLen(metadata_response, 1)
    self.assertEqual(
        metadata_response[0],
        dicom_store_mock._pydicom_file_dataset_to_json(dcm),
    )

  @parameterized.parameters(['', '?', '/?'])
  def test_series_metadata_request(self, url_suffix: str):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances'
                f'{url_suffix}'
            ),
            headers=headers,
        )
        metadata_response = json.loads(response.text)

        self.assertLen(metadata_response, 1)
    self.assertEqual(
        metadata_response[0],
        dicom_store_mock._pydicom_file_dataset_to_json(dcm),
    )

  def test_instance_metadata_request(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/'
                f'instances/{instanceuid}/metadata'
            ),
            headers=headers,
        )
        metadata_response = json.loads(response.text)

        self.assertLen(metadata_response, 1)
    self.assertEqual(
        metadata_response[0],
        dicom_store_mock._pydicom_file_dataset_to_json(dcm),
    )

  @parameterized.parameters(
      [('123.456.789', ''), ('', '123.456.789'), ('123.456.789', '123.456.789')]
  )
  def test_instance_metadata_search_for_study_series_that_does_not_exist_empty(
      self, studyuid, seriesuid
  ):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID if not studyuid else studyuid
    seriesuid = dcm.SeriesInstanceUID if not seriesuid else seriesuid

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/'
                'instances?'
            ),
            headers=headers,
        )
        self.assertEmpty(response.text)
        self.assertEqual(
            response.headers['Content-Type'],
            dicom_store_mock._ContentType.TEXT_HTML.value,
        )

  def test_instance_metadata_request_for_instance_does_not_exist_raises_error(
      self,
  ):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    invalid_instanceuid = '123.456.789'

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/'
                f'instances/{invalid_instanceuid}/metadata'
            ),
            headers=headers,
        )
        with self.assertRaises(requests.exceptions.HTTPError):
          response.raise_for_status()

  @parameterized.parameters(['', '/'])
  def test_dicom_store_frame_request(self, dicom_path_suffix: str):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    expected_frame_data = list(
        pydicom.encaps.generate_pixel_data_frame(
            dcm.PixelData, dcm.NumberOfFrames
        )
    )
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {
          'accept': (
              'multipart/related; type=application/octet-stream;'
              ' transfer-syntax=*'
          )
      }
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/'
                f'instances/{instanceuid}/frames/1{dicom_path_suffix}'
            ),
            headers=headers,
        )
        response.raise_for_status()
    multipart_data = requests_toolbelt.MultipartDecoder.from_response(response)
    self.assertLen(multipart_data.parts, 1)
    self.assertEqual(multipart_data.parts[0].content, expected_frame_data[0])

  @parameterized.named_parameters([
      dict(
          testcase_name='invalid_frame_index',
          instance_config={},
          header_config={},
          frame_request='2',
          error_status_code=http.client.BAD_REQUEST,
      ),
      dict(
          testcase_name='invalid_dicom_uid',
          instance_config=dict(instanceuid='1.2.3'),
          header_config={},
          frame_request='1',
          error_status_code=http.client.NOT_FOUND,
      ),
      dict(
          testcase_name='invalid_accept_header',
          instance_config={},
          header_config=dict(accept='foo'),
          frame_request='1',
          error_status_code=http.client.NOT_ACCEPTABLE,
      ),
  ])
  def test_dicom_store_invalid_frame_request(
      self,
      instance_config: Mapping[str, str],
      header_config: Mapping[str, str],
      frame_request: str,
      error_status_code: int,
  ):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    dicom_instance = dict(
        studyuid=dcm.StudyInstanceUID,
        seriesuid=dcm.SeriesInstanceUID,
        instanceuid=dcm.SOPInstanceUID,
    )
    dicom_instance.update(instance_config)

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {
          'accept': (
              'multipart/related; type=application/octet-stream;'
              ' transfer-syntax=*'
          )
      }
      headers.update(header_config)
      with requests.Session() as session:
        response = session.get(
            (
                f'{_MOCK_STORE_URL}/studies/{dicom_instance["studyuid"]}/series'
                f'/{dicom_instance["seriesuid"]}/instances/'
                f'{dicom_instance["instanceuid"]}/frames/{frame_request}'
            ),
            headers=headers,
        )
        self.assertEqual(response.status_code, error_status_code)
        with self.assertRaises(requests.exceptions.HTTPError):
          response.raise_for_status()

  def test_httplib2_series_metadata_request(self):
    test_dicom_path = dicom_test_utils.test_dicominstance_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      http_auth = google_auth_httplib2.AuthorizedHttp(None)
      try:
        _, response_data = http_auth.request(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances',
            headers={'accept': 'application/dicom+json; charset=utf-8'},
            method='GET',
            body=None,
        )
      finally:
        http_auth.close()
    metadata_response = json.loads(response_data)
    self.assertLen(metadata_response, 1)
    self.assertEqual(
        metadata_response[0],
        dicom_store_mock._pydicom_file_dataset_to_json(dcm),
    )

  def test_invalid_httplib2_request(self):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL):
      http_auth = google_auth_httplib2.AuthorizedHttp(None)
      try:
        response, response_data = http_auth.request(
            f'{_MOCK_STORE_URL}/bad_request',
            headers={'accept': 'application/dicom+json; charset=utf-8'},
            method='GET',
            body=None,
        )
      finally:
        http_auth.close()
    self.assertEqual(response.status, http.client.BAD_REQUEST)
    self.assertEqual(
        response_data, b'DICOM Store Error: Unhandled HTTPLib2 request.'
    )


if __name__ == '__main__':
  absltest.main()
