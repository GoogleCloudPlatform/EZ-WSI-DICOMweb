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
"""Tests dicom store mock."""
import http.client
import io
import json
import os
import shutil
import tempfile
from typing import Mapping
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import google_auth_httplib2
import pydicom
import requests
import requests_toolbelt

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock_types

_MOCK_STORE_URL = 'http://mock.dicom.store1.com/dicomweb'


def _test_file_path() -> str:
  return os.path.join(
      flags.FLAGS.test_srcdir,
      (
          '_main/ez_wsi_dicomweb/test_utils/'
          'dicom_store_mock/testdata/test_wsi.dcm'
      ),
  )


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
    dcm = pydicom.dcmread(_test_file_path())
    dcm_json = dicom_store_mock._pydicom_file_dataset_to_json(
        dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
    )
    dcm2 = dicom_store_mock._convert_to_pydicom_file_dataset(dcm_json)
    # Binary Tags are removed
    del dcm['PixelData']  # Pixel Data tag
    del dcm.file_meta['00020001']  # 	File Meta Information Version tag
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
    dcm = pydicom.dcmread(_test_file_path())

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
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      mock_ds[_MOCK_STORE_URL].assert_not_empty(self)

  def test_multiple_stores(self):
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(
        'store1', 'store2', 'store1'
    ) as mock_ds:
      self.assertLen(mock_ds, 2)
      mock_ds['store1'].add_instance(test_dicom_path)
      mock_ds['store1'].assert_not_empty(self)
      mock_ds['store2'].assert_empty(self)

  def test_assert_uid_not_in_store(self):
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, test_dicom_path)

  def test_assert_uid_in_store(self):
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_upload_dicom_to_store(self):
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      headers = {'Content-Type': 'application/dicom'}
      with open(test_dicom_path, 'rb') as infile:
        response = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=infile, headers=headers
        )
        response.raise_for_status()

      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_upload_invalid_dicom_to_store_raises_error(self):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL):
      headers = {'Content-Type': 'application/dicom'}
      response = requests.post(
          f'{_MOCK_STORE_URL}/studies', data=io.BytesIO(), headers=headers
      )
      with self.assertRaises(requests.exceptions.HTTPError):
        response.raise_for_status()

  def test_upload_duplicate_dicom_to_store_succeeds(self):
    test_dicom_path = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'Content-Type': 'application/dicom'}
      with open(test_dicom_path, 'rb') as infile:
        response = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=infile, headers=headers
        )
        response.raise_for_status()

      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, test_dicom_path)

  def test_delete_study_instance_uid_from_store(self):
    dcm = _test_file_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    dcm2.StudyInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      response = requests.delete(
          f'{_MOCK_STORE_URL}/studies/{studyuid}', headers=headers
      )

      response.raise_for_status()
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, dcm2)
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, dcm)

  def test_delete_series_instance_uid_from_store(self):
    dcm = _test_file_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    seriesuid = dcm2.SeriesInstanceUID
    dcm2.SeriesInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      response = requests.delete(
          f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}',
          headers=headers,
      )

      response.raise_for_status()
      mock_ds[_MOCK_STORE_URL].assert_uid_in_store(self, dcm2)
      mock_ds[_MOCK_STORE_URL].assert_uid_not_in_store(self, dcm)

  def test_delete_sop_instance_uid_from_store(self):
    dcm = _test_file_path()
    dcm2 = pydicom.dcmread(dcm)
    studyuid = dcm2.StudyInstanceUID
    seriesuid = dcm2.SeriesInstanceUID
    instanceuid = dcm2.SOPInstanceUID
    dcm2.SOPInstanceUID = '599.999.999'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(dcm)
      mock_ds[_MOCK_STORE_URL].add_instance(dcm2)
      headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}

      response = requests.delete(
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
    test_dicom_path = _test_file_path()
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
      response = requests.delete(
          (
              f'{_MOCK_STORE_URL}/studies/{study_instance_uid}/series/'
              f'{series_instance_uid}/instances/{sop_instance_uid}'
          ),
          headers=headers,
      )
      with self.assertRaises(requests.exceptions.HTTPError):
        response.raise_for_status()

  def test_download_instance_from_store(self):
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom; transfer-syntax=*'}

      response = requests.get(
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
    test_dicom_path = _test_file_path()
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
      response = requests.get(
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
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    with dicom_store_mock.MockDicomStores(
        _MOCK_STORE_URL, bulkdata_uri_enabled=False
    ) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      response = requests.get(
          f'{_MOCK_STORE_URL}/studies/{studyuid}/instances{url_suffix}',
          headers=headers,
      )
      metadata_response = json.loads(response.text)

      self.assertLen(metadata_response, 1)
    self.assertEqual(
        metadata_response[0],
        dicom_store_mock._pydicom_file_dataset_to_json(
            dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
        ),
    )

  @parameterized.parameters(['', '?', '/?'])
  def test_series_metadata_request(self, url_suffix: str):
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    with dicom_store_mock.MockDicomStores(
        _MOCK_STORE_URL, bulkdata_uri_enabled=False
    ) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      response = requests.get(
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
        dicom_store_mock._pydicom_file_dataset_to_json(
            dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
        ),
    )

  def test_instance_metadata_request(self):
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID
    with dicom_store_mock.MockDicomStores(
        _MOCK_STORE_URL, bulkdata_uri_enabled=False
    ) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      response = requests.get(
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
        dicom_store_mock._pydicom_file_dataset_to_json(
            dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
        ),
    )

  @parameterized.parameters(
      [('123.456.789', ''), ('', '123.456.789'), ('123.456.789', '123.456.789')]
  )
  def test_instance_metadata_search_for_study_series_that_does_not_exist_empty(
      self, studyuid, seriesuid
  ):
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID if not studyuid else studyuid
    seriesuid = dcm.SeriesInstanceUID if not seriesuid else seriesuid

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      response = requests.get(
          f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances?',
          headers=headers,
      )
      self.assertEmpty(response.text)
      self.assertEqual(
          response.headers['Content-Type'],
          dicom_store_mock.ContentType.TEXT_HTML.value,
      )

  def test_instance_metadata_request_for_instance_does_not_exist_raises_error(
      self,
  ):
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    invalid_instanceuid = '123.456.789'

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {'accept': 'application/dicom+json; charset=utf-8'}
      response = requests.get(
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
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    expected_frame_data = list(
        dicom_store_mock._generate_frames(dcm.PixelData, dcm.NumberOfFrames)
    )
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    instanceuid = dcm.SOPInstanceUID

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mock_ds:
      mock_ds[_MOCK_STORE_URL].add_instance(test_dicom_path)
      headers = {
          'accept': (
              'multipart/related; type="application/octet-stream";'
              ' transfer-syntax=*'
          )
      }
      response = requests.get(
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
    test_dicom_path = _test_file_path()
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
              'multipart/related; type="application/octet-stream";'
              ' transfer-syntax=*'
          )
      }
      headers.update(header_config)
      response = requests.get(
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
    test_dicom_path = _test_file_path()
    dcm = pydicom.dcmread(test_dicom_path)
    studyuid = dcm.StudyInstanceUID
    seriesuid = dcm.SeriesInstanceUID
    with dicom_store_mock.MockDicomStores(
        _MOCK_STORE_URL, bulkdata_uri_enabled=False
    ) as mock_ds:
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
        dicom_store_mock._pydicom_file_dataset_to_json(
            dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
        ),
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

  def test_set_dicom_store_mock_to_empty_file_system_interface(self):
    temp_dir = self.create_tempdir()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      mk[_MOCK_STORE_URL].assert_empty(self)

  def test_set_dicom_store_mock_ignores_non_dicom_files(self):
    temp_dir = self.create_tempdir()
    with open(os.path.join(temp_dir, 'foo.txt'), 'wt') as outfile:
      outfile.write('foo')
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      mk[_MOCK_STORE_URL].assert_empty(self)

  def test_set_dicom_store_mock_to_file_system_interface_with_dicom_file(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    dest_file = os.path.join(temp_dir, os.path.basename(source_file))
    shutil.copyfile(source_file, dest_file)
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      mk[_MOCK_STORE_URL].assert_not_empty(self)
      with pydicom.dcmread(source_file) as expected_instance:
        mk[_MOCK_STORE_URL].assert_uid_in_store(self, [expected_instance])

  def test_adding_dicom_to_dicom_store_mock_writes_to_file_file_system(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        expected_path = os.path.join(
            temp_dir,
            f'{dcm.StudyInstanceUID}/{dcm.SeriesInstanceUID}/{dcm.SOPInstanceUID}.dcm',
        )
        self.assertTrue(os.path.exists(expected_path))

  def test_removing_from_filesystem_mock_removes_files_and_dirs(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}
    studyuid = pydicom.dcmread(source_file).StudyInstanceUID
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
      result = requests.delete(
          f'{_MOCK_STORE_URL}/studies/{studyuid}', headers=headers
      )
      result.raise_for_status()

      mk[_MOCK_STORE_URL].assert_empty(self)
      self.assertEmpty(os.listdir(temp_dir))

  def test_delete_missing_files_from_filesystem_mock(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      dcm = pydicom.dcmread(source_file)
      mk[_MOCK_STORE_URL].add_instance(dcm)
      expected_path = os.path.join(
          temp_dir,
          f'{dcm.StudyInstanceUID}/{dcm.SeriesInstanceUID}/{dcm.SOPInstanceUID}.dcm',
      )
      result = requests.delete(
          f'{_MOCK_STORE_URL}/studies/1.2.3', headers=headers
      )
      self.assertEqual(result.status_code, http.HTTPStatus.NOT_FOUND)
      # validate stored dicom not removed.
      self.assertTrue(os.path.exists(expected_path))
      mk[_MOCK_STORE_URL].assert_uid_in_store(self, [dcm])

  def test_get_add_dicom_through_mocked_api_to_file_system_mock(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with open(source_file, 'rb') as data:
        result = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=data, headers=headers
        )
        result.raise_for_status()

      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].assert_uid_in_store(self, [dcm])
        expected_path = os.path.join(
            temp_dir,
            f'{dcm.StudyInstanceUID}/{dcm.SeriesInstanceUID}/{dcm.SOPInstanceUID}.dcm',
        )
        self.assertTrue(os.path.exists(expected_path))

  def test_get_add_same_dicom_twice_through_file_system_mock_succeeds(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with open(source_file, 'rb') as data:
        result = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=data, headers=headers
        )
        result.raise_for_status()
      with open(source_file, 'rb') as data:
        result = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=data, headers=headers
        )
        result.raise_for_status()

      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].assert_uid_in_store(self, [dcm])
        expected_path = os.path.join(
            temp_dir,
            f'{dcm.StudyInstanceUID}/{dcm.SeriesInstanceUID}/{dcm.SOPInstanceUID}.dcm',
        )
        self.assertTrue(os.path.exists(expected_path))

  def test_get_add_conflicting_dicom_through_file_system_mock_fails(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with open(source_file, 'rb') as data:
        result = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=data, headers=headers
        )
        result.raise_for_status()

      dcm = pydicom.dcmread(source_file)
      dcm.ImageType = 'FOO'
      with io.BytesIO() as data:
        dicom_store_mock._save_as(dcm, data)
        data.seek(0)
        result = requests.post(
            f'{_MOCK_STORE_URL}/studies', data=data, headers=headers
        )
        self.assertEqual(result.status_code, http.HTTPStatus.CONFLICT)
      # validate original still exists
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].assert_uid_in_store(self, [dcm])
        expected_path = os.path.join(
            temp_dir,
            f'{dcm.StudyInstanceUID}/{dcm.SeriesInstanceUID}/{dcm.SOPInstanceUID}.dcm',
        )
        self.assertTrue(os.path.exists(expected_path))

  def test_get_instance_bytes_using_file_system_mock(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom; transfer-syntax=*'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
      response = requests.get(
          f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}',
          headers=headers,
      )
      response.raise_for_status()
      with io.BytesIO(response.content) as data:
        with pydicom.dcmread(data) as dicom_read_from_store:
          with pydicom.dcmread(source_file) as dcm:
            self.assertEqual(dicom_read_from_store, dcm)

  def test_get_instance_metadata_using_file_system_mock(self):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(
        _MOCK_STORE_URL, bulkdata_uri_enabled=False
    ) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        response = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        self.assertEqual(
            response.json(),
            [
                dicom_store_mock._pydicom_file_dataset_to_json(
                    dcm, dicom_store_mock._FilterBinaryTagOperation.REMOVE
                ),
            ],
        )

  def test_get_repeated_calls_to_file_system_instance_metadata_return_same_val(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        response_2 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        self.assertEqual(response_1.json(), response_2.json())

  def test_metadata_query_with_missing_file_returns_nothing(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        os.remove(
            os.path.join(temp_dir, f'{studyuid}/{seriesuid}/{instanceuid}.dcm')
        )
        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        self.assertEqual(response_1.status_code, http.HTTPStatus.NOT_FOUND)
        mk[_MOCK_STORE_URL].assert_empty(self)

  def test_metadata_query_with_bad_file_returns_nothing(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        with open(
            os.path.join(temp_dir, f'{studyuid}/{seriesuid}/{instanceuid}.dcm'),
            'wt',
        ) as outfile:
          outfile.write('bad')
        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        self.assertEqual(response_1.status_code, http.HTTPStatus.NOT_FOUND)
        mk[_MOCK_STORE_URL].assert_empty(self)

  def test_metadata_query_with_missing_file_returns_nothing_clears_cache(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom+json; charset=utf-8'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        os.remove(
            os.path.join(temp_dir, f'{studyuid}/{seriesuid}/{instanceuid}.dcm')
        )
        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers=headers,
        )
        self.assertEqual(response_1.status_code, http.HTTPStatus.NOT_FOUND)
        mk[_MOCK_STORE_URL].assert_empty(self)

  def test_instance_retrieval_with_missing_file_returns_nothing(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom; transfer-syntax=*'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        os.remove(
            os.path.join(temp_dir, f'{studyuid}/{seriesuid}/{instanceuid}.dcm')
        )
        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}',
            headers=headers,
        )
        self.assertEqual(response_1.status_code, http.HTTPStatus.NOT_FOUND)
        mk[_MOCK_STORE_URL].assert_empty(self)

  def test_instance_retrieval_with_missing_file_clears_metadata_cache(
      self,
  ):
    temp_dir = self.create_tempdir()
    source_file = _test_file_path()
    headers = {'accept': 'application/dicom; transfer-syntax=*'}
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk[_MOCK_STORE_URL].set_dicom_store_disk_storage(temp_dir)
      with pydicom.dcmread(source_file) as dcm:
        mk[_MOCK_STORE_URL].add_instance(dcm)
        studyuid = dcm.StudyInstanceUID
        seriesuid = dcm.SeriesInstanceUID
        instanceuid = dcm.SOPInstanceUID
        requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers={'accept': 'application/dicom+json; charset=utf-8'},
        )
        os.remove(
            os.path.join(temp_dir, f'{studyuid}/{seriesuid}/{instanceuid}.dcm')
        )
        requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}',
            headers=headers,
        )

        response_1 = requests.get(
            f'{_MOCK_STORE_URL}/studies/{studyuid}/series/{seriesuid}/instances/{instanceuid}/metadata',
            headers={'accept': 'application/dicom+json; charset=utf-8'},
        )
        self.assertEqual(response_1.status_code, http.HTTPStatus.NOT_FOUND)
        mk[_MOCK_STORE_URL].assert_empty(self)

  def test_file_system_storage_load_instance_empty_raises(self):
    storage_mock = dicom_store_mock._MockDicomStoreFileSystemStorage(
        self.create_tempdir().full_path
    )
    with self.assertRaises(dicom_store_mock._UnableToReadDicomInstanceError):
      storage_mock._load_instance('')

  def test_file_system_storage_will_not_remove_files_not_in_base_dir(self):
    base_path = self.create_tempdir().full_path
    storage_mock = dicom_store_mock._MockDicomStoreFileSystemStorage(base_path)
    path = os.path.join(self.create_tempdir(), 'foo.dcm')
    with open(path, 'wt') as outfile:
      outfile.write('bad')
    storage_mock._remove_file_path(path)
    self.assertTrue(os.path.exists(path))

  def test_file_system_storage_remove_files_ignores_missing_dirs(self):
    base_path = self.create_tempdir().full_path
    storage_mock = dicom_store_mock._MockDicomStoreFileSystemStorage(base_path)
    path = os.path.join(base_path, 'baddir/foo.dcm')
    storage_mock._remove_file_path(path)
    self.assertTrue(os.path.exists(base_path))

  def test_store_instances_moved_between_memory_and_disk(self):
    source_file = _test_file_path()
    headers = {'Content-Type': 'application/dicom'}

    with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL) as mk:
      mk_store = mk.get(_MOCK_STORE_URL)
      mk_store.set_dicom_store_disk_storage(self.create_tempdir())
      with open(source_file, 'rb') as data:
        requests.post(f'{_MOCK_STORE_URL}/studies', data=data, headers=headers)
      with pydicom.dcmread(source_file) as dcm:
        mk_store.assert_uid_in_store(self, [dcm])
        mk_store.set_dicom_store_memory_storage()
        mk_store.assert_uid_in_store(self, [dcm])
        mk_store.set_dicom_store_memory_storage()
        mk_store.assert_uid_in_store(self, [dcm])
        temp_dir = self.create_tempdir()
        mk_store.set_dicom_store_disk_storage(temp_dir)
        mk_store.assert_uid_in_store(self, [dcm])
        mk_store.set_dicom_store_disk_storage(temp_dir)
        mk_store.assert_uid_in_store(self, [dcm])

  @mock.patch.object(
      dicom_store_mock, 'MockDicomStoreClient', side_effect=ValueError
  )
  def test_unexpected_error_closes_mock_and_raises(self, unused_mock):
    with self.assertRaises(ValueError):
      with dicom_store_mock.MockDicomStores(_MOCK_STORE_URL):
        pass


if __name__ == '__main__':
  absltest.main()
