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
"""Test Utils for dicom store utils."""
import io
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_store_utils
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.test_utils import dicom_test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
import pydicom
import requests


class DicomStoreUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'define_accept_header',
          {'accept': 'application/dicom; transfer-syntax=*'},
      ),
      (
          'default_headers',
          None,
      ),
  ])
  def test_download_instance_untranscoded_succeeds(self, headers):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = (
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    dicom_store_path = (
        f'{dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      with io.BytesIO() as downloaded_instance:
        self.assertTrue(
            dicom_store_utils.download_instance_untranscoded(
                instance_path, downloaded_instance, headers=headers
            )
        )
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
    instance_path = (
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    dicom_store_path = (
        f'{dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path):
      with io.BytesIO() as downloaded_instance:
        self.assertFalse(
            dicom_store_utils.download_instance_untranscoded(
                instance_path, downloaded_instance
            )
        )

  @mock.patch.object(requests.Session, 'get', side_effect=ValueError)
  def test_request_get_called_with_expected_accept_header(self, mock_get):
    instance_path = 'foo'
    query = f'{dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL}/{instance_path}'
    with io.BytesIO() as downloaded_instance:
      try:
        dicom_store_utils.download_instance_untranscoded(
            instance_path,
            downloaded_instance,
            headers={'accept': 'bar', 'google': 'test'},
        )
      except ValueError:
        # Throwing Value Error purposefully when mocked function is called.
        # To avoid further function execution. Test checks that request.get
        # is called with expected parameters.
        pass

    mock_get.assert_called_once_with(
        query,
        headers={
            'accept': 'application/dicom; transfer-syntax=*',
            'google': 'test',
        },
        stream=True,
    )

  def test_download_instance_frame_list_untranscoded(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    instance_path = (
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    number_of_frames = test_instance.NumberOfFrames
    dicom_store_path = (
        f'{dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      frames = dicom_store_utils.download_instance_frame_list_untranscoded(
          instance_path,
          range(1, number_of_frames + 1, 2),
          headers={'accept': 'application/dicom; transfer-syntax=*'},
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
    instance_path = (
        f'{dicom_store_path_instance_path}/'
        f'studies/{test_instance.StudyInstanceUID}/'
        f'series/{test_instance.SeriesInstanceUID}/'
        f'instances/{test_instance.SOPInstanceUID}'
    )
    number_of_frames = test_instance.NumberOfFrames
    dicom_store_path = (
        f'{dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      frames = dicom_store_utils.download_instance_frames_untranscoded(
          instance_path,
          1,
          number_of_frames,
          headers={'accept': 'application/dicom; transfer-syntax=*'},
      )
    self.assertLen(frames, number_of_frames)
    for index, frame in enumerate(frames):
      self.assertEqual(
          dicom_test_utils.test_dicom_instance_frame_bytes(index + 1), frame
      )


if __name__ == '__main__':
  absltest.main()
