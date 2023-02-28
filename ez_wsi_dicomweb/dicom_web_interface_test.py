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
"""Tests for ez_wsi_dicomweb.dicom_web_interface."""
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_test_utils
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
import mock

from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import dicom_web
from hcls_imaging_ml_toolkit import tags


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

  def test_dicom_object_constructor_with_invalid_input_raise_error(self):
    with self.assertRaises(ez_wsi_errors.DicomPathError):
      dicom_web_interface.DicomObject(self._dicom_store_path, {})

  def test_get_studies_with_invalid_input_raise_error(self):
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    with self.subTest(name='Study'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._study_path_1)
    with self.subTest(name='Series'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._series_path_1)
    with self.subTest(name='Instance'):
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_studies(self._instance_path_1)

  def test_get_studies_with_valid_input(self):
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
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_studies_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    studies = dwi.get_studies(self._dicom_store_path)
    self.assertCountEqual(
        studies,
        [
            dicom_web_interface.DicomObject(
                self._study_path_1, all_studies_json[0]
            ),
            dicom_web_interface.DicomObject(
                self._study_path_2, all_studies_json[1]
            ),
        ],
        (
            'All dicom objects returned should have a type of STUDY, with '
            'corresponding dicom path and original dicom tags attached.'
        ),
    )

  def test_get_series_with_invalid_input_raise_error(self):
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    invalid_path = [
        self._series_path_1,
        self._instance_path_1,
    ]
    for path in invalid_path:
      with self.assertRaises(ez_wsi_errors.DicomPathError):
        dwi.get_series(path)

  def test_get_series_with_store_level_path(self):
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
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_series_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    series = dwi.get_series(self._dicom_store_path)
    self.assertCountEqual(
        series,
        [
            dicom_web_interface.DicomObject(
                self._series_path_1, all_series_json[0]
            ),
            dicom_web_interface.DicomObject(
                self._series_path_2, all_series_json[1]
            ),
        ],
        (
            'All DICOM objects returned should have a type of SERIES, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  def test_get_series_with_study_level_path(self):
    all_series_json = [
        {
            '0020000E': {
                'Value': [dicom_test_utils.TEST_SERIES_UID_1],
                'vr': 'UI',
            }
        }
    ]
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_series_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    instances = dwi.get_series(self._study_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._series_path_1, all_series_json[0]
        ),
        (
            'The returned DICOM object should have a type of SERIES, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  def test_get_instances_with_invalid_input_raise_error(self):
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    with self.assertRaises(ez_wsi_errors.DicomPathError):
      dwi.get_instances(self._instance_path_1)

  def test_get_instances_with_store_level_path(self):
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
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_instances_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    instances = dwi.get_instances(self._dicom_store_path)
    self.assertCountEqual(
        instances,
        [
            dicom_web_interface.DicomObject(
                self._instance_path_1, all_instances_json[0]
            ),
            dicom_web_interface.DicomObject(
                self._instance_path_2, all_instances_json[1]
            ),
        ],
        (
            'All DICOM objects returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  def test_get_instances_with_study_level_path(self):
    all_instances_json = [{
        '0020000E': {'Value': [dicom_test_utils.TEST_SERIES_UID_1], 'vr': 'UI'},
        '00080018': {
            'Value': [dicom_test_utils.TEST_INSTANCE_UID_1],
            'vr': 'UI',
        },
    }]
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_instances_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    instances = dwi.get_instances(self._study_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._instance_path_1, all_instances_json[0]
        ),
        (
            'All DICOM objects returned should have a type of INSTANCE, with '
            'corresponding DICOM path and original DICOM tags attached.'
        ),
    )

  def test_get_instances_with_series_level_path(self):
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
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.QidoRs.return_value = all_instances_json
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    instances = dwi.get_instances(self._series_path_1)
    self.assertEqual(
        instances[0],
        dicom_web_interface.DicomObject(
            self._instance_path_1, all_instances_json[0]
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

  def test_get_frame_image_with_invalid_input_raise_error(self):
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
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

  def test_get_frame_image_with_valid_input(self):
    expected_image = b'\x01\x02\x03\x04'
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    mock_dicom_web_client.WadoRs.return_value = expected_image
    dwi = dicom_web_interface.DicomWebInterface(mock_dicom_web_client)
    self.assertEqual(
        expected_image,
        dwi.get_frame_image(
            self._instance_path_1,
            1,
            dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
        ),
    )

  def test_constructor_base_url_override(self):
    fake_base_url = 'fake_base_url'
    mock_dicom_web_client = mock.create_autospec(dicom_web.DicomWebClient)
    dwi = dicom_web_interface.DicomWebInterface(
        mock_dicom_web_client, fake_base_url
    )
    self.assertEqual(dwi._dicom_web_base_url, fake_base_url)


if __name__ == '__main__':
  absltest.main()
