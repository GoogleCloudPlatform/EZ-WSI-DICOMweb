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
"""Place for common constants and methods accross tests for EZ WSI DicomWeb."""
import json
from unittest import mock

from ez_wsi_dicomweb import dicom_web_interface

from hcls_imaging_ml_toolkit import dicom_path

# The path to the testing DICOM series data.
TEST_SLIDE_UID_1 = '1.22.333.101:1.22.333.201'
TEST_STORE_PATH = (
    'projects/project_name/locations/us-west1/datasets/'
    'dataset_name/dicomStores/dicom_store_name'
)
# The DICOMStore path to a testing series.
TEST_DICOM_SERIES = (
    TEST_STORE_PATH + '/dicomWeb/studies/1.22.333.101/series/1.22.333.201'
)

TEST_DICOM_SERIES_2 = (
    TEST_STORE_PATH + '/dicomWeb/studies/1.22.333.101/series/1.22.333.999'
)
# Test DICOM object UIDs.
TEST_STUDY_UID_1 = '1.22.333.101'
TEST_STUDY_UID_2 = '1.22.333.102'
TEST_SERIES_UID_1 = '1.22.333.201'
TEST_SERIES_UID_2 = '1.22.333.202'
TEST_INSTANCE_UID_1 = '1.22.333.301'
TEST_INSTANCE_UID_2 = '1.22.333.302'
# The path to the testing DICOM series data.
TEST_DATA_PATH = 'ez_wsi_dicomweb/testdata'
SAMPLE_INSTANCES_PATH = TEST_DATA_PATH + '/sample_instances.json'
INSTANCE_CONCATENATION_TEST_DATA_PATH = (
    TEST_DATA_PATH + '/concatenation_instance_test.json'
)
TEST_JPEG_PATH = TEST_DATA_PATH + '/google.jpg'
TEST_DICOM_STORE_PATH = TEST_DATA_PATH + '/test.dcm'


def create_mock_dicom_web_interface(test_data_path: str):
  """Helper function to create a mocked DicomWebInterface object."""
  with open(test_data_path, 'rt') as json_file:
    data = json.load(json_file)
    data = data['test_data']
    dicom_objects = [
        dicom_web_interface.DicomObject(
            dicom_path.FromString(x['path']), x['dicom_tags']
        )
        for x in data
    ]
  mock_dwi = mock.create_autospec(dicom_web_interface.DicomWebInterface)
  mock_dwi.get_instances.return_value = dicom_objects
  return mock_dwi
