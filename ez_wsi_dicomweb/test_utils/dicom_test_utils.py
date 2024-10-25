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
"""Place for common constants and methods across tests for EZ WSI DicomWeb."""
import json
import os
from unittest import mock

from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import dicomweb_credential_factory
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import pydicom

_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])

DEFAULT_DICOMWEB_BASE_URL = 'https://healthcare.googleapis.com/v1'

# The DICOMStore path to a testing series.
TEST_STORE_PATH = (
    'projects/project_name/locations/us-west1/datasets/'
    'dataset_name/dicomStores/dicom_store_name'
)

# Test DICOM object UIDs.
TEST_STUDY_UID_1 = '1.22.333.101'
TEST_STUDY_UID_2 = '1.22.333.102'
TEST_SERIES_UID_1 = '1.22.333.201'
TEST_SERIES_UID_2 = '1.22.333.202'
TEST_INSTANCE_UID_1 = '1.22.333.301'
TEST_INSTANCE_UID_2 = '1.22.333.302'
# The path to the testing DICOM series data.

TEST_DICOM_SERIES = (
    f'{TEST_STORE_PATH}/dicomWeb/studies/{TEST_STUDY_UID_1}'
    f'/series/{TEST_SERIES_UID_1}'
)
TEST_DICOM_SERIES_2 = (
    f'{TEST_STORE_PATH}/dicomWeb/studies/{TEST_STUDY_UID_2}'
    f'/series/{TEST_SERIES_UID_2}'
)


def _test_file_path(*args: str) -> str:
  base_path = [os.path.dirname(os.path.dirname(__file__))]
  base_path.extend(args)
  return os.path.join(*base_path)


def testdata_path(*args: str) -> str:
  return _test_file_path('testdata', *args)


def sample_instances_path() -> str:
  return testdata_path('sample_instances.json')


def instance_concatenation_test_data_path() -> str:
  return testdata_path('concatenation_instance_test.json')


def test_jpeg_path() -> str:
  return testdata_path('google.jpg')


def test_multi_frame_dicom_instance_path() -> str:
  return testdata_path('multiframe_camelyon_challenge_image.dcm')


def test_dicominstance_path() -> str:
  return testdata_path('test.dcm')


def test_dicom_instance_frame_bytes(frame_number: int) -> bytes:
  with open(testdata_path(f'dcm_frame_{frame_number}.jpg'), 'rb') as infile:
    return infile.read()


def create_mock_dicom_web_interface(test_data_path: str):
  """Helper function to create a mocked DicomWebInterface object."""
  with open(test_data_path, 'rt') as json_file:
    data = json.load(json_file)
    data = data['test_data']
    dicom_objects = [
        dicom_web_interface.DicomObject(
            dicom_path.FromString(x['path']), x['dicom_tags'], ''
        )
        for x in data
    ]
  dwi = dicom_web_interface.DicomWebInterface(
      dicomweb_credential_factory.CredentialFactory()
  )
  dwi.get_instances = mock.MagicMock()
  dwi.get_instances.return_value = dicom_objects
  return dwi


def create_test_dicom_instance(
    study_uid: str,
    series_uid: str,
    instance_uid: str,
    accession_number: str = '',
    frame_data: bytes = b'abc123abc123',
    sop_class_uid: str = '1.2.840.10008.5.1.4.1.1.77.1.6',
) -> pydicom.FileDataset:
  """Creates pydicom instance for testing."""
  file_meta = pydicom.dataset.FileMetaDataset()
  file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
  file_meta.MediaStorageSOPClassUID = sop_class_uid
  file_meta.MediaStorageSOPInstanceUID = instance_uid
  file_meta.ImplementationClassUID = '1.2.3'
  test_instance = pydicom.FileDataset(
      '', {}, file_meta=file_meta, preamble=b'\0' * 128
  )
  test_instance.StudyInstanceUID = study_uid
  test_instance.SeriesInstanceUID = series_uid
  test_instance.SOPInstanceUID = instance_uid
  test_instance.SOPClassUID = sop_class_uid
  test_instance.NumberOfFrames = 1
  test_instance.BitsAllocated = 8
  test_instance.InstanceNumber = 1
  test_instance.TotalPixelMatrixColumns = len(frame_data)
  test_instance.Columns = len(frame_data)
  test_instance.TotalPixelMatrixRows = 1
  test_instance.Rows = 1
  test_instance.SamplesPerPixel = 1
  test_instance.ImagedVolumeWidth = 1
  test_instance.ImagedVolumeHeight = 1.0 / len(frame_data)
  test_instance.HighBit = 7
  test_instance.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
  test_instance.PixelData = [frame_data]
  if accession_number:
    test_instance.AccessionNumber = accession_number
  if _PYDICOM_MAJOR_VERSION <= 2:
    test_instance.is_implicit_VR = False
    test_instance.is_little_endian = True
  return test_instance
