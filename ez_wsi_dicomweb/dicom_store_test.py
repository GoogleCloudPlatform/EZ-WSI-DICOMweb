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
"""Tests for ez_wsi_dicomweb.dicom_store."""
from absl.testing import absltest
from absl.testing import parameterized

from ez_wsi_dicomweb import dicom_store
from ez_wsi_dicomweb import dicom_test_utils


class DicomStoreTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Load DICOM objects for the testing slide.
    self.mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.SAMPLE_INSTANCES_PATH
    )

  def test_get_slide(self):
    slide = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH, True, self.mock_dwi
    ).get_slide(dicom_test_utils.TEST_SLIDE_UID_1)
    self.assertEqual('project_name', slide.path.project_id)
    self.assertEqual('us-west1', slide.path.location)
    self.assertEqual(
        dicom_test_utils.TEST_SLIDE_UID_1.split(':')[0], slide.path.study_uid
    )


if __name__ == '__main__':
  absltest.main()
