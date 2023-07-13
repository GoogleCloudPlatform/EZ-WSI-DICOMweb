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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_store
from ez_wsi_dicomweb import dicomweb_credential_factory
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import google.auth


class DicomStoreTest(parameterized.TestCase):

  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(
          mock.create_autospec(google.auth.credentials.Credentials),
          None,
      ),
  )
  def test_get_slide(self, unused_mock_auth):
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        dicomweb_credential_factory.CredentialFactory(),
        pixel_spacing_diff_tolerance=12.0,
    )
    store.dicomweb = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.sample_instances_path()
    )
    slide = store.get_slide(
        dicom_test_utils.TEST_STUDY_UID_1, dicom_test_utils.TEST_SERIES_UID_1
    )
    self.assertEqual('project_name', slide.path.project_id)
    self.assertEqual('us-west1', slide.path.location)
    self.assertEqual(12.0, slide.native_pixel_spacing._spacing_diff_tolerance)
    self.assertEqual(dicom_test_utils.TEST_STUDY_UID_1, slide.path.study_uid)
    self.assertEqual(dicom_test_utils.TEST_SERIES_UID_1, slide.path.series_uid)


if __name__ == '__main__':
  absltest.main()
