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
"""Tests for dicom store."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_store
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import google.auth

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


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
        credential_factory_module.CredentialFactory(),
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
    self.assertEqual(12.0, slide.native_pixel_spacing.spacing_diff_tolerance)
    self.assertEqual(dicom_test_utils.TEST_STUDY_UID_1, slide.path.study_uid)
    self.assertEqual(dicom_test_utils.TEST_SERIES_UID_1, slide.path.series_uid)

  @parameterized.named_parameters([
      dict(testcase_name='default_factory', credential_factory=None),
      dict(
          testcase_name='custom_factory',
          credential_factory=credential_factory_module.CredentialFactory(),
      ),
  ])
  def test_dicom_store_credential_factory_init(self, credential_factory):
    with mock.patch.object(
        google.auth,
        'default',
        autospec=True,
        return_value=(
            mock.create_autospec(google.auth.credentials.Credentials),
            None,
        ),
    ):
      store = dicom_store.DicomStore(
          dicom_test_utils.TEST_STORE_PATH,
          True,
          credential_factory,
          pixel_spacing_diff_tolerance=12.0,
      )
    self.assertIsInstance(
        store.dicomweb.credential_factory,
        credential_factory_module.CredentialFactory,
    )

  @parameterized.named_parameters([
      dict(testcase_name='default_factory', logging_factory=None),
      dict(
          testcase_name='custom_factory',
          logging_factory=ez_wsi_logging_factory.BasePythonLoggerFactory(),
      ),
  ])
  def test_dicom_store_loggingl_factory_init(self, logging_factory):
    with mock.patch.object(
        google.auth,
        'default',
        autospec=True,
        return_value=(
            mock.create_autospec(google.auth.credentials.Credentials),
            None,
        ),
    ):
      store = dicom_store.DicomStore(
          dicom_test_utils.TEST_STORE_PATH,
          True,
          credential_factory_module.CredentialFactory(),
          pixel_spacing_diff_tolerance=12.0,
          logging_factory=logging_factory,
      )
    self.assertIsInstance(
        store._logging_factory,
        ez_wsi_logging_factory.BasePythonLoggerFactory,
    )

  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(
          mock.create_autospec(google.auth.credentials.Credentials),
          None,
      ),
  )
  def test_dicom_store_cache_initialization(self, unused_mock_auth):
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        pixel_spacing_diff_tolerance=12.0,
    )
    self.assertIsNone(store.slide_frame_cache)
    val = store.init_slide_frame_cache(
        max_cache_frame_memory_lru_cache_size_bytes=1000000000
    )
    self.assertIsNotNone(store.slide_frame_cache)
    self.assertIs(val, store.slide_frame_cache)
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        credential_factory_module.CredentialFactory(),
        pixel_spacing_diff_tolerance=12.0,
    )
    store.dicomweb = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.sample_instances_path()
    )
    slide = store.get_slide(
        dicom_test_utils.TEST_STUDY_UID_1, dicom_test_utils.TEST_SERIES_UID_1
    )
    self.assertIs(slide.slide_frame_cache, store.slide_frame_cache)

  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(
          mock.create_autospec(google.auth.credentials.Credentials),
          None,
      ),
  )
  def test_dicom_store_cache_constructor_initialization(self, unused_mock_auth):
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        pixel_spacing_diff_tolerance=12.0,
        slide_frame_cache=mock.MagicMock(),
    )
    self.assertIsNotNone(store.slide_frame_cache)

  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(
          mock.create_autospec(google.auth.credentials.Credentials),
          None,
      ),
  )
  def test_remove_dicom_store_cache(self, unused_mock_auth):
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        pixel_spacing_diff_tolerance=12.0,
        slide_frame_cache=mock.MagicMock(),
    )
    store.remove_slide_frame_cache()
    self.assertIsNone(store.slide_frame_cache)

  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(
          mock.create_autospec(google.auth.credentials.Credentials),
          None,
      ),
  )
  def test_dicom_store_cache_setter(self, unused_mock_auth):
    store = dicom_store.DicomStore(
        dicom_test_utils.TEST_STORE_PATH,
        True,
        pixel_spacing_diff_tolerance=12.0,
    )
    val = mock.MagicMock()
    store.slide_frame_cache = val
    self.assertIs(store.slide_frame_cache, val)

  def test_get_slide_by_accession(self):
    accession_number = 'ABC123-acaf32'
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    study_uid = '1.2'
    series_uid = f'{study_uid}.3'
    instance_uid = f'{series_uid}.4'
    study_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        study_uid, series_uid, instance_uid, accession_number
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_store.DicomStore(dicomstore_path=study_path)
      slide = ds.get_slide_by_accession_number(accession_number)
    self.assertEqual(slide.path.study_uid, study_uid)
    self.assertEqual(slide.path.series_uid, series_uid)


if __name__ == '__main__':
  absltest.main()
