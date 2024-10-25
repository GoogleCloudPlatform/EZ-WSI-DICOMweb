# Copyright 2024 Google LLC
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

from unittest import mock

from absl.testing import absltest
import cachetools
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicomweb_credential_factory
import google.auth
from google.oauth2 import service_account
import requests


def _create_mock_credentials(is_valid: bool):
  mock_credentials = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  type(mock_credentials).valid = mock.PropertyMock(return_value=is_valid)
  return mock_credentials


class DicomwebCredientalFactoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    credential_factory._init_fork_module_state()

  def test_no_auth_credentials(self):
    headers = {}
    token = 'abc'

    fc = credential_factory.NoAuthCredentialsFactory()
    cred = fc.get_credentials()

    cred.refresh(requests.Request())
    cred.before_request(requests.Request(), 'get', 'abc', headers)
    self.assertEqual(headers, {})
    self.assertFalse(cred.expired)
    self.assertTrue(cred.valid)
    cred.apply(headers, token)
    self.assertEqual(headers, {})

  def test_no_auth_credentials_factory_undefined_hash(self):
    fc = credential_factory.NoAuthCredentialsFactory()
    self.assertEqual(fc.credential_source_hash(), '')

  def test_token_pass_through_factory_undefined_hash(self):
    fc = credential_factory.TokenPassthroughCredentialFactory('abc')
    self.assertEqual(fc.credential_source_hash(), '')

  def test_token_pass_through_apply(self):
    headers = {}
    token = 'abc'

    fc = credential_factory.TokenPassthroughCredentialFactory(token)
    cred = fc.get_credentials()

    self.assertFalse(cred.expired)
    self.assertTrue(cred.valid)
    cred.apply(headers, token)
    self.assertEqual(headers, {'authorization': 'Bearer abc'})

  def test_token_pass_through_before_request(self):
    headers = {}
    token = 'efg'

    fc = credential_factory.TokenPassthroughCredentialFactory(token)
    cred = fc.get_credentials()

    self.assertFalse(cred.expired)
    self.assertTrue(cred.valid)
    cred.before_request(requests.Request(), 'get', 'abc', headers)
    self.assertEqual(headers, {'authorization': 'Bearer efg'})

  def test_token_pass_through_refresh_nop(self):
    fc = credential_factory.TokenPassthroughCredentialFactory('efg')
    cred = fc.get_credentials()
    self.assertEqual(cred.token, 'efg')
    cred.refresh(requests.Request())
    self.assertEqual(cred.token, 'efg')

  def test_google_auth_factory_undefined_hash(self):
    fc = credential_factory.GoogleAuthCredentialFactory(
        _create_mock_credentials(True)
    )
    self.assertEqual(fc.credential_source_hash(), '')

  def test_default_factory_hash(self):
    fc = credential_factory.DefaultCredentialFactory()
    self.assertEqual(
        fc.credential_source_hash(), 'application_default_credentials'
    )

  def test_service_account_factory_hash(self):
    fc = credential_factory.ServiceAccountCredentialFactory({'ABC': 123})
    self.assertEqual(
        fc.credential_source_hash(),
        'e8a42b4cba471539197281b21310ffcabf8f96a3f9db422c01f4c75cf1a84e6835eeb4e59a919e7a6f470fa42724bc5bf66bef43b2d67d08dc4f10072d560b3d',
    )

  def test_core_credential_default_factory_hash(self):
    fc = credential_factory.CredentialFactory()
    self.assertEqual(
        fc.credential_source_hash(), 'application_default_credentials'
    )

  def test_core_credential_service_account_factory_hash(self):
    fc = credential_factory.CredentialFactory({'ABC': 123})
    self.assertEqual(
        fc.credential_source_hash(),
        'e8a42b4cba471539197281b21310ffcabf8f96a3f9db422c01f4c75cf1a84e6835eeb4e59a919e7a6f470fa42724bc5bf66bef43b2d67d08dc4f10072d560b3d',
    )

  def test_init_fork_module_state(self):
    credential_factory._cache_tools_lock = None
    credential_factory._credential_factory_cache = None
    credential_factory._init_fork_module_state()
    self.assertIsNotNone(credential_factory._cache_tools_lock)
    self.assertIsInstance(
        credential_factory._credential_factory_cache, cachetools.TTLCache
    )

  def test_refresh_invalid_credentials(self):
    mock_credentials = _create_mock_credentials(False)
    credential_factory.refresh_credentials(mock_credentials)
    mock_credentials.refresh.assert_called_once()
    self.assertEmpty(credential_factory._credential_factory_cache)

  def test_does_not_refresh_valid_credentials(self):
    mock_credentials = _create_mock_credentials(True)
    credential_factory.refresh_credentials(mock_credentials)
    mock_credentials.refresh.assert_not_called()
    self.assertEmpty(credential_factory._credential_factory_cache)

  def test_refresh_invalid_credentials_with_credential_factory_sets_cache(self):
    mock_credentials = _create_mock_credentials(False)
    cf = credential_factory.DefaultCredentialFactory()
    credential_factory.refresh_credentials(mock_credentials, cf)
    mock_credentials.refresh.assert_called_once()
    self.assertIs(
        credential_factory._credential_factory_cache[
            cf.credential_source_hash()
        ],
        mock_credentials,
    )

  def test_credential_factory_does_not_refresh_valid_credentials(self):
    mock_credentials = _create_mock_credentials(True)
    cf = credential_factory.DefaultCredentialFactory()
    credential_factory._credential_factory_cache[
        cf.credential_source_hash()
    ] = mock_credentials
    credential = credential_factory.DefaultCredentialFactory().get_credentials()
    self.assertIs(credential, mock_credentials)
    mock_credentials.refresh.assert_not_called()
    self.assertIs(
        credential_factory._credential_factory_cache[
            cf.credential_source_hash()
        ],
        mock_credentials,
    )

  def test_credential_factory_refreshes_invalid_cache_credentials(self):
    mock_credentials = _create_mock_credentials(False)
    cf = credential_factory.DefaultCredentialFactory()
    credential_factory._credential_factory_cache[
        cf.credential_source_hash()
    ] = mock_credentials
    credential = credential_factory.DefaultCredentialFactory().get_credentials()
    self.assertIs(credential, mock_credentials)
    mock_credentials.refresh.assert_called_once()
    self.assertIs(
        credential_factory._credential_factory_cache[
            cf.credential_source_hash()
        ],
        mock_credentials,
    )

  @mock.patch.object(google.auth, 'default', autospec=True)
  def test_credential_factory_generates_new_credentials_if_not_in_cache(
      self, mock_default
  ):
    mock_credentials = _create_mock_credentials(True)
    mock_default.return_value = (
        mock_credentials,
        'project',
    )
    cf = credential_factory.DefaultCredentialFactory()
    credential = cf.get_credentials()
    self.assertIs(credential, mock_credentials)
    mock_credentials.refresh.assert_not_called()
    self.assertIs(
        credential_factory._credential_factory_cache[
            cf.credential_source_hash()
        ],
        mock_credentials,
    )

  @mock.patch.object(google.auth, 'default', autospec=True)
  def test_credential_factory_get_project(self, mock_default):
    mock_credentials = _create_mock_credentials(True)
    mock_default.return_value = (
        mock_credentials,
        'project',
    )
    self.assertEqual(credential_factory.get_default_gcp_project(), 'project')

  @mock.patch.object(
      service_account.Credentials, 'from_service_account_info', autospec=True
  )
  def test_credential_factory_generates_new_credentials_from_json_if_not_in_cache(
      self, mock_default
  ):
    test_json = self.create_tempfile('test.json', '{"abc": 123}')
    mock_credentials = _create_mock_credentials(True)
    mock_default.return_value = mock_credentials
    cf = credential_factory.ServiceAccountCredentialFactory(test_json)
    credential = cf.get_credentials()
    self.assertIs(credential, mock_credentials)
    mock_credentials.refresh.assert_not_called()
    self.assertIs(
        credential_factory._credential_factory_cache[
            cf.credential_source_hash()
        ],
        mock_credentials,
    )

  @mock.patch.object(credential_factory, 'refresh_credentials', autospec=True)
  def test_legacy_dicomweb_refresh_credentials_pass_through(
      self, mock_refresh_credentials
  ):
    mock_credentials = _create_mock_credentials(True)
    cf = credential_factory.DefaultCredentialFactory()
    dicomweb_credential_factory.refresh_credentials(mock_credentials, cf)
    mock_refresh_credentials.assert_called_once_with(mock_credentials, cf)

  def test_google_auth_credential_factory_get_credentials(self):
    mock_credentials = _create_mock_credentials(True)
    cf = credential_factory.GoogleAuthCredentialFactory(mock_credentials)
    self.assertIs(cf.get_credentials(), mock_credentials)
    mock_credentials.refresh.assert_not_called()

  def test_google_auth_credential_factory_get_credentials_refreshes(self):
    mock_credentials = _create_mock_credentials(False)
    cf = credential_factory.GoogleAuthCredentialFactory(mock_credentials)
    self.assertIs(cf.get_credentials(), mock_credentials)
    mock_credentials.refresh.assert_called_once()


if __name__ == '__main__':
  absltest.main()
