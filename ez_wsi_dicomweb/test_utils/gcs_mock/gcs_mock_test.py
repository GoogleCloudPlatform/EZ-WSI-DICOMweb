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
"""GCS Mock Tests."""
import os

from absl.testing import absltest
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


class GcsMockTest(absltest.TestCase):

  def test_create_mock_context_mgr_no_buckets(self):
    with gcs_mock.GcsMock() as mk:
      self.assertEmpty(mk.list_bucket_names())

  def test_create_mock_context_mgr_list_buckets(self):
    with gcs_mock.GcsMock(['earth', 'mars']) as mk:
      self.assertEqual(mk.list_bucket_names(), ['earth', 'mars'])
      for bucket_name in mk.list_bucket_names():
        path = mk.get_bucket_path(bucket_name)
        self.assertTrue(os.path.isdir(path))
        self.assertTrue(os.path.exists(path))

  def test_create_mock_context_mgr_dict_buckets(self):
    tmp = self.create_tempdir()
    mars_path = tmp.full_path
    with gcs_mock.GcsMock({'earth': None, 'mars': mars_path}) as mk:
      self.assertEqual(mk.list_bucket_names(), ['earth', 'mars'])
      path = mk.get_bucket_path('earth')
      self.assertTrue(os.path.isdir(path))
      self.assertTrue(os.path.exists(path))
      path = mk.get_bucket_path('mars')
      self.assertTrue(os.path.isdir(path))
      self.assertTrue(os.path.exists(path))
      self.assertEqual(path, mars_path)

  def test_create_mock_context_mgr_dict_buckets_raises_path_does_not_exist(
      self,
  ):
    with self.assertRaisesRegex(
        gcs_mock.GcsMockError, 'Path: ".+" does not exist.'
    ):
      with gcs_mock.GcsMock({'earth': None, 'mars': 'invalid_path'}):
        pass

  def test_create_mock_context_mgr_dict_buckets_raises_invalid_path(self):
    temp_dir = self.create_tempdir()
    path = os.path.join(temp_dir.full_path, 'foo.txt')
    with open(path, 'wt') as outfile:
      outfile.write('exists.')
    with self.assertRaisesRegex(
        gcs_mock.GcsMockError, 'Path: ".+" does not reference a directory.'
    ):
      with gcs_mock.GcsMock({'earth': None, 'mars': path}):
        pass

  def test_context_mgr_returns_instance_of_mock_gcs(self):
    with gcs_mock.GcsMock() as gcs_mk:
      self.assertIsInstance(gcs_mk, gcs_mock.GcsMock)  # pytype: disable=attribute-error

  def test_client_mock_gcp_state_is_context_manages_gcp_state(self):
    with gcs_mock.GcsMock() as mk:
      cl = google.cloud.storage.Client()
      self.assertIs(mk._mock_state, cl.mock_state)  # pytype: disable=attribute-error


if __name__ == '__main__':
  absltest.main()
