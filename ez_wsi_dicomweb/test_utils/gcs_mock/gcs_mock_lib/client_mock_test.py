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
"""Client Mock Tests."""
import os
from typing import List, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.api_core import exceptions
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import client_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_state_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_test_utils


class ClientMockTest(parameterized.TestCase):
  """google.cloud.storage.Client Mock tests."""

  def _create_list_blobs_mock_test_state(self) -> Mapping[str, str]:
    bucket_name_1 = 'earth'
    test_files_1 = gcs_test_utils.create_mock_test_files(
        self.create_tempdir().full_path,
        [(f'{bucket_name_1}.txt', 10), (f'bar/{bucket_name_1}.txt', 20)],
    )
    bucket_name_2 = 'mars'
    test_files_2 = gcs_test_utils.create_mock_test_files(
        self.create_tempdir().full_path,
        [(f'{bucket_name_2}.txt', 30), (f'bar/{bucket_name_2}.txt', 40)],
    )
    return {bucket_name_1: test_files_1.root, bucket_name_2: test_files_2.root}

  def test_constructor_client_all_params(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client(
          project='fake_project',
          credentials=None,
          _http=None,
          client_info=None,
          client_options={'foo': 'bar'},
      )
      self.assertIsNotNone(cl)

  def test_constructor_client_client_all_params(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client(
          project='fake_project',
          credentials=None,
          _http=None,
          client_info=None,
          client_options={'foo': 'bar'},
      )
      self.assertIsNotNone(cl)

  def test_client_bucket(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()

      bucket = cl.bucket(bucket_name)

      self.assertIs(bucket.name, bucket_name)
      self.assertIs(bucket.client, cl)

  def test_create_anonymous_client(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client.create_anonymous_client()
      self.assertIsInstance(cl, client_mock.ClientMock)

  def test_get_bucket_by_name_exists(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = cl.get_bucket(bucket_name)
      self.assertEqual(bucket.path, f'/b/{bucket_name}')
      self.assertIs(bucket.client, cl)

  def test_get_bucket_from_bucket_exists(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(None, bucket_name)
      bucket = cl.get_bucket(bucket)
      self.assertEqual(bucket.path, f'/b/{bucket_name}')
      self.assertIs(bucket.client, cl)

  def test_get_bucket_by_name_raises_not_found(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      with self.assertRaises((exceptions.Forbidden, exceptions.NotFound)):
        cl.get_bucket('mars')

  def test_client_download_blob_from_uri(self):
    bucket_name = 'earth'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path,
            bucket_name,
            blob_name,
            blob_size,
        )
    ):
      cl = google.cloud.storage.Client()
      path = os.path.join(self.create_tempdir().full_path, 'test_out')
      with open(path, 'wb') as outfile:
        cl.download_blob_to_file(
            f'gs://{bucket_name}/{blob_name}', outfile, raw_download=True
        )
      self.assertEqual(os.path.getsize(path), blob_size)

  def test_client_download_blob_from_blob(self):
    bucket_name = 'earth'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path,
            bucket_name,
            blob_name,
            blob_size,
        )
    ):
      cl = google.cloud.storage.Client()
      path = os.path.join(self.create_tempdir().full_path, 'test_out')
      with open(path, 'wb') as outfile:
        blob = cl.bucket(bucket_name).blob(blob_name)
        cl.download_blob_to_file(blob, outfile, raw_download=True)
      self.assertEqual(os.path.getsize(path), blob_size)

  @parameterized.parameters([([],), (['earth', 'venus', 'mars', 'jupiter'],)])
  def test_list_buckets_succeeds(self, bucket_list: List[str]):
    with gcs_mock.GcsMock(bucket_list):
      cl = google.cloud.storage.Client()
      results = {bucket.name for bucket in cl.list_buckets()}
      self.assertEqual(results, set(bucket_list))

  def test_list_buckets_max_results_succeeds(self):
    bucket_list = ['earth', 'venus', 'mars', 'jupiter']
    with gcs_mock.GcsMock(bucket_list):
      cl = google.cloud.storage.Client()
      results = {bucket.name for bucket in cl.list_buckets(max_results=1)}
      self.assertLen(results, 1)
      self.assertIn(results.pop(), bucket_list)

  def test_list_buckets_prefix_succeeds(self):
    bucket_list = ['earth', 'earth_venus', 'mars', 'jupiter']
    with gcs_mock.GcsMock(bucket_list):
      cl = google.cloud.storage.Client()
      results = {bucket.name for bucket in cl.list_buckets(prefix='earth')}
      self.assertLen(results, 2)
      self.assertEqual({'earth', 'earth_venus'}, results)

  def test_list_buckets_set_page_token_throws(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      with self.assertRaises(gcs_mock_types.GcsMockError):
        list(cl.list_buckets(page_token='foo'))

  @parameterized.parameters(['full', 'noAcl'])
  def test_list_buckets_valid_projection(self, projection: str):
    bucket_list = ['earth', 'venus', 'mars', 'jupiter']
    with gcs_mock.GcsMock(bucket_list):
      cl = google.cloud.storage.Client()
      results = {
          bucket.name for bucket in cl.list_buckets(projection=projection)
      }
      self.assertEqual(results, set(bucket_list))

  def test_list_buckets_invalid_projection_throws(self):
    with gcs_mock.GcsMock(['earth', 'venus', 'mars', 'jupiter']):
      cl = google.cloud.storage.Client()
      with self.assertRaises(gcs_mock_types.GcsMockError):
        list(cl.list_buckets(projection='foo'))

  def test_lookup_bucket_exists(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = cl.lookup_bucket(bucket_name)
      self.assertEqual(bucket.path, f'/b/{bucket_name}')
      self.assertIs(bucket.client, cl)

  def test_lookup_bucket_not_exists(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      # When bucket cannot be found one of two errors can occur in wild.
      # These are simulated in mock to make sure code handles both.
      # 1) If bucket exists and but cannot be accessed (i.e. owned by someone
      # else then a exceptions.Forbidden is raised.
      # 2) If bucket does not exist then lookup returns None
      try:
        self.assertIsNone(cl.lookup_bucket('mars'))
      except exceptions.Forbidden:
        pass

  def test_list_blobs_empty_buckets_succeeds(self):
    blob_name = 'earth'
    with gcs_mock.GcsMock([blob_name]):
      cl = google.cloud.storage.Client()
      self.assertEmpty({blob.name for blob in cl.list_blobs(blob_name)})

  @parameterized.parameters(['full', 'noAcl'])
  def test_list_blobs_by_str_succeeds(self, projection: str):
    with gcs_mock.GcsMock(self._create_list_blobs_mock_test_state()):
      cl = google.cloud.storage.Client()
      results = {
          blob.name for blob in cl.list_blobs('earth', projection=projection)
      }
      self.assertEqual(results, {'bar/earth.txt', 'earth.txt'})

  def test_list_blobs_by_bucket_succeeds(self):
    with gcs_mock.GcsMock(self._create_list_blobs_mock_test_state()):
      cl = google.cloud.storage.Client()
      bucket = cl.bucket('earth')
      results = {blob.name for blob in cl.list_blobs(bucket)}
      self.assertEqual(results, {'bar/earth.txt', 'earth.txt'})

  def test_list_blobs_multiple_buckets_max_results_succeeds(self):
    with gcs_mock.GcsMock(self._create_list_blobs_mock_test_state()):
      cl = google.cloud.storage.Client()
      results = {blob.name for blob in cl.list_blobs('mars', max_results=1)}
      self.assertLen(results, 1)

  def test_list_blobs_prefix_succeeds(self):
    with gcs_mock.GcsMock(self._create_list_blobs_mock_test_state()):
      cl = google.cloud.storage.Client()
      results = {blob.name for blob in cl.list_blobs('earth', prefix='bar')}
      self.assertEqual(results, {'bar/earth.txt'})

  def test_list_blobs_prefix_succeeds_bucket_init_to_dir_on_fs(self):
    bucket_name = 'earth'
    dir_name = 'bar'
    blob_name = 'earth.txt'
    tmp = self.create_tempdir()
    pth = os.path.join(tmp.full_path, dir_name)
    os.mkdir(pth)
    with open(os.path.join(pth, blob_name), 'wt') as outfile:
      outfile.write('hello_world')

    with gcs_mock.GcsMock({bucket_name: tmp.full_path + '/'}):
      cl = google.cloud.storage.Client()
      results = {
          blob.name for blob in cl.list_blobs(bucket_name, prefix=dir_name)
      }
      self.assertEqual(results, {f'{dir_name}/{blob_name}'})

  def test_list_blobs_end_offset_succeeds(self):
    with gcs_mock.GcsMock(self._create_list_blobs_mock_test_state()):
      cl = google.cloud.storage.Client()
      results = {blob.name for blob in cl.list_blobs('earth', end_offset='dar')}
      self.assertEqual(results, {'bar/earth.txt'})

  def test_list_blobs_invalid_projection_throws(self):
    with gcs_mock.GcsMock(['earth']):
      cl = google.cloud.storage.Client()
      with self.assertRaises(gcs_mock_types.GcsMockError):
        list(cl.list_blobs('earth', projection='foo'))

  def test_list_blobs_raises_if_bucket_not_defined(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      with self.assertRaises((exceptions.NotFound, exceptions.Forbidden)):
        list(cl.list_blobs('earth'))

  def test_create_bucket_by_name(self):
    bucket_name = 'mars'
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      cl.create_bucket(bucket_name)
      bucket = cl.lookup_bucket(bucket_name)
      self.assertEqual(bucket.name, bucket_name)

  def test_create_bucket_by_bucket(self):
    bucket_name = 'mars'
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      cl.create_bucket(google.cloud.storage.Bucket(None, bucket_name))
      bucket = cl.lookup_bucket(bucket_name)
      self.assertEqual(bucket.name, bucket_name)

  @parameterized.parameters([True, False])
  def test_setting_requester_pays_raises(self, requester_pays: bool):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      with self.assertRaises(gcs_mock_types.GcsMockError):
        cl.create_bucket(
            google.cloud.storage.Bucket(None, 'foo'),
            requester_pays=requester_pays,
        )


if __name__ == '__main__':
  absltest.main()
