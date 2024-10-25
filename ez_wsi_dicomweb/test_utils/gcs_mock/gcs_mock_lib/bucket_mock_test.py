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
"""Bucket Mock Tests."""
from absl.testing import absltest
from absl.testing import parameterized
from google.api_core import exceptions
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import bucket_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_test_utils


class BucketMockTest(parameterized.TestCase):

  @parameterized.parameters(['abc/', '/abc', '/'])
  def test_invalid_bucket_name_raises(self, bucket_name: str):
    with self.assertRaisesRegex(
        ValueError, 'Bucket names must start and end with a number or letter.'
    ):
      google.cloud.storage.Bucket(client=None, name=bucket_name)

  def test_empty_bucket_name_raises(self):
    with self.assertRaises(IndexError):
      google.cloud.storage.Bucket(client=None, name='')

  def test_constructor_bucket_all_params(self):
    bucket_name = 'fake_bucket'
    user_project = 'project'
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket(
          client=google.cloud.storage.Client(),
          name=bucket_name,
          user_project=user_project,
      )
      self.assertIsNotNone(bucket)

  def test_constructor_bucket_bucket_all_params(self):
    bucket_name = 'fake_bucket'
    user_project = 'project'
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.bucket.Bucket(
          client=google.cloud.storage.Client(),
          name=bucket_name,
          user_project=user_project,
      )
      self.assertIsNotNone(bucket)

  def test_bucket_parameters_keyword_name(self):
    bucket_name = 'fake_bucket'
    user_project = 'project'
    with gcs_mock.GcsMock():
      bucket_by_position = google.cloud.storage.bucket.Bucket(
          None, bucket_name, user_project
      )
      bucket_by_name = google.cloud.storage.bucket.Bucket(
          client=None, name=bucket_name, user_project=user_project
      )
      self.assertEqual(bucket_by_position.name, bucket_by_name.name)
      self.assertEqual(bucket_by_position.client, bucket_by_name.client)
      self.assertEqual(
          bucket_by_position.user_project, bucket_by_name.user_project
      )

  @parameterized.named_parameters(
      [('No_user_project', None), ('user_project', 'project')]
  )
  def test_bucket_properties(self, user_project: str):
    bucket_name = 'fake_bucket'
    with gcs_mock.GcsMock():
      client = google.cloud.storage.Client()
      bucket = google.cloud.storage.bucket.Bucket(
          client, bucket_name, user_project
      )
      self.assertEqual(bucket.name, bucket_name)
      self.assertIs(bucket.client, client)
      self.assertEqual(bucket.user_project, user_project)
      self.assertEqual(bucket.path, f'/b/{bucket_name}')

  def test_get_blob_from_bucket(self):
    bucket_name = 'fake_bucket'
    blob_name = 'fake_blob'
    generation = 4
    with gcs_mock.GcsMock():
      client = google.cloud.storage.Client()
      bucket = google.cloud.storage.bucket.Bucket(client, bucket_name)
      blob = bucket.blob(
          blob_name,
          chunk_size=10,
          encryption_key=b'123',
          kms_key_name='foo',
          generation=generation,
      )
      self.assertEqual(blob.name, blob_name)
      self.assertIs(blob.bucket, bucket)
      self.assertIs(blob.generation, generation)

  def test_exists_true(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      self.assertTrue(cl.bucket(bucket_name).exists())

  def test_exists_false_raises(self):
    with gcs_mock.GcsMock(['earth']):
      cl = google.cloud.storage.Client()
      try:
        found = cl.bucket('mars').exists()
      except exceptions.Forbidden:
        found = False
      self.assertFalse(found)

  def test_bucket_from_string(self):
    with gcs_mock.GcsMock():
      client = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket.from_string(
          'gs://foo/bar/five.txt', client=client
      )
      self.assertEqual(bucket.name, 'foo')
      self.assertEqual(bucket.client, client)

  def test_bucket_from_string_no_client(self):
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket.from_string('gs://foo/bar/five.txt')
      self.assertEqual(bucket.name, 'foo')
      self.assertIsNone(bucket.client)

  def test_bucket_from_string_no_gs_uri_prefix(self):
    with gcs_mock.GcsMock():
      with self.assertRaises(ValueError):
        google.cloud.storage.Bucket.from_string('foo/bar/five.txt')

  def test_bucket_from_string_no_uri(self):
    with gcs_mock.GcsMock():
      with self.assertRaises(IndexError):
        google.cloud.storage.Bucket.from_string('gs://')

  def test_bucket_from_string_no_blob_name(self):
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket.from_string('gs://earth/')
      self.assertEqual(bucket.name, 'earth')
      self.assertIsNone(bucket.client)

  def test_list_blobs(self):
    bucket_name = 'earth'
    blob_name = 'mars.txt'
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name
        )
    ):
      bucket = google.cloud.storage.Bucket.from_string('gs://earth/')
      self.assertEqual(
          [
              blob.name
              for blob in bucket.list_blobs(
                  client=google.cloud.storage.Client()
              )
          ],
          ['mars.txt'],
      )

  def test_reload_no_client_raises(self):
    with gcs_mock.GcsMock(['earth']):
      with self.assertRaises(AttributeError):
        bucket = google.cloud.storage.Bucket(None, 'earth')
        bucket.reload()

  def test_reload_bucket_client(self):
    with gcs_mock.GcsMock(['earth']):
      bucket = google.cloud.storage.Bucket(
          google.cloud.storage.Client(), 'earth'
      )
      bucket.reload()

  @parameterized.parameters(['noAcl', 'full'])
  def test_reload_method_client(self, projection: str):
    with gcs_mock.GcsMock(['earth']):
      bucket = google.cloud.storage.Bucket(None, 'earth')
      bucket.reload(google.cloud.storage.Client(), projection=projection)

  def test_reload_method_bucket_does_not_exist_raises(self):
    with gcs_mock.GcsMock(['mars']):
      with self.assertRaises((exceptions.Forbidden, exceptions.NotFound)):
        bucket = google.cloud.storage.Bucket(None, 'earth')
        bucket.reload(google.cloud.storage.Client())

  def test_reload_method_bucket_invalid_projection_raises(self):
    bucket_name = 'mars'
    with gcs_mock.GcsMock([bucket_name]):
      bucket = google.cloud.storage.Bucket(None, bucket_name)
      with self.assertRaises(gcs_mock_types.GcsMockError):
        bucket.reload(google.cloud.storage.Client(), projection='foo')

  def test_get_blob_no_client_raises(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket(None, bucket_name)
      with self.assertRaises(AttributeError):
        bucket.get_blob('mars.txt')

  def test_get_blob_no_client_returns_none_or_raises(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      bucket = cl.bucket(bucket_name)
      try:
        self.assertIsNone(bucket.get_blob('mars.txt'))
      except exceptions.Forbidden:
        pass

  def test_get_blob_no_blob_returns_none(self):
    bucket_name = 'earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = cl.bucket(bucket_name)
      self.assertIsNone(bucket.get_blob('mars.txt'))

  def test_get_blob_no_blob_generation_returns_none(self):
    bucket_name = 'earth'
    blob_name = 'mars.txt'
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name
        )
    ):
      cl = google.cloud.storage.Client()
      bucket = cl.bucket(bucket_name)
      self.assertIsNone(bucket.get_blob(blob_name, generation=10))

  def test_get_blob_succeeds(self):
    bucket_name = 'earth'
    blob_name = 'mars.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      cl = google.cloud.storage.Client()
      bucket = cl.bucket(bucket_name)
      blob = bucket.get_blob(blob_name)
      self.assertIsNotNone(blob)
      self.assertEqual(blob.name, blob_name)
      self.assertEqual(blob.size, blob_size)

  def test_get_blob_succeeds_param_client_generation_param_succeeds(self):
    bucket_name = 'earth'
    blob_name = 'mars.txt'
    blob_size = 10
    generation = 1
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(None, bucket_name)
      blob = bucket.get_blob(blob_name, client=cl, generation=generation)
      self.assertIsNotNone(blob)
      self.assertEqual(blob.name, blob_name)
      self.assertEqual(blob.size, blob_size)
      self.assertEqual(blob.generation, generation)


if __name__ == '__main__':
  absltest.main()
