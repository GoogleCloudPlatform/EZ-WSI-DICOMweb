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
"""Blob Mock Tests."""
import datetime
import os
import typing
from typing import Optional, Union

from absl.testing import absltest
from absl.testing import parameterized
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_state_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import bucket_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_constants
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_test_utils


_TEST_BLOB_METHOD_NO_BUCKET_OR_NO_CLIENT_RAISES = [
    ('exists', {}),
    ('update', {}),
    ('reload', {}),
    ('delete', {}),
    ('download_as_bytes', {}),
    ('download_as_text', {}),
    ('upload_from_string', dict(data=b'')),
    ('compose', dict(sources=[])),
    ('rewrite', dict(source=google.cloud.storage.Blob('foo', None))),
]


class BlobMockTest(parameterized.TestCase):

  def test_constructor_blob_all_params(self):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob(
          name='foo',
          bucket=None,
          chunk_size=5,
          encryption_key=b'BADF00D',
          kms_key_name='bar',
          generation=5,
      )
      self.assertIsNotNone(blob)

  def test_constructor_blob_blob_all_params(self):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.blob.Blob(
          name='foo',
          bucket=None,
          chunk_size=5,
          encryption_key=b'BADF00D',
          kms_key_name='bar',
          generation=5,
      )
      self.assertIsNotNone(blob)

  def test_blob_parameters_keyword_name(self):
    blob_name = 'foo'
    bucket = None
    chunksize = 5
    encryption_key = b'BADF00D'
    kms_key_name = 'bar'
    generation = 5
    with gcs_mock.GcsMock():
      blob_by_position = google.cloud.storage.Blob(
          blob_name, bucket, chunksize, encryption_key, kms_key_name, generation
      )
      blob_by_name = google.cloud.storage.Blob(
          name=blob_name,
          bucket=bucket,
          chunk_size=chunksize,
          encryption_key=encryption_key,
          kms_key_name=kms_key_name,
          generation=generation,
      )
      self.assertEqual(blob_by_position.name, blob_by_name.name)
      self.assertEqual(blob_by_position.chunk_size, blob_by_name.chunk_size)
      self.assertEqual(blob_by_position.bucket, blob_by_name.bucket)
      self.assertEqual(blob_by_position.generation, blob_by_name.generation)

  def test_blob_properties(self):
    blob_name = 'foo'
    bucket_name = 'fake_bucket'
    chunksize = 5
    encryption_key = b'BADF00D'
    kms_key_name = 'bar'
    generation = 5
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket(
          google.cloud.storage.Client(), bucket_name
      )
      blob = google.cloud.storage.Blob(
          name=blob_name,
          bucket=bucket,
          chunk_size=chunksize,
          encryption_key=encryption_key,
          kms_key_name=kms_key_name,
          generation=generation,
      )
      self.assertEqual(blob.name, blob_name)
      self.assertEqual(blob.chunk_size, chunksize)
      self.assertIs(blob.bucket, bucket)
      self.assertEqual(blob.generation, generation)
      self.assertEqual(blob.path, f'/b/{bucket_name}/o/{blob_name}')

  def test_invalid_blob_name(self):
    with gcs_mock.GcsMock():
      with self.assertRaises(gcs_mock_types.GcsMockError):
        google.cloud.storage.Blob(
            'a' * (gcs_mock_constants.MAX_BLOB_NAME_LEN + 1), None
        )

  def test_blob_from_string(self):
    with gcs_mock.GcsMock():
      client = google.cloud.storage.Client()
      blob = google.cloud.storage.Blob.from_string(
          'gs://foo/bar/five.txt', client=client
      )
      self.assertEqual(blob.bucket.name, 'foo')
      self.assertEqual(blob.bucket.client, client)
      self.assertEqual(blob.name, 'bar/five.txt')

  def test_blob_from_string_no_client(self):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob.from_string('gs://foo/bar/five.txt')
      self.assertEqual(blob.bucket.name, 'foo')
      self.assertIsNone(blob.bucket.client)
      self.assertEqual(blob.name, 'bar/five.txt')

  def test_blob_from_string_no_gs_uri_prefix(self):
    with gcs_mock.GcsMock():
      with self.assertRaises(ValueError):
        google.cloud.storage.Blob.from_string('foo/bar/five.txt')

  def test_blob_from_string_no_uri(self):
    with gcs_mock.GcsMock():
      with self.assertRaises(IndexError):
        google.cloud.storage.Blob.from_string('gs://')

  def test_blob_from_string_no_blob_name(self):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob.from_string('gs://earth/')
      self.assertEqual(blob.bucket.name, 'earth')
      self.assertIsNone(blob.bucket.client)
      self.assertEqual(blob.name, '')

  def test_blob_with_no_chunk(self):
    mocked_blob_no_chunk = blob_mock.BlobMock('fake', None)
    with self.assertRaises(gcs_mock_types.GcsMockError):
      mocked_blob_no_chunk.path  # pylint: disable=pointless-statement

  @parameterized.named_parameters([
      dict(
          testcase_name='blob_exists',
          test_blob_name='test.txt',
          expected_exists=True,
      ),
      dict(
          testcase_name='blob_does_not_exist',
          test_blob_name='bar.txt',
          expected_exists=False,
      ),
  ])
  def test_blob_exists_true(self, test_blob_name: str, expected_exists: bool):
    bucket_name = 'Earth'
    blob_name = 'test.txt'
    with gcs_mock.GcsMock(
        {
            bucket_name: gcs_test_utils.create_mock_test_files(
                self.create_tempdir().full_path, [(blob_name, 10)]
            ).root
        }
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob(test_blob_name)
      self.assertEqual(blob.exists(), expected_exists)

  def test_blob_client_no_bucket(self):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob('test', None)
      self.assertIsNone(blob.client)

  def test_blob_set_state_and_property_accessors(self):
    """Blob level state on blob and in GcsMock."""
    blob_name = 'test.txt'
    size_in_bytes = 10
    md5_hash = 'ABC123'
    bucket_name = 'foo'
    metadata = {'foo': 'bar'}
    generation = 5
    metageneration = 12
    etag = '45678-9'
    content_type = 'text/plain'
    time_created = datetime.datetime.now()
    time_deleted = datetime.datetime.now()
    updated = datetime.datetime.now()
    component_count = 12
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = typing.cast(
          blob_mock.BlobMock, google.cloud.storage.Blob(blob_name, bucket)
      )
      # Set Blob State
      blob._update_state(
          blob_state_mock.BlobStateMock(
              size_in_bytes,
              md5_hash,
              metadata,
              generation,
              metageneration,
              etag,
              content_type,
              time_created,
              time_deleted,
              updated,
              component_count,
          )
      )
      # Validate Blob State
      self.assertEqual(blob.name, blob_name)
      self.assertEqual(blob.size, size_in_bytes)
      self.assertEqual(blob.md5_hash, md5_hash)
      self.assertIs(blob.bucket, bucket)
      self.assertIs(blob.client, cl)
      self.assertEqual(blob.bucket.name, bucket_name)  # pytype: disable=attribute-error
      self.assertEqual(blob.metadata, metadata)
      self.assertEqual(blob.generation, generation)
      self.assertEqual(blob.metageneration, metageneration)
      self.assertEqual(blob.etag, etag)
      self.assertEqual(blob.content_type, content_type)
      self.assertEqual(blob.time_created, time_created)
      self.assertEqual(blob.time_deleted, time_deleted)
      self.assertEqual(blob.updated, updated)
      self.assertEqual(blob.component_count, component_count)

  def test_calling_blob_update_state_with_none_does_not_change_blob_state(self):
    """Blob level state on blob and in GcsMock."""
    blob_name = 'test.txt'
    size_in_bytes = 10
    md5_hash = 'ABC123'
    bucket_name = 'foo'
    metadata = {'foo': 'bar'}
    generation = 5
    metageneration = 12
    etag = '45678-9'
    content_type = 'text/plain'
    time_created = datetime.datetime.now()
    time_deleted = datetime.datetime.now()
    updated = datetime.datetime.now()
    component_count = 12
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = typing.cast(
          blob_mock.BlobMock, google.cloud.storage.Blob(blob_name, bucket)
      )
      blob._update_state(
          blob_state_mock.BlobStateMock(
              size_in_bytes,
              md5_hash,
              metadata,
              generation,
              metageneration,
              etag,
              content_type,
              time_created,
              time_deleted,
              updated,
              component_count,
          )
      )

      blob._update_state(None)

      # Validate Blob state has not changed after setting None
      self.assertEqual(blob.name, blob_name)
      self.assertEqual(blob.size, size_in_bytes)
      self.assertEqual(blob.md5_hash, md5_hash)
      self.assertIs(blob.bucket, bucket)
      self.assertIs(blob.client, cl)
      self.assertEqual(blob.bucket.name, bucket_name)  # pytype: disable=attribute-error
      self.assertEqual(blob.metadata, metadata)
      self.assertEqual(blob.generation, generation)
      self.assertEqual(blob.metageneration, metageneration)
      self.assertEqual(blob.etag, etag)
      self.assertEqual(blob.content_type, content_type)
      self.assertEqual(blob.time_created, time_created)
      self.assertEqual(blob.time_deleted, time_deleted)
      self.assertEqual(blob.updated, updated)
      self.assertEqual(blob.component_count, component_count)

  def test_blob_content_type_setter(self):
    content_type = 'foo/bar'
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob('test', None)
      blob.content_type = content_type
      self.assertEqual(blob.content_type, content_type)

  def test_blob_metadata_setter(self):
    metadata = {'abc': '123'}
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob('test', None)
      blob.metadata = metadata
      self.assertEqual(blob.metadata, metadata)
      self.assertIsNot(blob.metadata, metadata)

  @parameterized.parameters(_TEST_BLOB_METHOD_NO_BUCKET_OR_NO_CLIENT_RAISES)
  def test_blob_method_no_bucket_raises(self, method: str, kwargs):
    with gcs_mock.GcsMock():
      blob = google.cloud.storage.Blob('test', None)
      with self.assertRaises(AttributeError):
        getattr(blob, method)(**kwargs)

  @parameterized.parameters(_TEST_BLOB_METHOD_NO_BUCKET_OR_NO_CLIENT_RAISES)
  def test_blob_method_no_client_raises(self, method: str, kwargs):
    with gcs_mock.GcsMock():
      bucket = google.cloud.storage.Bucket(None, 'foo')
      blob = google.cloud.storage.Blob('test', bucket)
      with self.assertRaises(AttributeError):
        getattr(blob, method)(**kwargs)

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', op_param_cl=False),
      dict(testcase_name='method_client_param', op_param_cl=True),
  ])
  def test_blob_update_success(self, op_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name
        )
    ):
      bucket_client = None if op_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if op_param_cl else None
      )

      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = bucket.blob(blob_name)
      blob.metadata = {'foo': 'bar'}
      blob.update(method_client_param)

      self.assertNotEqual(
          bucket_client is not None, method_client_param is not None
      )
      self.assertEqual(blob.generation, 1)
      self.assertEqual(blob.metageneration, 2)
      self.assertEqual(blob.metadata, {'foo': 'bar'})

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_blob_reload_success(self, method_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )

      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      # Locally set metadata replaced with what is on server
      blob.metadata = {'foo': 'bar'}

      blob.reload(method_client_param)
      self.assertEqual(blob.size, blob_size)
      # MD5 hash value generated of blob test file generate  with
      # create_mock_test_files.
      self.assertNotEqual(
          bucket_client is not None, method_client_param is not None
      )
      self.assertEqual(blob.md5_hash, 'P8JtyQqyluQbxfyPmip9QA==')
      self.assertEqual(blob.generation, 1)
      self.assertEqual(blob.metageneration, 1)
      self.assertEqual(blob.metadata, {})

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_blob_delete_success(self, method_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )

      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      blob.delete(method_client_param)
      self.assertNotEqual(
          bucket_client is not None, method_client_param is not None
      )
      self.assertFalse(blob.exists(method_client_param))

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_download_as_bytes_success(self, method_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      data = blob.download_as_bytes(method_client_param, raw_download=True)
      self.assertEqual(data, b'**********')
      self.assertIsNone(blob.size)
      self.assertIsNotNone(blob.md5_hash)

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_download_as_text(self, method_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      data = blob.download_as_text(method_client_param, raw_download=True)
      self.assertEqual(data, '**********')
      self.assertIsNone(blob.size)
      self.assertIsNotNone(blob.md5_hash)

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_download_to_filename(self, method_param_cl: bool):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      path = os.path.join(self.create_tempdir().full_path, 'foo.txt')
      blob.download_to_filename(path, method_client_param, raw_download=True)
      with open(path, 'rb') as infile:
        blob_data = infile.read()
      self.assertEqual(blob_data, b'**********')
      self.assertIsNone(blob.size)
      self.assertIsNotNone(blob.md5_hash)

  def test_deprecated_download_as_string_raise(self):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      blob = google.cloud.storage.Client().bucket(bucket_name).blob(blob_name)
      with self.assertRaises(gcs_mock_types.GcsMockError):
        blob.download_as_string()

  def test_deprecated_download_to_file_raise(self):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        gcs_test_utils.create_single_file_mock(
            self.create_tempdir().full_path, bucket_name, blob_name, blob_size
        )
    ):
      blob = google.cloud.storage.Client().bucket(bucket_name).blob(blob_name)
      with self.assertRaises(gcs_mock_types.GcsMockError):
        with open(
            os.path.join(self.create_tempdir().full_path, 'foo.txt'), 'wb'
        ) as outfile:
          blob.download_to_file(outfile)

  @parameterized.named_parameters([
      dict(
          testcase_name='bucket_client_param_and_byte_input',
          data=b'Hello',
          method_param_cl=False,
      ),
      dict(
          testcase_name='method_client_param_and_str_input',
          data='Hello',
          method_param_cl=True,
      ),
  ])
  def test_upload_from_string_success(
      self, data: Union[str, bytes], method_param_cl: bool
  ):
    start_time = datetime.datetime.now()
    bucket_name = 'foo'
    blob_name = 'test.txt'
    bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock({bucket_name: bucket_path}):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)

      blob.upload_from_string(data, client=method_client_param)

      expected_bytes = data.encode('utf-8') if isinstance(data, str) else data
      with open(os.path.join(bucket_path, blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertIsNotNone(blob.md5_hash)
      self.assertIsNotNone(blob.etag)
      self.assertEqual(blob.generation, 1)
      self.assertEqual(blob.metageneration, 1)
      self.assertGreaterEqual((blob.time_created - start_time).microseconds, 0)
      self.assertIsNone(blob.time_deleted)
      self.assertIsNone(blob.updated)
      self.assertEqual(blob.component_count, 1)
      self.assertEqual(blob.content_type, 'text/plain')

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_compose_success(self, method_param_cl: bool):
    start_time = datetime.datetime.now()
    bucket_name = 'foo'
    blob_name = 'test.txt'
    source_blob_name_1 = 'source_1.txt'
    source_blob_name_2 = 'source_2.txt'
    expected_bytes = b'Hello World!'
    bucket_path = self.create_tempdir().full_path
    with open(os.path.join(bucket_path, source_blob_name_1), 'wb') as outfile:
      outfile.write(expected_bytes[:6])
    with open(os.path.join(bucket_path, source_blob_name_2), 'wb') as outfile:
      outfile.write(expected_bytes[6:])

    with gcs_mock.GcsMock({bucket_name: bucket_path}):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      bucket = google.cloud.storage.Bucket(bucket_client, bucket_name)
      blob = bucket.blob(blob_name)
      blob.compose(
          [bucket.blob(source_blob_name_1), bucket.blob(source_blob_name_2)],
          client=method_client_param,
      )
      with open(os.path.join(bucket_path, blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertIsNone(blob.md5_hash)
      self.assertIsNotNone(blob.etag)
      self.assertEqual(blob.generation, 1)
      self.assertEqual(blob.metageneration, 1)
      self.assertGreaterEqual((blob.time_created - start_time).microseconds, 0)
      self.assertIsNone(blob.time_deleted)
      self.assertIsNone(blob.updated)
      self.assertEqual(blob.component_count, 2)
      self.assertEqual(blob.content_type, 'application/octet-stream')

  @parameterized.named_parameters([
      dict(testcase_name='bucket_client_param', method_param_cl=False),
      dict(testcase_name='method_client_param', method_param_cl=True),
  ])
  def test_rewrite_success(self, method_param_cl: bool):
    start_time = datetime.datetime.now()
    source_bucket_name = 'foo'
    source_blob_name = 'test.txt'
    source_bucket_path = self.create_tempdir().full_path
    expected_bytes = b'Hello_world!'
    with open(
        os.path.join(source_bucket_path, source_blob_name), 'wb'
    ) as outfile:
      outfile.write(expected_bytes)

    dest_bucket_name = 'bar'
    dest_blob_name = 'test2.txt'
    dest_bucket_path = self.create_tempdir().full_path

    with gcs_mock.GcsMock({
        source_bucket_name: source_bucket_path,
        dest_bucket_name: dest_bucket_path,
    }):
      bucket_client = None if method_param_cl else google.cloud.storage.Client()
      method_client_param = (
          google.cloud.storage.Client() if method_param_cl else None
      )
      source_bucket = google.cloud.storage.Bucket(None, source_bucket_name)
      source_blob = google.cloud.storage.Blob(source_blob_name, source_bucket)
      source_blob.content_type = 'text/plain'

      bucket = google.cloud.storage.Bucket(bucket_client, dest_bucket_name)
      blob = google.cloud.storage.Blob(dest_blob_name, bucket)

      token = None
      while True:
        token, len_bytes_written, file_size = blob.rewrite(
            source=source_blob, token=token, client=method_client_param
        )
        self.assertLen(expected_bytes, file_size)
        self.assertGreaterEqual(file_size, len_bytes_written)
        if token is None:
          break

      with open(os.path.join(dest_bucket_path, dest_blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertIsNotNone(blob.md5_hash)
      self.assertIsNotNone(blob.etag)
      self.assertEqual(blob.generation, 1)
      self.assertEqual(blob.metageneration, 1)
      self.assertGreaterEqual((blob.time_created - start_time).microseconds, 0)
      self.assertIsNone(blob.time_deleted)
      self.assertIsNone(blob.updated)
      self.assertEqual(blob.component_count, 1)
      self.assertEqual(blob.content_type, 'text/plain')

  @parameterized.named_parameters([
      dict(
          testcase_name='default_content_type',
          upload_file_ext='dat',
          upload_content_type=None,
          expected_content_type=gcs_mock_constants.DEFAULT_CONTENT_TYPE,
      ),
      dict(
          testcase_name='content_byte_determined_from_file_name',
          upload_file_ext='txt',
          upload_content_type=None,
          expected_content_type='text/plain',
      ),
      dict(
          testcase_name='explicit_content_type',
          upload_file_ext='txt',
          upload_content_type='hello/world',
          expected_content_type='hello/world',
      ),
  ])
  def test_upload_from_filename_content_type(
      self,
      upload_file_ext: str,
      upload_content_type: Optional[str],
      expected_content_type: str,
  ):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    expected_bytes = b'Hello World!'
    temp_input_file_path = os.path.join(
        self.create_tempdir().full_path, f'input_file.{upload_file_ext}'
    )
    with open(temp_input_file_path, 'wb') as outfile:
      outfile.write(expected_bytes)
    bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock({bucket_name: bucket_path}):
      bucket = google.cloud.storage.Client().bucket(bucket_name)
      blob = bucket.blob(blob_name)
      blob.upload_from_filename(
          temp_input_file_path, content_type=upload_content_type
      )

      with open(os.path.join(bucket_path, blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertEqual(blob.content_type, expected_content_type)

  @parameterized.named_parameters([
      dict(
          testcase_name='default_content_type',
          method_content_type=None,
          blob_content_type=None,
          expected_content_type=gcs_mock_constants.DEFAULT_CONTENT_TYPE,
      ),
      dict(
          testcase_name='determined_from_blob',
          method_content_type=None,
          blob_content_type='blob/content',
          expected_content_type='blob/content',
      ),
      dict(
          testcase_name='explicit_content_type',
          method_content_type='plain/text',
          blob_content_type='blob/content',
          expected_content_type='plain/text',
      ),
  ])
  def test_upload_from_file_content_type(
      self,
      method_content_type: Optional[str],
      blob_content_type: Optional[str],
      expected_content_type: str,
  ):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    expected_bytes = b'Hello World!'
    temp_input_file_path = os.path.join(
        self.create_tempdir().full_path, 'input_file.txt'
    )
    with open(temp_input_file_path, 'wb') as outfile:
      outfile.write(expected_bytes)
    with gcs_mock.GcsMock([bucket_name]):
      bucket = google.cloud.storage.Client().bucket(bucket_name)
      blob = bucket.blob(blob_name)
      blob.content_type = blob_content_type
      with open(temp_input_file_path, 'rb') as infile:
        blob.upload_from_file(infile, content_type=method_content_type)

      self.assertLen(expected_bytes, blob.size)
      self.assertEqual(blob.content_type, expected_content_type)

  @parameterized.named_parameters([
      dict(testcase_name='rewind_input', rewind=True),
      dict(testcase_name='read_from_file_position', rewind=False),
  ])
  def test_upload_from_file_rewind(
      self,
      rewind: bool,
  ):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    expected_bytes = b'Hello World!'
    read_ahead = 4
    temp_input_file_path = os.path.join(
        self.create_tempdir().full_path, 'input_file.txt'
    )
    with open(temp_input_file_path, 'wb') as outfile:
      outfile.write(expected_bytes)
    with gcs_mock.GcsMock([bucket_name]):
      bucket = google.cloud.storage.Client().bucket(bucket_name)
      blob = bucket.blob(blob_name)
      with open(temp_input_file_path, 'rb') as infile:
        infile.seek(read_ahead)
        blob.upload_from_file(infile, rewind=rewind)
      expected_size = len(expected_bytes) - (0 if rewind else read_ahead)
      self.assertEqual(expected_size, blob.size)

  @parameterized.named_parameters([
      dict(testcase_name='read_all_bytes', bytes_to_read=None),
      dict(testcase_name='read_byte_count', bytes_to_read=5),
  ])
  def test_upload_from_file_byte_count(
      self,
      bytes_to_read: Optional[int],
  ):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    expected_bytes = b'Hello World!'
    temp_input_file_path = os.path.join(
        self.create_tempdir().full_path, 'input_file.txt'
    )
    with open(temp_input_file_path, 'wb') as outfile:
      outfile.write(expected_bytes)
    with gcs_mock.GcsMock([bucket_name]):
      bucket = google.cloud.storage.Client().bucket(bucket_name)
      blob = bucket.blob(blob_name)
      with open(temp_input_file_path, 'rb') as infile:
        blob.upload_from_file(infile, size=bytes_to_read)
      if bytes_to_read is None:
        expected_size = len(expected_bytes)
      else:
        expected_size = bytes_to_read
      self.assertEqual(blob.size, expected_size)


if __name__ == '__main__':
  absltest.main()
