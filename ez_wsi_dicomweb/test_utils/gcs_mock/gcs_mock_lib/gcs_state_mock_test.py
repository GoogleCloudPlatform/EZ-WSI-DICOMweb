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
"""Tests for gcs_state_mock."""
import copy
import dataclasses
import datetime
import inspect
import os
from typing import Any, Mapping, Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.api_core import exceptions
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_state_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_test_utils


def _get_client_mock_state(
    cl: google.cloud.storage.Client,
) -> gcs_state_mock.GcsStateMock:
  return cl.mock_state  # pytype: disable=attribute-error


# Test Parameters
# Parameter 1) name of method
# Parameter 2) parameters values to set when method is called unset params
#   uninitalized params set to None unless set by test.
#
# Test call methods with blob and additional test specific parameters.

_BUCKET_DOES_NOT_EXIST_THROW_TEST_PARAMETERS = [
    ('blob_update', {}),
    ('blob_reload', dict(projection='noAcl')),
    ('blob_delete', {}),
    ('blob_download_as_bytes', dict(raw_download=True)),
    ('blob_upload_from_string', dict(data=b'')),
    ('blob_compose', dict(sources=[])),
    (
        'blob_rewrite',
        dict(
            source=google.cloud.storage.Blob(
                'foo', google.cloud.storage.Bucket(None, 'test')
            )
        ),
    ),
]

_BLOB_DOES_NOT_EXIST_THROW_TEST_PARAMETERS = [
    ('blob_update', {}),
    ('blob_reload', dict(projection='noAcl')),
    ('blob_delete', {}),
    ('blob_download_as_bytes', dict(raw_download=True)),
]

_ETAG_TEST_PARAMETERS = [
    ('has_blob', {}),
    ('blob_reload', dict(projection='noAcl')),
    ('blob_download_as_bytes', dict(raw_download=True)),
]

_GENERATION_TEST_PARAMETERS = [
    ('has_blob', {}),
    ('blob_update', {}),
    ('blob_reload', dict(projection='noAcl')),
    ('blob_delete', {}),
    ('blob_download_as_bytes', dict(raw_download=True)),
    ('blob_upload_from_string', dict(data=b'')),
]

_METAGENERATION_TEST_PARAMETERS = [
    ('has_blob', {}),
    ('blob_update', {}),
    ('blob_reload', dict(projection='noAcl')),
    ('blob_delete', {}),
    ('blob_download_as_bytes', dict(raw_download=True)),
    ('blob_upload_from_string', dict(data=b'')),
]


class GcsStateMockTest(parameterized.TestCase):

  def _setup_destination_source_bucket_state(
      self,
      dest_bucket_name: Optional[str] = 'dest_bucket',
      source_bucket_name: Optional[str] = 'source_bucket',
      source_blob_name: Optional[str] = 'test.txt',
      source_blob_data: bytes = b'TestData',
      dest_bucket_path: Optional[str] = None,
  ):
    config = {}
    if dest_bucket_name is not None:
      if dest_bucket_path is None:
        dest_bucket_path = self.create_tempdir().full_path
      config[dest_bucket_name] = dest_bucket_path
    if source_bucket_name is not None:
      if source_bucket_name in config:
        source_bucket_path = dest_bucket_path
      else:
        source_bucket_path = self.create_tempdir().full_path
        config[source_bucket_name] = source_bucket_path
      if source_blob_name is not None:
        source_blob_path = os.path.join(source_bucket_path, source_blob_name)
        with open(source_blob_path, 'wb') as outfile:
          outfile.write(source_blob_data)
    return config

  def test_called_inside_context_manager_throws_enter_before_context_enter(
      self,
  ):
    mk = gcs_state_mock.GcsStateMock(['Earth'])
    with self.assertRaisesRegex(
        gcs_mock_types.GcsMockError,
        'Method can only be called inside entered context manager.',
    ):
      mk.list_bucket_names()

  def test_called_inside_context_manager_throws_enter_after_context_exit(self):
    mk = gcs_state_mock.GcsStateMock()
    with mk:
      pass
    with self.assertRaisesRegex(
        gcs_mock_types.GcsMockError,
        'Method can only be called inside entered context manager.',
    ):
      mk.list_bucket_names()

  def test_called_inside_context_manager_inside_succeeds(self):
    with gcs_mock.GcsMock(['Earth']) as mk:
      self.assertEqual(mk.list_bucket_names(), ['Earth'])

  def test_context_manager_returns_instance_of_mock_gcs_state(self):
    with gcs_mock.GcsMock(['Earth']):
      cl = google.cloud.storage.Client()
      mk = _get_client_mock_state(cl)

      self.assertIsInstance(mk, gcs_state_mock.GcsStateMock)

  def test_context_manager_rentry_throws(self):
    with gcs_mock.GcsMock(['Earth']):
      cl = google.cloud.storage.Client()
      mk = _get_client_mock_state(cl)
    with self.assertRaisesRegex(
        gcs_mock_types.GcsMockError, 'GcsStateMock does not support re-entry.'
    ):
      with mk:
        pass

  def test_has_bucket(self):
    bucket_name = 'Earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      self.assertTrue(bucket.exists())

  def test_has_bucket_short_name_throws_bad_request(self):
    bucket_name = 'E'
    with self.assertRaisesRegex(
        exceptions.BadRequest,
        'Bucket names must be at least 3 characters in length',
    ):
      with gcs_mock.GcsMock([bucket_name]):
        pass

  def test_has_bucket_not_there_name_throws_forbidden(self):
    bucket_name = 'Earth'
    with gcs_mock.GcsMock(['Mars']):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      # When bucket cannot be found one of two errors can occur in wild.
      # These are simulated in mock to make sure code handles both.
      # 1) If bucket exists and but cannot be accessed (i.e. owned by someone
      # else then a exceptions.Forbidden is raised.
      # 2) If bucket does not exist then exists returns False
      try:
        self.assertFalse(bucket.exists())
      except exceptions.Forbidden:
        pass

  @parameterized.parameters([(['earth'],), ({'earth': None},)])
  def test_list_blobs_for_empty_dir_returns_no_blobs(self, bucket_def):
    with gcs_mock.GcsMock(bucket_def):
      cl = google.cloud.storage.Client()
      mk = _get_client_mock_state(cl)
      bucket = cl.bucket('earth')
      self.assertEmpty(mk.list_blobs(bucket))

  def test_list_blobs_for_dir_with_blobs_returns_blobs(self):
    bucket_name = 'earth'
    test_files = gcs_test_utils.create_mock_test_files(
        self.create_tempdir().full_path, [('test.txt', 10), ('bar/foo.txt', 20)]
    )
    with gcs_mock.GcsMock({bucket_name: test_files.root}):
      cl = google.cloud.storage.Client()
      mk = _get_client_mock_state(cl)
      bucket = cl.bucket(bucket_name)
      blob_list = mk.list_blobs(bucket)
      self.assertLen(blob_list, 2)
      self.assertEqual(blob_list, test_files.list_mocked_file_paths())

  def test_initalized_blob_state_from_filesystem(self):
    bucket_name = 'earth'
    test_files = gcs_test_utils.create_mock_test_files(
        self.create_tempdir().full_path, [('test.txt', 10), ('bar/foo.txt', 20)]
    )
    with gcs_mock.GcsMock({bucket_name: test_files.root}):
      cl = google.cloud.storage.Client()
      mk = _get_client_mock_state(cl)
      bucket = cl.bucket(bucket_name)
      blob_list = mk.list_blobs(bucket)
      self.assertLen(blob_list, 2)
      for blob_name in blob_list:
        blob = bucket.blob(blob_name)
        path = mk._get_blob_path(blob)
        blob_state = mk._blob_state[path]
        file_state = test_files.files[blob.name]
        self.assertEqual(blob_state.size_in_bytes, file_state.size)
        self.assertIsNotNone(blob_state.md5_hash)
        self.assertEmpty(blob_state.metadata)
        self.assertEqual(blob_state.generation, 1)
        self.assertEqual(blob_state.metageneration, 1)
        self.assertIsNotNone(blob_state.etag)
        self.assertEqual(blob_state.content_type, 'text/plain')
        self.assertIsNotNone(blob_state.time_created)
        self.assertIsNone(blob_state.time_deleted)
        self.assertIsNone(blob_state.updated)
        self.assertEqual(blob_state.component_count, 1)

  def test_validate_bucket_for_invalide_bucket_throws(self):
    with gcs_state_mock.GcsStateMock() as mk:
      bucket = google.cloud.storage.Bucket(None, 'foo')
      with self.assertRaises((exceptions.Forbidden, exceptions.NotFound)):
        mk._validate_bucket(bucket, gcs_mock_types.HttpMethod.GET)

  def test_get_blob_path(self):
    with gcs_state_mock.GcsStateMock() as mk:
      blob = google.cloud.storage.Blob('foo', None)
      with self.assertRaises(gcs_mock_types.GcsMockError):
        mk._get_blob_path(blob)

  def _get_single_file_mock(
      self, bucket_name: str, blob_name: str
  ) -> Mapping[str, str]:
    return gcs_test_utils.create_single_file_mock(
        self.create_tempdir().full_path, bucket_name, blob_name
    )

  def test_has_blob_true(self):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      self.assertTrue(blob.exists())

  def test_has_blob_no_client_raises(self):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      bucket = google.cloud.storage.Bucket(None, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      with self.assertRaises(AttributeError):
        blob.exists()

  @parameterized.named_parameters(
      [('Has_bucket_not_blob', 'Earth'), ('MissingBucket_and_Blob', 'Mars')]
  )
  def test_has_blob_false(self, test_bucket_name: str):
    bucket_name = 'Earth'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, test_bucket_name)
      blob = google.cloud.storage.Blob('temp.txt', bucket)
      self.assertFalse(blob.exists())

  def test_has_blob_conds_pass(self):
    if_etag_match = 'bar'
    if_generation_match = 1
    if_metageneration_match = 7
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      _get_client_mock_state(cl).set_blob_debug_state(
          blob, if_etag_match, if_generation_match, if_metageneration_match
      )

      self.assertTrue(
          blob.exists(
              if_etag_match=if_etag_match,
              if_etag_not_match='foo',
              if_generation_match=if_generation_match,
              if_generation_not_match=if_generation_match - 1,
              if_metageneration_match=if_metageneration_match,
              if_metageneration_not_match=if_metageneration_match - 1,
          )
      )

  def _call_method(self, cl, method_name, blob, **kwargs):
    kwargs = copy.copy(kwargs)
    test_method = getattr(_get_client_mock_state(cl), method_name)
    sig = inspect.signature(test_method)
    for param_name in list(sig.parameters)[1:]:
      if param_name not in kwargs:
        kwargs[param_name] = None
    ba = sig.bind(blob, **kwargs)
    test_method(*ba.args, **ba.kwargs)

  def _blob_condition(
      self,
      method_name: str,
      pre_condition: Mapping[str, Any],
      exception,
      exception_regex: str,
      **kwargs,
  ):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        {
            bucket_name: gcs_test_utils.create_mock_test_files(
                self.create_tempdir().full_path, [(blob_name, blob_size)]
            ).root
        }
    ):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      pre_kwargs = copy.copy(kwargs)
      pre_kwargs.update(pre_condition)
      with self.assertRaisesRegex(exception, exception_regex):
        self._call_method(cl, method_name, blob, **pre_kwargs)

  @parameterized.parameters(_BUCKET_DOES_NOT_EXIST_THROW_TEST_PARAMETERS)
  def test_blob_method_bucket_does_not_exist_raises(
      self, method_name: str, kwargs
  ):
    bucket_name = 'Earth'
    test_bucket_name = 'Mars'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, test_bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      with self.assertRaises((exceptions.Forbidden, exceptions.NotFound)):
        self._call_method(cl, method_name, blob, **kwargs)

  @parameterized.parameters(_BLOB_DOES_NOT_EXIST_THROW_TEST_PARAMETERS)
  def test_blob_method_blob_does_not_exist_raises(
      self, method_name: str, kwargs
  ):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    test_blob_name = 'temp2.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(test_blob_name, bucket)
      with self.assertRaises(exceptions.NotFound):
        self._call_method(cl, method_name, blob, **kwargs)

  @parameterized.parameters(_ETAG_TEST_PARAMETERS)
  def test_blob_method_etag_pre_condition(self, method_name: str, kwargs):
    self._blob_condition(
        method_name,
        dict(if_etag_match='foo'),
        exceptions.PreconditionFailed,
        '',
        **kwargs,
    )

  @parameterized.parameters(_ETAG_TEST_PARAMETERS)
  def test_blob_method_etag_not_modified(self, method_name: str, kwargs):
    self._blob_condition(
        method_name,
        dict(if_etag_not_match='bar'),
        exceptions.NotModified,
        '',
        **kwargs,
    )

  @parameterized.parameters(_GENERATION_TEST_PARAMETERS)
  def test_blob_method_generation_pre_condition(self, method_name: str, kwargs):
    self._blob_condition(
        method_name,
        dict(if_generation_match=4),
        exceptions.PreconditionFailed,
        '',
        **kwargs,
    )

  @parameterized.parameters(_GENERATION_TEST_PARAMETERS)
  def test_blob_method_generation_not_modified(self, method_name: str, kwargs):
    self._blob_condition(
        method_name,
        dict(if_generation_not_match=1),
        exceptions.NotModified,
        '',
        **kwargs,
    )

  @parameterized.parameters(_METAGENERATION_TEST_PARAMETERS)
  def test_blob_method_metageneration_pre_condition(
      self, method_name: str, kwargs
  ):
    self._blob_condition(
        method_name,
        dict(if_metageneration_match=5),
        exceptions.PreconditionFailed,
        '',
        **kwargs,
    )

  @parameterized.parameters(_METAGENERATION_TEST_PARAMETERS)
  def test_blob_method_metageneration_not_modified(
      self, method_name: str, kwargs
  ):
    self._blob_condition(
        method_name,
        dict(if_metageneration_not_match=7),
        exceptions.NotModified,
        '',
        **kwargs,
    )

  def test_blob_update_succeeds(self):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    blob_gen_init = 1
    blob_metagen_init = 7
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      blob.metadata = {'foo': 'bar'}
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(
          blob, 'bar', blob_gen_init, blob_metagen_init
      )
      mock_state._blob_state[mock_state._get_blob_path(blob)].updated = None

      with mock.patch.object(
          gcs_state_mock, '_gen_etag', return_value='ETAG_GEN'
      ):
        result = mock_state.blob_update(
            blob,
            if_generation_match=blob_gen_init,
            if_generation_not_match=blob_gen_init + 1,
            if_metageneration_match=blob_metagen_init,
            if_metageneration_not_match=blob_metagen_init + 1,
        )

      # internal mock state is as expected
      blob_state = mock_state._blob_state[mock_state._get_blob_path(blob)]
      self.assertEqual(blob_state.metageneration, blob_metagen_init + 1)
      self.assertEqual(blob_state.metadata, blob.metadata)
      self.assertEqual(blob_state.etag, 'ETAG_GEN')
      self.assertIsNotNone(blob_state.updated)
      # internal mock state matches the return blob state
      self.assertEqual(result, blob_state)

  @parameterized.parameters([
      ('foo', "projection must equal 'full' or 'noAcl'"),
      (
          'full',
          'Setting projection to value other than noAcl not supported in mock.',
      ),
  ])
  def test_blob_reload_invalid_projection_raises(
      self, projection: str, exp_msg: str
  ):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    test_blob_name = 'temp2.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(test_blob_name, bucket)
      with self.assertRaisesRegex(gcs_mock_types.GcsMockError, exp_msg):
        _get_client_mock_state(cl).blob_reload(
            blob, projection, None, None, None, None, None, None
        )

  def test_blob_reload_succeeds(
      self,
  ):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    test_start_time = datetime.datetime.now()
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      result = mock_state.blob_reload(
          blob,
          'noAcl',
          if_etag_match='bar',
          if_etag_not_match='foo',
          if_generation_match=1,
          if_generation_not_match=2,
          if_metageneration_match=7,
          if_metageneration_not_match=8,
      )
      # created after test started
      self.assertGreaterEqual(
          (result.time_created - test_start_time).microseconds, 0
      )
      self.assertEqual(
          dataclasses.asdict(result),
          {
              'component_count': 1,
              'content_type': 'text/plain',
              'etag': 'bar',
              'generation': 1,
              'md5_hash': 'P8JtyQqyluQbxfyPmip9QA==',
              'metadata': {},
              'metageneration': 7,
              'size_in_bytes': 10,
              'time_created': result.time_created,
              'time_deleted': None,
              'updated': None,
          },
      )

  def test_blob_delete_succeeds(
      self,
  ):
    bucket_name = 'Earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock(self._get_single_file_mock(bucket_name, blob_name)):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.blob_delete(
          blob,
          if_generation_match=1,
          if_generation_not_match=2,
          if_metageneration_match=7,
          if_metageneration_not_match=8,
      )
      self.assertFalse(blob.exists())

  @parameterized.named_parameters([
      ('Default read whole file', None, None, None, 10),
      ('read whole file', 0, 9, 'md5', 10),
      ('partial read', 1, 8, 'crc32c', 8),
  ])
  def test_blob_download_succeeds(
      self,
      start: Optional[int],
      end: Optional[int],
      checksum: Optional[str],
      expected_size: int,
  ):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    blob_size = 10
    with gcs_mock.GcsMock(
        {
            bucket_name: gcs_test_utils.create_mock_test_files(
                self.create_tempdir().full_path, [(blob_name, blob_size)]
            ).root
        }
    ):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      bytes_read, blob_state = mock_state.blob_download_as_bytes(
          blob,
          start,
          end,
          raw_download=True,
          if_etag_match='bar',
          if_etag_not_match='foo',
          if_generation_match=1,
          if_generation_not_match=2,
          if_metageneration_match=7,
          if_metageneration_not_match=8,
          checksum=checksum,
      )
      self.assertLen(bytes_read, expected_size)
      self.assertEqual(blob_state.size_in_bytes, blob_size)
      self.assertEqual(blob_state.etag, 'bar')
      self.assertEqual(blob_state.generation, 1)
      self.assertEqual(blob_state.metageneration, 7)

  @parameterized.named_parameters([
      dict(
          testcase_name='invalid_checksum',
          test_params=dict(checksum='bar', raw_download=True),
          exception=gcs_mock_types.GcsMockError,
          expected_regex='Invalid checksum',
      ),
      dict(
          testcase_name='invalid_byte_range',
          test_params=dict(start=12, end=15, raw_download=True),
          exception=google.api_core.exceptions.RequestRangeNotSatisfiable,
          expected_regex='',
      ),
  ])
  def test_blob_download_raises(
      self,
      test_params: Mapping[str, Any],
      exception,
      expected_regex: str,
  ):
    self._blob_condition(
        'blob_download_as_bytes', test_params, exception, expected_regex
    )

  def test_init_blob_state_and_inc_generation_new_blob(self):
    start_time = datetime.datetime.now()
    bucket_name = 'earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob(blob_name)
      mock_state = _get_client_mock_state(cl)

      mock_state._init_blob_state_and_inc_generation(blob, 'text/plain')

      blob_path = mock_state._get_blob_path(blob)
      metadata = mock_state._blob_state[blob_path]
    self.assertEqual(
        dataclasses.asdict(metadata),
        {
            'component_count': 1,
            'content_type': 'text/plain',
            'etag': mock.ANY,
            'generation': 1,
            'md5_hash': '1B2M2Y8AsgTpgAmY7PhCfg==',
            'metadata': {},
            'metageneration': 1,
            'size_in_bytes': 0,
            'time_created': mock.ANY,
            'time_deleted': None,
            'updated': None,
        },
    )
    self.assertGreaterEqual(
        (metadata.time_created - start_time).microseconds, 0
    )
    self.assertIsNotNone(metadata.etag)

  def test_init_blob_state_and_inc_generation_existing_blob(self):
    start_time = datetime.datetime.now()
    bucket_name = 'earth'
    blob_name = 'temp.txt'
    with gcs_mock.GcsMock([bucket_name]):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob(blob_name)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 4, 7)

      mock_state._init_blob_state_and_inc_generation(blob, 'text/plain')

      blob_path = mock_state._get_blob_path(blob)
      metadata = mock_state._blob_state[blob_path]
    self.assertEqual(
        dataclasses.asdict(metadata),
        {
            'component_count': 1,
            'content_type': 'text/plain',
            'etag': mock.ANY,
            'generation': 5,
            'md5_hash': '1B2M2Y8AsgTpgAmY7PhCfg==',
            'metadata': {},
            'metageneration': 1,
            'size_in_bytes': 0,
            'time_created': mock.ANY,
            'time_deleted': None,
            'updated': None,
        },
    )
    self.assertGreaterEqual(
        (metadata.time_created - start_time).microseconds, 0
    )
    self.assertIsNotNone(metadata.etag)

  @parameterized.parameters([None, 'md5', 'crc32c'])
  def test_blob_upload_from_string(self, checksum: Optional[str]):
    bucket_name = 'foo'
    blob_name = 'test.txt'
    data = b'TestData'
    bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock({bucket_name: bucket_path}):
      cl = google.cloud.storage.Client()
      bucket = google.cloud.storage.Bucket(cl, bucket_name)
      blob = google.cloud.storage.Blob(blob_name, bucket)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)

      blob.upload_from_string(
          data,
          if_generation_match=1,
          if_generation_not_match=2,
          if_metageneration_match=7,
          if_metageneration_not_match=8,
          checksum=checksum,
      )

      with open(os.path.join(bucket_path, blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), data)

  def test_blob_upload_from_string_invalid_checksum_raises(self):
    test_params = dict(data=b'', checksum='foo')
    exception = gcs_mock_types.GcsMockError
    expected_regex = ''
    self._blob_condition(
        'blob_upload_from_string', test_params, exception, expected_regex
    )

  @parameterized.named_parameters([
      ('dest_generation_match', dict(if_generation_match=3)),
      ('dest_metageneration_match', dict(if_metageneration_match=9)),
      ('source_generation_match', dict(if_source_generation_match=[4, 3])),
  ])
  def test_blob_compose_precondition_raises(self, kargs: Mapping[str, Any]):
    bucket_name = 'bucket'
    source_blob_name = 'source.txt'
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=bucket_name,
            source_bucket_name=bucket_name,
            source_blob_name=source_blob_name,
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob('out.txt')
      source = cl.bucket(bucket_name).blob(source_blob_name)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      with self.assertRaises(exceptions.PreconditionFailed):
        blob.compose([source, source], **kargs)

  @parameterized.named_parameters([
      ('to_few_sources', 0, 'Compose called with empty list.'),
      ('to_many_sources', 33, 'Cannot compose more than 32 blobs.'),
  ])
  def test_blob_compose_invalid_number_of_sources_raises(
      self, source_count: int, regex: str
  ):
    bucket_name = 'bucket'
    source_blob_name = 'source.txt'
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=bucket_name,
            source_bucket_name=bucket_name,
            source_blob_name=source_blob_name,
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob('out.txt')
      source = cl.bucket(bucket_name).blob(source_blob_name)
      with self.assertRaisesRegex(gcs_mock_types.GcsMockError, regex):
        blob.compose([source] * source_count)

  def test_blob_compose_source_and_dest_not_in_same_bucket_raises(self):
    dest_bucket_name = 'dest_bucket'
    source_bucket_name = 'source_bucket'
    source_blob_name = 'source.txt'
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=dest_bucket_name,
            source_bucket_name=source_bucket_name,
            source_blob_name=source_blob_name,
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(dest_bucket_name).blob('out.txt')
      source = cl.bucket(source_bucket_name).blob(source_blob_name)
      with self.assertRaisesRegex(
          gcs_mock_types.GcsMockError,
          'Destination and source blobs must be in the same bucket',
      ):
        blob.compose([source])

  def test_blob_compose_source_list_and_gen_match_not_same_len_raises(self):
    bucket_name = 'dest_bucket'
    source_blob_name = 'source.txt'
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=bucket_name,
            source_bucket_name=bucket_name,
            source_blob_name=source_blob_name,
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob('out.txt')
      source = cl.bucket(bucket_name).blob(source_blob_name)
      with self.assertRaisesRegex(
          gcs_mock_types.GcsMockError,
          r'len\(sources\) != len\(if_source_generation_match\)',
      ):
        blob.compose([source, source], if_source_generation_match=[1])

  def test_blob_compose_source_blob_does_not_exist_raises(self):
    dest_bucket_name = 'dest_bucket'
    source_bucket_name = 'source_bucket'
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=dest_bucket_name,
            source_bucket_name=source_bucket_name,
            source_blob_name='source.txt',
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(dest_bucket_name).blob('out.txt')
      source = cl.bucket(source_bucket_name).blob('foo_bar.txt')
      with self.assertRaises(gcs_mock_types.GcsMockError):
        blob.compose([source, source])

  @parameterized.named_parameters([
      (
          'dest_blob_content_type_set',
          'dest/blob',
          'source/blob',
          'dest/blob',
      ),
      (
          'source_blob_content_type_set',
          None,
          'source/blob',
          'source/blob',
      ),
  ])
  def test_compose_blob_succeeds(
      self,
      dest_content_type: str,
      source_content_type: str,
      expected_content_type: str,
  ):
    dest_blob_name = 'dest.txt'
    source_blob_name = 'source.txt'
    bucket_name = 'dest_bucket'
    source_blob_data = b'test_data'
    dest_bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            dest_bucket_name=bucket_name,
            source_bucket_name=bucket_name,
            source_blob_name=source_blob_name,
            source_blob_data=source_blob_data,
            dest_bucket_path=dest_bucket_path,
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket(bucket_name).blob(dest_blob_name)
      blob.content_type = dest_content_type
      source = cl.bucket(bucket_name).blob(source_blob_name)
      source.content_type = source_content_type
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      blob.compose(
          [source, source],
          if_generation_match=1,
          if_metageneration_match=7,
          if_source_generation_match=[2, 2],
      )
      self.assertLen(source_blob_data * 2, blob.size)
      self.assertEqual(2, blob.component_count)
      self.assertIsNone(blob.md5_hash)
      self.assertEqual(blob.content_type, expected_content_type)
      with open(os.path.join(dest_bucket_path, dest_blob_name), 'rb') as infile:
        self.assertEqual(infile.read(), source_blob_data * 2)

  def test_blob_rewrite_source_bucket_does_not_exist_raises(self):
    with gcs_mock.GcsMock(self._setup_destination_source_bucket_state()):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('error').blob('nope.txt')
      with self.assertRaises((exceptions.Forbidden, exceptions.NotFound)):
        blob.rewrite(source)

  def test_blob_rewrite_source_blob_does_not_exist_raises(self):
    with gcs_mock.GcsMock(self._setup_destination_source_bucket_state()):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('source_bucket').blob('nope.txt')
      with self.assertRaises(exceptions.NotFound):
        blob.rewrite(source)

  @parameterized.named_parameters([
      ('dest_generation_match', dict(if_generation_match=3)),
      ('dest_metageneration_match', dict(if_metageneration_match=9)),
      ('source_generation_match', dict(if_source_generation_match=4)),
      ('source_metageneration_match', dict(if_source_metageneration_match=12)),
  ])
  def test_blob_rewrite_precondition_raises(self, kargs: Mapping[str, Any]):
    with gcs_mock.GcsMock(self._setup_destination_source_bucket_state()):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('source_bucket').blob('test.txt')
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      with self.assertRaises(exceptions.PreconditionFailed):
        blob.rewrite(source, **kargs)

  @parameterized.named_parameters([
      ('dest_generation_not_match', dict(if_generation_not_match=1)),
      ('dest_metageneration_not_match', dict(if_metageneration_not_match=7)),
      ('source_generation_not_match', dict(if_source_generation_not_match=2)),
      (
          'source_metageneration_not_match',
          dict(if_source_metageneration_not_match=8),
      ),
  ])
  def test_blob_rewrite_not_modified_raises(self, kargs: Mapping[str, Any]):
    with gcs_mock.GcsMock(self._setup_destination_source_bucket_state()):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('source_bucket').blob('test.txt')
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      with self.assertRaises(exceptions.NotModified):
        blob.rewrite(source, **kargs)

  def test_blob_rewrite_not_raises_invalid_token(self):
    with gcs_mock.GcsMock(self._setup_destination_source_bucket_state()):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('source_bucket').blob('test.txt')
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 1, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      token = 'invalid_token'
      with self.assertRaises(gcs_mock_types.GcsMockError):
        blob.rewrite(
            source,
            token,
            client=cl,
            if_generation_match=1,
            if_generation_not_match=2,
            if_metageneration_match=7,
            if_metageneration_not_match=8,
            if_source_generation_match=2,
            if_source_generation_not_match=3,
            if_source_metageneration_match=8,
            if_source_metageneration_not_match=9,
            timeout=None,
            retry=None,
        )

  def test_blob_rewrite_succeeds(self):
    expected_bytes = b'TestData'
    start_time = datetime.datetime.now()
    dest_bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            source_blob_data=expected_bytes, dest_bucket_path=dest_bucket_path
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      source = cl.bucket('source_bucket').blob('test.txt')
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 4, 7)
      mock_state.set_blob_debug_state(source, 'foo', 2, 8)
      token = None
      loop_count = 0
      while True:
        token, len_bytes_written, file_size = blob.rewrite(
            source,
            token,
            client=cl,
            if_generation_match=4,
            if_generation_not_match=2,
            if_metageneration_match=7,
            if_metageneration_not_match=8,
            if_source_generation_match=2,
            if_source_generation_not_match=3,
            if_source_metageneration_match=8,
            if_source_metageneration_not_match=9,
            timeout=None,
            retry=None,
        )
        self.assertLen(expected_bytes, file_size)
        self.assertGreaterEqual(file_size, len_bytes_written)
        loop_count += 1
        if token is None:
          break

      self.assertEqual(loop_count, 2)
      with open(os.path.join(dest_bucket_path, blob.name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertIsNotNone(blob.md5_hash)
      self.assertIsNotNone(blob.etag)
      self.assertEqual(blob.generation, 5)
      self.assertEqual(blob.metageneration, 1)
      self.assertGreaterEqual((blob.time_created - start_time).microseconds, 0)
      self.assertIsNone(blob.time_deleted)
      self.assertIsNone(blob.updated)
      self.assertEqual(blob.component_count, 1)
      self.assertEqual(blob.content_type, 'application/octet-stream')

  def test_blob_rewrite_same_blob_succeeds(self):
    expected_bytes = b'TestData'
    start_time = datetime.datetime.now()
    dest_bucket_path = self.create_tempdir().full_path
    with gcs_mock.GcsMock(
        self._setup_destination_source_bucket_state(
            source_blob_data=expected_bytes, dest_bucket_path=dest_bucket_path
        )
    ):
      cl = google.cloud.storage.Client()
      blob = cl.bucket('dest_bucket').blob('out.txt')
      with open(os.path.join(dest_bucket_path, blob.name), 'wb') as outfile:
        outfile.write(expected_bytes)
      mock_state = _get_client_mock_state(cl)
      mock_state.set_blob_debug_state(blob, 'bar', 4, 7)
      token, len_bytes_written, file_size = blob.rewrite(
          blob,
          token=None,
          client=cl,
          if_generation_match=4,
          if_generation_not_match=2,
          if_metageneration_match=7,
          if_metageneration_not_match=8,
          if_source_generation_match=4,
          if_source_generation_not_match=2,
          if_source_metageneration_match=7,
          if_source_metageneration_not_match=8,
          timeout=None,
          retry=None,
      )
      self.assertLen(expected_bytes, file_size)
      self.assertEqual(file_size, len_bytes_written)
      self.assertIsNone(token)
      with open(os.path.join(dest_bucket_path, blob.name), 'rb') as infile:
        self.assertEqual(infile.read(), expected_bytes)
      self.assertLen(expected_bytes, blob.size)
      self.assertIsNotNone(blob.md5_hash)
      self.assertIsNotNone(blob.etag)
      self.assertEqual(blob.generation, 5)
      self.assertEqual(blob.metageneration, 1)
      self.assertGreaterEqual((blob.time_created - start_time).microseconds, 0)
      self.assertIsNone(blob.time_deleted)
      self.assertIsNone(blob.updated)
      self.assertEqual(blob.component_count, 1)
      self.assertEqual(blob.content_type, 'application/octet-stream')

  def test_create_bucket(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      mock_state = _get_client_mock_state(cl)
      bucket = cl.bucket('moon')
      mock_state.create_bucket(bucket, '')
      self.assertIsNone(
          mock_state._validate_bucket(
              bucket, http_method=gcs_mock_types.HttpMethod.GET
          )
      )

  def test_create_bucket_exists_raises(self):
    with gcs_mock.GcsMock():
      cl = google.cloud.storage.Client()
      mock_state = _get_client_mock_state(cl)
      bucket = cl.bucket('moon')
      mock_state.create_bucket(bucket, '')
      with self.assertRaises(exceptions.Conflict):
        mock_state.create_bucket(bucket, '')


if __name__ == '__main__':
  absltest.main()
