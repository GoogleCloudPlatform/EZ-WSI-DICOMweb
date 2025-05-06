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
"""Test for mock of google cloud storage transfer_manager."""
import tempfile

from absl.testing import absltest
import google.cloud.storage
import google.cloud.storage.transfer_manager

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


class TransferManagerMockTest(absltest.TestCase):

  def test_download_chunks_concurrently(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with open(temp_dir + '/test.txt', 'wt') as outfile:
        outfile.write('test')
      with gcs_mock.GcsMock({'Earth': temp_dir}):
        cl = google.cloud.storage.Client()
        blob = google.cloud.storage.Bucket(cl, name='Earth').blob('test.txt')
        google.cloud.storage.transfer_manager.download_chunks_concurrently(
            blob,
            temp_dir + '/out.txt',
            deadline=60,
            worker_type='thread',
            max_workers=10,
            crc32c_checksum=True,
        )
        with open(temp_dir + '/out.txt', 'rt') as infile:
          self.assertEqual(infile.read(), 'test')

  def test_raises_value_error_if_blob_missing_client(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with open(temp_dir + '/test.txt', 'wt') as outfile:
        outfile.write('test')
      with gcs_mock.GcsMock({'Earth': temp_dir}):
        blob = google.cloud.storage.Bucket(None, name='Earth').blob('test.txt')
        with self.assertRaises(ValueError):
          google.cloud.storage.transfer_manager.download_chunks_concurrently(
              blob,
              temp_dir + '/out.txt',
              deadline=60,
              worker_type='thread',
              max_workers=10,
              crc32c_checksum=True,
          )

  def test_raises_value_error_if_works_less_than_one(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with open(temp_dir + '/test.txt', 'wt') as outfile:
        outfile.write('test')
      with gcs_mock.GcsMock({'Earth': temp_dir}):
        cl = google.cloud.storage.Client()
        blob = google.cloud.storage.Bucket(cl, name='Earth').blob('test.txt')
        with self.assertRaises(ValueError):
          google.cloud.storage.transfer_manager.download_chunks_concurrently(
              blob,
              temp_dir + '/out.txt',
              deadline=60,
              worker_type='thread',
              max_workers=0,
              crc32c_checksum=True,
          )

  def test_raises_value_error_if_bad_worker_type(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with open(temp_dir + '/test.txt', 'wt') as outfile:
        outfile.write('test')
      with gcs_mock.GcsMock({'Earth': temp_dir}):
        cl = google.cloud.storage.Client()
        blob = google.cloud.storage.Bucket(cl, name='Earth').blob('test.txt')
        with self.assertRaises(ValueError):
          google.cloud.storage.transfer_manager.download_chunks_concurrently(
              blob,
              temp_dir + '/out.txt',
              deadline=60,
              worker_type='bad_worker_type',
              max_workers=10,
              crc32c_checksum=True,
          )


if __name__ == '__main__':
  absltest.main()
