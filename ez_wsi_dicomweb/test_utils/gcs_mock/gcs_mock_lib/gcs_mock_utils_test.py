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
"""Tests for gcs_test_utils."""
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_constants
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_utils


class BlobMockTest(parameterized.TestCase):

  @parameterized.named_parameters([
      (
          'full_bucket_blob_path',
          'gs://foo/bar/five.txt',
          'foo',
          'bar/five.txt',
      ),
      ('full_bucket_path_plus_splitter', 'gs://earth/', 'earth', ''),
      ('full_bucket_path_nosplitter', 'gs://earth', 'earth', ''),
  ])
  def test_get_bucket_blob_from_gs_uri(
      self, uri: str, expected_bucket_name: str, expected_blob_name: str
  ):
    bucket_name, blob_name = gcs_mock_utils.get_bucket_blob_from_gs_uri(uri)
    self.assertEqual(bucket_name, expected_bucket_name)
    self.assertEqual(blob_name, expected_blob_name)

  def test_get_bucket_blob_from_gs_uri_no_gs_uri_prefix(self):
    with self.assertRaises(ValueError):
      gcs_mock_utils.get_bucket_blob_from_gs_uri('foo/bar/five.txt')

  def test_get_bucket_blob_from_gs_uri_no_uri(self):
    with self.assertRaises(IndexError):
      gcs_mock_utils.get_bucket_blob_from_gs_uri('gs://')

  @parameterized.parameters([
      ('text.txt', 'text/plain'),
      (None, gcs_mock_constants.DEFAULT_CONTENT_TYPE),
      ('', gcs_mock_constants.DEFAULT_CONTENT_TYPE),
  ])
  def test_guess_content_type_from_file(
      self, filename: Optional[str], expected: str
  ):
    self.assertEqual(
        gcs_mock_utils.guess_content_type_from_file(filename), expected
    )

  @parameterized.parameters([
      ('text/plain', 'text/plain'),
      (None, gcs_mock_constants.DEFAULT_CONTENT_TYPE),
  ])
  def test_guess_content_type_blob(
      self, blob_content_type: Optional[str], expected: str
  ):
    blob = google.cloud.storage.Blob('foo', None)
    blob.content_type = blob_content_type
    self.assertEqual(gcs_mock_utils.get_content_type_from_blob(blob), expected)


if __name__ == '__main__':
  absltest.main()
