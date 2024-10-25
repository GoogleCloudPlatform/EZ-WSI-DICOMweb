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
"""Utility functions for gcs_mock."""
import mimetypes
from typing import Optional, Tuple

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_constants
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types


def get_bucket_blob_from_gs_uri(uri: str) -> Tuple[str, str]:
  """Returns google.cloud.storage.Blob represented by a gs:// formated uri.

  Returned exceptions reproduce actual behavior of:
    google.cloud.storage.Blob.from_string

  Args:
    uri: GS formated URI for blob in GCP.  gs://bucket_name/blob_name

  Returns:
    Tuple[bucket_name, blob_name]

  Raises:
    ValueError: URI does not start with gs:// prefix.
    IndexError: Bucket name cannot be parsed from uri.
  """
  gs_prefix = 'gs://'
  if not uri.startswith(gs_prefix):
    raise ValueError('URI scheme must be gs')
  uri = uri[len(gs_prefix) :]
  if '/' in uri:
    bucket_splitter = uri.index('/')
    bucket_name = uri[:bucket_splitter]
    blob_name = uri[bucket_splitter + 1 :]
  else:
    bucket_name = uri
    blob_name = ''
  if not bucket_name:
    raise IndexError('string index out of range')  # raise acutal error
  return bucket_name, blob_name


def guess_content_type_from_file(filename: Optional[str]) -> str:
  if filename is None:
    return gcs_mock_constants.DEFAULT_CONTENT_TYPE
  content_type, _ = mimetypes.guess_type(filename)
  if content_type is not None:
    return content_type
  return gcs_mock_constants.DEFAULT_CONTENT_TYPE


def get_content_type_from_blob(blob: gcs_mock_types.GcsBlobType) -> str:
  content_type = blob.content_type
  if content_type is not None:
    return content_type
  return gcs_mock_constants.DEFAULT_CONTENT_TYPE
