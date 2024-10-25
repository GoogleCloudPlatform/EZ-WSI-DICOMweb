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
"""Core module for high level GCS mock."""
from __future__ import annotations

import contextlib
import functools
from typing import List, Mapping, Optional, Union
from unittest import mock

from absl.testing import absltest
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import bucket_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import client_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_state_mock

# Forward exception declare to enable users of lib to import.
GcsMockError = gcs_mock_types.GcsMockError


class GcsMock(contextlib.ExitStack):
  """Mocks of GCS within the context manager."""

  def __init__(
      self,
      buckets: Union[
          None,
          List[str],
          Mapping[str, Optional[Union[str, absltest._TempDir]]],
      ] = None,
  ):
    """MockGcs Constructor.

    Args:
      buckets: List of names of mocked buckets or mapping of bucket names to
        directories on file system.  Buckets in list or bucket names mapped to
        None will be mapped to a temporary directory which lives for the
        lifetime of the mock.
    """
    super().__init__()
    if isinstance(buckets, Mapping):
      # If GCS bucket passed path to testing temp dir resolve full path.
      temp_buckets = {}
      for key, value in buckets.items():
        temp_buckets[key] = (
            value.full_path if isinstance(value, absltest._TempDir) else value
        )
      buckets = temp_buckets
    self._buckets = buckets
    self._mock_state = None

  def __enter__(self) -> GcsMock:
    """Context manager entry point."""
    super().__enter__()
    try:
      self._mock_state = gcs_state_mock.GcsStateMock(self._buckets)
      for client_path in [
          'google.cloud.storage.Client',
          'google.cloud.storage.client.Client',
      ]:
        self.enter_context(
            mock.patch(
                client_path,
                autospec=True,
                side_effect=functools.partial(
                    client_mock.ClientMock, self._mock_state
                ),
            )
        )
      for blob_path in [
          'google.cloud.storage.Blob',
          'google.cloud.storage.blob.Blob',
      ]:
        self.enter_context(
            mock.patch(blob_path, autospec=True, side_effect=blob_mock.BlobMock)
        )
      for bucket_path in [
          'google.cloud.storage.Bucket',
          'google.cloud.storage.bucket.Bucket',
      ]:
        self.enter_context(
            mock.patch(
                bucket_path, autospec=True, side_effect=bucket_mock.BucketMock
            )
        )
      self.enter_context(self._mock_state)

      # mock class methods on client, bucket and blob.
      for cls in (
          google.cloud.storage.Client,
          google.cloud.storage.client.Client,
      ):
        self.enter_context(
            mock.patch.object(
                cls,
                'create_anonymous_client',
                side_effect=client_mock.ClientMock.create_anonymous_client,
            )
        )
      for cls in (
          google.cloud.storage.Bucket,
          google.cloud.storage.bucket.Bucket,
      ):
        self.enter_context(
            mock.patch.object(
                cls,
                'from_string',
                side_effect=bucket_mock.BucketMock.from_string,
            )
        )

      for cls in (google.cloud.storage.Blob, google.cloud.storage.blob.Blob):
        self.enter_context(
            mock.patch.object(
                cls, 'from_string', side_effect=blob_mock.BlobMock.from_string
            )
        )
      return self
    except:
      # Exception occurred during context manager entry. Close any opened
      # context managers attached to this class.
      self.close()
      raise

  def __exit__(self, *args, **kwargs):
    super().__exit__(*args, **kwargs)
    self._mock_state = None  # Cleanup state

  def list_bucket_names(self) -> List[str]:
    return self._mock_state.list_bucket_names()  # pytype: disable=attribute-error

  def get_bucket_path(self, bucket_name: str) -> str:
    return self._mock_state.get_bucket_path(bucket_name)  # pytype: disable=attribute-error
