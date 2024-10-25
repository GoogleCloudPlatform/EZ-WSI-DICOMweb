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
"""Mock google.cloud.storage.Blob."""
from __future__ import annotations

import copy
import datetime
import typing
from typing import BinaryIO, List, Mapping, Optional, Tuple, Union

import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_state_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import client_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_constants
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_utils


class BlobMock:
  """Mock google.cloud.storage.Blob.

  Do not use class directly. Create class via mocked class constructor.
  google.cloud.storage.Blob (...)
  """

  def __init__(
      self,
      name: str,
      bucket: Optional[gcs_mock_types.GcsBucketType],
      chunk_size: Optional[int] = None,
      encryption_key: Optional[bytes] = None,
      kms_key_name: Optional[str] = None,
      generation: Optional[int] = None,
  ):
    """Mock google.cloud.storage.Blob constructor.

    Args:
      name: Name of the blob.
      bucket: The bucket to which this blob belongs.
      chunk_size: (Optional) The size of a chunk of data whenever iterating (in
        bytes).  Not used in mock.
      encryption_key: (Optional) 32 byte encryption key for customer-supplied
        encryption. Not implemented in mock.
      kms_key_name: (Optional) Resource name of Cloud KMS key used to encrypt
        the blob's contents. Not impelemented in mock.
      generation: (Optional) If present, selects a specific revision of this
        object.

    Raises:
      GcsMockError: Invalid parameter values.
    """
    del encryption_key, kms_key_name
    if len(name.encode('utf-8')) > gcs_mock_constants.MAX_BLOB_NAME_LEN:
      raise gcs_mock_types.GcsMockError('Blob name exceeds length limits.')
    self._name = name
    self._bucket = bucket
    self._chunk_size = chunk_size
    self._blob_state = blob_state_mock.BlobStateMock()
    self._blob_state.generation = generation

  @property
  def content_type(self) -> Optional[str]:
    return self._blob_state.content_type

  @content_type.setter
  def content_type(self, val: Optional[str]):
    self._blob_state.content_type = val

  @property
  def chunk_size(self) -> Optional[int]:
    """Returns blob chunk size."""
    return self._chunk_size

  @property
  def bucket(self) -> Optional[gcs_mock_types.GcsBucketType]:
    """Returns bucket within which blob is stored."""
    return self._bucket

  @property
  def time_created(self) -> Optional[datetime.datetime]:
    """Returns time blob was created."""
    return self._blob_state.time_created

  @property
  def time_deleted(self) -> Optional[datetime.datetime]:
    """Returns time blob was deleted."""
    return self._blob_state.time_deleted

  @property
  def updated(self) -> Optional[datetime.datetime]:
    """Returns time blob was updated."""
    return self._blob_state.updated

  @property
  def component_count(self) -> Optional[int]:
    """Returns number of components in blob is composed of ."""
    return self._blob_state.component_count

  @property
  def user_project(self) -> Optional[str]:
    """Returns user_project which would be conceptually billed."""
    return None if self._bucket is None else self._bucket.user_project

  @property
  def name(self) -> str:
    """Returns name of blob."""
    return self._name

  @property
  def etag(self) -> Optional[str]:
    """Returns etag for blob.

    https://cloud.google.com/python/docs/reference/storage/latest/generation_metageneration
    """
    return self._blob_state.etag

  @property
  def generation(self) -> Optional[int]:
    """Returns blob's generation number.

    https://cloud.google.com/python/docs/reference/storage/latest/generation_metageneration
    """
    return self._blob_state.generation

  @property
  def metageneration(self) -> Optional[int]:
    """Returns blob's metageneration number.

    https://cloud.google.com/python/docs/reference/storage/latest/generation_metageneration
    """
    return self._blob_state.metageneration

  @property
  def path(self) -> str:
    """Returns gcs path to blob."""
    if not self._bucket:
      raise gcs_mock_types.GcsMockError('Bucket is not set.')
    return f'{self._bucket.path}/o/{self.name}'

  @property
  def metadata(self) -> Optional[Mapping[str, str]]:
    """Returns blob metadata."""
    metadata = self._blob_state.metadata
    return None if metadata is None else copy.copy(metadata)

  @metadata.setter
  def metadata(self, val: Mapping[str, str]) -> None:
    """Sets blob metadata."""
    self._blob_state.metadata = dict(val)

  @property
  def md5_hash(self) -> Optional[str]:
    """Returns MD5 hash of the blob's data."""
    return self._blob_state.md5_hash

  @property
  def size(self) -> Optional[int]:
    """Returns size of the data the blob stores in bytes."""
    return self._blob_state.size_in_bytes

  def _update_state(
      self, state: Optional[blob_state_mock.BlobStateMock]
  ) -> None:
    """Sets blob state(metadata, hash, generation, metageneration, etc)."""
    if state is None:
      return
    self._blob_state = copy.copy(state)

  @classmethod
  def from_string(
      cls, uri: str, client: Optional[gcs_mock_types.GcsClientType] = None
  ) -> google.cloud.storage.Blob:
    """Returns google.cloud.storage.Blob represented by a gs:// formated uri.

    Returned exceptions reproduce actual behavior of:
      google.cloud.storage.Blob.from_string

    Args:
      uri: GS formated URI for blob in GCP.  gs://bucket_name/blob_name
      client: google.cloud.storage.Client which hosts bucket and blob.

    Returns:
      google.cloud.storage.Blob

    Raises:
      ValueError: URI does not start with gs:// prefix.
      IndexError: Bucket name cannot be parsed from uri.
    """
    bucket_name, blob_name = gcs_mock_utils.get_bucket_blob_from_gs_uri(uri)
    return google.cloud.storage.Blob(
        name=blob_name,
        bucket=google.cloud.storage.Bucket(client=client, name=bucket_name),
    )

  @property
  def client(self) -> Optional[gcs_mock_types.GcsClientType]:
    """Returns the google.cloud.storage.Client the blob is attached to."""
    if self._bucket is None:
      return None
    return self._bucket.client

  def _get_client(
      self,
      client: Union[None, gcs_mock_types.GcsClientType, client_mock.ClientMock],
  ) -> client_mock.ClientMock:
    """Returns google.cloud.storage.Client to execute internal blob operation.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.

    Returns:
      google.cloud.storage.Client

    Raises:
      AttributeError:
    """
    if client is None:
      client = self.client
    if client is None:
      raise AttributeError(
          f"'{type(client)}' object has no attribute '_get_resource'"
      )
    return typing.cast(client_mock.ClientMock, client)

  def exists(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
      soft_deleted: Optional[bool] = None,  # pylint:disable=unused-argument
  ) -> bool:
    """Returns true if blob exists on client in bucket.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy
      soft_deleted: (Optional) Not implemented in mock.

    Returns:
      True if blob exists.

    Raises:
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    return self._get_client(client).mock_state.has_blob(
        typing.cast(google.cloud.storage.Blob, self),
        if_etag_match,
        if_etag_not_match,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
    )

  def update(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
      override_unlocked_retention: bool = False,
  ) -> None:
    """Sends metadata set on blob to GCS.

    Metadata lifetime scoped to duration of GcsMock.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy
      override_unlocked_retention: Not supported by mock.

    Returns:
      None

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry, override_unlocked_retention
    self._update_state(
        self._get_client(client).mock_state.blob_update(
            typing.cast(google.cloud.storage.Blob, self),
            if_generation_match,
            if_generation_not_match,
            if_metageneration_match,
            if_metageneration_not_match,
        )
    )

  def reload(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      projection: str = 'noAcl',
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
      soft_deleted: Optional[bool] = None,  # pylint:disable=unused-argument
  ) -> None:
    """Reloads blob state from mocked cloud storage.

    Args:
      client: The client to use. If not passed, falls back to the client stored
        on the blob bucket.
      projection: (Optional) If used, must be 'full' or 'noAcl'. Defaults to
        'noAcl'. Specifies the set of properties to return.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy
      soft_deleted: (Optional) Not implemented in mock.

    Returns:
      None

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    self._update_state(
        self._get_client(client).mock_state.blob_reload(
            typing.cast(google.cloud.storage.Blob, self),
            projection,
            if_etag_match,
            if_etag_not_match,
            if_generation_match,
            if_generation_not_match,
            if_metageneration_match,
            if_metageneration_not_match,
        )
    )

  def delete(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Deletes a blob from Cloud Storage Mock.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    self._get_client(client).mock_state.blob_delete(
        typing.cast(google.cloud.storage.Blob, self),
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
    )

  def download_as_bytes(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      start: Optional[int] = None,
      end: Optional[int] = None,
      raw_download: bool = False,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      checksum: Optional[str] = 'md5',
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> bytes:
    """Download the contents of this blob as a bytes object.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      Downloaded bytes.

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    downloaded_bytes, source_state = self._get_client(
        client
    ).mock_state.blob_download_as_bytes(
        typing.cast(google.cloud.storage.Blob, self),
        start,
        end,
        raw_download,
        if_etag_match,
        if_etag_not_match,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        checksum,
    )
    # Blob doesn't set size in bytes after download.
    size_in_bytes = self._blob_state.size_in_bytes
    self._update_state(source_state)
    self._blob_state.size_in_bytes = size_in_bytes
    return downloaded_bytes

  def download_as_string(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      start: Optional[int] = None,
      end: Optional[int] = None,
      raw_download: bool = False,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """(Deprecated) Download the contents of this blob as a bytes object.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      gcs_mock_types.GcsMockError: Method is deprecated.
    """

    del (
        client,
        start,
        end,
        raw_download,
        if_etag_match,
        if_etag_not_match,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        timeout,
        retry,
    )
    raise gcs_mock_types.GcsMockError(
        'Method is deprecated use Blob.download_as_bytes'
    )

  def download_as_text(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      start: Optional[int] = None,
      end: Optional[int] = None,
      raw_download: bool = False,
      encoding: Optional[str] = None,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> str:
    """Download the contents of this blob as text.

    Args:
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      encoding: (Optional) encoding to be used to decode the downloaded bytes.
        Defaults to the charset param of attr:content_type, or else to "utf-8".
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      Downloaded text.

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    if encoding is None:
      encoding = 'utf-8'
    raw_bytes = self.download_as_bytes(
        client,
        start,
        end,
        raw_download,
        if_etag_match,
        if_etag_not_match,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        None,
    )
    return raw_bytes.decode(encoding)

  def download_to_filename(
      self,
      filename: str,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      start: Optional[int] = None,
      end: Optional[int] = None,
      raw_download: bool = False,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      checksum: Optional[str] = 'md5',
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Download the contents of this blob and stores as file.

    Args:
      filename: A filename to be passed to open.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del timeout, retry
    with open(filename, 'wb') as outfile:
      outfile.write(
          self.download_as_bytes(
              client,
              start,
              end,
              raw_download,
              if_etag_match,
              if_etag_not_match,
              if_generation_match,
              if_generation_not_match,
              if_metageneration_match,
              if_metageneration_not_match,
              checksum,
          )
      )

  def download_to_file(
      self,
      file_obj: BinaryIO,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      start: Optional[int] = None,
      end: Optional[int] = None,
      raw_download: bool = False,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      checksum: Optional[str] = 'md5',
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """DEPRECATED, Download the contents of this blob into a file-like object.

    Args:
      file_obj: A file handle to which to write the blob's data.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      gcs_mock_types.GcsMockError: Method is deprecated.
    """
    del (
        file_obj,
        client,
        start,
        end,
        raw_download,
        if_etag_match,
        if_etag_not_match,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        timeout,
        checksum,
        retry,
    )
    raise gcs_mock_types.GcsMockError('Method is deprecated')

  def upload_from_string(
      self,
      data: Union[bytes, str],
      content_type: Optional[str] = 'text/plain',
      num_retries: Optional[int] = None,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      predefined_acl: Optional[str] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      checksum: Optional[str] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Upload contents of this blob from a string.

    Args:
      data: The data to store in this blob. If the value is text, it will be
        encoded as UTF-8.
      content_type: (Optional) Type of content being uploaded. Defaults to
        'text/plain'.
      num_retries: Number of upload retries.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      predefined_acl: (Optional) Predefined access control list.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_generation_match, or if_metageneration_match) failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_generation_not_match, or if_metageneration_not_match) failed.
    """
    del num_retries, timeout, retry, predefined_acl
    if isinstance(data, str):
      data = data.encode('utf-8')
    self._update_state(
        self._get_client(client).mock_state.blob_upload_from_string(
            typing.cast(google.cloud.storage.Blob, self),
            data,
            content_type=content_type,
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            if_metageneration_match=if_metageneration_match,
            if_metageneration_not_match=if_metageneration_not_match,
            checksum=checksum,
        )
    )

  def upload_from_filename(
      self,
      filename: str,
      content_type: Optional[str] = None,
      num_retries: Optional[int] = None,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      predefined_acl: Optional[str] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      checksum: Optional[str] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Upload this blob's contents from a file.

    Args:
      filename: The path to the file.
      content_type: (Optional) Type of content being uploaded.
      num_retries: Number of upload retries.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      predefined_acl: (Optional) Predefined access control list.
      if_generation_match: (Optional) See :ref:using-if-generation-match.
      if_generation_not_match: (Optional) See :ref:
        using-if-generation-not-match.
      if_metageneration_match: (Optional) See :ref:
        using-if-metageneration-match.
      if_metageneration_not_match: (Optional) See :ref:
        using-if-metageneration-not-match.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_generation_match, or if_metageneration_match) failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_generation_not_match, or if_metageneration_not_match) failed.
    """
    if content_type is None:
      content_type = self.content_type
    if content_type is None:
      content_type = gcs_mock_utils.guess_content_type_from_file(filename)
    with open(filename, 'rb') as infile:
      self.upload_from_file(
          file_obj=infile,
          rewind=True,
          size=None,
          content_type=content_type,
          num_retries=num_retries,
          client=client,
          predefined_acl=predefined_acl,
          if_generation_match=if_generation_match,
          if_generation_not_match=if_generation_not_match,
          if_metageneration_match=if_metageneration_match,
          if_metageneration_not_match=if_metageneration_not_match,
          timeout=timeout,
          checksum=checksum,
          retry=retry,
      )

  def upload_from_file(
      self,
      file_obj: BinaryIO,
      rewind: bool = False,
      size: Optional[int] = None,
      content_type: Optional[str] = None,
      num_retries: Optional[int] = None,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      predefined_acl: Optional[str] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      checksum: Optional[str] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Upload this blob's contents from a open file-like-object.

    Args:
      file_obj: A file handle opened in binary mode for reading.
      rewind: If True, seek to the beginning of the file handle before writing.
      size: The number of bytes to be uploaded (which will be read file_obj).
      content_type: (Optional) Type of content being uploaded.
      num_retries: Number of upload retries.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      predefined_acl: (Optional) Predefined access control list.
      if_generation_match: (Optional) See :ref:using-if-generation-match.
      if_generation_not_match: (Optional) See :ref:
        using-if-generation-not-match.
      if_metageneration_match: (Optional) See :ref:
        using-if-metageneration-match.
      if_metageneration_not_match: (Optional) See :ref:
        using-if-metageneration-not-match.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_generation_match, or if_metageneration_match) failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_generation_not_match, or if_metageneration_not_match) failed.
    """
    if content_type is None:
      content_type = gcs_mock_utils.get_content_type_from_blob(
          typing.cast(google.cloud.storage.Blob, self)
      )
    if rewind:
      file_obj.seek(0)
    if size is not None:
      data = file_obj.read(size)
    else:
      data = file_obj.read()
    self.upload_from_string(
        data=data,
        content_type=content_type,
        num_retries=num_retries,
        client=client,
        predefined_acl=predefined_acl,
        if_generation_match=if_generation_match,
        if_generation_not_match=if_generation_not_match,
        if_metageneration_match=if_metageneration_match,
        if_metageneration_not_match=if_metageneration_not_match,
        timeout=timeout,
        checksum=checksum,
        retry=retry,
    )

  def compose(
      self,
      sources: List[gcs_mock_types.GcsBlobType],
      client: Optional[gcs_mock_types.GcsClientType] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      if_generation_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_source_generation_match: Optional[List[int]] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Concatenate source blobs into this one.

    Args:
      sources: Blobs whose contents will be composed into this blob.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response. No effect on mock.
      if_generation_match: (Optional) Makes the operation conditional on whether
        the destination object's current generation matches the given value.
        Setting to 0 makes the operation succeed only if there are no live
        versions of the object.
      if_metageneration_match: (Optional) Makes the operation conditional on
        whether the destination object's current metageneration matches the
        given value.
      if_source_generation_match: (Optional) Makes the operation conditional on
        whether the current generation of each source blob matches the
        corresponding generation.
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy
    """
    del timeout, retry
    self._update_state(
        self._get_client(client).mock_state.blob_compose(
            typing.cast(google.cloud.storage.Blob, self),
            sources,
            if_generation_match,
            if_metageneration_match,
            if_source_generation_match,
        )
    )

  def rewrite(
      self,
      source: Optional[gcs_mock_types.GcsBlobType],
      token: Optional[str] = None,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      if_source_generation_match: Optional[int] = None,
      if_source_generation_not_match: Optional[int] = None,
      if_source_metageneration_match: Optional[int] = None,
      if_source_metageneration_not_match: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> Tuple[Optional[str], int, int]:
    """Rewrite source blob into this one.

    Args:
      source: Blob whose contents will be rewritten into this blob.
      token: Token returned after first call, initally should be None.
      client: (Optional) The client to use. If not passed, falls back to the
        client stored on the blob's bucket.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      if_source_generation_match: (Optional) Makes the operation conditional on
        whether the source object's generation matches the given value.
      if_source_generation_not_match: (Optional) Makes the operation conditional
        on whether the source object's generation does not match the given
        value.
      if_source_metageneration_match: (Optional) Makes the operation conditional
        on whether the source object's current metageneration matches the given
        value.
      if_source_metageneration_not_match: (Optional) Makes the operation
        conditional on whether the source object's current metageneration does
        not match the given value.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response.  (No effect on mock)
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      Tuple[Token, bytes trasfered, total_bytes to transfer]
      Token is null when file transfer is complete.

    Raises:
      gcs_mock_types.GcsMockError: Rewrite called with invalid token.
      google.api_core.exceptions.Forbidden: If source/dest blob's bucket does
        not exist.
      google.api_core.exceptions.NotFound: Blob file does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_generation_match, or if_metageneration_match,
          if_source_generation_match, if_source_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_generation_not_match, if_metageneration_not_match,
          if_source_metageneration_match, if_source_metageneration_not_match)
          failed.
    """
    del timeout, retry
    token_bytes_transfered_total_size_tuple, dest_state = self._get_client(
        client
    ).mock_state.blob_rewrite(
        typing.cast(google.cloud.storage.Blob, self),
        source,
        token,
        if_generation_match,
        if_generation_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
        if_source_generation_match,
        if_source_generation_not_match,
        if_source_metageneration_match,
        if_source_metageneration_not_match,
    )
    self._update_state(dest_state)
    return token_bytes_transfered_total_size_tuple
