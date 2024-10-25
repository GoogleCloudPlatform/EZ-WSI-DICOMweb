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
"""Mock google.cloud.storage.Bucket."""
import re
import typing
from typing import Iterator, Optional, Union

from google.api_core import exceptions
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import client_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_utils

_VALID_BUCKET_NAME_START_END_CHAR_REGEX = re.compile('[0-9a-zA-Z]')


class BucketMock:
  """Mock google.cloud.storage.Bucket.

  Do not use class directly. Create class via mocked class constructor.
  google.cloud.storage.Bucket (...)
  """

  def __init__(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      name: Optional[str] = None,
      user_project: Optional[str] = None,
  ):
    """Mock google.cloud.storage.Bucket constructor.

    Args:
      client: A client which holds credentials and project configuration.
      name: The name of the bucket.
      user_project: (Optional) the project ID to be conceptually billed for API
        requests. Mock doesn't do transactions against GCP so no cost.

    Raises:
      ValueError: Bucket name starts or ends with invalid char.
    """
    self._client = client
    if name is not None:
      if not (
          _VALID_BUCKET_NAME_START_END_CHAR_REGEX.fullmatch(name[0])
          and _VALID_BUCKET_NAME_START_END_CHAR_REGEX.fullmatch(name[-1])
      ):
        raise ValueError(
            'Bucket names must start and end with a number or letter.'
        )
    self._name = name
    self._user_project = user_project

  @property
  def user_project(self) -> Optional[str]:
    """Returns user_project which would be conceptually billed."""
    return self._user_project

  @property
  def name(self) -> Optional[str]:
    """Returns bucket name."""
    return self._name

  @property
  def client(self) -> Optional[gcs_mock_types.GcsClientType]:
    """Returns bucket client."""
    return self._client

  @property
  def path(self) -> str:
    """Returns cloud storage path to bucket."""
    return f'/b/{self._name}'

  def blob(
      self,
      blob_name: str,
      chunk_size: Optional[int] = None,
      encryption_key: Optional[bytes] = None,
      kms_key_name: Optional[str] = None,
      generation: Optional[int] = None,
  ) -> gcs_mock_types.GcsBlobType:
    """Returns google.cloud.storage.Blob on bucket.

    Args:
      blob_name: Name of blob on bucket.
      chunk_size: (Optional) The size of a chunk of data whenever iterating (in
        bytes).  Not used in mock.
      encryption_key: (Optional) 32 byte encryption key for customer-supplied
        encryption. Not implemented in mock.
      kms_key_name: (Optional) Resource name of Cloud KMS key used to encrypt
        the blob's contents. Not impelemented in mock.
      generation: (Optional) If present, selects a specific revision of this
        object.

    Returns:
      google.cloud.storage.Blob
    """
    return google.cloud.storage.Blob(
        name=blob_name,
        bucket=self,
        chunk_size=chunk_size,
        encryption_key=encryption_key,
        kms_key_name=kms_key_name,
        generation=generation,
    )

  def get_blob(
      self,
      blob_name: str,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      encryption_key: Optional[bytes] = None,
      generation: Optional[int] = None,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_generation_match: Optional[int] = None,
      if_generation_not_match: Optional[int] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      retry: Optional[gcs_mock_types.RetryType] = None,
      soft_deleted: Optional[bool] = None,  # pylint:disable=unused-argument
      **kwargs,
  ) -> Optional[gcs_mock_types.GcsBlobType]:
    """Get a blob object by name.

    Args:
      blob_name: Name of blob.
      client: Optional client to use for operation.
      encryption_key: (Optional) 32 byte encryption key for customer-supplied
        encryption. Not implemented in mock.
      generation: Generation of blob to load. Not implemented in mock.
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
      **kwargs: Addtional keyword args; ignored.

    Returns:
      google.cloud.storage.Blob or None if not found.

    Raises:
      google.api_core.exceptions.Forbidden: If blob's bucket does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    del kwargs
    bucket = google.cloud.storage.Bucket(self._get_client(client), self.name)
    if not bucket.exists():
      return None
    blob = bucket.blob(
        blob_name, encryption_key=encryption_key, generation=generation
    )
    try:
      blob.reload(
          if_etag_match=if_etag_match,
          if_etag_not_match=if_etag_not_match,
          if_generation_match=if_generation_match,
          if_generation_not_match=if_generation_not_match,
          if_metageneration_match=if_metageneration_match,
          if_metageneration_not_match=if_metageneration_not_match,
          timeout=timeout,
          retry=retry,
      )
    except exceptions.NotFound:
      return None
    if generation is None or blob.generation == generation:
      return blob
    return None

  def _get_client(
      self,
      client: Union[None, gcs_mock_types.GcsClientType, client_mock.ClientMock],
  ) -> client_mock.ClientMock:
    if client is None:
      client = self.client
    return typing.cast(client_mock.ClientMock, client)

  def exists(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> bool:
    """Returns true if bucket exists on client otherwise may throw Forbidden.

      Actual behavior of: google.cloud.storage.Blob.exists():
        returns true if bucket exists and caller has permission to access.
        returns false if bucket does not exist anywhere.
        throws forbidden if bucket exists and caller does not have permission
        to access.

    Args:
      client: Optional client to use for operation.
      timeout: Operation timeout. Ingored by mock.
      if_etag_match: Raise if provided value != bucket.etag.
      if_etag_not_match: Raise if provided value == bucket.etag.
      if_metageneration_match: Raise if provided value != bucket.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        bucket.metageneration.
      retry: Ingored by mock.

    Raises:
      google.api_core.exceptions.BadRequest: Blob name is less than 3 chars.
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
    """
    del timeout, retry
    try:
      self._get_client(client).mock_state.has_bucket(
          self,
          if_etag_match,
          if_etag_not_match,
          if_metageneration_match,
          if_metageneration_not_match,
      )
      return True
    except exceptions.NotFound:
      return False

  def list_blobs(
      self,
      max_results: Optional[int] = None,
      page_token: Optional[str] = None,
      prefix: Optional[str] = None,
      delimiter: Optional[str] = None,
      start_offset: Optional[str] = None,
      end_offset: Optional[str] = None,
      include_trailing_delimiter: Optional[bool] = None,
      versions: Optional[bool] = None,
      projection: str = 'noAcl',
      fields: Optional[str] = None,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      timeout: gcs_mock_types.TimeoutType = 60.0,
      retry: Optional[gcs_mock_types.RetryType] = None,
      match_glob: Optional[str] = None,
      include_folders_as_prefixes: Optional[bool] = None,  # pylint:disable=unused-argument
      soft_deleted: Optional[bool] = None,  # pylint:disable=unused-argument
  ) -> Iterator[gcs_mock_types.GcsBlobType]:
    """Returns iterator of blobs in a bucket.

    Args:
      max_results: (Optional) The maximum number of blobs to return.
      page_token: (Optional) If present, return the next batch of blobs, using
        the value, which must correspond to the next PageToken value returned in
        the previous response.
      prefix: (Optional) Prefix used to filter blobs.
      delimiter: (Optional) Delimiter, used with prefix to emulate hierarchy.
      start_offset: (Optional) Filter results to objects whose names are
        lexicographically equal to or after startOffset. Not implemented in
        mock.
      end_offset: (Optional) Filter results to objects whose names are
        lexicographically before endOffset. Not implemented in mock.
      include_trailing_delimiter: If true, objects that end in exactly one
        instance of delimiter will have their metadata included in items in
        addition to prefixes.  Not implemented in mock.
      versions: (Optional) Whether object versions should be returned as
        separate blobs.  Not implemented in mock.
      projection: Optional) If used, must be 'full' or 'noAcl'. Defaults to
        'noAcl'.  Not implemented in mock.
      fields: (Optional) Selector specifying which fields to include in a
        partial response. Not implemented in mock.
      client: google.cloud.storage.Client which hosts bucket.
      timeout: The amount of time, in seconds, to wait for the server response.
        Not implemented in mock.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy.  Not implemented in
        mock.
      match_glob: Regular expression filter, not supported by mock.
      include_folders_as_prefixes: (Optional) Not implemented in mock.
      soft_deleted: (Optional) Not implemented in mock.

    Returns:
      Iterator of buckets on client.

    Raises:
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob bucket does not exist.
    """
    return self._get_client(client).list_blobs(
        self.name,
        max_results,
        page_token,
        prefix,
        delimiter,
        start_offset,
        end_offset,
        include_trailing_delimiter,
        versions,
        projection,
        fields,
        None,
        timeout,
        retry,
        match_glob,
    )

  @classmethod
  def from_string(
      cls, uri: str, client: Optional[gcs_mock_types.GcsClientType] = None
  ) -> google.cloud.storage.Bucket:
    """Returns google.cloud.storage.Bucket represented by a gs:// formated uri.

    Returned exceptions reproduce actual behavior of:
      google.cloud.storage.Bucket.from_string

    Args:
      uri: GS formated URI for blob in GCP.  gs://bucket_name
      client: google.cloud.storage.Client which hosts bucket.

    Returns:
      google.cloud.storage.Bucket

    Raises:
      ValueError: URI does not start with gs:// prefix.
      IndexError: Bucket name cannot be parsed from uri.
    """
    bucket_name, _ = gcs_mock_utils.get_bucket_blob_from_gs_uri(uri)
    return google.cloud.storage.Bucket(client=client, name=bucket_name)

  def reload(
      self,
      client: Optional[gcs_mock_types.GcsClientType] = None,
      projection: str = 'noAcl',
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Loads bucket state from GCS.

    Mock does not have concept of bucket state. Method validates projection
    parameter and that blob exists.

    Args:
      client: Optional client to use for operation.
      projection: Specifies the set of properties to return.
      timeout: Operation timeout. Ingored by mock.
      if_etag_match: Raise if provided value != bucket.etag.
      if_etag_not_match: Raise if provided value == bucket.etag.
      if_metageneration_match: Raise if provided value != bucket.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        bucket.metageneration.
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy; Ingored by mock.

    Raises:
      gcs_mock_types.GcsMockError: Invalid projection parameter
    """
    del timeout, retry
    if projection not in ('full', 'noAcl'):
      raise gcs_mock_types.GcsMockError(
          "projection must equal 'full' or 'noAcl'"
      )
    self._get_client(client).mock_state.has_bucket(
        self,
        if_etag_match,
        if_etag_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
    )
