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
"""Mock google.cloud.storage.Client."""
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Union

import google.api_core
import google.auth
import google.cloud.storage
import requests

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_state_mock


class ClientMock:
  """Mock google.cloud.storage.Client.

  Do not use class directly. Create class via mocked class constructor.
    google.cloud.storage.Client (...)
  """

  def __init__(
      self,
      mock_state: gcs_state_mock.GcsStateMock,
      project: Optional[str] = None,
      credentials: Optional[google.auth.credentials.Credentials] = None,
      _http: Optional[requests.Session] = None,  # pylint: disable=invalid-name
      client_info: Optional[google.api_core.client_info.ClientInfo] = None,
      client_options: Union[
          None, google.api_core.client_options.ClientOptions, Dict[Any, Any]
      ] = None,
  ):
    """Mock constructor for google.cloud.storage.Client.

    Args:
      mock_state: Context managed wrapper for that holds internal GCS mock
        state.
      project: The project which the client acts on behalf of.
      credentials: (Optional) The OAuth2 Credentials to use for this client.
      _http: (Optional) HTTP object to make requests.
      client_info: The client info used to send a user-agent string along with
        API requests.
      client_options: (Optional) Client options used to set user options on the
        client.
    """
    del project, credentials, _http, client_info, client_options
    self._mock_state = mock_state

  @property
  def mock_state(self) -> gcs_state_mock.GcsStateMock:
    return self._mock_state

  def bucket(
      self, bucket_name: str, user_project: Optional[str] = None
  ) -> gcs_mock_types.GcsBucketType:
    """Factory constructor for bucket object.

    Args:
      bucket_name: The name of the bucket to be instantiated.
      user_project: (Optional) The project ID to be billed for API requests made
        via the bucket.

    Returns:
      google.cloud.storage.Bucket
    """
    return google.cloud.storage.Bucket(
        self, bucket_name, user_project=user_project
    )

  @classmethod
  def create_anonymous_client(cls) -> gcs_mock_types.GcsClientType:
    return google.cloud.storage.Client()

  def get_bucket(
      self,
      bucket_or_name: Union[str, gcs_mock_types.GcsBucketType],
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> gcs_mock_types.GcsBucketType:
    """Get a bucket by name, returning None if not found.

    Args:
      bucket_or_name: The name of the bucket to get or
        google.cloud.storage.Bucket instance.
      timeout: Operation timeout. Ignored by mock.
      if_metageneration_match: Raise if provided value != bucket.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        bucket.metageneration.
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy; Ignored by mock.

    Returns:
      google.cloud.storage.Bucket
    """
    if isinstance(bucket_or_name, str):
      bucket_name = bucket_or_name
    else:
      bucket_name = bucket_or_name.name
    bucket = google.cloud.storage.Bucket(self, bucket_name)
    bucket.reload(
        timeout=timeout,
        if_metageneration_match=if_metageneration_match,
        if_metageneration_not_match=if_metageneration_not_match,
        retry=retry,
    )
    return bucket

  def download_blob_to_file(
      self,
      blob_or_uri: Union[gcs_mock_types.GcsBlobType, str],
      file_obj: BinaryIO,
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
    """Download the contents of a blob object or blob URI into a file object.

    Args:
      blob_or_uri: The blob resource to pass or URI to download.
      file_obj: A file handle to which to write the blob's data.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.
      timeout: (Optional) The amount of time, in seconds, to wait for the server
        response. See: configuring_timeouts
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object.
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy

    Returns:
      None
    """
    if isinstance(blob_or_uri, str):
      blob = google.cloud.storage.Blob.from_string(blob_or_uri, self)
    else:
      blob = blob_or_uri
    raw_bytes = blob.download_as_bytes(
        client=self,
        start=start,
        end=end,
        raw_download=raw_download,
        if_etag_match=if_etag_match,
        if_etag_not_match=if_etag_not_match,
        if_generation_match=if_generation_match,
        if_generation_not_match=if_generation_not_match,
        if_metageneration_match=if_metageneration_match,
        if_metageneration_not_match=if_metageneration_not_match,
        timeout=timeout,
        checksum=checksum,
        retry=retry,
    )
    file_obj.write(raw_bytes)

  def list_buckets(
      self,
      max_results: Optional[int] = None,
      page_token: Optional[str] = None,
      prefix: Optional[str] = None,
      projection: str = 'noAcl',
      fields: Optional[str] = None,
      project: Optional[str] = None,
      page_size: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> Iterator[gcs_mock_types.GcsBucketType]:
    """Yields client buckets.

    Behavior differes from actual google.cloud.storage.Client which returns:
    google.api_core.page_iterator.Iterator

    Args:
      max_results: Maximum number of buckets to return.
      page_token: Not implemented in mock.
      prefix: String prefix of bucket names to return.
      projection: Specifies the set of properties to return. If used, must be
        'full' or 'noAcl'. Defaults to 'noAcl'.
      fields: (Optional) Selector specifying which fields to include in a
        partial response. Not implemented in mock.
      project: The project whose buckets are to be listed. Not implemented in
        mock.
      page_size: Maximum number of buckets to return in each page. Not
        implemented in mock.
      timeout: The amount of time, in seconds, to wait for the server response.
        Not implemented in mock.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy.  Not implemented in
        mock.

    Yields:
      Iterator of buckets on client.

    Raises:
      GcsMockError: page_token set to non-default value. Functionality not
        implemented in mock.
    """
    del fields, project, page_size, timeout, retry
    if page_token is not None:
      raise gcs_mock_types.GcsMockError(
          'Deprecated: use the pages property of the returned iterator instead '
          'of manually passing the token.'
      )
    if projection not in ('full', 'noAcl'):
      raise gcs_mock_types.GcsMockError(
          "projection must equal 'full' or 'noAcl'"
      )
    returned_results = 0
    for bucket_name in self.mock_state.list_bucket_names():
      if max_results is not None and returned_results == max_results:
        return
      if prefix is not None and not bucket_name.startswith(prefix):
        continue
      yield google.cloud.storage.Bucket(self, bucket_name)
      returned_results += 1

  def lookup_bucket(
      self,
      bucket_name: str,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> Optional[gcs_mock_types.GcsBucketType]:
    """Get a bucket by name, returning None if not found.

    Args:
      bucket_name: The name of the bucket to get.
      timeout: Operation timeout. Ignored by mock.
      if_metageneration_match: Raise if provided value != bucket.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        bucket.metageneration.
      retry: google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy; Ignored by mock.

    Returns:
      google.cloud.storage.Bucket

    Raises:
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
    """
    bucket = google.cloud.storage.Bucket(self, bucket_name)
    if bucket.exists(
        timeout=timeout,
        if_metageneration_match=if_metageneration_match,
        if_metageneration_not_match=if_metageneration_not_match,
        retry=retry,
    ):
      return bucket
    return None

  def list_blobs(
      self,
      bucket_or_name: Union[str, gcs_mock_types.GcsBucketType],
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
      page_size: Optional[int] = None,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
      match_glob: Optional[str] = None,
      include_folders_as_prefixes: Optional[bool] = None,  # pylint:disable=unused-argument
      soft_deleted: Optional[bool] = None,  # pylint:disable=unused-argument
  ) -> Iterator[gcs_mock_types.GcsBlobType]:
    """Yields blobs on a bucket in client.

    Args:
      bucket_or_name: Bucket name or google.cloud.storage.Bucket
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
      page_size: Maximum number of buckets to return in each page. Not
        implemented in mock.
      timeout: The amount of time, in seconds, to wait for the server response.
        Not implemented in mock.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy.  Not implemented in
        mock.
      match_glob: A glob pattern used to filter results (for example, foo*bar).
        The string value must be UTF-8 encoded.
      include_folders_as_prefixes: (Optional) Not implemented in mock.
      soft_deleted: (Optional) Not implemented in mock.

    Yields:
      Iterator of buckets on client.

    Raises:
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Blob bucket does not exist.
    """
    del (
        include_trailing_delimiter,
        versions,
        fields,
        page_size,
        timeout,
        retry,
        page_token,
        delimiter,
        match_glob,
    )
    if projection not in ('full', 'noAcl'):
      raise gcs_mock_types.GcsMockError(
          "projection must equal 'full' or 'noAcl'"
      )
    if isinstance(bucket_or_name, str):
      bucket_name = bucket_or_name
    else:
      bucket_name = bucket_or_name.name
    returned_results = 0
    bucket = google.cloud.storage.Bucket(self, bucket_name)
    self.mock_state.has_bucket(bucket)
    for blob_name in self.mock_state.list_blobs(bucket):
      if max_results is not None and returned_results == max_results:
        return
      if prefix is not None and not blob_name.startswith(prefix):
        continue
      if start_offset is not None and blob_name < start_offset:
        continue
      if end_offset is not None and blob_name > end_offset:
        continue
      blob = google.cloud.storage.Blob(blob_name, bucket)
      blob.reload()
      yield blob
      returned_results += 1

  def create_bucket(
      self,
      bucket_or_name: Union[str, gcs_mock_types.GcsBucketType],
      requester_pays: Optional[bool] = None,
      project: Optional[str] = None,
      user_project: Optional[str] = None,
      location: Optional[str] = None,
      data_locations: Optional[List[str]] = None,
      predefined_acl: Optional[str] = None,
      predefined_default_object_acl: Optional[str] = None,
      enable_object_retention: bool = False,
      timeout: Optional[gcs_mock_types.TimeoutType] = 60,
      retry: Optional[gcs_mock_types.RetryType] = None,
  ) -> None:
    """Create a new bucket.

    Args:
      bucket_or_name: google.cloud.storage.Bucket instance or name of bucket to
        create.
      requester_pays: DEPRECATED.
      project: (Optional) The project under which the bucket is to be created.
        Not implemented in mock.
      user_project: (Optional) The project ID to be billed for API requests made
        via created bucket. Not implemented in mock.
      location: (Optional) The location of the bucket.  Not implemented in mock.
      data_locations: (Optional) The list of regional locations of a custom
        dual-region bucket.  Not implemented in mock.
      predefined_acl: (Optional) Name of predefined ACL to apply to bucket. Not
        implemented in mock.
      predefined_default_object_acl: (Optional) Name of predefined ACL to apply
        to bucket's objects.  Not implemented in mock.
      enable_object_retention: (Optional) Whether object retention should be
        enabled on this bucket. Not supported by Mock.
      timeout: The amount of time, in seconds, to wait for the server response.
        Not implemented in mock.
      retry:    google.api_core.retry.Retry or
        google.cloud.storage.retry.ConditionalRetryPolicy.  Not implemented in
        mock.

    Raises:
      google.cloud.exceptions.Conflict: If bucket already exists.
    """
    del (
        project,
        location,
        data_locations,
        predefined_acl,
        predefined_default_object_acl,
        timeout,
        retry,
        enable_object_retention,
    )
    if requester_pays is not None:
      raise gcs_mock_types.GcsMockError(
          'requester_pays parameter is DEPRECATED. '
          'Use Bucket().requester_pays instead.'
      )
    if isinstance(bucket_or_name, str):
      bucket = google.cloud.storage.Bucket(self, bucket_or_name)
    else:
      bucket = bucket_or_name
    self.mock_state.create_bucket(bucket, user_project)
