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
"""Holds Gcs State for mock implementation."""
import base64
import contextlib
import copy
import dataclasses
import datetime
import enum
import hashlib
import io
import os
import random
import tempfile
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union
import uuid

from google.api_core import exceptions
import google.cloud.storage

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_state_mock
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_constants
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_types
from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import gcs_mock_utils


class ContextManagerState(enum.Enum):
  NOT_ENTERED = 1
  ENTERED = 2
  EXITED = 3


def _gen_etag(blob: gcs_mock_types.GcsBlobType) -> str:
  """Returns mock etag for blob."""
  return f'etag_{blob.path}_{str(uuid.uuid1())}'


def _get_file_md5_hash(path: str) -> str:
  """Returns md5 hash of file as string."""
  hash_function = hashlib.md5()
  try:
    with open(path, 'rb') as infile:
      block = infile.read(16384)
      while block:
        hash_function.update(block)
        block = infile.read(16384)
  except FileNotFoundError:
    pass
  hash_val = hash_function.hexdigest()
  return base64.b64encode(bytes.fromhex(hash_val)).decode('utf-8')


def called_inside_context_manager(
    method: Callable[..., Any]
) -> Callable[..., Any]:
  """Decorator tests if method is called within context manager block.

  Raises error if context manager is not in ENTERED state.

  Args:
    method: Method to decorate.

  Returns:
    Called method results.

  Raises:
    gcs_mock_types.GcsMockError: Method not called within context mgr block.
  """

  def _decorator(self, *args, **kwargs) -> Any:
    if self._context_manager_state != ContextManagerState.ENTERED:  # pylint: disable=protected-access
      raise gcs_mock_types.GcsMockError(
          'Method can only be called inside entered context manager.'
      )
    return method(self, *args, **kwargs)

  return _decorator


class GcsStateMock(contextlib.ExitStack):
  """Holds Gcs State for mock implementation."""

  def __init__(
      self, buckets: Union[None, List[str], Mapping[str, Optional[str]]] = None
  ):
    super().__init__()
    self._rewrite_temp_buffer: Dict[str, io.BytesIO] = {}
    if buckets is None:
      self._buckets = {}
    elif isinstance(buckets, dict):
      self._buckets = copy.copy(buckets)
    else:
      self._buckets = {bucket_name: None for bucket_name in buckets}
    self._blob_state: Dict[str, blob_state_mock.BlobStateMock] = {}
    self._lock = threading.Lock()  # Lock to provide thread safety to mock
    self._temp_directory = None
    self._context_manager_state = ContextManagerState.NOT_ENTERED

  def __enter__(self):
    super().__enter__()
    try:
      if self._context_manager_state != ContextManagerState.NOT_ENTERED:
        raise gcs_mock_types.GcsMockError(
            'GcsStateMock does not support re-entry.'
        )
      self._context_manager_state = ContextManagerState.ENTERED
      for bucket_name in list(self._buckets):
        path = self._buckets.get(bucket_name)
        if path is None or not path:
          path = os.path.join(self._get_temp_dir_path(), bucket_name)
          os.mkdir(path)
        elif not os.path.exists(path):
          raise gcs_mock_types.GcsMockError(f'Path: "{path}" does not exist.')
        elif not os.path.isdir(path):
          raise gcs_mock_types.GcsMockError(
              f'Path: "{path}" does not reference a directory.'
          )
        self._buckets[bucket_name] = path.rstrip('/')
      self._init_mock_metadata_state()
      return self
    except:
      # Exception occurred during context manager entry. Force close any opened
      # context managers attached to this class.
      self.close()
      raise

  def __exit__(self, *args, **kwargs):
    """Closes context manager."""
    super().__exit__(*args, **kwargs)
    self._context_manager_state = ContextManagerState.EXITED

  def _get_temp_dir_path(self) -> str:
    if self._temp_directory is None:
      self._temp_directory = self.enter_context(tempfile.TemporaryDirectory())
    return self._temp_directory

  @called_inside_context_manager
  def get_bucket_path(self, bucket_name: str) -> Optional[str]:
    with self._lock:
      return self._buckets.get(bucket_name)

  @called_inside_context_manager
  def list_bucket_names(self) -> List[str]:
    """Returns the list of bucket names on mock interface."""
    with self._lock:
      return sorted(list(self._buckets))

  @called_inside_context_manager
  def _init_blob_state(
      self,
      blob: gcs_mock_types.GcsBlobType,
      generation: int,
      content_type: Optional[str],
  ) -> blob_state_mock.BlobStateMock:
    """Init blob metadata state called on: init, copy, compose, or upload.

    Args:
      blob: Blob mock.
      generation: Generation of blob.
      content_type: Content type.

    Returns:
      Blob state.
    """
    path = self._get_blob_path(blob)
    try:
      file_size_bytes = os.path.getsize(path)
    except FileNotFoundError:
      file_size_bytes = 0
    blob_state = blob_state_mock.BlobStateMock(
        size_in_bytes=file_size_bytes,
        md5_hash=_get_file_md5_hash(path),
        metadata={},
        generation=generation,
        metageneration=1,
        etag=_gen_etag(blob),
        content_type=content_type,
        time_created=datetime.datetime.now(),
        time_deleted=None,
        updated=None,
        component_count=1,
    )
    self._blob_state[path] = blob_state
    return blob_state

  @called_inside_context_manager
  def create_bucket(
      self, bucket: gcs_mock_types.GcsBucketType, user_project: str
  ) -> None:
    """Creates bucket.

    Method should not be called directly. Call using method
    on google.cloud.storage.Client, google.cloud.storage.Bucket, and
    google.cloud.storage.Blob

    Args:
      bucket: Mocked Bucket.
      user_project: User project to bill. Not used in Mock.

    Raises:
      google.cloud.exceptions.Conflict: If bucket exists.
    """
    del user_project
    with self._lock:
      if bucket.name not in self._buckets:
        try:
          bucket_path = os.path.join(self._get_temp_dir_path(), bucket.name)
          os.mkdir(bucket_path)
          self._buckets[bucket.name] = bucket_path
          return
        except FileExistsError:
          pass
      raise google.cloud.exceptions.Conflict(
          f'{gcs_mock_types.HttpMethod.GET.value} '
          f'{gcs_mock_constants.GCS_BASE_HTTPS_URL}/{bucket.path} The'
          ' requested bucketname is not available. The bucket namespace is'
          ' shared by all users of the system. Please select a different'
          ' name and try again'
      )

  @called_inside_context_manager
  def _validate_bucket(
      self,
      bucket: gcs_mock_types.GcsBucketType,
      http_method: gcs_mock_types.HttpMethod,
  ) -> None:
    """Validates if a bucket exists.

    Args:
      bucket: Mocked Bucket.
      http_method: HTTP method being checked.

    Returns:
      None

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
    """
    if len(bucket.name) < 3:
      raise exceptions.BadRequest(
          f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}/storage/v1{{bucket.path}}'
          ' Bucket names must be at least 3 characters in length, got'
          f" {len(bucket.name)}: '{bucket.name}'"
      )
    if bucket.name in self._buckets:
      return
    # When bucket cannot be found one of two errors can occur in wild.
    # These are simulated in mock to make sure code handles both.
    # 1) If bucket exists and but cannot be accessed (i.e. owned by someone
    # else then a exceptions.Forbidden is raised.
    # 2) If bucket does not exist then raise exceptions.NotFound
    if random.choice(('NotFound', 'Forbidden')) == 'NotFound':
      raise exceptions.NotFound(
          f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}'
          f'/storage/v1{bucket.path} The specified bucket does not exist.'
      )
    raise exceptions.Forbidden(
        f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}'
        f"/storage/v1{bucket.path} Permission 'storage.objects.get' denied on "
        'resource (or it may not exist).'
    )

  @called_inside_context_manager
  def list_blobs(self, bucket: gcs_mock_types.GcsBucketType) -> List[str]:
    """Returns the list of blob names in bucket on mock interface.

    Method should not be called directly. Call using method
    on google.cloud.storage.Client, google.cloud.storage.Bucket, and
    google.cloud.storage.Blob

    Args:
      bucket: Mocked bucket.

    Returns:
      List of names of blobs on bucket.

    Raises:
      google.api_core.exceptions.Forbidden: Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
    """
    file_paths = []
    with self._lock:
      self._validate_bucket(bucket, http_method=gcs_mock_types.HttpMethod.GET)
      bucket_path = self._buckets[bucket.name]
      for dir_struct in os.walk(bucket_path):
        dirpath, _, dir_filenames = dir_struct
        # add one for file seperator
        blob_dir = dirpath[len(bucket_path) :].lstrip('/')
        for name in dir_filenames:
          if not blob_dir:
            file_paths.append(name)
          else:
            file_paths.append(os.path.join(blob_dir, name))
      return file_paths

  @called_inside_context_manager
  def _get_blob_path(self, blob: gcs_mock_types.GcsBlobType) -> str:
    """Returns path to blob on file-system.

    Args:
      blob: Mocked Blob.

    Returns:
      Path to blob on file system.

    Raises:
      GcsMockError: Blob's bucket is not initalized with a name and client.
    """
    bucket = blob.bucket
    if bucket is None or bucket.name is None:
      raise gcs_mock_types.GcsMockError(
          'Cannot determine path without bucket name'
      )
    bucket_path = self._buckets.get(bucket.name)
    if bucket_path is None:
      return ''
    return os.path.join(bucket_path, blob.name)

  @called_inside_context_manager
  def _init_mock_metadata_state(self) -> None:
    """Initalizes mock metadata state at mock creation."""
    self._blob_state = {}
    bucket_names = self.list_bucket_names()
    if not bucket_names:
      return
    client = google.cloud.storage.Client()
    for bucket_name in bucket_names:
      bucket = google.cloud.storage.Bucket(client, bucket_name)
      for blob_name in self.list_blobs(bucket):
        blob = bucket.blob(blob_name)
        path = self._get_blob_path(blob)
        self._init_blob_state(
            blob, 1, gcs_mock_utils.guess_content_type_from_file(path)
        )

  @called_inside_context_manager
  def has_bucket(
      self,
      bucket: gcs_mock_types.GcsBucketType,
      if_etag_match: Optional[gcs_mock_types.EtagType] = None,
      if_etag_not_match: Optional[gcs_mock_types.EtagType] = None,
      if_metageneration_match: Optional[int] = None,
      if_metageneration_not_match: Optional[int] = None,
  ) -> bool:
    """Returns True if bucket exists.

    Args:
      bucket: Mocked Bucket.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.

    Raises:
      google.api_core.exceptions.BadRequest: Blob name is less than 3 chars.
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
    """
    del (
        if_etag_match,
        if_etag_not_match,
        if_metageneration_match,
        if_metageneration_not_match,
    )
    with self._lock:
      self._validate_bucket(bucket, http_method=gcs_mock_types.HttpMethod.GET)
    return True

  @called_inside_context_manager
  def set_blob_debug_state(
      self,
      blob: gcs_mock_types.GcsBlobType,
      etag: str,
      generation: int,
      metageneration: int,
  ) -> None:
    """Debug access point set blob state for testing."""
    path = self._get_blob_path(blob)
    if path not in self._blob_state:
      self._blob_state[path] = blob_state_mock.BlobStateMock()
    self._blob_state[path].etag = etag
    self._blob_state[path].generation = generation
    self._blob_state[path].metageneration = metageneration

  @called_inside_context_manager
  def _metadata_val_in(
      self,
      metadata_value: Union[int, str],
      test_metadata_match: Union[int, str, Set[str]],
  ) -> bool:
    if isinstance(test_metadata_match, set):
      return metadata_value in test_metadata_match
    return metadata_value == test_metadata_match

  @called_inside_context_manager
  def _test_valid_blob_metadata(
      self,
      blob: gcs_mock_types.GcsBlobType,
      http_method: gcs_mock_types.HttpMethod,
      element_to_check: str,
      if_match: Union[None, gcs_mock_types.EtagType, int],
      if_not_match: Union[None, gcs_mock_types.EtagType, int],
      default: Union[None, int, str] = None,
  ) -> None:
    """Test if blob metadata matches pre-post conditions if not raises.

    Used to check if blob, etag, generation, and metageneration match expected
    values.

    Args:
      blob: Mocked blob to test.
      http_method: HTTP method being checked.
      element_to_check: string name of element to test.
      if_match: value to require to match value on blob.
      if_not_match: value to require to not match value on blob.
      default: Default value to use if blob metadata is undefined (None).
    """
    path = self._get_blob_path(blob)
    blob_val = default
    blob_metadata = self._blob_state.get(path)
    if blob_metadata is not None:
      element_val = dataclasses.asdict(blob_metadata)[element_to_check]
      if element_val is not None:
        blob_val = element_val
    if blob_val is None:
      return
    if if_match is not None and not self._metadata_val_in(blob_val, if_match):
      raise exceptions.PreconditionFailed(
          f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}'
          f'/storage/v1{blob.path}: precondition failed'
      )
    if if_not_match is not None and self._metadata_val_in(
        blob_val, if_not_match
    ):
      raise exceptions.NotModified(
          f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}'
          f'/storage/v1{blob.path}: unknown error'
      )

  @called_inside_context_manager
  def _validate_blob_exists_and_return_mock_path(
      self,
      blob: gcs_mock_types.GcsBlobType,
      http_method: gcs_mock_types.HttpMethod,
      storage_op: str = '',
  ) -> str:
    """Validates if a blob exists and returns path to blob on file system.

    Args:
      blob: Mocked Blob.
      http_method: Method to include in exception message if blob not found.
      storage_op: Optional text to add to blob path in exception message

    Returns:
      path to blob on file-system.

    Raises:
      google.api_core.exceptions.NotFound: Blob does not exist.
    """
    blob_path = self._get_blob_path(blob)
    if os.path.isfile(blob_path):
      if blob_path not in self._blob_state:
        self._init_blob_state(
            blob, 1, gcs_mock_utils.guess_content_type_from_file(blob_path)
        )
      return blob_path
    request_path = [gcs_mock_constants.GCS_BASE_HTTPS_URL]
    if storage_op:
      request_path.append(storage_op)
    request_path.append('storage/v1')
    request_path = '/'.join(request_path)
    bucket_name = '' if blob.bucket is None else blob.bucket.name  # pytype: disable=attribute-error
    raise exceptions.NotFound(
        f'{http_method.value} {request_path}{blob.path} No such object:'
        f' {bucket_name}/{blob.name}'
    )

  def has_blob(
      self,
      blob: gcs_mock_types.GcsBlobType,
      if_etag_match: Optional[gcs_mock_types.EtagType],
      if_etag_not_match: Optional[gcs_mock_types.EtagType],
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
  ) -> bool:
    """Returns true if blob exists as file in gcs mock.

    Does not update blob state in caller.

    Method should not be called directly. Call using method
    on google.cloud.storage.Client, google.cloud.storage.Bucket, and
    google.cloud.storage.Blob

    Args:
      blob: Mocked Blob.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.

    Returns:
      True if blob exists; False if not.

    Raises:
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_etag_match, if_generation_match, or if_metageneration_match)
          failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_etag_not_match, if_generation_not_match, or
          if_metageneration_not_match) failed.
    """
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.GET
      try:
        self._validate_bucket(blob.bucket, http_method)
      except (exceptions.Forbidden, exceptions.NotFound) as _:
        return False
      try:
        self._validate_blob_exists_and_return_mock_path(blob, http_method)
      except exceptions.NotFound:
        return False
      self._test_valid_blob_metadata(
          blob, http_method, 'etag', if_etag_match, if_etag_not_match
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      return True

  def blob_update(
      self,
      blob: gcs_mock_types.GcsBlobType,
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
  ) -> blob_state_mock.BlobStateMock:
    """Updates sets blob's metadata on gcs mock.

    Method should not be called directly. Call using method
    on google.cloud.storage.Blob

    Args:
      blob: Mocked Blob.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.

    Returns:
      Blobs state on gcs mock following update.

    Raises:
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
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
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.PUT
      self._validate_bucket(blob.bucket, http_method)
      path = self._validate_blob_exists_and_return_mock_path(blob, http_method)
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      blob_state = self._blob_state[path]
      blob_state.metageneration += 1
      blob_state.metadata = (
          {} if blob.metadata is None else copy.copy(blob.metadata)
      )
      blob_state.etag = _gen_etag(blob)
      blob_state.updated = datetime.datetime.now()
      return copy.copy(blob_state)

  def blob_reload(
      self,
      blob: gcs_mock_types.GcsBlobType,
      projection: str,
      if_etag_match: Optional[gcs_mock_types.EtagType],
      if_etag_not_match: Optional[gcs_mock_types.EtagType],
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
  ) -> blob_state_mock.BlobStateMock:
    """Returns  blob's state on gcs mock.

    Method should not be called directly. Call using method
    on google.cloud.storage.Blob

    Args:
      blob: Mocked Blob.
      projection: (Optional) If used, must be 'full' or 'noAcl'. Defaults to
        'noAcl'. Specifies the set of properties to return.
      if_etag_match: Raise if provided value != blob.etag.
      if_etag_not_match: Raise if provided value == blob.etag.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.

    Returns:
      Blobs state on gcs mock.

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
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
    if projection not in ('full', 'noAcl'):
      raise gcs_mock_types.GcsMockError(
          "projection must equal 'full' or 'noAcl'"
      )
    if projection != 'noAcl':
      raise gcs_mock_types.GcsMockError(
          'Setting projection to value other than noAcl not supported in mock.'
      )
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.GET
      self._validate_bucket(blob.bucket, http_method)
      path = self._validate_blob_exists_and_return_mock_path(blob, http_method)
      self._test_valid_blob_metadata(
          blob, http_method, 'etag', if_etag_match, if_etag_not_match
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      return self._blob_state[path]

  def blob_delete(
      self,
      blob: gcs_mock_types.GcsBlobType,
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
  ) -> None:
    """Deletes blob from gcs mock.

    Method should not be called directly. Call using method
    on google.cloud.storage.Blob

    Args:
      blob: Mocked Blob.
      if_generation_match: Raise if provided value != blob.generation.
      if_generation_not_match: Raise if provided value == blob.generation.
      if_metageneration_match: Raise if provided value != blob.metageneration.
      if_metageneration_not_match: Raise if provided value ==
        blob.metageneration.

    Raises:
      gcs_mock_types.GcsMockError: Reload called with invalid parameter values.
      google.api_core.exceptions.NotFound: Bucket does not exist.
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
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.DELETE
      self._validate_bucket(blob.bucket, http_method)
      path = self._validate_blob_exists_and_return_mock_path(blob, http_method)
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      os.remove(path)
      del self._blob_state[path]

  def blob_download_as_bytes(
      self,
      blob: gcs_mock_types.GcsBlobType,
      start: Optional[int],
      end: Optional[int],
      raw_download: bool,
      if_etag_match: Optional[gcs_mock_types.EtagType],
      if_etag_not_match: Optional[gcs_mock_types.EtagType],
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
      checksum: Optional[str],
  ) -> Tuple[bytes, blob_state_mock.BlobStateMock]:
    """Downloads blob bytes.

    Method should not be called directly. Call using method
    on google.cloud.storage.Client, google.cloud.storage.Bucket, and
    google.cloud.storage.Blob

    Args:
      blob: Source blob to download.
      start: (Optional) The first byte in a range to be downloaded.
      end: (Optional) The last byte in a range to be downloaded.
      raw_download: (Optional) If true, download the object without any
        expansion. Mock only supports raw_download = True.
      if_etag_match: (Optional) See :ref:using-if-etag-match.
      if_etag_not_match: (Optional) See :ref:using-if-etag-not-match.
      if_generation_match: (Optional) See :ref:using-if-generation-match.
      if_generation_not_match: (Optional) See :ref:
        using-if-generation-not-match.
      if_metageneration_match: (Optional) See :ref:
        using-if-metageneration-match.
      if_metageneration_not_match: (Optional) See :ref:
        using-if-metageneration-not-match.
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)

    Returns:
      Blobs state of destination blob on gcs mock following download.

    Raises:
      MockGcsError: Called with invalid parameter values.
      google.api_core.exceptions.Forbidden:  Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Source blob file does not exist.
      google.api_core.exceptions.RequestRangeNotSatisfiable: Invalid download
        byte range.
    """
    del raw_download
    if checksum not in (None, 'md5', 'crc32c'):
      raise gcs_mock_types.GcsMockError('Invalid checksum parameter value.')
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.GET
      self._validate_bucket(blob.bucket, http_method)
      path = self._validate_blob_exists_and_return_mock_path(
          blob, http_method, 'download'
      )
      self._test_valid_blob_metadata(
          blob, http_method, 'etag', if_etag_match, if_etag_not_match
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      source_blob_state = self._blob_state[path]
      with open(path, 'rb') as infile:
        if start is not None and not (end is not None and start > end):
          if start >= os.path.getsize(path):
            raise google.api_core.exceptions.RequestRangeNotSatisfiable(
                f'{http_method.value} {gcs_mock_constants.GCS_BASE_HTTPS_URL}'
                f'/download/storage/v1{blob.path}'
            )
          infile.seek(start)
        if end is not None:
          return (infile.read(end - start + 1), source_blob_state)
        return (infile.read(), source_blob_state)

  @called_inside_context_manager
  def _init_blob_state_and_inc_generation(
      self, blob: gcs_mock_types.GcsBlobType, content_type: Optional[str]
  ) -> blob_state_mock.BlobStateMock:
    """Init blob metadata state and increments blob's generation.

    Args:
      blob: Blob mock.
      content_type: Content type.

    Returns:
      Blob state.
    """
    path = self._get_blob_path(blob)
    metadata = self._blob_state.get(path)
    generation = (
        1
        if (metadata is None or metadata.generation is None)
        else metadata.generation + 1
    )
    return self._init_blob_state(blob, generation, content_type)

  def blob_upload_from_string(
      self,
      blob: gcs_mock_types.GcsBlobType,
      data: bytes,
      content_type: Optional[str],
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
      checksum: Optional[str],
  ) -> blob_state_mock.BlobStateMock:
    """Uploads bytes to blob.

    Method should not be called directly. Call using method
    on google.cloud.storage.Client, google.cloud.storage.Bucket, and
    google.cloud.storage.Blob

    Args:
      blob: Desination blob to upload bytes to
      data: The data to store in this blob. If the value is text, it will be
        encoded as UTF-8.
      content_type: (Optional) Type of content being uploaded. Defaults to
        'text/plain'.
      if_generation_match: (Optional) See :ref:using-if-generation-match.
      if_generation_not_match: (Optional) See :ref:
        using-if-generation-not-match.
      if_metageneration_match: (Optional) See :ref:
        using-if-metageneration-match.
      if_metageneration_not_match: (Optional) See :ref:
        using-if-metageneration-not-match.
      checksum: (Optional) The type of checksum to compute to verify the
        integrity of the object. (No effect on mock)

    Returns:
      Blobs state of destination blob on gcs mock following upload.

    Raises:
      google.api_core.exceptions.Forbidden:   Bucket exists but client does not
        have permission to access; In mock generated as possible outcome when
        bucket does not exist to ensure code implemented in mock handles case.
      google.api_core.exceptions.NotFound: Bucket does not exist.
      google.api_core.exceptions.PreconditionFailed: Condition defined by (
          if_generation_match, or if_metageneration_match) failed.
      google.api_core.exceptions.NotModified: Condition defined by (
          if_generation_not_match, or if_metageneration_not_match) failed.
    """
    with self._lock:
      if checksum not in (None, 'md5', 'crc32c'):
        raise gcs_mock_types.GcsMockError('Invalid checksum parameter value.')
      http_method = gcs_mock_types.HttpMethod.PUT
      self._validate_bucket(blob.bucket, http_method)
      path = self._get_blob_path(blob)
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      os.makedirs(os.path.split(path)[0], exist_ok=True)
      with open(path, 'wb') as outfile:
        outfile.write(data)
      return self._init_blob_state_and_inc_generation(blob, content_type)

  def blob_compose(
      self,
      blob: gcs_mock_types.GcsBlobType,
      sources: List[gcs_mock_types.GcsBlobType],
      if_generation_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_source_generation_match: Optional[List[int]],
  ) -> blob_state_mock.BlobStateMock:
    """Concatenates multiple blobs to create a new blob.

    Method should not be called directly. Call using method
    on google.cloud.storage.Blob

    Args:
      blob: Destination blob.
      sources: Lost of source blob.
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

    Returns:
      Blobs state of destination blob on gcs mock following compose.

    Raises:
      MockGcsError: Called with invalid parameter values.
      google.api_core.exceptions.Forbidden: Blob bucket does not exist.
      google.api_core.exceptions.NotFound: Source blob file does not exist.
    """
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.POST
      self._validate_bucket(blob.bucket, http_method)
      self._test_valid_blob_metadata(
          blob, http_method, 'generation', if_generation_match, None
      )
      self._test_valid_blob_metadata(
          blob,
          http_method,
          'metageneration',
          if_metageneration_match,
          None,
      )
      if len(sources) < 1:
        raise gcs_mock_types.GcsMockError('Compose called with empty list.')
      if len(sources) > 32:
        raise gcs_mock_types.GcsMockError('Cannot compose more than 32 blobs.')
      # source blobs must be in destination bucket.
      for source_blob in sources:
        if source_blob.bucket.name != blob.bucket.name:
          raise gcs_mock_types.GcsMockError(
              'Destination and source blobs must be in the same bucket.'
          )
      content_type = blob.content_type
      if content_type is None:
        content_type = sources[0].content_type
        if content_type is None:
          content_type = gcs_mock_constants.DEFAULT_CONTENT_TYPE

      if if_source_generation_match is not None:
        if len(sources) != len(if_source_generation_match):
          raise gcs_mock_types.GcsMockError(
              'len(sources) != len(if_source_generation_match).'
          )
        for idx, source_blob in enumerate(sources):
          self._test_valid_blob_metadata(
              source_blob,
              http_method,
              'generation',
              if_source_generation_match[idx],
              None,
          )
      component_count = 0
      with io.BytesIO() as source_bytes:
        for source_blob in sources:
          self._validate_bucket(source_blob.bucket, http_method)
          path = self._validate_blob_exists_and_return_mock_path(
              source_blob, http_method
          )
          component_count += self._blob_state[path].component_count
          with open(path, 'rb') as infile:
            source_bytes.write(infile.read())
        path = self._get_blob_path(blob)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'wb') as outfile:
          outfile.write(source_bytes.getvalue())
      state = self._init_blob_state_and_inc_generation(blob, content_type)
      state.md5_hash = None
      state.component_count = component_count
      return state

  def blob_rewrite(
      self,
      dest: gcs_mock_types.GcsBlobType,
      source: gcs_mock_types.GcsBlobType,
      token: Optional[str],
      if_generation_match: Optional[int],
      if_generation_not_match: Optional[int],
      if_metageneration_match: Optional[int],
      if_metageneration_not_match: Optional[int],
      if_source_generation_match: Optional[int],
      if_source_generation_not_match: Optional[int],
      if_source_metageneration_match: Optional[int],
      if_source_metageneration_not_match: Optional[int],
  ) -> Tuple[
      Tuple[Optional[str], int, int], Optional[blob_state_mock.BlobStateMock]
  ]:
    """Rewrites a blob over another blob.

    Method should not be called directly. Call using method
    on google.cloud.storage.Blob

    Args:
      dest: Destiniation blob.
      source: Source blob
      token: Token returned after first call, initally should be None.
      if_generation_match: (Optional) See :ref:using-if-generation-match Note
        that the generation to be matched is that of the destination blob.
      if_generation_not_match: (Optional) See :ref:using-if-generation-not-match
        Note that the generation to be matched is that of the destination blob.
      if_metageneration_match: (Optional) See :ref:using-if-metageneration-match
        Note that the metageneration to be matched is that of the destination
        blob.
      if_metageneration_not_match: (Optional) See :ref:
        using-if-metageneration-not-match Note that the metageneration to be
        matched is that of the destination blob.
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

    Returns:
      Tuple (rewrite return tuple and state of destination blob).

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
    with self._lock:
      http_method = gcs_mock_types.HttpMethod.POST
      self._validate_bucket(dest.bucket, http_method)
      self._validate_bucket(source.bucket, http_method)

      self._test_valid_blob_metadata(
          dest,
          http_method,
          'generation',
          if_generation_match,
          if_generation_not_match,
      )
      self._test_valid_blob_metadata(
          dest,
          http_method,
          'metageneration',
          if_metageneration_match,
          if_metageneration_not_match,
      )
      self._test_valid_blob_metadata(
          source,
          http_method,
          'generation',
          if_source_generation_match,
          if_source_generation_not_match,
      )
      self._test_valid_blob_metadata(
          source,
          http_method,
          'metageneration',
          if_source_metageneration_match,
          if_source_metageneration_not_match,
      )
      content_type = dest.content_type
      if content_type is None:
        content_type = gcs_mock_utils.get_content_type_from_blob(source)
      # Simulate a rewrite loop in 2 cycles:
      source_path = self._validate_blob_exists_and_return_mock_path(
          source, http_method
      )
      file_size = os.path.getsize(source_path)
      if source.path == dest.path:
        blob_state = self._init_blob_state_and_inc_generation(
            dest, content_type
        )
        return ((None, file_size, file_size), blob_state)
      first_half = int(file_size / 2)
      if token is None:
        with open(source_path, 'rb') as infile:
          data = infile.read(first_half)
        token = f'{source.path}:{dest.path}_{str(uuid.uuid1())}'
        self._rewrite_temp_buffer[token] = io.BytesIO()
        self._rewrite_temp_buffer[token].write(data)
        len_bytes_written = len(data)
        blob_state = None
        return ((token, len_bytes_written, file_size), blob_state)

      with open(source_path, 'rb') as infile:
        infile.seek(first_half)
        data = infile.read()
      rewrite_buffer = self._rewrite_temp_buffer.get(token)
      if rewrite_buffer is None:
        raise gcs_mock_types.GcsMockError('Invalid blob.rewrite token')
      rewrite_buffer.write(data)
      bytes_written = rewrite_buffer.getvalue()
      len_bytes_written = len(bytes_written)
      dest_path = self._get_blob_path(dest)
      os.makedirs(os.path.split(dest_path)[0], exist_ok=True)
      with open(dest_path, 'wb') as outfile:
        outfile.write(bytes_written)
      del self._rewrite_temp_buffer[token]
      blob_state = self._init_blob_state_and_inc_generation(dest, content_type)
      token = None
      return ((token, len_bytes_written, file_size), blob_state)
