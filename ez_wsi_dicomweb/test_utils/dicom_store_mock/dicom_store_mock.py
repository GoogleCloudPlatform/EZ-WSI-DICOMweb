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
"""Context manager to mock http rest transactions against DICOM store for tests.

Limitations:
  Metadata returned using instance search will return full instance metadata
  regardless of query.
  https://cloud.google.com/healthcare-api/docs/how-tos/dicomweb#searching_using_dicom_tags
"""

from __future__ import annotations

import abc
import collections
import contextlib
import copy
import dataclasses
import enum
import http.client
import io
import json
import os
import re
from typing import Any, BinaryIO, Iterator, List, Mapping, Match, MutableMapping, NewType, Optional, Set, Tuple, Union
from unittest import mock

from absl.testing import absltest
import cachetools
import google.auth
import google.auth.credentials
import google_auth_httplib2
import httplib2
import pydicom
import requests
import requests_mock
import requests_toolbelt

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock_types
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_uid_value_map


_MAX_FRAME_CACHE_INSTANCE_SIZE = 10
_MAX_METADATA_CACHE_INSTANCE_SIZE = 100

# Operations empty string are used to referer to all instances in mock.
# Omiting study, series, and instance used to refer to all instances in store.
_ALL_DICOM_INSTANCES_IN_STORE = ('', '', '')

_BINARY_DICOM_TAG_VR_CODES = ('OB', 'OD', 'OF', 'OL', 'OV', 'OW')
_BULK_DATA_URI = 'BulkDataURI'
_DICOM_PIXEL_DATA_TAG = '7FE00010'
_DICOM_SERIES_INSTANCE_UID_TAG = '0020000E'
_DICOM_SOP_INSTANCE_UID_TAG = '00080018'
_DICOM_STUDY_INSTANCE_UID_TAG = '0020000D'
_INLINE_BINARY = 'InlineBinary'
_SQ = 'SQ'
_VALUE = 'Value'
_VR = 'vr'
_GET_REQUESTED_TRANSFER_SYNTAX = re.compile(
    r'.*;[ ]+transfer-syntax=(.*)', re.IGNORECASE
)
_MOCK_TOKEN = 'MOCK_BEARER_TOKEN'

_DicomUidTriple = dicom_store_mock_types.DicomUidTriple

_DicomUidTripleSourceType = Union[
    str,
    Mapping[str, Any],
    pydicom.Dataset,
    pydicom.FileDataset,
    _DicomUidTriple,
    dicom_store_mock_types.AbstractGetDicomUidTripleInterface,
]


class RequestMethod(enum.Enum):
  GET = 'GET'
  POST = 'POST'
  DELETE = 'DELETE'


CustomContentType = NewType('CustomContentType', str)
_GET_DICOM_INSTANCE_BASE_CONTENT = 'application/dicom; transfer-syntax='

# Mime type mapping associated for DICOM transfer syntax uid.
# https://cloud.google.com/healthcare-api/docs/dicom#json_metadata_and_bulk_data_requests
_DICOM_TRANSFER_SYNTAX_TO_MIME_TYPE = collections.defaultdict(
    lambda: 'application/octet-stream',
    {
        '1.2.840.10008.1.2.4.50': 'image/jpeg',
        '1.2.840.10008.1.2.​4.​51': 'image/jpeg',
        '1.2.840.10008.1.2.​4.​57': 'image/jpeg',
        '1.2.840.10008.1.2.4.70': 'image/jpeg',
        '1.2.840.10008.1.2.4.90': 'image/jp2',
        '1.2.840.10008.1.2.4.91': 'image/jp2',
        '1.2.840.10008.1.2.4.92': 'image/jpx',
        '1.2.840.10008.1.2.4.93': 'image/jpx',
        '1.2.840.10008.1.2.​4.​80': 'image/x-jls',
        '1.2.840.10008.1.2.4.81': 'image/x-jls',
        '1.2.840.10008.1.2.5': 'image/x-dicom-rle',
    },
)

_UNENCAPSULATED_TRANSFER_SYNTAXES = frozenset([
    '1.2.840.10008.1.2.1',  # 	Explicit VR Little Endian
    '1.2.840.10008.1.2',  # Implicit VR Endian: Default Transfer Syntax
    '1.2.840.10008.1.2.1.99',  # Deflated Explicit VR Little Endian
    '1.2.840.10008.1.2.2',  # Explicit VR Big Endian
])


class ContentType(enum.Enum):
  TEXT_PLAIN = 'text/plain; charset=us-ascii'
  APPLICATION_DICOM_JSON = 'application/dicom+json; charset=utf-8'
  APPLICATION_DICOM_NO_TRANSCODE = f'{_GET_DICOM_INSTANCE_BASE_CONTENT}*'
  TEXT_HTML = 'text/html'
  APPLICATION_DICOM_XML = 'application/dicom+xml'
  APPLICATION_JSON = 'application/json; charset=UTF-8'
  MULTIPART_RELATED = 'multipart/related'
  UNTRANSCODED_FRAME_REQUEST = (
      'multipart/related; type="application/octet-stream"; transfer-syntax=*'
  )


@dataclasses.dataclass(frozen=True)
class MockHttpResponse:
  regex: str
  http_method: RequestMethod
  http_status: int
  http_body: Union[str, bytes]
  http_content_type: Union[ContentType, CustomContentType] = (
      ContentType.TEXT_PLAIN
  )


# pydicom version 3 is deprecating some pydicom v2 functionality.
# This code duplicates code in pydicom_version_util.py to avoid dependency.

# TODO: b/373988830 - Remove when pydicom version 2 is no longer used.

_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])


def _generate_frames(buffer: bytes, number_of_frames: int) -> Iterator[bytes]:
  number_of_frames = int(number_of_frames)
  # pytype: disable=module-attr
  if _PYDICOM_MAJOR_VERSION <= 2:
    return pydicom.encaps.generate_pixel_data_frame(buffer, number_of_frames)
  return pydicom.encaps.generate_frames(
      buffer, number_of_frames=number_of_frames
  )
  # pytype: enable=module-attr


def _set_little_endian_explicit_vr(dcm: pydicom.FileDataset) -> None:
  if _PYDICOM_MAJOR_VERSION > 2:
    return
  dcm.is_little_endian = True
  dcm.is_implicit_VR = False


def _save_as(dcm: pydicom.FileDataset, filename: Union[str, BinaryIO]) -> None:
  if _PYDICOM_MAJOR_VERSION <= 2:
    dcm.save_as(filename)
  else:
    # pylint: disable=unexpected-keyword-arg
    # pytype: disable=wrong-keyword-args
    dcm.save_as(filename, little_endian=True, implicit_vr=False)
    # pytype: enable=wrong-keyword-args
    # pylint: enable=unexpected-keyword-arg


# End code duplicates code in pydicom_version_util.py to avoid dependency.


def _is_iterator_empty(itr: Iterator[Any]) -> bool:
  try:
    next(itr)
    return False
  except StopIteration:
    return True


def _does_iterator_have_items(itr: Iterator[Any]) -> bool:
  return not _is_iterator_empty(itr)


class _FilterBinaryTagOperation(enum.Enum):
  NONE = 0
  REMOVE = 1
  CREATE_BULKDATA_URI = 2


def _filter_binary_tags(
    metadata: MutableMapping[str, Any],
    operation: _FilterBinaryTagOperation,
    path: str = '',
) -> MutableMapping[str, Any]:
  """Filter DICOM json metadata to remove binary tags or add buldata uri."""
  remove_tag_list = []
  for key, value in metadata.items():
    vr_code = value.get(_VR, '').upper()
    if vr_code == _SQ:
      cr_uri = operation == _FilterBinaryTagOperation.CREATE_BULKDATA_URI
      for index, sq_value in enumerate(value.get(_VALUE, [])):
        new_path = _join_url_parts(path, f'{key}/{index}') if cr_uri else path
        _filter_binary_tags(sq_value, operation, new_path)
      continue
    elif vr_code in _BINARY_DICOM_TAG_VR_CODES:
      if operation == _FilterBinaryTagOperation.REMOVE:
        remove_tag_list.append(key)
      elif operation == _FilterBinaryTagOperation.CREATE_BULKDATA_URI:
        del value[_INLINE_BINARY]
        value[_BULK_DATA_URI] = _join_url_parts(path, key)
      else:
        raise ValueError(f'Unsupported operation: {operation}')
  for key in remove_tag_list:
    del metadata[key]
  return metadata


def _pydicom_file_dataset_to_json(
    dcm: pydicom.FileDataset,
    operation: _FilterBinaryTagOperation,
    dicomweb_path: str = '',
) -> Mapping[str, Any]:
  """Coverts pydicom to json metadata representation.

  Strips PixelData tag and icc profile tag to model what store returns and
  save internal memory in file system dicom store metadata cache.

  Args:
    dcm: Pydicom file dataset.
    operation: Operation to perform on binary tags.
    dicomweb_path: Dicom web path to DICOM store.

  Returns:
    JSON representation.
  """
  path = ''
  if operation == _FilterBinaryTagOperation.CREATE_BULKDATA_URI:
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    sop_instance_uid = dcm.SOPInstanceUID
    path = f'{dicomweb_path}/studies/{study_uid}/series/{series_uid}/instances/{sop_instance_uid}/bulkdata'
  result = dcm.to_json_dict()
  # always remove pixel data from metadata representation.
  try:
    del result[_DICOM_PIXEL_DATA_TAG]
  except KeyError:
    pass
  result.update(dcm.file_meta.to_json_dict())
  if operation != _FilterBinaryTagOperation.NONE:
    _filter_binary_tags(result, operation, path)
  return result


class _MockDicomStoreAbstractStorage(metaclass=abc.ABCMeta):
  """Abstract class for storage backing DICOM store mock."""

  def __init__(self):
    # self._frame_cache stores a list of the encapsulated frames
    # bytes stored in DICOM instances to enable rapid random access.
    # The cache is initialized for a DICOM instance on the first frame request.
    # PyDICOM uses a generator style interface to access blob bytes stored
    # within encapsulated frames. Without the cache this makes repeated random
    # access using to frame bytes inefficient.
    self._frame_cache = cachetools.LRUCache(
        maxsize=_MAX_FRAME_CACHE_INSTANCE_SIZE
    )

  def _get_uid_tuple(
      self, file_dataset: pydicom.FileDataset
  ) -> Tuple[str, str, str]:
    return (
        file_dataset.StudyInstanceUID,
        file_dataset.SeriesInstanceUID,
        file_dataset.SOPInstanceUID,
    )

  @abc.abstractmethod
  def add_instance(self, file_dataset: pydicom.FileDataset) -> bool:
    """Returns True if instance added to storage."""

  @abc.abstractmethod
  def remove_instance(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> bool:
    """Returns True if instance(s) removed from storage."""

  def _remove_instance_from_frame_cache(
      self, uid: Tuple[str, str, str]
  ) -> None:
    """Removes instances from frame cache.

    Ok if frames not in cache, called when instance is removed from store.
    Instance may not have cached frames.

    Args:
      uid: StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID to remove.
    """
    try:
      del self._frame_cache[uid]
    except KeyError:
      pass

  def get_instance_frame_bytes(
      self, dcm: pydicom.FileDataset
  ) -> Optional[List[bytes]]:
    return self._frame_cache.get(self._get_uid_tuple(dcm))

  def set_instance_frame_bytes(
      self, dcm: pydicom.FileDataset, frame_bytes: List[bytes]
  ) -> None:
    self._frame_cache[self._get_uid_tuple(dcm)] = frame_bytes

  @abc.abstractmethod
  def get_instances(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[pydicom.FileDataset]:
    """Returns iterator of pydicom dataset with uid values that match query."""

  @abc.abstractmethod
  def get_metadata(
      self,
      dicomweb_path: str,
      bulkdata_uri_enabled: bool,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[Mapping[str, Any]]:
    """Returns iterator of DICOM json with uid values that match query."""

  def has_instances_in_store(self) -> bool:
    return _does_iterator_have_items(
        self.get_instances(*_ALL_DICOM_INSTANCES_IN_STORE)
    )

  def is_empty(self) -> bool:
    return _is_iterator_empty(
        self.get_instances(*_ALL_DICOM_INSTANCES_IN_STORE)
    )


class _MockDicomStoreMemoryStorage(_MockDicomStoreAbstractStorage):
  """Implementation to back mock storage in memory."""

  def __init__(self):
    super().__init__()
    self._instance_map = dicom_uid_value_map.DicomUidValueMap[
        pydicom.FileDataset
    ]()

  def add_instance(self, file_dataset: pydicom.FileDataset) -> bool:
    uid = self._get_uid_tuple(file_dataset)
    if self._instance_map.add_instance(uid, file_dataset):
      return True
    existing_instance = list(self._instance_map.get_instances(uid))
    return existing_instance[0] == file_dataset

  def remove_instance(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> bool:
    uid = (study_instance_uid, series_instance_uid, sop_instance_uid)
    uids_to_removed = [
        self._get_uid_tuple(ds) for ds in self._instance_map.get_instances(uid)
    ]
    if not uids_to_removed:
      return False
    self._instance_map.remove_instances(uid)
    for uid_to_remove in uids_to_removed:
      self._remove_instance_from_frame_cache(uid_to_remove)
    return True

  def get_instances(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[pydicom.FileDataset]:
    uid = (study_instance_uid, series_instance_uid, sop_instance_uid)
    return self._instance_map.get_instances(uid)

  def get_metadata(
      self,
      dicomweb_path: str,
      bulkdata_uri_enabled: bool,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[Mapping[str, Any]]:
    uid = (study_instance_uid, series_instance_uid, sop_instance_uid)
    operation = (
        _FilterBinaryTagOperation.CREATE_BULKDATA_URI
        if bulkdata_uri_enabled
        else _FilterBinaryTagOperation.REMOVE
    )
    for instance in self._instance_map.get_instances(uid):
      yield _pydicom_file_dataset_to_json(
          instance, operation, dicomweb_path=dicomweb_path
      )


class _UnableToReadDicomInstanceError(Exception):
  pass


@dataclasses.dataclass(frozen=True)
class _FileSystemDicomData:
  dicom_data_path: str
  uid: Tuple[str, str, str]


class _MockDicomStoreFileSystemStorage(_MockDicomStoreAbstractStorage):
  """Implementation to back mock storage on disk."""

  def __init__(self, file_system_backed_storage_path: str):
    super().__init__()
    self._metadata_cache: MutableMapping[
        Tuple[str, str, str], Mapping[str, Any]
    ] = cachetools.LRUCache(maxsize=_MAX_METADATA_CACHE_INSTANCE_SIZE)
    self._file_system_backed_storage_path = file_system_backed_storage_path
    self._instance_map = dicom_uid_value_map.DicomUidValueMap[
        _FileSystemDicomData
    ]()
    for root, _, filelist in os.walk(self._file_system_backed_storage_path):
      for filename in filelist:
        path = os.path.join(root, filename)
        try:
          uid = self._get_dicom_file_uid(path)
        except _UnableToReadDicomInstanceError:
          continue
        self._instance_map.add_instance(uid, _FileSystemDicomData(path, uid))

  @property
  def dicom_storagefile_system_backed_storage_path(self) -> str:
    return self._file_system_backed_storage_path

  def _get_dicom_file_uid(self, path: str) -> Tuple[str, str, str]:
    """Returns uid tuple stored in DICOM instance.

    Args:
      path: File path to binary Part10 DICOM instance.

    Raises:
      _UnableToReadDicomInstanceError: Unable to read dicom instance.
    """
    try:
      with pydicom.dcmread(
          path,
          specific_tags=[
              _DICOM_STUDY_INSTANCE_UID_TAG,
              _DICOM_SERIES_INSTANCE_UID_TAG,
              _DICOM_SOP_INSTANCE_UID_TAG,
          ],
      ) as dcm:
        return self._get_uid_tuple(dcm)
    except (
        OSError,
        FileNotFoundError,
        pydicom.errors.InvalidDicomError,
    ) as exp:
      raise _UnableToReadDicomInstanceError() from exp

  def _load_instance(self, path: str) -> pydicom.FileDataset:
    """Returns DICOM instance for UID tripple.

    Args:
      path: File path of desired instance.

    Returns:
      Instance loaded from file system.

    Raises:
      _UnableToReadDicomInstanceError: Unable to find or read instance.
    """
    if not path:
      raise _UnableToReadDicomInstanceError()
    try:
      return pydicom.dcmread(path, defer_size='100 KB')
    except (
        OSError,
        FileNotFoundError,
        pydicom.errors.InvalidDicomError,
    ) as exp:
      raise _UnableToReadDicomInstanceError() from exp

  def _remove_file_path(self, path: str) -> None:
    """Removes file path from file system."""
    base_dir = self._file_system_backed_storage_path.rstrip('/')
    if not path.startswith(base_dir):
      return
    try:
      os.remove(path)
    except (OSError, FileNotFoundError):
      pass
    dirname = os.path.dirname(path)
    # remove empty dirs
    try:
      while dirname.rstrip('/') != base_dir and dirname.startswith(base_dir):
        os.rmdir(dirname)
        dirname = os.path.dirname(dirname)
    except (OSError, FileNotFoundError):
      pass

  def _create_dicom(self, file_dataset: pydicom.FileDataset, path: str) -> bool:
    try:
      os.makedirs(os.path.dirname(path), exist_ok=True)
      _save_as(file_dataset, path)
      return True
    except (
        OSError,
        FileNotFoundError,
        pydicom.errors.InvalidDicomError,
    ):
      self._remove_file_path(path)

  def _create_instance_file_path(
      self, file_dataset: pydicom.FileDataset
  ) -> bool:
    uid = self._get_uid_tuple(file_dataset)
    path = os.path.join(
        self._file_system_backed_storage_path, uid[0], uid[1], f'{uid[2]}.dcm'
    )
    self._instance_map.add_instance(uid, _FileSystemDicomData(path, uid))
    if self._create_dicom(file_dataset, path):
      return True
    self._instance_map.remove_instances(uid)
    return False

  def add_instance(self, file_dataset: pydicom.FileDataset) -> bool:
    uid = self._get_uid_tuple(file_dataset)
    existing_instance = list(self.get_instances(*uid))
    if not existing_instance:
      return self._create_instance_file_path(file_dataset)
    return existing_instance[0] == file_dataset

  def _remove_cached_instances_and_file(
      self, instance: _FileSystemDicomData
  ) -> None:
    self._remove_instance_from_frame_cache(instance.uid)
    try:
      del self._metadata_cache[instance.uid]
    except KeyError:
      pass
    self._remove_file_path(instance.dicom_data_path)

  def remove_instance(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> bool:
    uid = (study_instance_uid, series_instance_uid, sop_instance_uid)
    uids_to_removed = list(self._instance_map.get_instances(uid))
    if not uids_to_removed:
      return False
    self._instance_map.remove_instances(uid)
    for uid_to_remove in uids_to_removed:
      self._remove_cached_instances_and_file(uid_to_remove)
    return True

  def _remove_list_of_instances(
      self, instances_to_remove: List[_FileSystemDicomData]
  ) -> None:
    for instance in instances_to_remove:
      self._instance_map.remove_instances(instance.uid)
      self._remove_cached_instances_and_file(instance)

  def get_instances(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[pydicom.FileDataset]:
    instances_to_remove = []
    for instance in self._instance_map.get_instances(
        (study_instance_uid, series_instance_uid, sop_instance_uid)
    ):
      try:
        yield self._load_instance(instance.dicom_data_path)
      except _UnableToReadDicomInstanceError:
        instances_to_remove.append(instance)
    self._remove_list_of_instances(instances_to_remove)

  def get_metadata(
      self,
      dicomweb_path: str,
      bulkdata_uri_enabled: bool,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Iterator[Mapping[str, Any]]:
    instances_to_remove = []
    operation = (
        _FilterBinaryTagOperation.CREATE_BULKDATA_URI
        if bulkdata_uri_enabled
        else _FilterBinaryTagOperation.REMOVE
    )
    for instance in self._instance_map.get_instances(
        (study_instance_uid, series_instance_uid, sop_instance_uid)
    ):
      if not os.path.exists(instance.dicom_data_path):
        instances_to_remove.append(instance)
        continue
      metadata = self._metadata_cache.get(instance.uid)
      if metadata is not None:
        yield metadata
        continue
      try:
        dataset = self._load_instance(instance.dicom_data_path)
        metadata = _pydicom_file_dataset_to_json(
            dataset, operation, dicomweb_path=dicomweb_path
        )
        self._metadata_cache[instance.uid] = metadata
        yield metadata
      except _UnableToReadDicomInstanceError:
        instances_to_remove.append(instance)
    self._remove_list_of_instances(instances_to_remove)


def _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
    accept_header: str, dcm: pydicom.FileDataset
) -> bool:
  """Test if accept header matches DICOM Transfer Syntax."""
  if (
      accept_header == 'image/jpeg'
      and dcm.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.50'
  ):
    return True
  result = _GET_REQUESTED_TRANSFER_SYNTAX.fullmatch(accept_header)
  if result is None:
    return False
  return result.groups()[0] == dcm.file_meta.TransferSyntaxUID


def _get_dicom_uid(
    instance_identifier: _DicomUidTripleSourceType,
) -> _DicomUidTriple:
  """Converts DICOM instance into DicomUidTriple."""
  if isinstance(instance_identifier, _DicomUidTriple):
    return instance_identifier
  if isinstance(
      instance_identifier,
      dicom_store_mock_types.AbstractGetDicomUidTripleInterface,
  ):
    return instance_identifier.get_dicom_uid_triple()

  if isinstance(instance_identifier, str):
    dcm = pydicom.dcmread(instance_identifier, defer_size='512 KB')
  elif isinstance(instance_identifier, Mapping):
    dcm = pydicom.Dataset.from_json(json.dumps(instance_identifier))
  else:
    dcm = instance_identifier
  return _DicomUidTriple(
      dcm.StudyInstanceUID, dcm.SeriesInstanceUID, dcm.SOPInstanceUID
  )


def _build_response(
    status_code: int,
    msg: Union[str, bytes],
    content: Union[ContentType, CustomContentType] = ContentType.TEXT_PLAIN,
) -> requests.Response:
  """Generates request Response to return."""
  resp = requests.Response()
  resp.status_code = status_code
  if isinstance(content, ContentType):
    resp.headers['Content-Type'] = content.value
  else:
    resp.headers['Content-Type'] = str(content)
  if isinstance(msg, str):
    msg = msg.encode('utf-8')
  resp.raw = io.BytesIO(msg)
  return resp


def _convert_to_pydicom_file_dataset(
    dcm: Union[str, Mapping[str, Any], pydicom.FileDataset],
) -> pydicom.FileDataset:
  """Converts input to pydicom.FileDataset."""
  if isinstance(dcm, pydicom.FileDataset):
    return dcm
  if isinstance(dcm, str):
    return pydicom.dcmread(dcm)
  # isinstance(dataset, Mapping[str, Any])
  file_meta = {}
  main_body = {}
  for tag, value in dcm.items():
    if tag.startswith('0002'):  # pytype: disable=attribute-error
      file_meta[tag] = value
    else:
      main_body[tag] = value
  file_meta = pydicom.dataset.FileMetaDataset(
      pydicom.Dataset.from_json(file_meta)
  )
  dicom_instance = pydicom.dataset.FileDataset(
      'mock_file.dcm',
      dataset=pydicom.Dataset.from_json(main_body),
      file_meta=file_meta,
  )
  _set_little_endian_explicit_vr(dicom_instance)
  dicom_instance.file_meta.MediaStorageSOPClassUID = dicom_instance.SOPClassUID
  dicom_instance.file_meta.MediaStorageSOPInstanceUID = (
      dicom_instance.SOPInstanceUID
  )
  return dicom_instance


def _pydicom_file_dataset_to_bytes(dcm: pydicom.FileDataset) -> bytes:
  instance_bytes = io.BytesIO()
  _save_as(dcm, instance_bytes)
  return instance_bytes.getvalue()


def _join_url_parts(p1: str, p2: str) -> str:
  p1 = p1.rstrip('/')
  p2 = p2.lstrip('/')
  return f'{p1}/{p2}'


def _get_accept(headers: Mapping[str, str]) -> str:
  for key, value in headers.items():
    if key.lower() == 'accept':
      return value
  return ''


def _get_content_type(headers: Mapping[str, str]) -> str:
  for key, value in headers.items():
    if key.lower() == 'content-type':
      return value
  return ''


def _replace_bulkdata(
    dicom_json: MutableMapping[str, Any],
    content_location_map: Mapping[str, bytes],
) -> None:
  """Replace bulkdata tags with content defined in multipart content locations.

  Args:
    dicom_json: DICOM Json.
    content_location_map: Map between BulkURI content location and binary blob
      to encode at location.

  Returns:
    None
  """
  value_list = list(dicom_json.values())
  while value_list:
    value = value_list.pop()
    if isinstance(value, list):
      for dataset in value:
        for dataset_value in dataset.values():
          value_list.append(dataset_value)
      continue
    bulkdata_uri = value.get(_BULK_DATA_URI)
    if bulkdata_uri is None:
      continue
    content = content_location_map.get(bulkdata_uri)
    if content is not None:
      del value[_BULK_DATA_URI]
      value[_INLINE_BINARY] = content


def _build_pydicom_dicom_from_request_json(
    content: bytes,
    content_location_map: Mapping[str, bytes],
) -> pydicom.FileDataset:
  """Converts DICOM store formatted json into PyDicom FileDataset.

  Args:
    content: bytes received in multipart DICOM json data.
    content_location_map: Mapping of content-locations to bytes for data in
      multi-part related request.

  Returns:
    PyDicom FileDataset represented by JSON

  Raises:
    ValueError: Invalid DICOM JSON.
    KeyError: Invalid DICOM JSON.
    IndexError: Invalid DICOM JSON.
    pydicom.errors.InvalidDicomError: Invalid DICOM
    json.decoder.JSONDecodeError: Invalid DICOM JSON.
  """
  all_tags = json.loads(content)
  if isinstance(all_tags, list):
    if len(all_tags) != 1:
      raise ValueError(f'Error found {len(all_tags)} instances in part.')
    all_tags = all_tags[0]
  if not isinstance(all_tags, dict):
    raise ValueError('Invalid formatted DICOM JSON.')
  file_meta_tags = {}
  dataset_tags = {}
  for address, value in all_tags.items():
    if address.startswith('0002'):
      file_meta_tags[address] = value
    else:
      dataset_tags[address] = value
  _replace_bulkdata(dataset_tags, content_location_map)
  file_meta = pydicom.dataset.Dataset().from_json(json.dumps(file_meta_tags))
  base_dataset = pydicom.Dataset().from_json(json.dumps(dataset_tags))
  return pydicom.dataset.FileDataset(
      '',
      base_dataset,
      preamble=b'\0' * 128,
      file_meta=pydicom.dataset.FileMetaDataset(file_meta),
  )


def _decode_pydicom(data: bytes) -> pydicom.FileDataset:
  with io.BytesIO(data) as instance_bytes:
    return pydicom.dcmread(instance_bytes)


def _mock_apply_credentials(
    headers: MutableMapping[Any, Any], token: Optional[str] = None
) -> None:
  headers['authorization'] = 'Bearer {}'.format(token or _MOCK_TOKEN)


class MockDicomStoreClient(contextlib.ContextDecorator):
  """Context manager mocks individual DICOM Store."""

  def __init__(
      self,
      dicomweb_path: str,
      mock_credential: bool = True,
      mock_credential_project: str = 'mock_gcp_project',
      mock_request: Optional[requests_mock.Mocker] = None,
      read_auth_bearer_tokens: Optional[List[str]] = None,
      write_auth_bearer_tokens: Optional[List[str]] = None,
      real_http: bool = False,
      bulkdata_uri_enabled: bool = True,
  ):
    self._read_auth_bearer_tokens = read_auth_bearer_tokens
    self._write_auth_bearer_tokens = write_auth_bearer_tokens
    self._context_manager_entered = False
    self._mock_dicom_store_request_entered = False
    self._dicomweb_path = dicomweb_path
    self._bulkdata_uri_enabled = bulkdata_uri_enabled
    if not mock_credential:
      self._credentials_mock = None
    else:
      credentials_mock = mock.create_autospec(
          google.auth.credentials.Credentials, instance=True
      )
      type(credentials_mock).token = mock.PropertyMock(return_value=_MOCK_TOKEN)
      type(credentials_mock).valid = mock.PropertyMock(return_value='True')
      type(credentials_mock).expired = mock.PropertyMock(return_value='False')
      credentials_mock.apply.side_effect = _mock_apply_credentials
      self._credentials_mock = mock.patch(
          'google.auth.default',
          return_value=(credentials_mock, mock_credential_project),
      )
    self._dicom_storage = _MockDicomStoreMemoryStorage()
    if mock_request is None:
      self._mock_dicom_store_request = requests_mock.Mocker(real_http=real_http)
      self._request_mock_started = False
    else:
      self._mock_dicom_store_request = mock_request
      self._request_mock_started = True
    self._mock_responses = []
    self._request_handlers = [
        self._instance_metadata_request,
        self._study_level_series_metadata_request,
        self._series_instance_metadata_request,
        self._study_metadata_request,
        self._add_study_instance,
        self._download_instance,
        self._delete_sop_instance_uid,
        self._delete_series_instance_uid,
        self._delete_study_instance_uid,
        self._download_frame,
        self._store_level_study_request,
        self._bulkdata_request,
    ]

  @property
  def bulkdata_uri_support_enabled(self) -> bool:
    return self._bulkdata_uri_enabled

  def set_dicom_store_memory_storage(self) -> None:
    """Set mock DICOM store to back instance storage in memory."""
    if isinstance(self._dicom_storage, _MockDicomStoreMemoryStorage):
      return
    old_interface = self._dicom_storage
    self._dicom_storage = _MockDicomStoreMemoryStorage()
    # Copy all instances from old interface to new interface.
    for instance in old_interface.get_instances(*_ALL_DICOM_INSTANCES_IN_STORE):
      self._dicom_storage.add_instance(instance)

  def set_dicom_store_disk_storage(
      self, path: Union[str, absltest._TempDir]
  ) -> None:
    """Set mock DICOM store to back instance storage using file system."""
    if isinstance(path, absltest._TempDir):  # pylint: disable=protected-access
      path = path.full_path
    if (
        isinstance(self._dicom_storage, _MockDicomStoreFileSystemStorage)
        and self._dicom_storage.dicom_storagefile_system_backed_storage_path
        == path
    ):
      return
    old_interface = self._dicom_storage
    self._dicom_storage = _MockDicomStoreFileSystemStorage(path)
    # Copy all instances from old interface to new interface.
    for instance in old_interface.get_instances(*_ALL_DICOM_INSTANCES_IN_STORE):
      self._dicom_storage.add_instance(instance)

  def _can_read(self, headers: Mapping[str, str]) -> bool:
    if self._read_auth_bearer_tokens is None:
      return True
    for key, value in headers.items():
      if (
          key.lower() == 'authorization'
          and value in self._read_auth_bearer_tokens
      ):
        return True
    return False

  def _can_write(self, headers: Mapping[str, str]) -> bool:
    if self._write_auth_bearer_tokens is None:
      return True
    for key, value in headers.items():
      if (
          key.lower() == 'authorization'
          and value in self._write_auth_bearer_tokens
      ):
        return True
    return False

  @property
  def mock_request(self) -> requests_mock.Mocker:
    """Returns requests mock used by the mock."""
    return self._mock_dicom_store_request

  def add_instance(
      self, instance_data: Union[str, Mapping[str, Any], pydicom.FileDataset]
  ) -> bool:
    """Adds an instance to the mocked store."""
    return self._dicom_storage.add_instance(
        _convert_to_pydicom_file_dataset(instance_data)
    )

  def remove_instance(
      self, instance_data: Union[str, Mapping[str, Any], pydicom.FileDataset]
  ) -> bool:
    """Adds an instance to the mocked store."""
    instance = _convert_to_pydicom_file_dataset(instance_data)
    return self._dicom_storage.remove_instance(
        instance.StudyInstanceUID,
        instance.SeriesInstanceUID,
        instance.SOPInstanceUID,
    )

  def __enter__(self) -> MockDicomStoreClient:
    if self._context_manager_entered:
      raise ValueError('Context manager has already been entered')
    self._context_manager_entered = True
    if self._credentials_mock is not None:
      self._credentials_mock.__enter__()
    if not self._request_mock_started:
      self._mock_dicom_store_request.__enter__()
      self._mock_dicom_store_request_entered = True
    self._mock_dicom_store_request.add_matcher(self.handle_request)
    return self

  def __exit__(self, exc_type, exc, exc_tb):
    if not self._context_manager_entered:
      raise ValueError('Context manager has not been entered')
    self._context_manager_entered = False
    if self._mock_dicom_store_request_entered:
      self._mock_dicom_store_request.__exit__(exc_type, exc, exc_tb)
    if self._credentials_mock is not None:
      self._credentials_mock.__exit__(exc_type, exc, exc_tb)

  def handle_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Apply request to DICOM store request handlers.

    Args:
      request: Request to apply to handler.

    Returns:
      Result of first handler to handle request or None.
    """
    for mock_response in self._mock_responses:
      if (
          self._parse_url(
              request, mock_response.regex, mock_response.http_method
          )
          is not None
      ):
        return _build_response(
            mock_response.http_status,
            mock_response.http_body,
            mock_response.http_content_type,
        )
    for handler in self._request_handlers:
      response = handler(request)
      if response is not None:
        return response
    return None

  def _get_pydicom_file_dataset_frame_bytes(
      self, dcm: pydicom.FileDataset, index: int
  ) -> Optional[bytes]:
    """Returns encapsulated frame from loaded DICOM file.

    Args:
      dcm: Loaded DICOM file.
      index: DICOM Frame Index to return (1 = First frame)

    Returns:
      Frame bytes.

    Raises:
      IndexError: Requested DICOM frame index is out of bounds.
    """
    index -= 1  # DICOM frame numbering start at 1
    if index < 0:
      raise IndexError('Invalid Index')
    frame_bytes = self._dicom_storage.get_instance_frame_bytes(dcm)
    if frame_bytes is not None:
      return frame_bytes[index]
    if dcm.file_meta.TransferSyntaxUID in _UNENCAPSULATED_TRANSFER_SYNTAXES:
      number_of_frames = dcm.NumberOfFrames
      step = int(len(dcm.PixelData) / number_of_frames)
      frame_bytes = [
          dcm.PixelData[fnum * step : (fnum + 1) * step]
          for fnum in range(number_of_frames)
      ]
    else:
      frame_bytes = list(_generate_frames(dcm.PixelData, dcm.NumberOfFrames))
    self._dicom_storage.set_instance_frame_bytes(dcm, frame_bytes)
    return frame_bytes[index]

  def _get_instances(
      self,
      study_instance_uid: Optional[str],
      series_instance_uid: Optional[str],
      sop_instance_uid: Optional[str],
  ) -> Iterator[pydicom.FileDataset]:
    """Returns list of instances in mocked store with matching UIDs."""
    if study_instance_uid is None:
      study_instance_uid = ''
    if series_instance_uid is None:
      series_instance_uid = ''
    if sop_instance_uid is None:
      sop_instance_uid = ''
    return self._dicom_storage.get_instances(
        study_instance_uid, series_instance_uid, sop_instance_uid
    )

  def _get_dicom_metadata_iterator(
      self,
      study_instance_uid: Optional[str] = '',
      series_instance_uid: Optional[str] = '',
      sop_instance_uid: Optional[str] = '',
  ) -> Iterator[Mapping[str, Any]]:
    """Returns DICOM metadata associated with study or series in mocked store."""
    if study_instance_uid is None:
      study_instance_uid = ''
    if series_instance_uid is None:
      series_instance_uid = ''
    if sop_instance_uid is None:
      sop_instance_uid = ''
    return self._dicom_storage.get_metadata(
        self._dicomweb_path,
        self._bulkdata_uri_enabled,
        study_instance_uid,
        series_instance_uid,
        sop_instance_uid,
    )

  def _parse_url(
      self,
      request: Union[requests.Request, requests.PreparedRequest],
      reg_ex: str,
      method: RequestMethod,
  ) -> Optional[Match[str]]:
    if request.method != method.value:
      return None
    url_regex = _join_url_parts(self._dicomweb_path, reg_ex)
    return re.fullmatch(url_regex, request.url, re.IGNORECASE)

  def _study_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a study metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/instances/{0,1}(\?.*)?',
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    series_instance_uid = self._parse_url_parameters(
        parts, 1, 'SeriesInstanceUID'
    )
    sop_instance_uid = self._parse_url_parameters(parts, 1, 'SOPInstanceUID')
    metadata_iterator = self._get_dicom_metadata_iterator(
        parts[0], series_instance_uid, sop_instance_uid
    )
    metadata = self._filter_metadata_response(parts, 1, metadata_iterator)
    if metadata:
      return _build_response(
          http.HTTPStatus.OK,
          json.dumps(metadata),
          ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.HTTPStatus.NO_CONTENT, '', ContentType.TEXT_HTML
    )

  def _parse_url_parameters(
      self,
      parsed_url_parts: Tuple[str, ...],
      index: int,
      parameter: str,
  ) -> Optional[str]:
    """Parses parameter(s) from url parameter string.

    Args:
      parsed_url_parts: Tuple containing strings parsed from url.
      index: Index of string to parse from url_part.
      parameter: Single parameter to search for.

    Returns:
      Parameter value or None.
    """
    if len(parsed_url_parts) <= index:
      return None
    url_part = parsed_url_parts[index]
    if url_part is None or not url_part:
      return None
    parameter_search = re.fullmatch(
        r'\?.*' f'{parameter}' r'=([^&]*).*', url_part, re.IGNORECASE
    )
    if parameter_search is not None:
      return parameter_search.groups()[0]
    return None

  def _limit_metadata(
      self, metadata: Iterator[Mapping[str, Any]], limit: Optional[str]
  ) -> List[Mapping[str, Any]]:
    """Limit length of returned JSON metadata.

    Args:
      metadata: List DICOM metadata.
      limit: Max number of datasets to return.

    Returns:
      Clipped List DICOM metadata.
    """
    if limit is None or not limit:
      return list(metadata)
    limit = int(limit)
    limited_metadata = []
    for count, data in enumerate(metadata):
      if count >= limit:
        break
      limited_metadata.append(data)
    return limited_metadata

  def _filter_metadata(
      self,
      metadata: Iterator[Mapping[str, Any]],
      tag_address: str,
      tag_value: Optional[str],
      limit_study: bool = False,
      limit_series: bool = False,
  ) -> Iterator[Mapping[str, Any]]:
    """Filter metadata by tag value.

    Default = returns all instances that pass tag filter; SOPInstanceUID level.

    Args:
      metadata: List of DICOM JSON metadata to return.
      tag_address: Tag address to filter on.
      tag_value: Tag value to match.
      limit_study: Optional bool returns metadata at study level.
      limit_series: Optional bool returns metadata at series level.

    Yields:
      Iterator of filtered DICOM JSON metadata.
    """
    if tag_value is None or not tag_value:
      for data in metadata:
        yield data
    filtered_metadata = set()
    for data in metadata:
      try:
        if data[tag_address][_VALUE][0] != tag_value:
          continue
        if limit_study:
          data = dict(data)
          for address in [
              _DICOM_SERIES_INSTANCE_UID_TAG,
              _DICOM_SOP_INSTANCE_UID_TAG,
          ]:
            if address in data:
              del data[address]
          # Limit results to single instance per study.
          study_instance_uid = data[_DICOM_STUDY_INSTANCE_UID_TAG][_VALUE][0]
          if study_instance_uid not in filtered_metadata:
            filtered_metadata.add(study_instance_uid)
            yield data
          continue
        if limit_series:
          data = dict(data)
          # Remove SOPInstanceUID metadata
          for address in [_DICOM_SOP_INSTANCE_UID_TAG]:
            if address in data:
              del data[address]
          # Limit results to single instance per series.
          series_instance_uid = data[_DICOM_SERIES_INSTANCE_UID_TAG][_VALUE][0]
          if series_instance_uid not in filtered_metadata:
            filtered_metadata.add(series_instance_uid)
            yield data
          continue
        sop_instance_uid = data[_DICOM_SOP_INSTANCE_UID_TAG][_VALUE][0]
        if sop_instance_uid not in filtered_metadata:
          filtered_metadata.add(sop_instance_uid)
          yield data
      except (KeyError, IndexError) as _:
        continue

  def _remove_binary_tag_metadata(
      self, dataset: Iterator[Mapping[str, Any]]
  ) -> Iterator[Mapping[str, Any]]:
    for metadata in dataset:
      yield _filter_binary_tags(
          dict(metadata), _FilterBinaryTagOperation.REMOVE, self._dicomweb_path
      )

  def _filter_metadata_response(
      self,
      parsed_url_parts: Tuple[str, ...],
      parameter_index: int,
      metadata: Iterator[Mapping[str, Any]],
      limit_study: bool = False,
      limit_series: bool = False,
  ) -> List[Mapping[str, Any]]:
    """Filters metadata response based on url parameters.

    Default = returns all instances that pass tag filter; SOPInstanceUID level.

    Args:
      parsed_url_parts: Tuple containing strings parsed from url.
      parameter_index: Index of string to parse from url_part.
      metadata: List of DICOM instance JSON metadata.
      limit_study: Optional bool returns metadata at study level.
      limit_series: Optional bool returns metadata at series level.

    Returns:
      Filtered DICOM Metadata Response.
    """
    metadata = self._filter_metadata(
        metadata,
        '00080050',
        self._parse_url_parameters(
            parsed_url_parts, parameter_index, 'AccessionNumber'
        ),
        limit_study=limit_study,
        limit_series=limit_series,
    )
    metadata = self._filter_metadata(
        metadata,
        '00100020',
        self._parse_url_parameters(
            parsed_url_parts, parameter_index, 'PatientID'
        ),
        limit_study=limit_study,
        limit_series=limit_series,
    )
    # Google DICOM store does not return binary tags inline for tag search
    # requests. i.g. for DICOMweb instances, studies, or series requests.
    # If bulk uri support not is enabled strip binary tags from metadata.
    if not self._bulkdata_uri_enabled:
      metadata = self._remove_binary_tag_metadata(metadata)
    return self._limit_metadata(
        metadata,
        self._parse_url_parameters(parsed_url_parts, parameter_index, 'Limit'),
    )

  def _store_level_study_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a store level study metadata request."""
    result = self._parse_url(
        request,
        r'/studies(\?.*)?',
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    study_instance_uid = self._parse_url_parameters(
        parts, 0, 'StudyInstanceUID'
    )
    metadata_iterator = self._get_dicom_metadata_iterator(study_instance_uid)
    metadata = self._filter_metadata_response(
        parts, 0, metadata_iterator, limit_study=True
    )
    if metadata:
      return _build_response(
          http.HTTPStatus.OK,
          json.dumps(metadata),
          ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.HTTPStatus.NO_CONTENT, '', ContentType.TEXT_HTML
    )

  def _get_binary_tag_data(
      self, metadata: pydicom.FileDataset, path: str
  ) -> Optional[bytes]:
    """Returns binary data referenced by bulkdata uri path."""
    bulkuri_parts = path.split('/')
    index = 0
    while index < len(bulkuri_parts):
      try:
        metadata = metadata[bulkuri_parts[index]]
      except KeyError:
        return None
      index += 1
      if index == len(bulkuri_parts):
        try:
          return metadata.value
        except KeyError:
          return None
      try:
        sq_index = int(bulkuri_parts[index])
      except ValueError:
        return None
      index += 1
      try:
        metadata = metadata[sq_index]
      except (IndexError, KeyError) as _:
        return None
    return None

  def _bulkdata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a store level study metadata request."""
    if not self._bulkdata_uri_enabled:
      return None
    result = self._parse_url(
        request,
        r'/studies/(.*?)/series/(.*?)/instances/(.*?)/bulkdata/(.*)',
        RequestMethod.GET,
    )
    if result is None:
      return None
    parts = result.groups()
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    metadata = list(self._get_instances(parts[0], parts[1], parts[2]))
    if not metadata:
      return _build_response(
          http.HTTPStatus.BAD_REQUEST,
          'Error poorly formatted multipart content.',
          ContentType.TEXT_HTML,
      )
    response = self._get_binary_tag_data(metadata[0], parts[3])
    if response is not None:
      content = CustomContentType('application/octet-stream; transfer-syntax=*')
      return _build_response(http.HTTPStatus.OK, response, content)
    return _build_response(
        http.HTTPStatus.NO_CONTENT, '', ContentType.TEXT_HTML
    )

  def _study_level_series_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for returning study level series metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series{0,1}(\?.*)?',
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    series_instance_uid = self._parse_url_parameters(
        parts, 1, 'SeriesInstanceUID'
    )
    metadata_iterator = self._get_dicom_metadata_iterator(
        parts[0], series_instance_uid
    )
    metadata = self._filter_metadata_response(
        parts, 1, metadata_iterator, limit_series=True
    )
    if metadata:
      return _build_response(
          http.HTTPStatus.OK,
          json.dumps(metadata),
          ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.HTTPStatus.NO_CONTENT, '', ContentType.TEXT_HTML
    )

  def _series_instance_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a study/uid/series/uid metadata request."""
    for test_param in ('instances', 'metadata'):
      result = self._parse_url(
          request,
          r'/studies/([0-9.]+)/series/([0-9.]+)/'
          f'{test_param}'
          r'/{0,1}(\?.*)?',
          RequestMethod.GET,
      )
      if result is not None:
        break
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    sop_instance_uid = (
        self._parse_url_parameters(parts, 2, 'SOPInstanceUID')
        if test_param == 'instances'
        else ''
    )
    metadata_iterator = self._get_dicom_metadata_iterator(
        parts[0], parts[1], sop_instance_uid
    )
    if test_param == 'instances':
      metadata = self._filter_metadata_response(parts, 2, metadata_iterator)
    else:
      metadata = list(metadata_iterator)
    if metadata:
      return _build_response(
          http.HTTPStatus.OK,
          json.dumps(metadata),
          ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.HTTPStatus.NO_CONTENT, '', ContentType.TEXT_HTML
    )

  def _instance_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for a study/uid/series/uid/instances/uid/metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)/metadata',
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    metadata = list(
        self._get_dicom_metadata_iterator(parts[0], parts[1], parts[2])
    )
    if metadata:
      return _build_response(
          http.HTTPStatus.OK,
          json.dumps(metadata),
          ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.HTTPStatus.NOT_FOUND,
        'resource not found',
        ContentType.APPLICATION_DICOM_JSON,
    )

  def _add_study_instance(
      self, request: requests.PreparedRequest
  ) -> Optional[requests.Response]:
    """Entry point for handling adding an instance to a study."""
    result = self._parse_url(
        request, r'/studies(/([0-9.]+))?', RequestMethod.POST
    )
    if result is None:
      return None
    add_study_instance_uid_filter = result.groups()[1]
    if not self._can_write(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    request_content_type = _get_content_type(request.headers)
    lower_content_type = request_content_type.lower()
    dcm_instance_list = []
    try:
      if isinstance(request.body, bytes):
        # If body is bytes.
        request_content = request.body
      else:
        # If streaming Read
        request_content = request.body.read()
      if lower_content_type.startswith('application/dicom'):
        dcm_instance_list = [_decode_pydicom(request_content)]
      elif lower_content_type.startswith('multipart/related;'):
        try:
          mp_response = requests_toolbelt.MultipartDecoder(
              request_content, request_content_type
          )
        except requests_toolbelt.NonMultipartContentTypeException:
          return _build_response(
              http.HTTPStatus.BAD_REQUEST,
              'Error poorly formatted multipart content.',
              ContentType.TEXT_HTML,
          )
        dicom_parts = []
        content_location_map = {}
        for part in mp_response.parts:
          content_location = None
          for key, value in part.headers.items():
            if key.decode('utf-8').lower().strip() == 'content-location':
              content_location = value.decode('utf-8')
              break
          if content_location is None:
            dicom_parts.append(part)
          else:
            content_location_map[content_location] = part.content
        for part in dicom_parts:
          part_content_type = None
          for key, value in part.headers.items():
            if key.decode('utf-8').lower().strip() == 'content-type':
              part_content_type = value.decode('utf-8')
              break
          if part_content_type is None:
            return _build_response(
                http.HTTPStatus.BAD_REQUEST,
                'Error poorly formatted multipart content.',
                ContentType.TEXT_HTML,
            )
          if part_content_type.lower() == 'application/dicom':
            dcm_instance_list.append(_decode_pydicom(part.content))
          elif part_content_type.lower().startswith('application/dicom+json'):
            dcm_instance_list.append(
                _build_pydicom_dicom_from_request_json(
                    part.content, content_location_map
                )
            )
    except (
        pydicom.errors.InvalidDicomError,
        json.decoder.JSONDecodeError,
    ) as _:
      return _build_response(
          http.HTTPStatus.BAD_REQUEST,
          'Invalid DICOM Instance',
          ContentType.TEXT_PLAIN,
      )
    http_status = http.HTTPStatus.OK
    for dcm in dcm_instance_list:
      if (
          add_study_instance_uid_filter is not None
          and dcm.StudyInstanceUID != add_study_instance_uid_filter
      ):
        http_status = http.HTTPStatus.BAD_REQUEST
        continue
      try:
        if not self.add_instance(dcm):
          if http_status != http.HTTPStatus.BAD_REQUEST:
            http_status = http.HTTPStatus.CONFLICT
          continue
      except dicom_uid_value_map.UidValueMapError:
        http_status = http.HTTPStatus.BAD_REQUEST
    return _build_response(
        http_status, 'Instances Added', ContentType.TEXT_PLAIN
    )

  def _download_instance(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling downloading instance from store."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)',
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    instance_list = list(self._get_instances(parts[0], parts[1], parts[2]))
    if not instance_list:
      return _build_response(
          http.HTTPStatus.NOT_FOUND,
          '"DICOM instance does not exist."',
          ContentType.APPLICATION_JSON,
      )
    if len(instance_list) > 1:
      return _build_response(
          http.HTTPStatus.NOT_FOUND,
          '"Error in DICOM store mock, multiple instances returned."',
          ContentType.APPLICATION_JSON,
      )

    dcm = instance_list[0]
    accept_header_value = _get_accept(request.headers)
    if (
        accept_header_value != ContentType.APPLICATION_DICOM_NO_TRANSCODE.value
        and not _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
            accept_header_value, dcm
        )
    ):
      return _build_response(
          http.HTTPStatus.NOT_FOUND,
          (
              'DICOM store mock does not support downloading DICOM with '
              f'accept != "{ContentType.APPLICATION_DICOM_NO_TRANSCODE.value}"'
              f'; passed accept == "{accept_header_value}".'
          ),
      )
    content = CustomContentType(
        f'{_GET_DICOM_INSTANCE_BASE_CONTENT}{dcm.file_meta.TransferSyntaxUID}'
    )
    return _build_response(
        http.HTTPStatus.OK, _pydicom_file_dataset_to_bytes(dcm), content
    )

  def _download_frame(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling downloading untranscoded frames from store."""
    result = self._parse_url(
        request,
        (
            r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)/frames/(('
            r' *[0-9]+ *,{0,1})+)($|/.*)'
        ),
        RequestMethod.GET,
    )
    if result is None:
      return None
    if not self._can_read(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    (
        study_instance_uid,
        series_instance_uid,
        sop_instance_uid,
        frame_indexes,
    ) = result.groups()[:4]
    instance_list = list(
        self._get_instances(
            study_instance_uid, series_instance_uid, sop_instance_uid
        )
    )
    if not instance_list:
      return _build_response(
          http.HTTPStatus.NOT_FOUND,
          '"DICOM instance does not exist."',
          ContentType.APPLICATION_JSON,
      )
    if len(instance_list) > 1:
      return _build_response(
          http.HTTPStatus.NOT_FOUND,
          '"Error in DICOM store mock, multiple instances returned."',
          ContentType.APPLICATION_JSON,
      )
    dcm = instance_list[0]
    accept_header_value = _get_accept(request.headers)
    if (
        accept_header_value != ContentType.UNTRANSCODED_FRAME_REQUEST.value
        and not _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
            accept_header_value, dcm
        )
    ):
      return _build_response(
          http.HTTPStatus.NOT_ACCEPTABLE,
          (
              'DICOM store mock does not support downloading DICOM with '
              f'accept != "{ContentType.APPLICATION_DICOM_NO_TRANSCODE.value}"'
              f'; passed accept == "{accept_header_value}".'
          ),
      )
    frame_list = [
        int(f_num) for f_num in frame_indexes.replace(' ', '').split(',')
    ]
    fields = []
    mime_type = _DICOM_TRANSFER_SYNTAX_TO_MIME_TYPE[
        dcm.file_meta.TransferSyntaxUID
    ]
    frame_content_type = (
        f'{mime_type}; transfer-syntax={dcm.file_meta.TransferSyntaxUID}'
    )
    for fnum in frame_list:
      try:
        frame_image_bytes = self._get_pydicom_file_dataset_frame_bytes(
            dcm, fnum
        )
      except IndexError:
        return _build_response(
            http.HTTPStatus.BAD_REQUEST,
            f'Invalid frame number: {fnum}',
            ContentType.APPLICATION_JSON,
        )
      # None, first tuple parameter represents a "name", e.g. filename
      # in multipart response, not used here. see source.
      # https://github.com/requests/toolbelt/blob/master/requests_toolbelt/multipart/encoder.py
      fields.append((str(fnum), (None, frame_image_bytes, frame_content_type)))
    frame_data = requests_toolbelt.MultipartEncoder(fields=fields)
    result = _build_response(
        http.HTTPStatus.OK,
        frame_data.read(),
        CustomContentType(
            f'{ContentType.MULTIPART_RELATED.value};'
            f' boundary={frame_data.boundary_value}'
        ),
    )
    return result

  def _del_dicom(
      self,
      study_instance_uid: str,
      series_instance_uid: str,
      sop_instance_uid: str,
  ) -> Optional[requests.Response]:
    """Deletes a mocked instance from the mocked store."""
    if self._dicom_storage.remove_instance(
        study_instance_uid, series_instance_uid, sop_instance_uid
    ):
      return _build_response(http.HTTPStatus.OK, 'Deleted')
    return _build_response(
        http.HTTPStatus.NOT_FOUND,
        'Resource not found',
        ContentType.APPLICATION_JSON,
    )

  def _delete_study_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting study from store."""
    result = self._parse_url(
        request, r'/studies/([0-9.]+)', RequestMethod.DELETE
    )
    if result is None:
      return None
    if not self._can_write(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    return self._del_dicom(result.groups()[0], '', '')

  def _delete_series_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting series from store."""
    result = self._parse_url(
        request, r'/studies/([0-9.]+)/series/([0-9.]+)', RequestMethod.DELETE
    )
    if result is None:
      return None
    if not self._can_write(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    return self._del_dicom(parts[0], parts[1], '')

  def _delete_sop_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting sop instance from store."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)',
        RequestMethod.DELETE,
    )
    if result is None:
      return None
    if not self._can_write(request.headers):
      return _build_response(
          http.HTTPStatus.UNAUTHORIZED,
          '',
          ContentType.TEXT_HTML,
      )
    parts = result.groups()
    return self._del_dicom(parts[0], parts[1], parts[2])

  def in_store(
      self,
      dicom_instances: Union[
          _DicomUidTripleSourceType,
          List[_DicomUidTripleSourceType],
          Set[_DicomUidTripleSourceType],
      ],
  ) -> List[bool]:
    """Returns list of bool indicating if instances are in store."""
    if not isinstance(dicom_instances, set) and not isinstance(
        dicom_instances, list
    ):
      dicom_instances = [dicom_instances]
    instances_found_list = []
    for dicom_instance in dicom_instances:
      uid = _get_dicom_uid(dicom_instance)
      instance_found = self._get_instances(
          uid.study_instance_uid,
          uid.series_instance_uid,
          uid.sop_instance_uid,
      )
      instances_found_list.append(_does_iterator_have_items(instance_found))
    return instances_found_list

  def assert_uid_in_store(
      self,
      test_lib: absltest.TestCase,
      dicom_instances: Union[
          _DicomUidTripleSourceType,
          List[_DicomUidTripleSourceType],
          Set[_DicomUidTripleSourceType],
      ],
  ) -> None:
    """Tests that DICOM instance(s) UID are in mocked DICOM Store."""
    if not isinstance(dicom_instances, set) and not isinstance(
        dicom_instances, list
    ):
      dicom_instances = [dicom_instances]
    for dicom_instance in dicom_instances:
      uid = _get_dicom_uid(dicom_instance)
      instances_found = self._get_instances(
          uid.study_instance_uid,
          uid.series_instance_uid,
          uid.sop_instance_uid,
      )
      test_lib.assertTrue(
          _does_iterator_have_items(instances_found),
          (
              f'StudyInstanceUID: {uid.study_instance_uid}, '
              f'SeriesInstanceUID: {uid.series_instance_uid}, '
              f'SOPInstanceUID: {uid.sop_instance_uid} is not in DICOM Store.'
          ),
      )

  def assert_uid_not_in_store(
      self,
      test_lib: absltest.TestCase,
      dicom_instances: Union[
          _DicomUidTripleSourceType,
          List[_DicomUidTripleSourceType],
          Set[_DicomUidTripleSourceType],
      ],
  ) -> None:
    """Tests that DICOM instance(s) UID are not in mocked DICOM Store."""
    if not isinstance(dicom_instances, set) and not isinstance(
        dicom_instances, list
    ):
      dicom_instances = [dicom_instances]
    for dicom_instance in dicom_instances:
      uid = _get_dicom_uid(dicom_instance)
      instances_found = self._get_instances(
          uid.study_instance_uid,
          uid.series_instance_uid,
          uid.sop_instance_uid,
      )
      test_lib.assertTrue(
          _is_iterator_empty(instances_found),
          (
              f'StudyInstanceUID: {uid.study_instance_uid}, '
              f'SeriesInstanceUID: {uid.series_instance_uid}, '
              f'SOPInstanceUID: {uid.sop_instance_uid} is in DICOM Store.'
          ),
      )

  def assert_empty(self, test_lib: absltest.TestCase) -> None:
    """Asserts if mocked DICOM store is not empty."""
    test_lib.assertTrue(
        self._dicom_storage.is_empty(),
        'DICOM Store has instances.',
    )

  def assert_not_empty(self, test_lib: absltest.TestCase) -> None:
    """Asserts if mocked DICOM store is empty."""
    test_lib.assertTrue(
        self._dicom_storage.has_instances_in_store(),
        'DICOM Store is empty.',
    )

  def set_mock_response(self, mock_request: MockHttpResponse) -> None:
    self._mock_responses.append(mock_request)


class MockDicomStores(contextlib.ExitStack):
  """Context manager enables simultaneous mocking of multiple stores.

  Context manager accepts one or more store paths. If used using "with" syntax
  the manager returns a dictionary which maps the mocked paths to mocked DCIOM
  store instances. These instances can be accessed via the request api mock.

  Mocked store instances have methods to insert a dicom instance into the store.
  To enable the store to be pre-filled prior to its use and assertions to test
  if instances are in the store and the general state of the store.

  The DICOM store mocks implement the subset of the DICOMweb API necessary to
  run DPAS ingestion.

  A key limitation: metadata queries, will always return an instances full-
  metadata. This will result in the mocked instances returning more metadata
  than the store would for study_instance_uid and series_instance_uid metadata
  queries.
  """

  def __init__(
      self,
      *dicomstore_web_path: str,
      mock_credential: bool = True,
      mock_credential_project: str = 'mock_gcp_project',
      mock_request: Optional[requests_mock.Mocker] = None,
      read_auth_bearer_tokens: Optional[List[str]] = None,
      write_auth_bearer_tokens: Optional[List[str]] = None,
      real_http: bool = False,
      bulkdata_uri_enabled: bool = True,
  ):
    """Constructor.

    Args:
      *dicomstore_web_path: DicomWebPath to one or more mocked dicom stores.
      mock_credential: Bool flag, if credentials should also be mocked.
      mock_credential_project: Name of GCP project if credentials mocked.
      mock_request: Optional pre-existing requests mock, pass in mock_request
        chain context manager to pre-existing mocks.
      read_auth_bearer_tokens: Optional list of bearer tokens which can read
        from mock store.
      write_auth_bearer_tokens: Optional list of bearer tokens which can write
        to mock store.
      real_http: If true pass unhandled http requests through to server,
        real_http is not compatible with httplib2 mock and disables the mock
        support for httplib2 mediated transactions.
      bulkdata_uri_enabled: If True store will encode and support binary data
        bulkdata uri.
    """
    super().__init__()
    self._real_http = real_http
    self._read_auth_bearer_tokens = copy.copy(read_auth_bearer_tokens)
    self._write_auth_bearer_tokens = copy.copy(write_auth_bearer_tokens)
    self._dicomweb_paths = list(dict.fromkeys(dicomstore_web_path))
    self._credentials_mock = mock_credential
    self._mock_credential_project = mock_credential_project
    self._mock_request = mock_request
    self._mocked_dicom_stores = {}
    self._bulkdata_uri_enabled = bulkdata_uri_enabled

  def _httplib2_request_handler(
      self, uri: str, method: str, body: str, headers: Mapping[str, str]
  ) -> Tuple[httplib2.Response, bytes]:
    """Catch httplib2 requests and translate into requests.

    Args:
      uri: URL requested.
      method: HTTP method used for request.
      body: Request msg.
      headers: Headers supplied with request.

    Returns:
      Tuple[httplib2.Response, body of returned message(bytes)]
    """
    req = requests.Request(method, uri, headers, body)
    for (
        dicom_store_path,
        dicom_store_instance,
    ) in self._mocked_dicom_stores.items():
      if not uri.startswith(dicom_store_path):
        continue
      result = dicom_store_instance.handle_request(req)
      if result is None:
        continue
      # Convert requests response into httplib2 response.
      result_response_dict = {
          'status': result.status_code,
          'reason': result.reason,
      }
      result_response_dict.update(result.headers)
      raw_result = result.raw
      if raw_result is not None:
        return (
            httplib2.Response(result_response_dict),
            raw_result.getvalue(),
        )
    # Default unhandled response.
    return (
        httplib2.Response({
            'status': http.HTTPStatus.BAD_REQUEST,
            'content-type': ContentType.TEXT_PLAIN.value,
        }),
        b'DICOM Store Error: Unhandled HTTPLib2 request.',
    )

  def __getitem__(self, mocked_store_path: str) -> MockDicomStoreClient:
    return self._mocked_dicom_stores[mocked_store_path]

  def get(self, mocked_store_path: str, default: Any = None) -> Any:
    return self._mocked_dicom_stores.get(mocked_store_path, default)

  def __len__(self) -> int:
    return len(self._mocked_dicom_stores)

  def __enter__(self) -> MockDicomStores:
    """Enter context manager.

    Returns:
       Mapping[DICOMweb storepath, StoreMockInterface]
    """
    super().__enter__()
    try:
      mock_credential = self._credentials_mock
      mock_request = self._mock_request
      for mocked_store_path in self._dicomweb_paths:
        mocked_store = MockDicomStoreClient(
            mocked_store_path,
            mock_credential,
            self._mock_credential_project,
            mock_request,
            self._read_auth_bearer_tokens,
            self._write_auth_bearer_tokens,
            real_http=self._real_http,
            bulkdata_uri_enabled=self._bulkdata_uri_enabled,
        )
        self._mocked_dicom_stores[mocked_store_path] = mocked_store
        self.enter_context(mocked_store)
        mock_request = mocked_store.mock_request
        mock_credential = False
      if not self._real_http:
        self.enter_context(
            mock.patch.object(
                google_auth_httplib2.AuthorizedHttp,
                'request',
                side_effect=self._httplib2_request_handler,
            )
        )
      return self
    except:
      # Exception occurred during context manager entry. Close any opened
      # context managers attached to this class.
      self.close()
      raise
