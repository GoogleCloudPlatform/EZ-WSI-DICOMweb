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

import collections
import contextlib
import enum
import http.client
import io
import json
import re
from typing import Any, List, Mapping, Match, NewType, Optional, Set, Tuple, Union
from unittest import mock

from absl.testing import absltest
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock_types
import google.auth
import google.auth.credentials
import google_auth_httplib2
import httplib2
import pydicom
import requests
import requests_mock
import requests_toolbelt


_DicomUidTriple = dicom_store_mock_types.DicomUidTriple

_DicomUidTripleSourceType = Union[
    str,
    Mapping[str, Any],
    pydicom.Dataset,
    pydicom.FileDataset,
    _DicomUidTriple,
    dicom_store_mock_types.AbstractGetDicomUidTripleInterface,
]


class _RequestMethod(enum.Enum):
  GET = 'GET'
  POST = 'POST'
  DELETE = 'DELETE'


_CustomContentType = NewType('CustomContentType', str)
_GET_DICOM_INSTANCE_BASE_CONTENT = 'application/dicom; transfer-syntax='

# Mime type mapping assoicated for DICOM transfer syntax uid.
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


class _ContentType(enum.Enum):
  TEXT_PLAIN = 'text/plain; charset=us-ascii'
  APPLICATION_DICOM_JSON = 'application/dicom+json; charset=utf-8'
  APPLICATION_DICOM_NO_TRANSCODE = f'{_GET_DICOM_INSTANCE_BASE_CONTENT}*'
  TEXT_HTML = 'text/html'
  APPLICATION_DICOM_XML = 'application/dicom+xml'
  APPLICATION_JSON = 'application/json; charset=UTF-8'
  MULTIPART_RELATED = 'multipart/related'
  UNTRANSCODED_FRAME_REQUEST = (
      'multipart/related; type=application/octet-stream; transfer-syntax=*'
  )


def _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
    accept_header: str, dcm: pydicom.FileDataset
) -> bool:
  result = re.fullmatch(
      r'.*type=application/octet-stream;[ ]+transfer-syntax=(.*)', accept_header
  )
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
    content: Union[_ContentType, _CustomContentType] = _ContentType.TEXT_PLAIN,
) -> requests.Response:
  """Generates request Response to return."""
  resp = requests.Response()
  resp.status_code = status_code
  if isinstance(content, _ContentType):
    resp.headers['Content-Type'] = content.value
  else:
    resp.headers['Content-Type'] = str(content)
  if isinstance(msg, str):
    msg = msg.encode('utf-8')
  resp.raw = io.BytesIO(msg)
  return resp


def _convert_to_pydicom_file_dataset(
    dcm: Union[str, Mapping[str, Any], pydicom.FileDataset]
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
      is_implicit_VR=False,
      is_little_endian=True,
  )
  dicom_instance.file_meta.MediaStorageSOPClassUID = dicom_instance.SOPClassUID
  dicom_instance.file_meta.MediaStorageSOPInstanceUID = (
      dicom_instance.SOPInstanceUID
  )
  return dicom_instance


def _pydicom_file_dataset_to_json(
    dcm: pydicom.FileDataset,
) -> Mapping[str, Any]:
  result = dcm.to_json_dict()
  file_meta = dcm.file_meta.to_json_dict()
  result.update(file_meta)
  return result


def _pydicom_file_dataset_to_bytes(dcm: pydicom.FileDataset) -> bytes:
  instance_bytes = io.BytesIO()
  dcm.save_as(instance_bytes, write_like_original=True)
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


class _MockDicomStoreClient(contextlib.ContextDecorator):
  """Context manager mocks individual DICOM Store."""

  def __init__(
      self,
      dicomweb_path: str,
      mock_credential: bool = True,
      mock_credential_project: str = 'mock_gcp_project',
      mock_request: Optional[requests_mock.Mocker] = None,
  ):
    self._context_manager_entered = False
    # self._frame_cache stores a list representations of encapsulated frames
    # bytes stored in DICOM instances to enable rapid, index based,
    # random access. Frame cache is initialized for a DICOM instance
    # on the first frame request. PyDICOM uses a generator style interface
    # to access blob bytes stored within encapsulated frames. This makes
    # repeated random access to frame bytes inefficent, e.g., used directly to
    # return value stored in the last frame itteration over all N - 1 prior
    # frames.
    self._frame_cache = {}
    self._mock_dicom_store_request_entered = False
    self._dicomweb_path = dicomweb_path
    if not mock_credential:
      self._credentials_mock = None
    else:
      credentials_mock = mock.create_autospec(
          google.auth.credentials.Credentials, instance=True
      )
      self._credentials_mock = mock.patch(
          'google.auth.default',
          return_value=(credentials_mock, mock_credential_project),
      )
    self._instance_list: List[pydicom.FileDataset] = []
    if mock_request is None:
      self._mock_dicom_store_request = requests_mock.Mocker()
      self._request_mock_started = False
    else:
      self._mock_dicom_store_request = mock_request
      self._request_mock_started = True
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
    ]

  @property
  def mock_request(self) -> requests_mock.Mocker:
    """Returns requests mock used by the mock."""
    return self._mock_dicom_store_request

  def add_instance(
      self, instance_data: Union[str, Mapping[str, Any], pydicom.FileDataset]
  ):
    """Adds an instance to the mocked store."""
    self._instance_list.append(_convert_to_pydicom_file_dataset(instance_data))

  def __enter__(self) -> _MockDicomStoreClient:
    if self._context_manager_entered:
      raise ValueError('Context manager has already been entered')
    self._context_manager_entered = True
    if self._credentials_mock is not None:
      self._credentials_mock.__enter__()
    if not self._request_mock_started:
      self._mock_dicom_store_request.__enter__()
      self._mock_dicom_store_request_entered = True
    for handler in self._request_handlers:
      self._mock_dicom_store_request.add_matcher(handler)
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
    dcm_key = (dcm.StudyInstanceUID, dcm.SeriesInstanceUID, dcm.SOPInstanceUID)
    frame_byte_offset = self._frame_cache.get(dcm_key)
    if frame_byte_offset is None:
      if (
          dcm.file_meta.TransferSyntaxUID
          not in _UNENCAPSULATED_TRANSFER_SYNTAXES
      ):
        frame_byte_offset = list(
            pydicom.encaps.generate_pixel_data_frame(
                dcm.PixelData, dcm.NumberOfFrames
            )
        )
      else:
        number_of_frames = dcm.NumberOfFrames
        step = int(len(dcm.PixelData) / number_of_frames)
        frame_byte_offset = [
            dcm.PixelData[fnum * step : (fnum + 1) * step]
            for fnum in range(number_of_frames)
        ]
      self._frame_cache[dcm_key] = frame_byte_offset
    index -= 1  # DICOM frame numbering start at 1
    if index < 0:
      raise IndexError('Invalid Index')
    return frame_byte_offset[index]

  def _get_instances(
      self,
      study_instance_uid: str,
      series_instance_uid: Optional[str],
      sop_instance_uid: Optional[str],
  ) -> List[pydicom.FileDataset]:
    """Returns list of instances in mocked store with matching UIDs."""
    if series_instance_uid is None:
      series_instance_uid = ''
    if sop_instance_uid is None:
      sop_instance_uid = ''
    if study_instance_uid and not series_instance_uid and sop_instance_uid:
      raise ValueError('Invalid query')
    instances_found = []
    for instance in self._instance_list:
      if instance.StudyInstanceUID != study_instance_uid:
        continue
      if not series_instance_uid:
        instances_found.append(instance)
        continue
      if series_instance_uid == instance.SeriesInstanceUID:
        if not sop_instance_uid or sop_instance_uid == instance.SOPInstanceUID:
          instances_found.append(instance)
    return instances_found

  def _get_dicom_metadata(
      self,
      study_instance_uid: str,
      series_instance_uid: str = '',
      sop_instance_uid: str = '',
  ) -> List[Mapping[str, Any]]:
    """Returns DICOM metadata associated with study or series in mocked store."""
    return [
        _pydicom_file_dataset_to_json(instance)
        for instance in self._get_instances(
            study_instance_uid, series_instance_uid, sop_instance_uid
        )
    ]

  def _parse_url(
      self, request: requests.Request, reg_ex: str, method: _RequestMethod
  ) -> Optional[Match[str]]:
    if request.method != method.value:
      return None
    url_regex = _join_url_parts(self._dicomweb_path, reg_ex)
    return re.fullmatch(url_regex, request.url)

  def _study_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a study metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/instances/{0,1}(\?.*)?',
        _RequestMethod.GET,
    )
    if result is None:
      return None
    metadata = self._get_dicom_metadata(result.groups()[0])
    if metadata:
      return _build_response(
          http.client.OK,
          json.dumps(metadata),
          _ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(http.client.NO_CONTENT, '', _ContentType.TEXT_HTML)

  def _study_level_series_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for returning study level series metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series{0,1}(\?.*)?',
        _RequestMethod.GET,
    )
    if result is None:
      return None
    parts = result.groups()
    metadata = self._get_dicom_metadata(parts[0])
    if len(parts) == 2:
      accession_modifier = re.fullmatch(
          r'\?.*AccessionNumber=([^, ]*).*', parts[1]
      )
      if accession_modifier is not None:
        filtered_metadata = {}
        accession_number = accession_modifier.groups()[0]
        for data in metadata:
          try:
            if data['00080050']['Value'][0] != accession_number:
              continue
            if '00080018' in data:
              del data['00080018']  # Remove SOPInstanceUID metadata
            # Limit results to single instance per series.
            series_instance_uid = data['0020000E']['Value'][0]
            filtered_metadata[series_instance_uid] = data
          except (KeyError, IndexError) as _:
            continue
          metadata = list(filtered_metadata.values())
    if metadata:
      return _build_response(
          http.client.OK,
          json.dumps(metadata),
          _ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(http.client.NO_CONTENT, '', _ContentType.TEXT_HTML)

  def _series_instance_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling a study/uid/series/uid metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/{0,1}(\?.*)?',
        _RequestMethod.GET,
    )
    if result is None:
      return None
    parts = result.groups()
    metadata = self._get_dicom_metadata(parts[0], parts[1])
    if metadata:
      return _build_response(
          http.client.OK,
          json.dumps(metadata),
          _ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(http.client.NO_CONTENT, '', _ContentType.TEXT_HTML)

  def _instance_metadata_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for a study/uid/series/uid/instances/uid/metadata request."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)/metadata',
        _RequestMethod.GET,
    )
    if result is None:
      return None
    parts = result.groups()
    metadata = self._get_dicom_metadata(parts[0], parts[1], parts[2])
    if metadata:
      return _build_response(
          http.client.OK,
          json.dumps(metadata),
          _ContentType.APPLICATION_DICOM_JSON,
      )
    return _build_response(
        http.client.NOT_FOUND,
        'resource not found',
        _ContentType.APPLICATION_DICOM_JSON,
    )

  def _add_study_instance(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling adding an instance to a study."""
    result = self._parse_url(request, '/studies', _RequestMethod.POST)
    if result is None:
      return None
    try:
      dcm = pydicom.dcmread(request.text)  # pytype: disable=attribute-error
    except pydicom.errors.InvalidDicomError:
      return _build_response(
          http.client.CONFLICT,
          'Invalid DICOM Instance',
          _ContentType.APPLICATION_DICOM_XML,
      )

    instance_list = self._get_instances(
        dcm.StudyInstanceUID, dcm.SeriesInstanceUID, dcm.SOPInstanceUID
    )
    if instance_list:
      if len(instance_list) == 1 and _pydicom_file_dataset_to_bytes(
          instance_list[0]
      ) == _pydicom_file_dataset_to_bytes(dcm):
        return _build_response(
            http.client.OK, 'Instance already exists in the Store'
        )
      return _build_response(
          http.client.CONFLICT,
          'Instance with (study, series, instance) already exists',
      )
    self.add_instance(dcm)
    return _build_response(http.client.OK, 'Instance Added')

  def _download_instance(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling downloading instance from store."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)',
        _RequestMethod.GET,
    )
    if result is None:
      return None
    parts = result.groups()
    instance_list = self._get_instances(parts[0], parts[1], parts[2])
    if not instance_list:
      return _build_response(
          http.client.NOT_FOUND,
          '"DICOM instance does not exist."',
          _ContentType.APPLICATION_JSON,
      )
    if len(instance_list) > 1:
      return _build_response(
          http.client.NOT_FOUND,
          '"Error in DICOM store mock, multiple instances returned."',
          _ContentType.APPLICATION_JSON,
      )

    dcm = instance_list[0]
    accept_header_value = _get_accept(request.headers)
    if (
        accept_header_value != _ContentType.APPLICATION_DICOM_NO_TRANSCODE.value
        and not _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
            accept_header_value, dcm
        )
    ):
      return _build_response(
          http.client.NOT_FOUND,
          (
              'DICOM store mock does not support downloading DICOM with '
              f'accept != "{_ContentType.APPLICATION_DICOM_NO_TRANSCODE.value}"'
              f'; passed accept == "{accept_header_value}".'
          ),
      )
    content = _CustomContentType(
        f'{_GET_DICOM_INSTANCE_BASE_CONTENT}{dcm.file_meta.TransferSyntaxUID}'
    )
    return _build_response(
        http.client.OK, _pydicom_file_dataset_to_bytes(dcm), content
    )

  def _download_frame(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling downloading untranscoded frames from store."""
    result = self._parse_url(
        request,
        (
            r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)/frames/(('
            r' *[0-9]+ *,{0,1})+)/{0,1}'
        ),
        _RequestMethod.GET,
    )
    if result is None:
      return None
    (
        study_instance_uid,
        series_instance_uid,
        sop_instance_uid,
        frame_indexes,
        _,
    ) = result.groups()
    instance_list = self._get_instances(
        study_instance_uid, series_instance_uid, sop_instance_uid
    )
    if not instance_list:
      return _build_response(
          http.client.NOT_FOUND,
          '"DICOM instance does not exist."',
          _ContentType.APPLICATION_JSON,
      )
    if len(instance_list) > 1:
      return _build_response(
          http.client.NOT_FOUND,
          '"Error in DICOM store mock, multiple instances returned."',
          _ContentType.APPLICATION_JSON,
      )
    dcm = instance_list[0]
    accept_header_value = _get_accept(request.headers)
    if (
        accept_header_value != _ContentType.UNTRANSCODED_FRAME_REQUEST.value
        and not _accept_header_transfer_syntax_matches_dcm_transfer_syntax(
            accept_header_value, dcm
        )
    ):
      return _build_response(
          http.client.NOT_ACCEPTABLE,
          (
              'DICOM store mock does not support downloading DICOM with '
              f'accept != "{_ContentType.APPLICATION_DICOM_NO_TRANSCODE.value}"'
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
            http.client.BAD_REQUEST,
            f'Invalid frame number: {fnum}',
            _ContentType.APPLICATION_JSON,
        )
      # None, first tuple parameter represents a "name", e.g. filename
      # in multipart response, not used here. see source.
      # https://github.com/requests/toolbelt/blob/master/requests_toolbelt/multipart/encoder.py
      fields.append((str(fnum), (None, frame_image_bytes, frame_content_type)))
    frame_data = requests_toolbelt.MultipartEncoder(fields=fields)
    result = _build_response(
        http.client.OK,
        frame_data.read(),
        _CustomContentType(
            f'{_ContentType.MULTIPART_RELATED.value};'
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
    instance_list = self._get_instances(
        study_instance_uid, series_instance_uid, sop_instance_uid
    )
    for instance in instance_list:
      self._instance_list.remove(instance)
    if instance_list:
      return _build_response(http.client.OK, 'Deleted')
    return _build_response(
        http.client.NOT_FOUND,
        'Resource not found',
        _ContentType.APPLICATION_JSON,
    )

  def _delete_study_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting study from store."""
    result = self._parse_url(
        request, r'/studies/([0-9.]+)', _RequestMethod.DELETE
    )
    if result is None:
      return None
    return self._del_dicom(result.groups()[0], '', '')

  def _delete_series_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting series from store."""
    result = self._parse_url(
        request, r'/studies/([0-9.]+)/series/([0-9.]+)', _RequestMethod.DELETE
    )
    if result is None:
      return None
    parts = result.groups()
    return self._del_dicom(parts[0], parts[1], '')

  def _delete_sop_instance_uid(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Entry point for handling deleting sop instance from store."""
    result = self._parse_url(
        request,
        r'/studies/([0-9.]+)/series/([0-9.]+)/instances/([0-9.]+)',
        _RequestMethod.DELETE,
    )
    if result is None:
      return None
    parts = result.groups()
    return self._del_dicom(parts[0], parts[1], parts[2])

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
      instance_found = self._get_instances(
          uid.study_instance_uid, uid.series_instance_uid, uid.sop_instance_uid
      )
      test_lib.assertNotEmpty(
          instance_found,
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
      instance_found = self._get_instances(
          uid.study_instance_uid, uid.series_instance_uid, uid.sop_instance_uid
      )
      test_lib.assertEmpty(
          instance_found,
          (
              f'StudyInstanceUID: {uid.study_instance_uid}, '
              f'SeriesInstanceUID: {uid.series_instance_uid}, '
              f'SOPInstanceUID: {uid.sop_instance_uid} is in DICOM Store.'
          ),
      )

  def assert_empty(self, test_lib: absltest.TestCase) -> None:
    """Asserts if mocked DICOM store is not empty."""
    test_lib.assertEmpty(self._instance_list, 'DICOM Store has instances.')

  def assert_not_empty(self, test_lib: absltest.TestCase) -> None:
    """Asserts if mocked DICOM store is empty."""
    test_lib.assertNotEmpty(self._instance_list, 'DICOM Store is empty.')


class MockDicomStores(contextlib.ContextDecorator):
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
  ):
    """Constructor.

    Args:
      *dicomstore_web_path: DicomWebPath to one or more mocked dicom stores.
      mock_credential: Bool flag, if credentials should also be mocked.
      mock_credential_project: Name of GCP project if credentials mocked.
      mock_request: Optional pre-existing requests mock, pass in mock_request
        chain context manager to pre-existing mocks.
    """
    self._dicomweb_paths = list(dict.fromkeys(dicomstore_web_path))
    self._credentials_mock = mock_credential
    self._mock_credential_project = mock_credential_project
    self._mock_request = mock_request
    self._httplib2_request_mock = mock.patch.object(
        google_auth_httplib2.AuthorizedHttp,
        'request',
        side_effect=self._httplib2_request_handler,
    )
    self._mocked_dicom_stores = {}

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
            'status': http.client.BAD_REQUEST,
            'content-type': _ContentType.TEXT_PLAIN.value,
        }),
        b'DICOM Store Error: Unhandled HTTPLib2 request.',
    )

  def __enter__(self) -> Mapping[str, _MockDicomStoreClient]:
    """Enter context manager.

    Returns:
       Mapping[DICOMweb storepath, StoreMockInterface]
    """
    mock_credential = self._credentials_mock
    mock_request = self._mock_request
    for mocked_store_path in self._dicomweb_paths:
      mocked_store = _MockDicomStoreClient(
          mocked_store_path,
          mock_credential,
          self._mock_credential_project,
          mock_request,
      )
      mocked_store.__enter__()
      self._mocked_dicom_stores[mocked_store_path] = mocked_store
      mock_request = mocked_store.mock_request
      mock_credential = False
    self._httplib2_request_mock.__enter__()
    return self._mocked_dicom_stores

  def __exit__(self, exc_type, exc, exc_tb):
    self._httplib2_request_mock.__exit__(exc_type, exc, exc_tb)
    for mocked_store_path in reversed(self._dicomweb_paths):
      self._mocked_dicom_stores[mocked_store_path].__exit__(
          exc_type, exc, exc_tb
      )
