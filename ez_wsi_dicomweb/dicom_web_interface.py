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
"""DICOMWeb API Interface."""
import copy
import dataclasses
import enum
import http.client
import io
import json
import threading
from typing import Any, BinaryIO, Collection, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union
import urllib

import dataclasses_json
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import error_retry_util
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_json
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
import google.auth.credentials
import requests
from requests_toolbelt.multipart import decoder
import retrying

# The default Google HealthCare DICOMWeb API entry.
_BULKDATA_URI = 'BulkDataURI'
_VALUE = 'Value'

# The default DICOM tags to retrieve for an instance.
_DEFAULT_INSTANCE_TAGS = (
    tags.SOP_INSTANCE_UID,
    tags.INSTANCE_NUMBER,
    tags.ROWS,
    tags.COLUMNS,
    tags.SAMPLES_PER_PIXEL,
    tags.BITS_ALLOCATED,
    tags.HIGH_BIT,
    tags.PIXEL_REPRESENTATION,
    tags.PIXEL_SPACING,
    tags.NUMBER_OF_FRAMES,
    tags.FRAME_INCREMENT_POINTER,
    tags.IMAGE_VOLUME_WIDTH,
    tags.IMAGE_VOLUME_HEIGHT,
    tags.IMAGE_VOLUME_DEPTH,
    tags.TOTAL_PIXEL_MATRIX_COLUMNS,
    tags.TOTAL_PIXEL_MATRIX_ROWS,
    tags.TOTAL_PIXEL_MATRIX_ORIGIN_SEQUENCE,
    tags.SPECIMEN_LABEL_IN_IMAGE,
    tags.EXTENDED_DEPTH_OF_FIELD,
    tags.LABEL_TEXT,
    tags.BARCODE_VALUE,
    tags.CONCATENATION_FRAME_OFFSET_NUMBER,
    tags.IMAGE_TYPE,
    tags.TRANSFER_SYNTAX_UID,
    tags.ICC_PROFILE,
    tags.OPTICAL_PATH_SEQUENCE,
    tags.REFERENCED_SERIES_SEQUENCE,
    tags.SERIES_INSTANCE_UID,
    tags.REFERENCED_IMAGE_SEQUENCE,
    tags.REFERENCED_SOP_INSTANCE_UID,
    tags.REFERENCED_IMAGE_SEQUENCE,
    tags.REFERENCED_SOP_INSTANCE_UID,
    tags.OPERATOR_IDENTIFICATION_SEQUENCE,
    tags.PERSON_IDENTIFICATION_CODE_SEQUENCE,
    tags.LONG_CODE_VALUE,
    tags.SHARED_FUNCTIONAL_GROUP_SEQUENCE,
    tags.DIMENSION_ORGANIZATION_TYPE,
)


# Chunk size (in bytes) for streaming instance downloads.
# A larger chunk size uses more memory but may increase download speed.
# A smaller chunk size uses less memory but may decrease download speed.
# Adjust this value based on your memory constraints and network conditions.
_STREAMING_CHUNKSIZE = 102400


# Transfer syntax defining how DICOM frames should be transcoded.
# https://cloud.google.com/healthcare-api/docs/dicom#dicom_frames
class TranscodeDicomFrame(enum.Enum):
  DO_NOT_TRANSCODE = '*'
  UNCOMPRESSED_LITTLE_ENDIAN = '1.2.840.10008.1.2.1'


@retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
def _invoke_http_request(
    query: str,
    credentials: google.auth.credentials.Credentials,
    uri: str,
    headers: Mapping[str, Any],
    timeout: Optional[int],
) -> requests.Response:
  """Invokes a Http request to DICOMWeb API client.

  Args:
    query: DICOM Query Type
    credentials: Credentials to use for the request.
    uri: URI of Http request.
    headers: Http request headers.
    timeout: Http timeout in seconds.

  Returns:
    Tuple of httplib2.Response and string content.
  """
  credentials.apply(headers)
  try:
    response = requests.get(uri, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response
  except requests.exceptions.HTTPError as exp:
    status_code = exp.response.status_code
    content = exp.response.text
    ez_wsi_errors.raise_ez_wsi_http_exception(
        f'{query} error. Response Status: {status_code},\nURL:'
        f' {uri},\nContent: {content}.',
        exp,
    )


def _qido_rs(
    credentials: google.auth.credentials.Credentials,
    qido_url: str,
    timeout: Optional[int] = 3600,
) -> List[Dict[str, Any]]:
  """Performs the request, and returns the parsed JSON response.

  https://www.dicomstandard.org/using/dicomweb/query-qido-rs

  Args:
    credentials: Credentials to use for the request.
    qido_url: URL for the QIDO request.
    timeout: Http timeout in seconds.

  Returns:
    The parsed JSON response content or empty list if no contents are found.

  Raises:
    HttpError: If the response status was not success.
  """
  response = _invoke_http_request(
      'QidoRs', credentials, qido_url, {}, timeout=timeout
  )
  if response.status_code == http.client.NO_CONTENT:  # Empty query
    return []
  return json.loads(response.text)


def _wado_rs(
    credentials: google.auth.credentials.Credentials,
    wado_url: str,
    accept_header: Optional[str] = None,
    timeout: Optional[int] = 3600,
) -> bytes:
  """Performs the request, parses the multipart response, and returns content.

  https://www.dicomstandard.org/using/dicomweb/retrieve-wado-rs-and-wado-uri

  Args:
    credentials: Credentials to use for the request.
    wado_url: URL for the WADO request.
    accept_header: Value of the Accept header to use. If set to None, no Accept
      header will be used.
    timeout: Http timeout in seconds.

  Returns:
    The content of the first (and only) part as a string.

  Raises:
    HttpError : If the response status was not success or the
      number of parts in the multipart response is different from 1.
  """
  response = _invoke_http_request(
      'WadoRs',
      credentials,
      wado_url,
      headers={'Accept': accept_header} if accept_header is not None else {},
      timeout=timeout,
  )
  try:
    multipart_data = decoder.MultipartDecoder.from_response(response)
  except decoder.NonMultipartContentTypeException as exp:
    raise ez_wsi_errors.InvalidWadoRsResponseError(
        'Received invalid WadoRs multipart response.'
    ) from exp
  num_parts = len(multipart_data.parts)
  if num_parts != 1:
    raise ez_wsi_errors.InvalidWadoRsResponseError(
        'WadoRs multipart response expected to have a single part.'
        f' Actual: {num_parts}.\nURL: {wado_url}'
    )
  return multipart_data.parts[0].content


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class DicomObject:
  """Represents a DICOM object by its path and DICOM tags.

  Attributes:
    path: The DICOM path to the object in a DICOMStore.
    dicom_tags: A list of DICOM tags that are associated with the DICOM object.
    icc_profile_bulkdata_url: Bulkdata URI to return instance ICC Profile
  """

  path: dicom_path.Path
  dicom_tags: Dict[str, Any]
  icc_profile_bulkdata_url: str

  def __post_init__(self):
    """Post __init__.

    Raises:
      DicomPathError if self.path type is not STUDY, SERIES or
      INSTANCE.
    """
    if self.path.type not in (
        dicom_path.Type.STUDY,
        dicom_path.Type.SERIES,
        dicom_path.Type.INSTANCE,
    ):
      raise ez_wsi_errors.DicomPathError(
          f'The DICOM path has an invalid type: {self.path.type}'
      )

  def type(self) -> dicom_path.Type:
    """Returns the type of the DICOM  path."""
    return self.path.type

  def get_value(self, tag: tags.DicomTag) -> Any:
    """Returns the first value for the tag in dicom_tags.

    For many DICOM tags this is going to be the only value in the list. If you
    wish to fetch the entire list of tags instead, use get_list_value().

    Args:
      tag: The tag to lookup values for.

    Returns:
      The first value from dicom_tags corresponding to the tag or None if the
      tag is not present in dicom_tags.
    """
    return dicom_json.GetValue(self.dicom_tags, tag)

  def get_list_value(self, tag: tags.DicomTag) -> Optional[List[Any]]:
    """Returns the value list for the tag from dicom_tags.

    Args:
      tag: The tag to lookup values for.

    Returns:
      The value list corresponding to the tag or None if the tag is not present
      in dicom_tags.
    """
    return dicom_json.GetList(self.dicom_tags, tag)


class _IntToStringConverter:
  """Utility to convert int to string and count number of vals processed."""

  def __init__(self):
    self._count = 0

  def convert(
      self, values: Union[Sequence[int], Iterator[int]]
  ) -> Iterator[str]:
    for value in values:
      self._count += 1
      yield str(value)

  @property
  def count(self) -> int:
    return self._count


class DicomWebInterface:
  """A Python interface of the DICOMWeb API.

  It wraps around the HealthCare DICOMWeb API to fetch DICOM objects, such as
  studies, series, instances, and dicom_tags associated with those objects.
  """

  def __init__(
      self,
      credential_factory: credential_factory_module.AbstractCredentialFactory,
  ):
    """Constructor.

    Args:
      credential_factory: Factory that creates DICOMweb credentials.
    """
    self._interface_lock = threading.RLock()
    self._credentials = None
    self._credential_factory = credential_factory

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = copy.copy(self.__dict__)
    del state['_credentials']
    del state['_interface_lock']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._interface_lock = threading.RLock()
    self._credentials = None

  @property
  def credential_factory(
      self,
  ) -> credential_factory_module.AbstractCredentialFactory:
    return self._credential_factory

  def credentials(self) -> google.auth.credentials.Credentials:
    """Returns credentials used to access DICOM store."""
    with self._interface_lock:
      if self._credentials is None:
        self._credentials = self._credential_factory.get_credentials()
      self._credentials = credential_factory_module.refresh_credentials(
          self._credentials, self._credential_factory
      )
      return self._credentials

  def _make_api_url(self, base_path: dicom_path.Path, api_entry) -> str:
    """Constructs a full http url for the input base path and api entry.

    Args:
      base_path: The base path of the resource upon which the api will be
        called. This path should take the form of:
        'projects/{}/locations/{}/datasets/{}/dicomStores/{}'
      api_entry: The api path to attach after the base path.

    Returns:
      The full http url for calling the API.
    """
    return dicom_path.DicomPathJoin(base_path.complete_url, api_entry)

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def get_studies(
      self, dicom_store_path: dicom_path.Path
  ) -> Sequence[DicomObject]:
    """Gets all study objects from the input DICOMStore.

    Args:
      dicom_store_path: The path to the DICOMStore to fetch studies from.

    Returns:
      A sequence of the DICOM study objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level
      dicom path.
    """
    if dicom_store_path.type != dicom_path.Type.STORE:
      raise ez_wsi_errors.DicomPathError(
          'A store-level path is required.'
          f'Found: {dicom_store_path.type} path found in {dicom_store_path}'
      )
    api_url = self._make_api_url(dicom_store_path, 'studies')
    json_results = _qido_rs(self.credentials(), api_url)
    if not json_results:
      return []

    return [
        _build_dicom_object(dicom_path.Type.STUDY, dicom_store_path, dicom_tags)
        for dicom_tags in json_results
    ]

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def get_series(
      self,
      parent_path: dicom_path.Path,
      dicom_tags: Optional[Dict[str, Any]] = None,
  ) -> Sequence[DicomObject]:
    """Gets all series objects under the input parent path.

    Args:
      parent_path: The path to a DICOMStore or a study to fetch series from.
      dicom_tags: Tags to search for, if provided.

    Returns:
      A sequence of the series objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level or
      study-level
      DICOM path.
    """
    if parent_path.type not in (dicom_path.Type.STORE, dicom_path.Type.STUDY):
      raise ez_wsi_errors.DicomPathError(
          'A store-level or study-level path is required.'
          f'Found: {parent_path.type} path found in {parent_path}'
      )
    series_query = (
        f'series?{urllib.parse.urlencode(dicom_tags)}'
        if dicom_tags
        else 'series'
    )

    api_url = self._make_api_url(parent_path, series_query)
    json_results = _qido_rs(self.credentials(), api_url)
    if not json_results:
      return []

    return [
        _build_dicom_object(dicom_path.Type.SERIES, parent_path, dicom_tags)
        for dicom_tags in json_results
    ]

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def get_instances(
      self, parent_path: dicom_path.Path, **dicomweb_filter: str | int  # kwargs
  ) -> Sequence[DicomObject]:
    """Gets all instances under the input parent path.

    Args:
      parent_path: The path to a DICOMStore, a study or a series.
      **dicomweb_filter: Additional DICOMweb query parameters to include in the
        QIDO-RS query.

    Returns:
      A sequence of the DICOM instance objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level,
      study-level or series-level DICOM path.
    """
    if parent_path.type not in (
        dicom_path.Type.STORE,
        dicom_path.Type.STUDY,
        dicom_path.Type.SERIES,
    ):
      raise ez_wsi_errors.DicomPathError(
          'A store-level, study-level or series-level path is required.'
          f'Found: {parent_path.type} path found in {parent_path}'
      )
    api_url = self._make_api_url(
        parent_path,
        f'instances/?{_get_qido_suffix(_DEFAULT_INSTANCE_TAGS, **dicomweb_filter)}',
    )
    json_results = _qido_rs(self.credentials(), api_url)
    if not json_results:
      return []

    return [
        _build_dicom_object(dicom_path.Type.INSTANCE, parent_path, dicom_tags)
        for dicom_tags in json_results
    ]

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def get_frame_image(
      self,
      instance_path: dicom_path.Path,
      frame_index: int,
      transcode_frame: TranscodeDicomFrame,
  ) -> bytes:
    """Retrieves the BGR image pixels of a frame from an instance.

    Args:
      instance_path: The path to a DICOM instance.
      frame_index: The index of the DICOM frame within the requested instance.
      transcode_frame: How DICOM frame should be transcoded.

    Returns:
      The image buffer that contains the pixels of the frame.

    Raises:
      DicomPathError if the input path is not an instance.
    """
    if instance_path.type != dicom_path.Type.INSTANCE:
      raise ez_wsi_errors.DicomPathError(
          'An instance-level path is required. '
          f'Found: {instance_path.type} path found in {instance_path}'
      )
    api_url = self._make_api_url(instance_path, f'frames/{frame_index}')
    return _wado_rs(
        self.credentials(),
        api_url,
        (
            'multipart/related; type="application/octet-stream"; '
            f'transfer-syntax={transcode_frame.value}'
        ),
    )

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  @retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
  def get_bulkdata(self, uri: str, chunk_size=_STREAMING_CHUNKSIZE) -> bytes:
    """Returns binary data stored stored at bulkdata uri.

    https://dicom.nema.org/medical/dicom/current/output/html/part18.html#sect_10.4.1.1.5

    Args:
      uri: DICOM store bulkdata uri to query.
      chunk_size: Streaming chunk size.

    Returns:
      bytes from bulkdata uri.

    Raises:
      ez_wsi_errors.HTTPError if http error occurs.
    """
    try:
      headers = {'Accept': 'application/octet-stream; transfer-syntax=*'}
      self.credentials().apply(headers)
      with requests.Session() as session:
        response = session.get(uri, headers=headers, stream=True)
        with io.BytesIO() as output_stream:
          response.raise_for_status()
          for chunk in response.iter_content(chunk_size=max(1, chunk_size)):
            output_stream.write(chunk)
          return output_stream.getvalue()
    except requests.exceptions.HTTPError as exp:
      ez_wsi_errors.raise_ez_wsi_http_exception(exp.response.reason, exp)

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def download_instance_untranscoded(
      self,
      instance_path: dicom_path.Path,
      output_stream: BinaryIO,
      chunk_size: int = _STREAMING_CHUNKSIZE,
      retry: bool = True,
  ) -> None:
    """Downloads DICOM instance from store in native format to output stream.

    Args:
      instance_path: DICOM web path to instance to download.
      output_stream: Output stream to write to.
      chunk_size: Streaming download chunksize.
      retry: Retry on retriable HTTP errors.
    """

    @retrying.retry(
        **error_retry_util.enable_config(
            retry, error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG
        )
    )
    def _inner_func() -> None:
      headers = {'Accept': 'application/dicom; transfer-syntax=*'}
      self.credentials().apply(headers)
      try:
        with requests.Session() as session:
          response = session.get(
              instance_path.complete_url, headers=headers, stream=True
          )
          response.raise_for_status()
          for chunk in response.iter_content(chunk_size=max(1, chunk_size)):
            output_stream.write(chunk)
      except requests.exceptions.HTTPError as exp:
        ez_wsi_errors.raise_ez_wsi_http_exception(exp.response.reason, exp)

    _inner_func()

  @retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
  def _get_download_instance_frame_list_untranscoded_core_request(
      self, query: str
  ) -> requests.Response:
    """Returns response for download_instance_frame_list_untranscoded.

    Will retry and refresh credentials if needed.

    Args:
      query: Query to send to DICOM store.

    Returns:
      Response from DICOM store.

    Raises:
      ez_wsi_errors.HTTPError if http error occurs.
    """
    headers = {
        'Accept': (
            'multipart/related; type="application/octet-stream";'
            ' transfer-syntax=*'
        )
    }
    self.credentials().apply(headers)
    try:
      response = requests.get(query, headers=headers)
      response.raise_for_status()
      return response
    except requests.exceptions.HTTPError as exp:
      ez_wsi_errors.raise_ez_wsi_http_exception(exp.response.reason, exp)

  def download_instance_frame_list_untranscoded(
      self,
      instance_path: dicom_path.Path,
      frame_numbers: Union[Sequence[int], Iterator[int]],
      retry: bool = True,
  ) -> List[bytes]:
    """Downloads a list of frames from a DICOM instance untranscoded.

    Args:
      instance_path: DICOM web path to instance to download.
      frame_numbers: Sequence or iterator of frame numbers.
      retry: Retry on retriable HTTP errors.

    Returns:
      List of bytes downloaded.

    Raises:
      DownloadInstanceFrameError: Number of frames returned by server != what
        was requested.
    """

    @retrying.retry(
        **error_retry_util.enable_config(
            retry, error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG
        )
    )
    def _inner_func() -> List[bytes]:
      if not frame_numbers:
        return []
      converter = _IntToStringConverter()
      query = (
          f'{instance_path}/frames/{",".join(converter.convert(frame_numbers))}'
      )
      frame_number_count = converter.count
      try:
        response = (
            self._get_download_instance_frame_list_untranscoded_core_request(
                query
            )
        )
        try:
          multipart_data = decoder.MultipartDecoder.from_response(response)
        except decoder.NonMultipartContentTypeException as exp:
          raise ez_wsi_errors.DownloadInstanceFrameError(
              'Received invalid multipart response.'
          ) from exp
        received_frame_count = len(multipart_data.parts)
        if received_frame_count != frame_number_count:
          raise ez_wsi_errors.DownloadInstanceFrameError(
              'DICOM Store returned incorrect number of frames. Expected:'
              f' {frame_number_count}, Received: {received_frame_count}.'
          )
        return [frame_bytes.content for frame_bytes in multipart_data.parts]
      except ez_wsi_errors.HttpError as exp:
        raise ez_wsi_errors.DownloadInstanceFrameError(
            'HTTP Error downloading DICOM frames.'
        ) from exp

    return _inner_func()

  def download_instance_frames_untranscoded(
      self,
      instance_path: dicom_path.Path,
      first_frame: int,
      last_frame: int,
      retry: bool = True,
  ) -> List[bytes]:
    """Downloads DICOM instance from store in native format to output stream.

    Args:
      instance_path: DICOM web path to instance to download.
      first_frame: index (base 1) inclusive.
      last_frame: index (base 1) inclusive.
      retry: Retry on retriable HTTP errors.

    Returns:
      List of bytes downloaded.

    Raises:
      DownloadInstanceFrameError: Number of frames returned by server != what
        was requested.
    """
    return self.download_instance_frame_list_untranscoded(
        instance_path, range(first_frame, last_frame + 1), retry=retry
    )


def _get_icc_profile_bulkdata_uri(dcm_tags: Mapping[str, Any]) -> str:
  """Returns icc profile bulkdata uri or empty str."""
  icc_profile = dcm_tags.get(tags.ICC_PROFILE.number)
  if icc_profile is None:
    return ''
  bulkdata_uri = icc_profile.get(_BULKDATA_URI)
  return bulkdata_uri if bulkdata_uri is not None else ''


def _find_icc_profile_bulkdata_uri(dcm_tags: Mapping[str, Any]) -> str:
  """Searches DICOM json for icc profile and returns bulkdata uri or empty str."""
  seq = dcm_tags.get(tags.OPTICAL_PATH_SEQUENCE.number)
  if seq is not None:
    seq = seq.get(_VALUE, [])
    if isinstance(seq, list):
      for dataset in seq:
        uri = _get_icc_profile_bulkdata_uri(dataset)
        if uri:
          return uri
  uri = _get_icc_profile_bulkdata_uri(dcm_tags)
  if uri:
    return uri
  return ''


def _build_dicom_object(
    object_type: dicom_path.Type,
    parent_path: dicom_path.Path,
    dicom_tags: Dict[str, Any],
) -> DicomObject:
  """Creates a DICOM object from a DICOM parent path and a set of tags.

  Args:
    object_type: The type of the object to construct.
    parent_path: the DICOM path pointing to the parent of the DICOM object to be
      constructed.
    dicom_tags: DICOM tags associated for the object construction.

  Returns:
    An instance of DicomObject constructed with proper type.
  Raises:
    DicomPathError if the type of the path constructed for the DICOM
    object does not match the input object_type.
  """
  path = dicom_path.FromPath(
      parent_path,
      # the study_uid is None for a store-level path
      study_uid=dicom_json.GetValue(dicom_tags, tags.STUDY_INSTANCE_UID),
      # the series_uid is None for a store-level or tudy-level path
      series_uid=dicom_json.GetValue(dicom_tags, tags.SERIES_INSTANCE_UID),
      instance_uid=dicom_json.GetValue(dicom_tags, tags.SOP_INSTANCE_UID),
  )
  if path.type != object_type:
    raise ez_wsi_errors.DicomPathError(
        f'The parent path and DICOM tags do not yield a path({path.type}) as '
        f'expected: {object_type}'
    )
  # test metadata for icc profile bulkdata uri
  icc_profile_bulkdata_uri = _find_icc_profile_bulkdata_uri(dicom_tags)
  # If returned remove tags used to find ICCProfile from instance
  for remove_tag in (tags.OPTICAL_PATH_SEQUENCE, tags.ICC_PROFILE):
    if remove_tag.number in dicom_tags:
      del dicom_tags[remove_tag.number]
  # remove unessential tags from multi-frame groups
  try:
    shared_functional_group_sequence = dicom_tags[
        tags.SHARED_FUNCTIONAL_GROUP_SEQUENCE.number
    ]['Value'][0]
    for tag_address in list(shared_functional_group_sequence):
      if tag_address != tags.PIXEL_MEASURES_SEQUENCE.number:
        del shared_functional_group_sequence[tag_address]
  except (KeyError, IndexError) as _:
    pass
  return DicomObject(path, dicom_tags, icc_profile_bulkdata_uri)


def _get_qido_suffix(
    tags_to_fetch: Collection[tags.DicomTag], **wargs: str | int
) -> str:
  """Returns the suffix to be used in a QIDO-RS query.

  Args:
    tags_to_fetch: DICOM tags to be fetched for each instance.
    **wargs: Additional query parameters to include in the suffix.

  Example:
    suffix = _get_qido_suffix(
        tags_to_fetch=[tags.PatientName, tags.StudyDate],
        Modality="SM",
        limit=1
    )
  """
  query_params = {'includefield': [tag.number for tag in tags_to_fetch]}

  # Add any additional keyword arguments to the query parameters
  query_params.update(wargs)

  return urllib.parse.urlencode(query_params, doseq=True)
