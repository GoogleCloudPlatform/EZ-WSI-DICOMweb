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
"""Utility to interface with DICOM Store."""
from typing import BinaryIO, Iterator, List, Mapping, Optional, Sequence, Union

from ez_wsi_dicomweb import dicom_web_interface
import requests
from requests_toolbelt.multipart import decoder

# Default chunksize for instance downloads.
# Smaller or larger chunksizes can be used to use more or less
# memory in the streaming download.
_STREAMING_CHUNKSIZE = 102400


class DownloadInstanceFrameError(Exception):
  pass


def _get_accept_key(headers: Mapping[str, str]) -> str:
  for key in headers:
    if key.lower() == 'accept':
      return key
  return 'Accept'


def download_instance_untranscoded(
    instance_path: str,
    output_stream: BinaryIO,
    headers: Optional[Mapping[str, str]] = None,
    chunk_size: int = _STREAMING_CHUNKSIZE,
    dicom_store_url: str = dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL,
) -> bool:
  """Downloads DICOM instance from store in native format to output stream.

  Args:
    instance_path: DICOM web path to instance to download.
    output_stream: Output stream to write to.
    headers: Optional headers to include with HTTP request.
    chunk_size: Streaming download chunksize.
    dicom_store_url: Base URL to DICOM store.

  Returns:
    True if download succeeds.
  """
  headers = {} if headers is None else dict(headers)
  headers[_get_accept_key(headers)] = 'application/dicom; transfer-syntax=*'
  query = f'{dicom_store_url}/{instance_path}'
  with requests.Session() as session:
    response = session.get(query, headers=headers, stream=True)
    try:
      response.raise_for_status()
      for chunk in response.iter_content(chunk_size=max(1, chunk_size)):
        output_stream.write(chunk)
      return True
    except requests.exceptions.HTTPError:
      return False


def download_instance_frame_list_untranscoded(
    instance_path: str,
    frame_numbers: Union[Sequence[int], Iterator[int]],
    headers: Optional[Mapping[str, str]] = None,
    dicom_store_url: str = dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL,
) -> List[bytes]:
  """Downloads a list of frames from a DICOM instance untranscoded.

  Args:
    instance_path: DICOM web path to instance to download.
    frame_numbers: Sequence of frame numbers.
    headers: Optional headers to include with HTTP request.
    dicom_store_url: Base URL to DICOM store.

  Returns:
    List of bytes downloaded.

  Raises:
    DownloadInstanceFrameError: Number of frames returned by server != what
      was requested.
  """

  def int_to_str(
      frame_numbers: Union[Sequence[int], Iterator[int]]
  ) -> Iterator[str]:
    """Converts int seq/iter to string iter and counts total number of frames."""
    nonlocal frame_number_count
    for frame_number in frame_numbers:
      frame_number_count += 1
      yield str(frame_number)

  if not frame_numbers:
    return []
  headers = {} if headers is None else dict(headers)
  headers[_get_accept_key(headers)] = (
      'multipart/related; type=application/octet-stream; transfer-syntax=*'
  )
  frame_number_count = 0
  query = f'{dicom_store_url}/{instance_path}/frames/{",".join(int_to_str(frame_numbers))}'
  try:
    with requests.Session() as session:
      response = session.get(query, headers=headers, stream=True)
      response.raise_for_status()
      try:
        multipart_data = decoder.MultipartDecoder.from_response(response)
      except decoder.NonMultipartContentTypeException as exp:
        raise DownloadInstanceFrameError(
            'Received invalid multipart response.'
        ) from exp
      received_frame_count = len(multipart_data.parts)
      if received_frame_count != frame_number_count:
        raise DownloadInstanceFrameError(
            'DICOM Store returned incorrect number of frames. Expected:'
            f' {frame_number_count}, Received: {received_frame_count}.'
        )
    return [frame_bytes.content for frame_bytes in multipart_data.parts]
  except requests.exceptions.HTTPError as exp:
    raise DownloadInstanceFrameError(
        'HTTP Error downloading DICOM frames.'
    ) from exp


def download_instance_frames_untranscoded(
    instance_path: str,
    first_frame: int,
    last_frame: int,
    headers: Optional[Mapping[str, str]] = None,
    dicom_store_url: str = dicom_web_interface.DEFAULT_DICOMWEB_BASE_URL,
) -> List[bytes]:
  """Downloads DICOM instance from store in native format to output stream.

  Args:
    instance_path: DICOM web path to instance to download.
    first_frame: index (base 1) inclusive.
    last_frame: index (base 1) inclusive.
    headers: Optional headers to include with HTTP request.
    dicom_store_url: Base URL to DICOM store.

  Returns:
    List of bytes downloaded.

  Raises:
    DownloadInstanceFrameError: Number of frames returned by server != what
      was requested.
  """
  return download_instance_frame_list_untranscoded(
      instance_path,
      range(first_frame, last_frame + 1),
      headers,
      dicom_store_url,
  )
