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
"""Error classes for EZ WSI DicomWeb."""
import http.client
from typing import NoReturn, Union

import google.api_core.exceptions
import requests


class EZWsiError(Exception):
  pass


class CoordinateOutofImageDimensionsError(EZWsiError):
  pass


class DicomImageMissingRegionError(EZWsiError):
  pass


class DicomPathError(EZWsiError):
  pass


class DicomInstanceReadError(EZWsiError):
  pass


class DicomSlideMissingError(EZWsiError):
  pass


class DicomTagNotFoundError(EZWsiError):
  pass


class FrameNumberOutofBoundsError(EZWsiError):
  pass


class InputFrameNumberOutOfRangeError(EZWsiError):
  pass


class InternalError(EZWsiError):
  pass


class InvalidDicomTagError(EZWsiError):
  pass


class InvalidMagnificationStringError(EZWsiError):
  pass


class LevelNotFoundError(EZWsiError):
  pass


class MagnificationLevelNotFoundError(EZWsiError):
  pass


class NoDicomLevelsDetectedError(EZWsiError):
  pass


class PatchIntersectionNotFoundError(EZWsiError):
  pass


class InvalidPatchDimensionError(EZWsiError):
  pass


class PixelSpacingLevelNotFoundError(EZWsiError):
  pass


class PatchOutsideOfImageDimensionsError(EZWsiError):
  pass


class PixelSpacingNotFoundForInstanceError(EZWsiError):
  pass


class SectionOutOfImageBoundsError(EZWsiError):
  pass


class SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError(
    EZWsiError
):

  def __init__(self):
    super().__init__(
        'DICOM level instances do not have the same TransferSyntaxUID'
    )


class UnexpectedDicomObjectInstanceError(EZWsiError):
  pass


class UnexpectedDicomSlideCountError(EZWsiError):
  pass


class UndefinedPixelSpacingError(EZWsiError):
  pass


class UnsupportedPixelFormatError(EZWsiError):
  pass


class DicomPatchGenerationError(EZWsiError):
  pass


class DicomSlideInitError(EZWsiError):
  pass


class InvalidSlideJsonMetadataError(EZWsiError):
  pass


class SlidePathDoesNotMatchJsonMetadataError(EZWsiError):
  pass


class PatchEmbeddingDimensionError(EZWsiError):
  pass


class SinglePatchEmbeddingEnsembleError(EZWsiError):
  pass


class SinglePatchEmbeddingEnsemblePositionError(EZWsiError):
  pass


class MeanPatchEmbeddingEnsembleError(EZWsiError):
  pass


class PatchEmbeddingEndpointError(EZWsiError):
  pass


class InvalidTissueMaskError(EZWsiError):
  pass


class GcsImageError(EZWsiError):
  pass


class GcsImagePathFormatError(GcsImageError):
  pass


class DownloadInstanceFrameError(EZWsiError):
  pass


class HttpError(EZWsiError):
  """Base class for HTTP errors."""

  def __init__(
      self,
      message: str = '',
      status_code: int = http.client.INTERNAL_SERVER_ERROR,
      reason: str = '',
  ):
    super().__init__(message)
    self._status_code = status_code
    self._reason = reason

  @property
  def status_code(self) -> int:
    return self._status_code

  @property
  def reason(self) -> str:
    return self._reason


class HttpTooManyRequestsError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.TOO_MANY_REQUESTS, reason)


class HttpRequestTimeoutError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.REQUEST_TIMEOUT, reason)


class HttpServiceUnavailableError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.SERVICE_UNAVAILABLE, reason)


class HttpGatewayTimeoutError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.GATEWAY_TIMEOUT, reason)


class HttpInternalServerError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.INTERNAL_SERVER_ERROR, reason)


class HttpUnauthorizedError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.UNAUTHORIZED, reason)


class HttpForbiddenError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.FORBIDDEN, reason)


class HttpBadRequestError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.BAD_REQUEST, reason)


class HttpNotFoundError(HttpError):

  def __init__(self, message: str = '', reason: str = ''):
    super().__init__(message, http.client.NOT_FOUND, reason)


class InvalidWadoRsResponseError(HttpError):
  pass


class HttpUnexpectedResponseError(HttpError):
  pass


_HTTP_ERROR_CODE_EXCEPTION = {
    # Retriable errors
    http.client.INTERNAL_SERVER_ERROR: HttpInternalServerError,
    http.client.TOO_MANY_REQUESTS: HttpTooManyRequestsError,
    http.client.REQUEST_TIMEOUT: HttpRequestTimeoutError,
    http.client.SERVICE_UNAVAILABLE: HttpServiceUnavailableError,
    http.client.GATEWAY_TIMEOUT: HttpGatewayTimeoutError,
    # retry auth
    http.client.UNAUTHORIZED: HttpUnauthorizedError,
    http.client.FORBIDDEN: HttpForbiddenError,
    # non-retriable errors
    http.client.BAD_REQUEST: HttpBadRequestError,
    http.client.NOT_FOUND: HttpNotFoundError,
}


def raise_ez_wsi_http_exception(
    message: str,
    trigger_exception: Union[
        requests.exceptions.HTTPError,
        google.api_core.exceptions.GoogleAPICallError,
    ],
) -> NoReturn:
  """Raises an EZ WSI HttpError from a requests.HTTPError or GoogleAPICallError."""
  try:
    status_code = trigger_exception.response.status_code
    reason = trigger_exception.response.reason
  except AttributeError:
    status_code = http.client.INTERNAL_SERVER_ERROR
    reason = ''
  exception_class = _HTTP_ERROR_CODE_EXCEPTION.get(status_code)
  if exception_class is not None:
    raise exception_class(message, reason) from trigger_exception
  raise HttpUnexpectedResponseError(message, status_code, reason) from (
      trigger_exception
  )
