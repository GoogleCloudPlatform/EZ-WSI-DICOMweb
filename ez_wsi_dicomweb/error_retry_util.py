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
"""Configuration settings for retrying HTTP errors."""

from typing import Any, Mapping, Union

from ez_wsi_dicomweb import ez_wsi_errors

_RetriableErrors = Union[
    ez_wsi_errors.HttpInternalServerError,
    ez_wsi_errors.HttpTooManyRequestsError,
    ez_wsi_errors.HttpRequestTimeoutError,
    ez_wsi_errors.HttpServiceUnavailableError,
    ez_wsi_errors.HttpGatewayTimeoutError,
]

_AuthErrors = Union[
    ez_wsi_errors.HttpUnauthorizedError, ez_wsi_errors.HttpForbiddenError
]


def _is_retriable_http_error(exception: Exception) -> bool:
  return isinstance(exception, _RetriableErrors)


def _is_retriable_http_auth_error(exception: Exception) -> bool:
  return isinstance(exception, _AuthErrors)


def _is_other_http_exception(exp: Exception) -> bool:
  if not isinstance(exp, ez_wsi_errors.HttpError):
    return False
  return not _is_retriable_http_error(
      exp
  ) and not _is_retriable_http_auth_error(exp)


HTTP_AUTH_ERROR_RETRY_CONFIG = dict(
    retry_on_exception=_is_retriable_http_auth_error,
    stop_max_attempt_number=3,
)

HTTP_SERVER_ERROR_RETRY_CONFIG = dict(
    retry_on_exception=_is_retriable_http_error,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)


def enable_config(retry: bool, config: Mapping[str, Any]) -> Mapping[str, Any]:
  if retry:
    return config
  config = dict(config)
  config['stop_max_attempt_number'] = 1
  return config


def other_http_exception_retry_config(retry_count: int) -> Mapping[str, Any]:
  return dict(
      retry_on_exception=_is_other_http_exception,
      wait_exponential_multiplier=500,
      wait_exponential_max=5000,
      stop_max_attempt_number=max(1, retry_count),
  )
