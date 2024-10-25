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
"""Types used in GCS mock."""
import enum
from typing import Set, Tuple, Union

import google.api_core
import google.cloud.storage

GcsClientType = Union[
    google.cloud.storage.Client, google.cloud.storage.client.Client
]
GcsBucketType = Union[
    google.cloud.storage.Bucket, google.cloud.storage.bucket.Bucket
]
GcsBlobType = Union[google.cloud.storage.Blob, google.cloud.storage.blob.Blob]

EtagType = Union[str, Set[str]]

RetryType = Union[
    google.api_core.retry.Retry,
    google.cloud.storage.retry.ConditionalRetryPolicy,
]

TimeoutType = Union[float, Tuple[float, float]]


class GcsMockError(Exception):
  """Base class for all exceptions thrown by GcsMock."""


class HttpMethod(enum.Enum):
  DELETE = 'DELETE'
  GET = 'GET'
  POST = 'POST'
  PUT = 'PUT'
