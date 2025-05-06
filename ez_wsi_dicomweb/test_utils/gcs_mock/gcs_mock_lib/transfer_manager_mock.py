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
"""Mock of google cloud storage transfer_manager."""
from typing import Any, Mapping, Optional

import google.cloud.storage.transfer_manager

_pre_mock_download_chunks_concurrently = (
    google.cloud.storage.transfer_manager.download_chunks_concurrently
)


def download_chunks_concurrently(
    blob: google.cloud.storage.Blob,
    filename: str,
    chunk_size: int = 33554432,
    download_kwargs: Optional[Mapping[str, Any]] = None,
    deadline: Optional[float] = None,
    worker_type: str = 'process',
    max_workers: int = 8,
    *,
    crc32c_checksum: bool = True,
) -> None:
  """Mock download_chunks_concurrently."""
  if worker_type not in ('thread', 'process'):
    raise ValueError('worker_type must be one of "thread" or "process".')
  if max_workers < 1:
    raise ValueError('max_workers must be at least 1.')
  del chunk_size, download_kwargs, deadline, crc32c_checksum
  if blob.client is None:
    raise ValueError('Blob client is None.')
  blob.download_to_filename(filename)
