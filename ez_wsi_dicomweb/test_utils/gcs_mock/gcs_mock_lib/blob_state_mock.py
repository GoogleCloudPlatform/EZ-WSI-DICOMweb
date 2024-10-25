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
from __future__ import annotations

import dataclasses
import datetime
from typing import MutableMapping, Optional


@dataclasses.dataclass
class BlobStateMock:
  """Blob level state on blob and in GcsMock."""

  size_in_bytes: Optional[int] = None
  md5_hash: Optional[str] = None
  metadata: Optional[MutableMapping[str, str]] = None
  generation: Optional[int] = None
  metageneration: Optional[int] = None
  etag: Optional[str] = None
  content_type: Optional[str] = None
  time_created: Optional[datetime.datetime] = None
  time_deleted: Optional[datetime.datetime] = None
  updated: Optional[datetime.datetime] = None
  component_count: Optional[int] = None

  def __copy__(self) -> BlobStateMock:
    return BlobStateMock(
        self.size_in_bytes,
        self.md5_hash,
        None if self.metadata is None else dict(self.metadata),
        self.generation,
        self.metageneration,
        self.etag,
        self.content_type,
        self.time_created,
        self.time_deleted,
        self.updated,
        self.component_count,
    )
