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
"""Types used in local dicom slide cache."""
import dataclasses
import enum
from typing import Any, Optional, Union

from ez_wsi_dicomweb import slide_level_map


InstancePathType = Union[slide_level_map.Level, slide_level_map.Instance]


class LocalDicomSlideCacheError(Exception):
  pass


class InvalidFrameNumberError(LocalDicomSlideCacheError):

  def __init__(
      self,
      msg: str = 'Encountered DICOM frame number < 1; DICOM frame numbers are >= 1.',
  ):
    super().__init__(msg)


class UnexpectedTypeError(LocalDicomSlideCacheError):
  pass


class InvalidLRUMaxCacheSizeError(LocalDicomSlideCacheError):

  def __init__(
      self,
      msg: str = 'LRU max cache size in bytes is less than 1. Optimally cache max size should be in megabyte - gigabyte range.',
  ):
    super().__init__(msg)


class CacheConfigOptimizationHint(enum.Enum):
  """Optimization hints provided to inference_cache."""

  # Block and wait for cache loading to complete if cache miss occurs.
  MINIMIZE_DICOM_STORE_QPM = 'MINIMIZE_DICOM_STORE_QPM'
  # On cache miss load frame block in a background thread and read and return
  # missed frame. Cache miss frame will be provided as fast as possible with
  # second query.
  MINIMIZE_LATENCY = 'MINIMIZE_LATENCY'
  DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE = (
      'DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE'
  )


@dataclasses.dataclass
class CacheStats:
  """Select metrics and logs to monitor local_dicom_slide_cache use and perf."""

  # Range of the frame numbers requested
  first_frame_number_read: Optional[int] = None
  last_frame_number_read: Optional[int] = None

  # Metrics for cache hit and miss
  frame_cache_miss_count: int = 0
  frame_cache_hit_count: int = 0

  # Metrics associated with async retrieval, of block of frames and pre-fetching
  # frames. Read will occur async with other operations if MINIMIZE_LATENCY
  # optimization hint is used. If operating async frame retrieval
  # frame_block_read_time will not represent frame serving latency.
  frame_block_read_time: float = 0.0
  number_of_frame_blocks_read: int = 0
  number_of_frames_read_in_frame_blocks: int = 0
  number_of_frame_bytes_read_in_frame_blocks: int = 0

  # Metrics associated with whole instance retrieval. Read may occur async with
  # other operations if MINIMIZE_LATENCY optimization hint is used. If operating
  # async frame retrieval frame_block_read_time will not represent frame
  # serving latency.
  dicom_instance_read_time: float = 0.0
  number_of_dicom_instances_read: int = 0
  number_of_frames_read_in_dicom_instances: int = 0
  number_of_frame_bytes_read_in_dicom_instances: int = 0

  # Metrics associated with cache miss latency reduction frame retrieval.
  # If MINIMIZE_LATENCY optimization hint is used
  # time_spent_downloading_frames_to_reduce_latency is a measure of the total
  # frame read time for the served frames.
  number_of_frames_downloaded_to_reduce_latency: int = 0
  time_spent_downloading_frames_to_reduce_latency: float = 0.0

  # Metrics associated with time spent waiting for cache loading operations to
  # complete. MINIMIZE_DICOM_STORE_QPM optmization hint causes cache to wait
  # for cache loading operations to complete when a cache miss occurs.
  # Metric time may not be the total sum of all read times if some of the cache
  # loading operations completed without a cache miss.
  time_spent_blocked_waiting_for_cache_loading_to_complete: float = 0.0

  # Memory limits and system state set when cache status is returned.
  frame_cache_memory_size_limit: Optional[int] = None  # None == unlimited
  current_frame_cache_memory_size: Optional[int] = None  # None == NA
  # system memory reported by psutil
  system_memory: Optional[Any] = None
