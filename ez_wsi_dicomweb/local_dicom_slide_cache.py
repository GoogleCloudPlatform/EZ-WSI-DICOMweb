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
"""Local DICOM Slide Read Cache."""

from __future__ import annotations

import collections
from concurrent import futures
import copy
import functools
import heapq
import io
import itertools
import threading
import time
import typing
from typing import Any, BinaryIO, Iterator, List, Mapping, MutableMapping, Optional, Tuple
import uuid

import cachetools
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import psutil
import pydicom


_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])


class _LogKeywords:
  DICOM_FRAME_NUMBER_RANGE_LIST = 'dicom_frame_number_range_list'
  DICOM_WEB_INSTANCE_PATH = 'dicom_web_instance_path'
  EXECUTION_TIME_SEC = 'execution_time_sec'
  FRAME_NUMBER = 'frame_number'
  INSTANCE_CACHE_WORKER_TRACE_UID = 'instance_cache_worker_trace_uid'
  INSTANCE_CACHE_LIFETIME_TRACE_UID = 'instance_cache_lifetime_trace_uid'
  NUMBER_OF_FRAMES_IN_DICOM_INSTANCE = 'number_of_frames_in_dicom_instance'
  RUNNING_AS_THREAD = 'running_as_thread'


# Number of frames to read when frame request misses the cache.
DEFAULT_NUMBER_OF_FRAMES_TO_READ_ON_CACHE_MISS = 500

# Control total number of threads which can be executed concurrently to
# orchestrate operations. Orchestrate operations queue operations to
# uploaded and download queues. Orchestrator functions queued on a
# separate thread pool from upload and downloads to make it impossible for
# running orchestrator threads to block execution of upload and download
# threads.
_MAX_ORCHESTRATOR_WORKER_THREADS = int(4)

# Prefer whole instance downloads over frame retrieval if total instance
# frame size is smaller than threshold.
MAX_INSTANCE_NUMBER_OF_FRAMES_TO_PREFER_WHOLE_INSTANCE_DOWNLOAD = int(10000)

# https://www.dicomlibrary.com/dicom/transfer-syntax/
_UNENCAPSULATED_TRANSFER_SYNTAXES = frozenset([
    '1.2.840.10008.1.2.1',  # 	Explicit VR Little Endian
    '1.2.840.10008.1.2',  # Implicit VR Endian: Default Transfer Syntax
    '1.2.840.10008.1.2.1.99',  # Deflated Explicit VR Little Endian
    '1.2.840.10008.1.2.2',  # Explicit VR Big Endian
])

_InstanceFameKey = Tuple[int, str]
_SharedFrameMemory = MutableMapping[_InstanceFameKey, bytes]


def _future_list_built_test_hook(
    future_list: List[futures.Future[None]],
) -> List[futures.Future[None]]:
  """Function is a NOP in prod;  Acts as mock target for unit test.

  Used in test_block_until_all_instance_frame_futures_are_loaded mocks.
    Test validates that block_until_frames_are_loaded waits correctly for
    futures to complete. Mock enables unit test to ensure that threads will not
    complete before waiting has begun (the actual unit test case). Other test
    cases test that block_until_all_instance_frame_futures_are_loaded functions
    correctly if the tests complete before waiting starts.

  Args:
    future_list: List of futures.

  Returns:
    List of futures.
  """
  return future_list


def _frame_number_key(
    instance_path: dicom_path.Path, frame_number: int
) -> _InstanceFameKey:
  """Returns instance frame hash key for shared memory dict."""
  return (frame_number, instance_path.complete_url)


def _is_unencapsulated_image_transfer_syntax(uid: str) -> bool:
  """Returns True if uid is in the list of raw transfer syntaxes."""
  return uid in _UNENCAPSULATED_TRANSFER_SYNTAXES


def _load_frame_list(buffer: BinaryIO) -> List[bytes]:
  """Loads DICOM instance frames into memory as a list of frames(bytes).

  Args:
    buffer: Buffer containing binary DICOM instance data.

  Returns:
    List of bytes encoded in DICOM instance frames.
  """
  with pydicom.dcmread(buffer) as ds:
    if 'PixelData' not in ds or not ds.PixelData:
      return []
    try:
      number_of_frames = int(ds.NumberOfFrames)
    except (ValueError, AttributeError) as _:
      return []
    if number_of_frames < 1:
      return []
    if _is_unencapsulated_image_transfer_syntax(ds.file_meta.TransferSyntaxUID):
      step = int(len(ds.PixelData) / number_of_frames)
      return [
          ds.PixelData[fnum * step : (fnum + 1) * step]
          for fnum in range(number_of_frames)
      ]
    if _PYDICOM_MAJOR_VERSION <= 2:
      # pytype: disable=module-attr
      frame_bytes_generator = pydicom.encaps.generate_pixel_data_frame(
          ds.PixelData, number_of_frames
      )
      # pytype: enable=module-attr
    else:
      # pytype: disable=module-attr
      frame_bytes_generator = pydicom.encaps.generate_frames(
          ds.PixelData, number_of_frames=number_of_frames
      )
      # pytype: enable=module-attr
    return [frame_bytes for frame_bytes in frame_bytes_generator]


def _get_frame_number_range_list(
    frame_list: List[int],
    logger: ez_wsi_logging_factory.AbstractLoggingInterface,
) -> List[Tuple[int, int]]:
  """Converts list of frame numbers into inclusive list ranges of frame numbers.

  Converts [1, 2, 3, 8, 9, 10] -> [(1, 3), (8, 10)]

  Args:
    frame_list: List of frames to load.
    logger: CloudLoggingClientInstance

  Returns:
    List of Frame number ranges to load.

  Raises:
    InvalidFrameNumberError: Frame number values < 1 were provided.
  """
  frame_number_range_list = []
  start_frame_number = None
  prior_frame_number = -1
  for fnum in frame_list:
    if fnum < 1:
      raise local_dicom_slide_cache_types.InvalidFrameNumberError(
          f'DICOM Frame numbers must be >= 1; encountered: {fnum}.'
      )
    if fnum < prior_frame_number:
      logger.warning(
          'Performance could be improved by providing the list of frame numbers'
          ' in sorted order.'
      )
      return _get_frame_number_range_list(sorted(frame_list), logger)
    if start_frame_number is None:
      start_frame_number = fnum
      prior_frame_number = fnum
    elif prior_frame_number == fnum:
      continue
    elif prior_frame_number + 1 == fnum:
      prior_frame_number = fnum
    else:
      frame_number_range_list.append((start_frame_number, prior_frame_number))
      start_frame_number = fnum
      prior_frame_number = fnum
  if start_frame_number is not None:
    frame_number_range_list.append((start_frame_number, prior_frame_number))
  return frame_number_range_list


def _get_instance_path_list(
    instance_path: local_dicom_slide_cache_types.InstancePathType,
) -> List[Tuple[dicom_path.Path, int]]:
  """Returns a list of paths to DICOM instances specified by param.

  Args:
    instance_path: Value which represents one or more DICOM web instances.

  Returns:
    List of tuples[DICOM web instance paths, number of frames in instance].

  Raises:
    local_dicom_slide_cache_types.UnexpectedTypeError: Instance_path is an
      unexpected type.
  """
  if isinstance(instance_path, slide_level_map.Level):
    return [
        (instance.dicom_object.path, instance.frame_count)
        for instance in instance_path.instances.values()
    ]
  elif isinstance(instance_path, slide_level_map.Instance):
    return [(
        instance_path.dicom_object.path,
        instance_path.frame_count,
    )]
  else:
    raise local_dicom_slide_cache_types.UnexpectedTypeError(
        'Unexpected DICOM web instance path type.'
    )


def _log_elapsed_time(start_time: float) -> Mapping[str, Any]:
  return {_LogKeywords.EXECUTION_TIME_SEC: time.time() - start_time}


class InMemoryDicomSlideCache:
  """In memory cache for EZ-WSI library access of DICOM Instances.

  Cache functions by downloading DICOM instance pixel data from DICOM store
  in blocks or entire instances. The total size of the frame cache can be
  managed via a size (bytes) limited LRU. The primary purposes of the cache is
  to speed access to DICOM frame data and reduce the total number of DICOM
  queries required to fetch digital pathology frame data from the DICOM store.

  Attributes:
    _dicom_instance_frame_bytes: Shared memory used to hold DICOM instance frame
      bytes.
    _number_of_frames_to_read: Number of frames of data to read on cache miss.
    _dicom_web_interface: Interface for DICOMweb.
    _lock: Threading lock to make cache access thread safe.
    _cache_stats: Cache state and logs.
    _orchestrator_thread_pool: Thread pool for high level ops which queue upload
      and download ops.
    lru_caching_enabled: True if cache a LRU cache that has a defined max size
      (bytes) | or False no defined max size.
    cache_instance_uid: UID for the cache instance preserved for lifetime of the
      cache and across pickle operations.
    optimization_hint: Cache optmization hint.
  """

  def __init__(
      self,
      credential_factory: credential_factory_module.AbstractCredentialFactory,
      max_cache_frame_memory_lru_cache_size_bytes: Optional[int] = None,
      number_of_frames_to_read: int = DEFAULT_NUMBER_OF_FRAMES_TO_READ_ON_CACHE_MISS,
      max_instance_number_of_frames_to_prefer_whole_instance_download: int = MAX_INSTANCE_NUMBER_OF_FRAMES_TO_PREFER_WHOLE_INSTANCE_DOWNLOAD,
      optimization_hint: local_dicom_slide_cache_types.CacheConfigOptimizationHint = local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
  ):
    """Initializes InMemoryDicomSlideCache.

    Args:
      credential_factory: Factory to create credentials to use to access the
        DICOM store.
      max_cache_frame_memory_lru_cache_size_bytes: Maximum size of cache in
        bytes.  Ideally should be in hundreds of megabyts-to-gigabyte size. If
        None, no limit to size.
      number_of_frames_to_read: Number of frames to read on cache miss.
      max_instance_number_of_frames_to_prefer_whole_instance_download: Max
        number of frames to prefer downloading whole instances over retrieving
        frames in batch (Typically faster for small instances e.g. < 10,0000).
        Optimal threshold will depend on average size of instance frame data and
        the size of non frame instance metadata.
      optimization_hint: Optimize cache to minimize data latency or total
        queries to the DICOM store.
      logging_factory: Factory to create logging interface defaults to Python
        logger.

    Raises:
      InvalidLRUMaxCacheSizeError: Invalid LRU max cache size.
    """
    if max_cache_frame_memory_lru_cache_size_bytes is None:
      self._max_cache_frame_memory_lru_cache_size_bytes = None
    elif max_cache_frame_memory_lru_cache_size_bytes < 1:
      raise local_dicom_slide_cache_types.InvalidLRUMaxCacheSizeError()
    else:
      self._max_cache_frame_memory_lru_cache_size_bytes = (
          max_cache_frame_memory_lru_cache_size_bytes
      )
    self._dicom_instance_frame_bytes: _SharedFrameMemory = (
        self._init_dicom_instance_frame_bytes()
    )
    if logging_factory is None:
      self._logging_factory = ez_wsi_logging_factory.BasePythonLoggerFactory()
    else:
      self._logging_factory = logging_factory
    self._cache_instance_uid = uuid.uuid1()
    self._logger = None
    self._number_of_frames_to_read = int(max(number_of_frames_to_read, 1))
    self._dicom_web_interface = dicom_web_interface.DicomWebInterface(
        credential_factory
    )
    # Primary lock used to protect shared class state aross threads.
    # Required to be RLock to protect against future add_done_callback
    # callback finishing while locked causing a deadlock.
    self._lock = threading.RLock()
    # Protects lazy initialized class state (logger & authentication)
    # Different lock from self._lock to enable state to be safely initialized
    # independently from broader lock.
    self._initialization_lock = threading.Lock()
    self._cache_stats = local_dicom_slide_cache_types.CacheStats()
    self._orchestrator_thread_pool = None
    # Maps dicom_web_instance_path to dict of thread futures being run to load
    # instance frames. This dict in turn maps these thread futures to list of
    # frame ranges being loaded by the thread.
    self._running_futures: MutableMapping[
        str, MutableMapping[futures.Future[None], List[Tuple[int, int]]]
    ] = collections.defaultdict(dict)
    self._optimization_hint = optimization_hint
    self._max_instance_number_of_frames_to_prefer_whole_instance_download = (
        max_instance_number_of_frames_to_prefer_whole_instance_download
    )
    self._init_thread_pool()

  @property
  def optimization_hint(
      self,
  ) -> local_dicom_slide_cache_types.CacheConfigOptimizationHint:
    return self._optimization_hint

  @optimization_hint.setter
  def optimization_hint(
      self,
      optimization_hint: local_dicom_slide_cache_types.CacheConfigOptimizationHint,
  ) -> None:
    self._optimization_hint = optimization_hint

  @property
  def cache_instance_uid(self) -> str:
    """Returns cache instance UID."""
    return str(self._cache_instance_uid)

  @property
  def lru_caching_enabled(self) -> bool:
    return isinstance(self._dicom_instance_frame_bytes, cachetools.LRUCache)

  def cache_externally_acquired_bytes(self, key: str, data: bytes) -> bool:
    """Adds externally acquired bytes to cache.

    Args:
      key: Cache key for external bytes.
      data: Bytes to add.

    Returns:
      True if data added to cache.
    """
    if (
        self._max_cache_frame_memory_lru_cache_size_bytes is not None
        and len(data) > self._max_cache_frame_memory_lru_cache_size_bytes
        and self.lru_caching_enabled
    ):
      self._get_logger().warning(
          'Data not cached. The maximum size in bytes of the LRU cache is '
          'smaller than the total size in bytes of the data. Data size: '
          f'{len(data)} bytes; Maximum size of cache: '
          f'{self._max_cache_frame_memory_lru_cache_size_bytes}.'
      )
      return False
    self._dicom_instance_frame_bytes[f'ext:{key}'] = data
    return True

  def get_cached_externally_acquired_bytes(self, key: str) -> Optional[bytes]:
    """Returns acquired bytes from cache or None if key not found."""
    return self._dicom_instance_frame_bytes.get(f'ext:{key}')

  def _init_dicom_instance_frame_bytes(
      self,
  ) -> _SharedFrameMemory:
    if self._max_cache_frame_memory_lru_cache_size_bytes is not None:
      return cachetools.LRUCache(
          maxsize=self._max_cache_frame_memory_lru_cache_size_bytes,
          getsizeof=len,
      )
    else:
      return dict()

  def _set_cached_frame(
      self,
      instance_path: dicom_path.Path,
      frame_number: int,
      frame_bytes: bytes,
  ) -> None:
    """Set instance frame bytes in cache."""
    cache_key = _frame_number_key(instance_path, frame_number)
    try:
      self._dicom_instance_frame_bytes[cache_key] = frame_bytes
    except ValueError as exp:
      if (
          self.lru_caching_enabled
          and len(frame_bytes)
          > self._max_cache_frame_memory_lru_cache_size_bytes
      ):
        self._get_logger().warning(
            'The maximum size in bytes of the LRU cache is smaller than the '
            'size of a single DICOM frame; LRU maximum cache size (bytes): '
            f'{self._max_cache_frame_memory_lru_cache_size_bytes}; '
            f'Size in bytes of DICOM frame: {len(frame_bytes)}. For optimal '
            "performance LRU cache should be large enough to store 10's-100's"
            ' of thousands of DICOM frames.',
            exp,
        )
      else:
        self._get_logger().warning(
            'Unexpected value error occurred adding frame data to cache.', exp
        )

  def _add_cached_instance_frames(
      self,
      instance_path: dicom_path.Path,
      first_frame_number: int,
      frame_list: List[bytes],
  ) -> None:
    """Adds instance frame data to frame cache.

    Args:
      instance_path: DICOMweb instance path.
      first_frame_number: First frame number of frame to set.
      frame_list: List of bytes for consecutive frames.

    Returns:
      None
    """
    for frame_number in range(
        first_frame_number, first_frame_number + len(frame_list)
    ):
      frame_bytes = frame_list[frame_number - first_frame_number]
      self._set_cached_frame(instance_path, frame_number, frame_bytes)

  def _total_frame_bytes_read(self, dicom_frames: List[bytes]) -> int:
    """Returns total number of bytes in list of frames."""
    total_frame_bytes_read = 0
    for frame_bytes in dicom_frames:
      total_frame_bytes_read += len(frame_bytes)
    if (
        self._max_cache_frame_memory_lru_cache_size_bytes is not None
        and total_frame_bytes_read
        > self._max_cache_frame_memory_lru_cache_size_bytes
        and self.lru_caching_enabled
    ):
      self._get_logger().warning(
          'The maximum size in bytes of the LRU cache is smaller than the '
          f'total size in bytes of the block of {len(dicom_frames)} DICOM '
          'frame(s) that was added to the cache in a single cache miss event; '
          'LRU maximum cache size (bytes): '
          f'{self._max_cache_frame_memory_lru_cache_size_bytes}; '
          f'Total size in bytes of DICOM frame(s): {total_frame_bytes_read}. '
          'For optimal performance LRU cache size should be be increased to a '
          'size much larger than the total number of frame bytes read at a '
          'single cache miss event.'
      )
    return total_frame_bytes_read

  def _get_logger_signature(self) -> Mapping[str, Any]:
    return {
        _LogKeywords.INSTANCE_CACHE_WORKER_TRACE_UID: str(uuid.uuid1()),
        _LogKeywords.INSTANCE_CACHE_LIFETIME_TRACE_UID: str(
            self._cache_instance_uid
        ),
    }

  def _get_logger(
      self,
  ) -> ez_wsi_logging_factory.AbstractLoggingInterface:
    if self._logger is not None:
      return self._logger
    with self._initialization_lock:
      if self._logger is None:
        self._logger = self._logging_factory.create_logger(
            self._get_logger_signature()
        )
      return self._logger

  def _init_thread_pool(self) -> None:
    """Init cache thread pools."""
    # Threads in this pool conduct high level background cache loading ops.
    self._orchestrator_thread_pool = futures.ThreadPoolExecutor(
        _MAX_ORCHESTRATOR_WORKER_THREADS
    )

  def __copy__(self) -> InMemoryDicomSlideCache:
    """Returns shallow copy of cache settings; does not cached data."""
    cache_copy = InMemoryDicomSlideCache(
        credential_factory=self._dicom_web_interface.credential_factory,
        max_cache_frame_memory_lru_cache_size_bytes=self._max_cache_frame_memory_lru_cache_size_bytes,
        number_of_frames_to_read=self._number_of_frames_to_read,
        max_instance_number_of_frames_to_prefer_whole_instance_download=self._max_instance_number_of_frames_to_prefer_whole_instance_download,
        optimization_hint=self._optimization_hint,
        logging_factory=self._logging_factory,
    )
    # maintain instance_cache_trace_uid across copy
    cache_copy._cache_instance_uid = self._cache_instance_uid
    return cache_copy

  def __deepcopy__(self, memo) -> InMemoryDicomSlideCache:
    return self.__copy__()

  def __getstate__(self):
    """Prepares class dictionary for pickeling.

    Make sure any operations loading GCS cache are completed before pickling.
    Deletes class state with thread pools, GCP clients, auth, and status.

    Returns:
      class dict
    """
    state = self.__dict__.copy()
    del state['_dicom_instance_frame_bytes']
    del state['_lock']
    del state['_initialization_lock']
    del state['_cache_stats']
    del state['_orchestrator_thread_pool']
    del state['_running_futures']
    del state['_logger']
    return state

  def __setstate__(self, dct):
    """Un-pickles class and re-initializes non-pickled properties."""
    self.__dict__ = dct
    self._dicom_instance_frame_bytes = self._init_dicom_instance_frame_bytes()
    self._cache_stats = local_dicom_slide_cache_types.CacheStats()
    self._running_futures = collections.defaultdict(dict)
    self._init_thread_pool()
    self._logger = None
    self._lock = threading.RLock()
    self._initialization_lock = threading.Lock()

  def _get_frame_bytes(
      self, instance_path: dicom_path.Path, frame_number: int
  ) -> Optional[bytes]:
    """Gets frame bytes for frame from cache.

    Args:
      instance_path: DICOMweb instance path.
      frame_number: Frame number to return data for.

    Returns:
      Frame bytes or None if not set
    """
    return self._dicom_instance_frame_bytes.get(
        _frame_number_key(instance_path, frame_number)
    )

  def _is_frame_number_loaded(
      self, instance_path: dicom_path.Path, frame_number: int
  ) -> bool:
    """Returns True if frame number is loaded."""
    return (
        _frame_number_key(instance_path, frame_number)
        in self._dicom_instance_frame_bytes
    )

  def _remove_finished_future(
      self, instance_path: dicom_path.Path, future: futures.Future[None]
  ) -> None:
    """Call back called by futures to remove self from loading future list."""
    with self._lock:
      instance_url = instance_path.complete_url
      instance_futures = self._running_futures.get(instance_url)
      if instance_futures is None:
        return
      try:
        del instance_futures[future]
      except KeyError:
        pass
      if not instance_futures:
        del self._running_futures[instance_url]

  def _handle_future(
      self,
      instance_path: dicom_path.Path,
      loading_frames: List[Tuple[int, int]],
      future: futures.Future[None],
  ) -> None:
    """Adds callback to future to remove future from monitor list."""
    # Add future to running list
    remove_future_partial = functools.partial(
        self._remove_finished_future, instance_path
    )
    self._running_futures[instance_path.complete_url][future] = loading_frames
    # Add call back to remove future from running list
    future.add_done_callback(remove_future_partial)

  def _is_frame_number_loading(
      self, instance_path: dicom_path.Path, frame_number: int
  ) -> bool:
    """Returns True if instance frame_number is being loaded in a thread."""
    for frame_range_list in self._running_futures[
        instance_path.complete_url
    ].values():
      for start_frame_number, end_frame_number in frame_range_list:
        if (
            start_frame_number <= frame_number
            and frame_number <= end_frame_number
        ):
          return True
    return False

  def _get_instance_future_loading_frame_ranges(
      self, instance_path: dicom_path.Path
  ) -> Iterator[List[Tuple[int, int]]]:
    """Returns iterator of lists of frame ranges being loaded for an instance."""
    return typing.cast(
        Iterator[List[Tuple[int, int]]],
        self._running_futures[instance_path.complete_url].values(),
    )

  def _clip_frame_range_to_loading_frames(
      self,
      instance_path: dicom_path.Path,
      frame_range: Optional[Tuple[int, int]],
  ) -> Optional[Tuple[int, int]]:
    """Clips starting and ending frame range based on loading frames.

    If cache miss occurs a range of frames is requested around the missed
    frame; by default ~500 frames will be retrieved. Rapid repeated cache
    misses can result in duplicate concurrent async requests for overlapping
    ranges of frames. This code clips a frame range to exclude frames at the
    start and end of the range which are being currently loaded in another
    thread. If a frame range is bound by frames which are not currently
    being loaded but were to contain frames within the range then the frames
    within and the range will that are being loaded will not be excluded. This
    limitation is not expected to occur in practice.

    Example: Range: (1, 6) is clipped Ranges (1, 1) and (6,6) are being loaded
      on another thread.  (1, 6) will be clipped to (2, 5).  However,
      if frame (1, 6) is clipped and range (2, 4) is being loaded on another
      thread then (1, 6) will be returned (frames 1 and 6 are not loading).

    Args:
      instance_path: DICOM web path to instance.
      frame_range: Inclusive frame number range to load.

    Returns:
      Clipped starting frame number and end frame number or None if entire frame
      range is already being loaded.
    """
    # Merge all loading frame range lists into single frame list sorted by
    # starting frame range position.
    if frame_range is None:
      return None
    start_frame_range, end_frame_range = frame_range
    for range_tpl in heapq.merge(
        *self._get_instance_future_loading_frame_ranges(instance_path),
        key=lambda x: x[0],
    ):
      start_frame_number, end_frame_number = range_tpl
      if (
          start_frame_number <= start_frame_range
          and start_frame_range <= end_frame_number
      ):
        start_frame_range = end_frame_number + 1
        if end_frame_range < start_frame_range:
          return None
        continue
      if start_frame_range < start_frame_number:
        break
    # Merge all loading frame range lists into single frame list sorted in
    # reverse by ending frame range position.
    range_end_frame_list = list(
        heapq.merge(
            *self._get_instance_future_loading_frame_ranges(instance_path),
            key=lambda x: x[1],
        )
    )
    range_end_frame_list.reverse()
    for range_tpl in range_end_frame_list:
      start_frame_number, end_frame_number = range_tpl
      if (
          start_frame_number <= end_frame_range
          and end_frame_range <= end_frame_number
      ):
        end_frame_range = start_frame_number - 1
        continue
      if end_frame_range > end_frame_number:
        break
    return (start_frame_range, end_frame_range)

  def block_until_frames_are_loaded(
      self,
      instance_path: Optional[dicom_path.Path] = None,
      timeout: Optional[float] = 600.0,
  ) -> float:
    """Blocks until all futures in future list, at time of call, are done.

    Args:
      instance_path: If defined blocks on futures associated with instance.
      timeout: Time(sec) to wait for cache loading blocks to finish; None=inf.

    Returns:
      Time waiting for cache loading to complete.
    """
    start_time = time.time()
    with self._lock:
      if not self._running_futures:
        return 0.0
      if instance_path is not None:
        instance_futures = self._running_futures.get(instance_path.complete_url)
      else:
        instance_futures = None
      if instance_path is not None and instance_futures is not None:
        # Block only on instance futures.
        future_list = list(instance_futures)
      elif instance_path is None:
        # Block for all running futures.
        future_list = []
        for instance_futures in self._running_futures.values():
          future_list.extend(instance_futures)
      else:
        # Instance path does not describe currently running future.
        future_list = []
    for future in _future_list_built_test_hook(future_list):
      future.result(timeout=timeout)
    elapsed_time = time.time() - start_time
    self._get_logger().debug(
        f'Blocked until frame loading completed ({elapsed_time}(sec)).',
        {_LogKeywords.EXECUTION_TIME_SEC: elapsed_time},
    )
    with self._lock:
      self._cache_stats.time_spent_blocked_waiting_for_cache_loading_to_complete += (
          elapsed_time
      )
    return elapsed_time

  def _update_frame_block_cache_stats_bytes_read(
      self, start_time: float, dicom_frames: List[bytes]
  ) -> None:
    self._cache_stats.number_of_frame_bytes_read_in_frame_blocks += (
        self._total_frame_bytes_read(dicom_frames)
    )
    self._cache_stats.frame_block_read_time += time.time() - start_time
    self._cache_stats.number_of_frame_blocks_read += 1
    self._cache_stats.number_of_frames_read_in_frame_blocks += len(dicom_frames)

  def _load_frame_number_ranges_thread(
      self,
      instance_path: dicom_path.Path,
      frame_number_range_list: List[Tuple[int, int]],
  ) -> None:
    """Loads list of DICOM instance frames numbers in batch into the cache.

    Args:
      instance_path: DICOM instance path.
      frame_number_range_list: List of Frame Number ranges to load.

    Returns:
      None
    """
    if not frame_number_range_list:
      return
    start_time = time.time()
    log_structure = {
        _LogKeywords.DICOM_WEB_INSTANCE_PATH: instance_path,
        _LogKeywords.DICOM_FRAME_NUMBER_RANGE_LIST: frame_number_range_list,
    }
    try:
      dicom_frames = self._dicom_web_interface.download_instance_frame_list_untranscoded(
          instance_path,
          itertools.chain(*[
              range(start_frame_number, end_frame_number + 1)
              for start_frame_number, end_frame_number in frame_number_range_list
          ]),
          retry=False,
      )
      with self._lock:
        offset = 0
        for start_frame_number, end_frame_number in frame_number_range_list:
          length = end_frame_number - start_frame_number + 1
          self._add_cached_instance_frames(
              instance_path,
              start_frame_number,
              dicom_frames[offset : offset + length],
          )
          offset += length
        self._update_frame_block_cache_stats_bytes_read(
            start_time, dicom_frames
        )
      self._get_logger().info(
          'Finished asyc loading DICOM frame number range(s) into cache.',
          log_structure,
          _log_elapsed_time(start_time),
      )
    except (
        ez_wsi_errors.HttpError,
        ez_wsi_errors.DownloadInstanceFrameError,
    ) as exp:
      self._get_logger().error(
          'Exception occurred caching DICOM instance frames.',
          exp,
          log_structure,
          _log_elapsed_time(start_time),
      )
      return
    except Exception as exp:
      self._get_logger().error(
          'Exception occurred caching DICOM instance frames.',
          exp,
          log_structure,
          _log_elapsed_time(start_time),
      )
      raise

  def _update_dicom_instance_cache_stats_bytes_read(
      self, start_time: float, dicom_frames: List[bytes]
  ) -> None:
    self._cache_stats.number_of_frame_bytes_read_in_dicom_instances += (
        self._total_frame_bytes_read(dicom_frames)
    )
    self._cache_stats.dicom_instance_read_time += time.time() - start_time
    self._cache_stats.number_of_dicom_instances_read += 1
    self._cache_stats.number_of_frames_read_in_dicom_instances += len(
        dicom_frames
    )

  def _cache_whole_instance_in_memory_thread(
      self,
      instance_path: dicom_path.Path,
      number_of_frames: int,
      running_as_thread: bool,
  ) -> None:
    """Loads frames from whole instance in thread.

    Args:
      instance_path: DICOM instance path to load.
      number_of_frames: number of frames in DICOM instance.
      running_as_thread: True if method running in thread (async).
    """
    start_time = time.time()
    log_structure = {
        _LogKeywords.DICOM_WEB_INSTANCE_PATH: instance_path,
        _LogKeywords.NUMBER_OF_FRAMES_IN_DICOM_INSTANCE: number_of_frames,
        _LogKeywords.RUNNING_AS_THREAD: running_as_thread,
    }
    try:
      with io.BytesIO() as buffer:
        try:
          self._dicom_web_interface.download_instance_untranscoded(
              instance_path, buffer, retry=False
          )
        except ez_wsi_errors.HttpError as exp:
          self._get_logger().error(
              'Could not download DICOM instance.',
              log_structure,
              _log_elapsed_time(start_time),
              exp,
          )
          return
        buffer.seek(0)
        dicom_frames = _load_frame_list(buffer)
      with self._lock:
        number_of_frames_downloaded = len(dicom_frames)
        if number_of_frames != number_of_frames_downloaded:
          self._get_logger().warning(
              'Expected number of frames does not match actual DICOM instance;'
              ' Number of frames in DICOM instance:'
              f' {number_of_frames_downloaded}; Expected number of frames:'
              f' {number_of_frames}.',
              log_structure,
          )
          number_of_frames = number_of_frames_downloaded
        if number_of_frames == 0:
          self._get_logger().warning(
              'Cached whole instance DICOM instance contains zero frames.',
              log_structure,
              _log_elapsed_time(start_time),
          )
        else:
          self._add_cached_instance_frames(instance_path, 1, dicom_frames)
        self._get_logger().info(
            'Cached whole instance DICOM instance.',
            log_structure,
            _log_elapsed_time(start_time),
        )
        self._update_dicom_instance_cache_stats_bytes_read(
            start_time, dicom_frames
        )
    except Exception as exp:
      self._get_logger().error(
          'Exception occurred caching whole DICOM instance.',
          exp,
          log_structure,
          _log_elapsed_time(start_time),
      )
      raise

  def _filter_loaded_or_loading_frame_numbers(
      self,
      instance_path: dicom_path.Path,
      frame_numbers: List[int],
  ) -> List[int]:
    """Returns frames numbers in list which are not loaded or loading."""
    filtered_frame_numbers = []
    for fnum in frame_numbers:
      if self._is_frame_number_loading(instance_path, fnum):
        continue
      if self._is_frame_number_loaded(instance_path, fnum):
        continue
      filtered_frame_numbers.append(fnum)
    return filtered_frame_numbers

  def preload_instance_frame_numbers(
      self,
      instance_frame_numbers: Mapping[str, List[int]],
      copy_from_cache: Optional[InMemoryDicomSlideCache] = None,
  ) -> None:
    """Preloads select instance frames from DICOM Store into cache.

    Args:
      instance_frame_numbers: Map instance path to frame numbers.
      copy_from_cache: Optional cache to copy frames from.

    Returns:
      None.
    """
    with self._lock:
      logger = self._get_logger()
      logger.info('Preloading DICOM instance frames')
      for instance_path, frame_numbers in instance_frame_numbers.items():
        instance_path = dicom_path.FromString(instance_path)
        if not frame_numbers:
          continue
        if copy_from_cache is not None:
          frame_numbers_not_copied = []
          for frame_number in frame_numbers:
            frame_bytes = copy_from_cache.get_cached_frame(
                instance_path, frame_number
            )
            if frame_bytes is not None:
              self._set_cached_frame(instance_path, frame_number, frame_bytes)
            else:
              frame_numbers_not_copied.append(frame_number)
          frame_numbers = frame_numbers_not_copied
        frame_number_range_list = _get_frame_number_range_list(
            self._filter_loaded_or_loading_frame_numbers(
                instance_path, frame_numbers
            ),
            logger,
        )
        if not frame_number_range_list:
          continue
        ex = typing.cast(
            futures.ThreadPoolExecutor, self._orchestrator_thread_pool
        )
        self._handle_future(
            instance_path,
            frame_number_range_list,
            ex.submit(
                self._load_frame_number_ranges_thread,
                instance_path,
                frame_number_range_list,
            ),
        )

  def _cache_whole_instance_in_memory(
      self,
      instance_path: dicom_path.Path,
      number_of_frames: int,
      blocking: bool,
  ) -> Optional[futures.Future[None]]:
    """Loads whole DICOM instance into memory.

    Args:
      instance_path: DICOMweb path to instance.
      number_of_frames: Number of frames in DICOM instance.
      blocking: Load cache as blocking operation.

    Returns:
      Future if async operation started or None.
    """
    if number_of_frames <= 0:
      return None
    # if entire instance is currently loading skip.
    if (
        self._clip_frame_range_to_loading_frames(
            instance_path, (1, number_of_frames)
        )
        is None
    ):
      return None
    if blocking:
      self._cache_whole_instance_in_memory_thread(
          instance_path, number_of_frames, running_as_thread=False
      )
      return None
    ex = typing.cast(futures.ThreadPoolExecutor, self._orchestrator_thread_pool)
    return ex.submit(
        self._cache_whole_instance_in_memory_thread,
        instance_path,
        number_of_frames,
        running_as_thread=True,
    )

  def cache_whole_instance_in_memory(
      self,
      instance_paths: local_dicom_slide_cache_types.InstancePathType,
      blocking: bool,
  ) -> None:
    """Caches whole DICOM instance in memory.

    Args:
      instance_paths: DICOMweb path to instance.
      blocking: Load cache as blocking operation.
    """
    for instance_path, number_of_frames in _get_instance_path_list(
        instance_paths
    ):
      future = self._cache_whole_instance_in_memory(
          instance_path, number_of_frames, blocking
      )
      if future is None:
        continue
      with self._lock:
        frame_range_to_load = (1, number_of_frames)
        self._handle_future(instance_path, [frame_range_to_load], future)

  def _get_frame_range_to_load(
      self,
      instance_path: dicom_path.Path,
      number_of_frames: int,
      frame_number: int,
      max_request: int,
  ) -> Optional[Tuple[int, int]]:
    """Get the range of frames to load in cached frame block.

    Always return frame_number in returned range.

    Args:
      instance_path: DICOMweb instance path.
      number_of_frames: Number of Frames in DICOM instance.
      frame_number: Frame number triggering caching.
      max_request: Maximum size of frame cache block.

    Returns:
      Tuple[Initialized starting frame number, ending frame number] or None if
      no valid frame range.
    """
    if (
        number_of_frames < 1
        or max_request < 1
        or frame_number > number_of_frames
        or frame_number < 1
    ):
      return None
    last_frame_number = min(number_of_frames, frame_number + max_request - 1)
    for last_frame_number in range(last_frame_number, frame_number - 1, -1):
      cache_key = _frame_number_key(instance_path, last_frame_number)
      if cache_key not in self._dicom_instance_frame_bytes:
        break
    # DICOM frame numbers start at 1
    first_frame_number = max(last_frame_number - max_request + 1, 1)
    for first_frame_number in range(first_frame_number, frame_number + 1):
      cache_key = _frame_number_key(instance_path, first_frame_number)
      if cache_key not in self._dicom_instance_frame_bytes:
        break
    return first_frame_number, last_frame_number

  def _start_load_frame_number_range(
      self,
      instance_path: dicom_path.Path,
      number_of_frames: int,
      frame_number: int,
      max_request: int,
  ) -> None:
    """Call to start background thread loading DICOM instance into cache.

    Args:
      instance_path: DICOM web instance path.
      number_of_frames: Total number of frames in the instance.
      frame_number: Missing instance frame number which triggered load instance.
      max_request: Maximum number of frames to load.
    """
    if number_of_frames <= 0:
      # if no frames do nothing.
      return
    if (
        number_of_frames
        < self._max_instance_number_of_frames_to_prefer_whole_instance_download
    ):
      future = self._cache_whole_instance_in_memory(
          instance_path, number_of_frames, blocking=False
      )
      if future is None:
        return
      frame_range_to_load = (1, number_of_frames)
      self._handle_future(instance_path, [frame_range_to_load], future)
      return
    frame_range_to_load = self._get_frame_range_to_load(
        instance_path,
        number_of_frames,
        frame_number,
        max_request,
    )
    frame_range_to_load = self._clip_frame_range_to_loading_frames(
        instance_path, frame_range_to_load
    )
    if frame_range_to_load is None:
      return
    ex = typing.cast(futures.ThreadPoolExecutor, self._orchestrator_thread_pool)
    self._handle_future(
        instance_path,
        [frame_range_to_load],
        ex.submit(
            self._load_frame_number_ranges_thread,
            instance_path,
            [frame_range_to_load],
        ),
    )

  def _get_optmization_hint(
      self,
      number_of_frames: int,
  ) -> local_dicom_slide_cache_types.CacheConfigOptimizationHint:
    """Initializes  hint based on class settings and number of frames."""
    if (
        number_of_frames == 1
        and self._optimization_hint
        == local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
    ):
      # Performance optimization.  If number of frames in DICOM instance is 1
      # then automatically wait for frame data to return.
      return (
          local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
      )
    return self._optimization_hint

  def _handle_cache_miss_minimize_dicom_store_qpm_optimization(
      self,
      instance_path: dicom_path.Path,
      frame_number: int,
      log_struct: Mapping[str, Any],
  ) -> Optional[bytes]:
    """Returns requested frame after waiting for futures to finish.

    Reduces store queries per mininute (QPM) by utilizing batch mechanism for
    filling cache and waiting for cache to fill before returning.

    Args:
      instance_path: DICOM web instance path.
      frame_number: Frame number requested.
      log_struct: Additional items to include in structured logs.

    Returns:
      Frame bytes or None if futures failed to load frame bytes.
    """
    self.block_until_frames_are_loaded(instance_path)
    self._get_logger().debug(
        'Reducing DICOM store queries by waiting for background thread frame'
        ' requests to complete.',
        log_struct,
    )
    with self._lock:
      return self._get_frame_bytes(instance_path, frame_number)

  def _handle_cache_miss_minimize_latency_optimization(
      self,
      instance_path: dicom_path.Path,
      frame_number: int,
      log_struct: Mapping[str, Any],
  ) -> Optional[bytes]:
    """Retrieve and return requested frame indivually to reduce latency.

    Single frame requests from the store increase store queries per minute and
    are slow proportionally (base on bytes transferred / time). However these
    requests are still fast. Request the frame data and return to unblock other
    operations. Cache will likely be filled on subsequent read. The method
    log_cache_stats can be used to log and inspect cache behavior and perf.

    Args:
      instance_path: DICOM web instance path.
      frame_number: Frame number requested.
      log_struct: Additional items to include in structured logs.

    Returns:
      Frame bytes or None if futures failed to load frame bytes.
    """

    self._get_logger().debug(
        'Reducing latency by downloading DICOM frame immediately from the'
        ' DICOM store.',
        log_struct,
    )
    start_time = time.time()
    try:
      dcm_frames = (
          self._dicom_web_interface.download_instance_frames_untranscoded(
              instance_path, frame_number, frame_number, retry=False
          )
      )
    except (
        ez_wsi_errors.HttpError,
        ez_wsi_errors.DownloadInstanceFrameError,
    ) as exp:
      self._get_logger().error(
          'Exception occurred caching DICOM instance frames.',
          exp,
          log_struct,
          _log_elapsed_time(start_time),
      )
      return None
    if len(dcm_frames) != 1:
      return None
    with self._lock:
      # Add retrieved frame to cache, to avoid re-retrieval if frame
      # requested again before background caching operation completes.
      self._add_cached_instance_frames(instance_path, frame_number, dcm_frames)
      self._cache_stats.number_of_frames_downloaded_to_reduce_latency += 1
      self._cache_stats.time_spent_downloading_frames_to_reduce_latency += (
          time.time() - start_time
      )
    return dcm_frames[0]

  def get_cached_frame(
      self,
      instance_path: dicom_path.Path,
      frame_number: int,
  ) -> Optional[bytes]:
    """Returns instance frame bytes from cache or None if not found.

    Args:
      instance_path: DICOM web instance path.
      frame_number: Frame number with in DICOM instance, First frame = 1

    Returns:
      Frames bytes or None
    """
    if frame_number < 1:
      return None
    with self._lock:
      return self._get_frame_bytes(instance_path, frame_number)

  def get_frame(
      self,
      instance_path: dicom_path.Path,
      number_of_frames: int,
      frame_number: int,
      optimization_hint: Optional[
          local_dicom_slide_cache_types.CacheConfigOptimizationHint
      ] = None,
  ) -> Optional[bytes]:
    """Returns frame bytes from DICOM instance or None if not found.

    If frame bytes not found and instance is in GCS cache, bytes for a frame
    block that includes the requested frame will be loaded on a background
    thread.

    Args:
      instance_path: DICOM web instance path.
      number_of_frames: Number of frames in DICOM instance.
      frame_number: Frame number with in DICOM instance, First frame = 1
      optimization_hint: Optimization to control performance optimizations that
        occure on a cache miss.

    Returns:
      Frames bytes or None
    """
    if frame_number < 1 or frame_number > number_of_frames:
      return None
    if optimization_hint is None:
      optimization_hint = self._get_optmization_hint(number_of_frames)
    with self._lock:
      if self._cache_stats.first_frame_number_read is None:
        self._cache_stats.first_frame_number_read = frame_number
        self._cache_stats.last_frame_number_read = frame_number
      else:
        self._cache_stats.first_frame_number_read = min(
            frame_number, self._cache_stats.first_frame_number_read
        )
        self._cache_stats.last_frame_number_read = max(
            frame_number, self._cache_stats.last_frame_number_read
        )

      frame_bytes = self._get_frame_bytes(instance_path, frame_number)
      if frame_bytes is not None:
        self._cache_stats.frame_cache_hit_count += 1
        return frame_bytes
      log_struct = {
          _LogKeywords.DICOM_WEB_INSTANCE_PATH: instance_path,
          _LogKeywords.FRAME_NUMBER: frame_number,
      }
      if not self._is_frame_number_loading(instance_path, frame_number):
        self._start_load_frame_number_range(
            instance_path,
            number_of_frames,
            frame_number,
            self._number_of_frames_to_read,
        )
        self._get_logger().debug(
            f'Cache Miss; Frame: {frame_number}; Starting frame loading',
            log_struct,
        )
      else:
        self._get_logger().debug(
            f'Cache Miss; Frame: {frame_number}; Frame loading in progress',
            log_struct,
        )
      self._cache_stats.frame_cache_miss_count += 1

    if (
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
        == optimization_hint
    ):
      return self._handle_cache_miss_minimize_dicom_store_qpm_optimization(
          instance_path, frame_number, log_struct
      )
    elif (
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
        == optimization_hint
    ):
      return self._handle_cache_miss_minimize_latency_optimization(
          instance_path, frame_number, log_struct
      )
    return None

  @property
  def cache_stats(self) -> local_dicom_slide_cache_types.CacheStats:
    """Returns cache stats metrics dataclass."""
    with self._lock:
      cache_stats = copy.copy(self._cache_stats)
    if not self.lru_caching_enabled:
      cache_stats.frame_cache_memory_size_limit = None
    else:
      cache_tools_lru = typing.cast(
          cachetools.LRUCache, self._dicom_instance_frame_bytes
      )
      cache_stats.frame_cache_memory_size_limit = cache_tools_lru.maxsize
      cache_stats.current_frame_Cache_memory_size = cache_tools_lru.currsize
    cache_stats.system_memory = psutil.virtual_memory()
    return cache_stats

  def reset_cache_stats(self) -> None:
    """Resets cache status metrics dataclass."""
    with self._lock:
      self._cache_stats = local_dicom_slide_cache_types.CacheStats()
      with self._initialization_lock:
        # Force logger to re-initialize to update logger signatures
        self._logger = None
