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
"""Tests for local dicom slide cache."""

from concurrent import futures
import copy
import functools
import io
import logging
import re
import threading
import time
from typing import List, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import pydicom

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


TEST_PATH = dicom_path.FromString(
    f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3'
)
BAD_TEST_PATH = dicom_path.FromString(
    f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.4'
)


def _test_inference_cache(
    *args,
    **kwargs,
) -> local_dicom_slide_cache.InMemoryDicomSlideCache:
  return local_dicom_slide_cache.InMemoryDicomSlideCache(
      credential_factory_module.CredentialFactory(), *args, **kwargs
  )


def _future_list_built_test_hook_signal(
    signal: threading.Event, future_list: List[futures.Future[None]]
) -> List[futures.Future[None]]:
  signal.set()
  return future_list


def _get_mock_ez_wsi_dicom_instance(
    path: dicom_path.Path, frame_count: int
) -> slide_level_map.Instance:
  mock_instance = mock.create_autospec(slide_level_map.Instance)
  mock_instance.dicom_object = mock.create_autospec(
      dicom_web_interface.DicomObject
  )
  mock_instance.dicom_object.path = path
  mock_instance.frame_count = frame_count
  return mock_instance


def _future_thread(running_signal: threading.Event):
  running_signal.wait()


def _block_on_running_futures(
    cache: local_dicom_slide_cache.InMemoryDicomSlideCache,
    instance_path: dicom_path.Path,
) -> float:
  return cache.block_until_frames_are_loaded(instance_path)


class LocalDicomSlideCacheTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dicom_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    self.test_dicom_instance_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/'
        f'studies/{self.test_dicom_instance.StudyInstanceUID}/'
        f'series/{self.test_dicom_instance.SeriesInstanceUID}/'
        f'instances/{self.test_dicom_instance.SOPInstanceUID}'
    )
    self.dicom_store_path = f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'

  @parameterized.named_parameters([
      dict(
          testcase_name='jpeg_baseline_process_1',
          transfer_syntax='1.2.840.10008.1.2.4.50',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg2000_lossless',
          transfer_syntax='1.2.840.10008.1.2.4.90',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg2000',
          transfer_syntax='1.2.840.10008.1.2.4.91',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='explicit_vr_little_endian',
          transfer_syntax='1.2.840.10008.1.2.1',
          is_unencapsulated=True,
      ),
      dict(
          testcase_name='implicit_vr_endian_default_transfer_syntax_for_dicom',
          transfer_syntax='1.2.840.10008.1.2',
          is_unencapsulated=True,
      ),
      dict(
          testcase_name='undefined_transfer_syntax',
          transfer_syntax='1.2.3',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='deflated_explicit_vr_little_endian',
          transfer_syntax='1.2.840.10008.1.2.1.99',
          is_unencapsulated=True,
      ),
      dict(
          testcase_name='explicit_vr_big_endian',
          transfer_syntax='1.2.840.10008.1.2.2',
          is_unencapsulated=True,
      ),
      dict(
          testcase_name='jpeg_baseline_process_2_4',
          transfer_syntax='1.2.840.10008.1.2.4.51',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg_lossless_process_14',
          transfer_syntax='1.2.840.10008.1.2.4.70',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg_ls',
          transfer_syntax='1.2.840.10008.1.2.4.80',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg_ls_lossy',
          transfer_syntax='1.2.840.10008.1.2.4.81',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg2000_multicomponent_lossless',
          transfer_syntax='1.2.840.10008.1.2.4.92',
          is_unencapsulated=False,
      ),
      dict(
          testcase_name='jpeg2000_multicomponent',
          transfer_syntax='1.2.840.10008.1.2.4.93',
          is_unencapsulated=False,
      ),
  ])
  def test_is_unencapsulated_image_transfer_syntax(
      self, transfer_syntax, is_unencapsulated
  ):
    self.assertEqual(
        local_dicom_slide_cache._is_unencapsulated_image_transfer_syntax(
            transfer_syntax
        ),
        is_unencapsulated,
    )

  def test_load_frame_list(self):
    with open(
        dicom_test_utils.test_multi_frame_dicom_instance_path(), 'rb'
    ) as dcm_file:
      list_of_frames = local_dicom_slide_cache._load_frame_list(dcm_file)
    self.assertLen(list_of_frames, self.test_dicom_instance.NumberOfFrames)
    self.assertEqual(
        [len(frame_bytes) for frame_bytes in list_of_frames],
        [
            8418,
            9094,
            7348,
            7710,
            5172,
            11654,
            13040,
            9712,
            14188,
            3492,
            5796,
            4220,
            5184,
            5380,
            2558,
        ],
    )

  @parameterized.parameters(['PixelData', 'NumberOfFrames'])
  def test_load_frame_returns_none_missing_dicom_tag(self, dicom_tag: str):
    with io.BytesIO() as bytes_buffer:
      with pydicom.dcmread(
          dicom_test_utils.test_multi_frame_dicom_instance_path()
      ) as dcm_file:
        del dcm_file[dicom_tag]
        dcm_file.save_as(bytes_buffer)
      bytes_buffer.seek(0)
      frame_byte_list = local_dicom_slide_cache._load_frame_list(bytes_buffer)
    self.assertEmpty(frame_byte_list)

  def test_load_frame_returns_none_dicom_frame_number_zero(self):
    with io.BytesIO() as bytes_buffer:
      with pydicom.dcmread(
          dicom_test_utils.test_multi_frame_dicom_instance_path()
      ) as dcm_file:
        dcm_file.NumberOfFrames = 0
        dcm_file.save_as(bytes_buffer)
      bytes_buffer.seek(0)
      frame_byte_list = local_dicom_slide_cache._load_frame_list(bytes_buffer)
    self.assertEmpty(frame_byte_list)

  @parameterized.named_parameters([
      dict(testcase_name='empty_list', input_list=[], expected=[]),
      dict(
          testcase_name='single_list',
          input_list=list(range(1, 9)),
          expected=[(1, 8)],
      ),
      dict(
          testcase_name='single_list_repeated_values',
          input_list=[1, 2, 3, 3, 3, 4],
          expected=[(1, 4)],
      ),
      dict(
          testcase_name='list_with_gaps',
          input_list=[1, 4, 9],
          expected=[(1, 1), (4, 4), (9, 9)],
      ),
      dict(
          testcase_name='list_with_runs_and_gaps',
          input_list=[1, 2, 4, 5, 6, 9],
          expected=[(1, 2), (4, 6), (9, 9)],
      ),
      dict(
          testcase_name='single_value',
          input_list=[1],
          expected=[(1, 1)],
      ),
      dict(
          testcase_name='single_value_repeated',
          input_list=[1, 1, 1, 1, 1, 1],
          expected=[(1, 1)],
      ),
  ])
  def test_get_frame_number_range_list_succeeds(self, input_list, expected):
    with mock.patch.object(
        ez_wsi_logging_factory._BasePythonLogger,
        'warning',
        autospec=True,
    ) as mock_log:
      self.assertEqual(
          local_dicom_slide_cache._get_frame_number_range_list(
              input_list,
              ez_wsi_logging_factory._BasePythonLogger(logging.getLogger()),
          ),
          expected,
      )
      self.assertEqual(mock_log.call_count, 0)

  @mock.patch.object(
      ez_wsi_logging_factory._BasePythonLogger,
      'warning',
      autospec=True,
  )
  def test_get_frame_number_range_list_if_not_sorted(self, mock_log):
    self.assertEqual(
        local_dicom_slide_cache._get_frame_number_range_list(
            [1, 1, 2, 3, 2, 1],
            ez_wsi_logging_factory._BasePythonLogger(logging.getLogger()),
        ),
        [(1, 3)],
    )
    self.assertEqual(mock_log.call_count, 1)

  @parameterized.parameters([-1, 0])
  def test_get_frame_number_range_list_throws_if_passed_values_less_than_1(
      self, invalid_frame_number
  ):
    with self.assertRaises(
        local_dicom_slide_cache_types.InvalidFrameNumberError
    ):
      local_dicom_slide_cache._get_frame_number_range_list(
          [invalid_frame_number],
          ez_wsi_logging_factory._BasePythonLogger(logging.getLogger()),
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='empty_list', input_list=[], preloaded=[], expected=[]
      ),
      dict(
          testcase_name='no_preloaded',
          input_list=list(range(1, 9)),
          preloaded=[],
          expected=list(range(1, 9)),
      ),
      dict(
          testcase_name='no_input_with_preloaded_values',
          input_list=[],
          preloaded=list(range(10)),
          expected=[],
      ),
      dict(
          testcase_name='all_values_filtered_input',
          input_list=list(range(1, 9)),
          preloaded=list(range(1, 9)),
          expected=[],
      ),
      dict(
          testcase_name='first_half_filtered',
          input_list=list(range(1, 9)),
          preloaded=list(range(1, 5)),
          expected=list(range(5, 9)),
      ),
      dict(
          testcase_name='second_half_filtered',
          input_list=list(range(1, 9)),
          preloaded=list(range(5, 9)),
          expected=list(range(1, 5)),
      ),
      dict(
          testcase_name='ever_other_odd_filtered',
          input_list=list(range(1, 9)),
          preloaded=list(range(1, 9, 2)),
          expected=list(range(2, 9, 2)),
      ),
      dict(
          testcase_name='ever_other_even_filtered',
          input_list=list(range(1, 9)),
          preloaded=list(range(2, 9, 2)),
          expected=list(range(1, 9, 2)),
      ),
  ])
  def test_filter_loaded_or_loading_numbers(
      self, input_list, preloaded, expected
  ):
    cache = _test_inference_cache()
    for preloaded_framenumber in preloaded:
      cache._dicom_instance_frame_bytes[
          local_dicom_slide_cache._frame_number_key(
              self.test_dicom_instance_path, preloaded_framenumber
          )
      ] = b'1'
    self.assertEqual(
        cache._filter_loaded_or_loading_frame_numbers(
            self.test_dicom_instance_path, input_list
        ),
        expected,
    )

  def test_load_whole_instance_to_memory(self):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache()
      cache.cache_whole_instance_in_memory(
          _get_mock_ez_wsi_dicom_instance(
              self.test_dicom_instance_path,
              self.test_dicom_instance.NumberOfFrames,
          ),
          blocking=True,
      )
      for frame_number in range(
          1, int(self.test_dicom_instance.NumberOfFrames + 1)
      ):
        frame_bytes = cache.get_frame(
            self.test_dicom_instance_path,
            self.test_dicom_instance.NumberOfFrames,
            frame_number,
        )
        self.assertEqual(
            frame_bytes,
            dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
        )

  def test_load_cache_from_dicom_store(self):
    optimization_hint = (
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE
    )
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          number_of_frames_to_read=1,
          max_instance_number_of_frames_to_prefer_whole_instance_download=0,
      )

      for frame_number in range(
          1, int(self.test_dicom_instance.NumberOfFrames + 1)
      ):
        self.assertIsNone(
            cache.get_frame(
                self.test_dicom_instance_path,
                self.test_dicom_instance.NumberOfFrames,
                frame_number,
                optimization_hint=optimization_hint,
            )
        )
      # Wait for frame to finish loading.
      cache.block_until_frames_are_loaded(self.test_dicom_instance_path)
      for frame_number in range(
          1, int(self.test_dicom_instance.NumberOfFrames + 1)
      ):
        frame_bytes = cache.get_frame(
            self.test_dicom_instance_path,
            self.test_dicom_instance.NumberOfFrames,
            frame_number,
            optimization_hint=optimization_hint,
        )
        self.assertEqual(
            frame_bytes,
            dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
        )
      status = cache.cache_stats
      self.assertEqual(
          status.frame_cache_miss_count, self.test_dicom_instance.NumberOfFrames
      )
      self.assertEqual(
          status.frame_cache_hit_count, self.test_dicom_instance.NumberOfFrames
      )

  def test_preload_cache_from_dicom_store(self):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(number_of_frames_to_read=1)
      cache.preload_instance_frame_numbers(
          {self.test_dicom_instance_path.complete_url: list(range(1, 16))}
      )
      cache.block_until_frames_are_loaded(self.test_dicom_instance_path)
      for frame_number in range(
          1, int(self.test_dicom_instance.NumberOfFrames + 1)
      ):
        frame_bytes = cache.get_frame(
            self.test_dicom_instance_path,
            self.test_dicom_instance.NumberOfFrames,
            frame_number,
            optimization_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE,
        )
        self.assertEqual(
            frame_bytes,
            dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
        )
      status = cache.cache_stats
      self.assertEqual(status.frame_cache_miss_count, 0)
      self.assertEqual(
          status.frame_cache_hit_count, self.test_dicom_instance.NumberOfFrames
      )

  def test_get_state_success(self):
    cache = _test_inference_cache(number_of_frames_to_read=1)
    state = cache.__getstate__()
    for item in (
        '_lock',
        '_cache_stats',
        '_orchestrator_thread_pool',
        '_dicom_instance_frame_bytes',
        '_running_futures',
    ):
      self.assertNotIn(item, state)

  def test_get_state_set_state_(self):
    cache = _test_inference_cache(number_of_frames_to_read=1)
    state = cache.__getstate__()
    cache.__setstate__(state)
    self.assertEmpty(cache._dicom_instance_frame_bytes)
    self.assertEmpty(cache._running_futures)
    self.assertIsNotNone(cache._lock)
    self.assertIsNotNone(cache._cache_stats)
    self.assertIsNotNone(cache._orchestrator_thread_pool)

  def test_copy(self):
    cache = _test_inference_cache(number_of_frames_to_read=1)
    cache2 = copy.copy(cache)
    self.assertEqual(cache._cache_instance_uid, cache2._cache_instance_uid)
    self.assertEqual(
        cache._dicom_web_interface.credential_factory,
        cache2._dicom_web_interface.credential_factory,
    )
    self.assertEqual(
        cache._number_of_frames_to_read, cache2._number_of_frames_to_read
    )

  def test_deepcopy(self):
    cache = _test_inference_cache(number_of_frames_to_read=1)
    cache2 = copy.deepcopy(cache)
    self.assertEqual(cache._cache_instance_uid, cache2._cache_instance_uid)
    self.assertEqual(
        cache._dicom_web_interface.credential_factory,
        cache2._dicom_web_interface.credential_factory,
    )
    self.assertEqual(
        cache._number_of_frames_to_read, cache2._number_of_frames_to_read
    )

  def _setup_test_slide_cache_with_logger(
      self, **kwargs
  ) -> local_dicom_slide_cache.InMemoryDicomSlideCache:
    return _test_inference_cache(
        logging_factory=ez_wsi_logging_factory.BasePythonLoggerFactory(None),
        **kwargs,
    )

  @mock.patch.object(logging.Logger, 'info', autospec=True)
  def test_get_logger(self, mock_logger):
    slide_cache = self._setup_test_slide_cache_with_logger()
    slide_cache._get_logger().info('Test_Message')
    self.assertRegex(
        mock_logger.call_args.args[1],
        r'Test_Message; instance_cache_lifetime_trace_uid: .*;'
        r' instance_cache_worker_trace_uid: .*',
    )

  def _log_and_match_regex(
      self,
      slide_cache: local_dicom_slide_cache.InMemoryDicomSlideCache,
      log_msg: str,
      uid_key: str,
  ) -> str:
    with mock.patch.object(logging.Logger, 'info', autospec=True) as mk_logger:
      slide_cache._get_logger().info(log_msg)
      match = re.fullmatch(
          f'{log_msg}.*instance_cache_lifetime_trace_uid: (.*?);'
          ' instance_cache_worker_trace_uid:(.*?)',
          mk_logger.call_args.args[1],
      )
    results = match.groups()  # pytype: disable=attribute-error
    return {
        'instance_cache_lifetime_trace_uid': results[0],
        'instance_cache_worker_trace_uid': results[1],
    }[uid_key]

  @parameterized.named_parameters([
      dict(
          testcase_name='lifetime_uid_static',
          uid_key='instance_cache_lifetime_trace_uid',
          expected=True,
      ),
      dict(
          testcase_name='worker_uid_changes',
          uid_key='instance_cache_worker_trace_uid',
          expected=False,
      ),
  ])
  def test_get_logger_trace_uids_across_pickle(self, uid_key, expected):
    slide_cache = self._setup_test_slide_cache_with_logger()
    pre_pickle_match = self._log_and_match_regex(slide_cache, 'Test_1', uid_key)
    state = slide_cache.__getstate__()
    slide_cache.__setstate__(state)
    post_pickle_match = self._log_and_match_regex(
        slide_cache, 'Test_2', uid_key
    )
    self.assertEqual(pre_pickle_match == post_pickle_match, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='lifetime_uid_static',
          uid_key='instance_cache_lifetime_trace_uid',
          expected=True,
      ),
      dict(
          testcase_name='worker_uid_changes',
          uid_key='instance_cache_worker_trace_uid',
          expected=False,
      ),
  ])
  def test_get_logger_trace_uids_across_reset_cache_stats(
      self, uid_key, expected
  ):
    slide_cache = self._setup_test_slide_cache_with_logger()
    pre_reset_state_match = self._log_and_match_regex(
        slide_cache, 'Test_1', uid_key
    )
    slide_cache.reset_cache_stats()
    post_reset_state_match = self._log_and_match_regex(
        slide_cache, 'Test_2', uid_key
    )
    self.assertEqual(
        pre_reset_state_match == post_reset_state_match,
        expected,
    )

  @mock.patch('time.time', autospec=True, return_value=10.0)
  def test_log_elapsed_time(self, unused_mock_time):
    self.assertEqual(
        local_dicom_slide_cache._log_elapsed_time(5.5),
        {'execution_time_sec': 4.5},
    )

  def test_load_frame_number_ranges_thread_success(self):
    dicom_frame_number_ranges_to_read = [(1, 5), (7, 10)]
    cache = _test_inference_cache()
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache._load_frame_number_ranges_thread(
          self.test_dicom_instance_path, dicom_frame_number_ranges_to_read
      )
    # test stats take about frame read operation match expectations.
    self.assertEqual(cache.cache_stats.number_of_frame_blocks_read, 1)
    self.assertEqual(cache.cache_stats.number_of_frames_read_in_frame_blocks, 9)
    self.assertEqual(
        cache.cache_stats.number_of_frame_bytes_read_in_frame_blocks,
        78174,
    )
    self.assertGreater(cache.cache_stats.frame_block_read_time, 0)
    # Test returned frame bytes match expectation
    for (
        start_frame_number,
        end_frame_number,
    ) in dicom_frame_number_ranges_to_read:
      for frame_number in range(start_frame_number, end_frame_number + 1):
        frame_bytes = cache._get_frame_bytes(
            self.test_dicom_instance_path, frame_number
        )
        self.assertEqual(
            frame_bytes,
            dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
        )

  def test_load_frame_number_ranges_thread_empty_list_nop(
      self,
  ):
    dicom_frame_number_ranges_to_read = []
    cache = _test_inference_cache()
    cache._load_frame_number_ranges_thread(
        self.test_dicom_instance_path, dicom_frame_number_ranges_to_read
    )
    # test stats take about frame read operation match expectations.
    self.assertEqual(cache.cache_stats.number_of_frame_blocks_read, 0)
    self.assertEqual(cache.cache_stats.number_of_frames_read_in_frame_blocks, 0)
    self.assertEqual(
        cache.cache_stats.number_of_frame_bytes_read_in_frame_blocks,
        0,
    )
    self.assertEqual(cache.cache_stats.frame_block_read_time, 0)

  def test_cache_whole_instance_in_memory_thread_success(self):
    cache = _test_inference_cache()
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache._cache_whole_instance_in_memory_thread(
          self.test_dicom_instance_path,
          self.test_dicom_instance.NumberOfFrames,
          running_as_thread=True,
      )
    # test stats take about frame read operation match expectations.
    self.assertEqual(cache.cache_stats.number_of_dicom_instances_read, 1)
    self.assertEqual(
        cache.cache_stats.number_of_frames_read_in_dicom_instances,
        self.test_dicom_instance.NumberOfFrames,
    )
    self.assertEqual(
        cache.cache_stats.number_of_frame_bytes_read_in_dicom_instances,
        112966,
    )
    self.assertGreater(cache.cache_stats.dicom_instance_read_time, 0)
    # Test returned frame bytes match expectation
    for frame_number in range(1, self.test_dicom_instance.NumberOfFrames + 1):
      frame_bytes = cache._get_frame_bytes(
          self.test_dicom_instance_path, frame_number
      )
      self.assertEqual(
          frame_bytes,
          dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
      )

  def test_future_completes_after_added_to_mon_list_removes_self(self):
    frame_range_list = [(1, 10)]
    cache = _test_inference_cache()
    with futures.ThreadPoolExecutor(1) as thread_pool:
      future_running_signal = threading.Event()
      future = thread_pool.submit(_future_thread, future_running_signal)
      # Add add running future to monitor list
      cache._handle_future(
          self.test_dicom_instance_path, frame_range_list, future
      )
      # Check future is in list.
      self.assertLen(cache._running_futures, 1)
      future_running_signal.set()
      future.result()
      # Test that future removes self from monitor list after finishing.
      start_time = time.time()
      while cache._running_futures and time.time() - start_time < 60.0:
        time.sleep(0.25)
      self.assertEmpty(cache._running_futures)

  def test_block_until_frames_are_loaded_nothing_running(self):
    cache = _test_inference_cache()
    self.assertEmpty(cache._running_futures)
    self.assertEqual(cache.block_until_frames_are_loaded(), 0.0)
    self.assertEqual(
        cache._cache_stats.time_spent_blocked_waiting_for_cache_loading_to_complete,
        0.0,
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='block_for_first_instance',
          instances_futures_to_wait_for=dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1'
          ),
          expected_unfinished_count=1,
      ),
      dict(
          testcase_name='block_for_second_instance',
          instances_futures_to_wait_for=dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/2'
          ),
          expected_unfinished_count=1,
      ),
      dict(
          testcase_name='block_for_all_instances',
          instances_futures_to_wait_for=None,
          expected_unfinished_count=0,
      ),
      dict(
          testcase_name='block_for_finished_instance',
          instances_futures_to_wait_for=BAD_TEST_PATH,
          expected_unfinished_count=2,
      ),
  ])
  def test_block_until_all_instance_frame_futures_are_loaded(
      self, instances_futures_to_wait_for, expected_unfinished_count
  ):
    cache = _test_inference_cache()
    hook_event = threading.Event()
    with futures.ThreadPoolExecutor(3) as thread_pool:
      with mock.patch.object(
          local_dicom_slide_cache,
          '_future_list_built_test_hook',
          autospec=True,
          side_effect=functools.partial(
              _future_list_built_test_hook_signal, hook_event
          ),
      ):
        start_time = time.time()
        started_futures = []
        for idx in range(1, 3):
          future_event = threading.Event()
          future = thread_pool.submit(_future_thread, future_event)
          future_path = dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{idx}'
          )
          cache._handle_future(future_path, [(1, 10)], future)
          started_futures.append((future, future_event, future_path))
        # approx time started waiting for future to finish
        block_on_future = thread_pool.submit(
            _block_on_running_futures, cache, instances_futures_to_wait_for
        )
        hook_event.wait()
        for future, future_event, instance_path in started_futures:
          if (
              instances_futures_to_wait_for
              and instance_path != instances_futures_to_wait_for
          ):
            continue
          future_event.set()
          future.result()
        # wait for block waiting of future to complete
        blocked_time = block_on_future.result()
        elapsed = time.time() - start_time
        for future, _, instance_path in started_futures:
          if (
              instances_futures_to_wait_for
              and instance_path != instances_futures_to_wait_for
          ):
            continue
          self.assertTrue(future.done())
        self.assertLess(0.0, blocked_time)
        self.assertLess(blocked_time, elapsed)
        self.assertEqual(
            cache._cache_stats.time_spent_blocked_waiting_for_cache_loading_to_complete,
            blocked_time,
        )
        unfinished_future_count = 0
        for future_tuple in started_futures:
          future, future_event, _ = future_tuple
          if not future.done():
            future_event.set()
            unfinished_future_count += 1
        self.assertEqual(unfinished_future_count, expected_unfinished_count)

  @parameterized.named_parameters([
      dict(
          testcase_name='remove_first_future',
          remove_futures=[(TEST_PATH, 'mock_future_1')],
          expected_running_futures_len=1,
          expected_instance_running_future_len=1,
      ),
      dict(
          testcase_name='remove_second_future',
          remove_futures=[(TEST_PATH, 'mock_future_2')],
          expected_running_futures_len=1,
          expected_instance_running_future_len=1,
      ),
      dict(
          testcase_name='remove_both_empty_instance_futures',
          remove_futures=[
              (TEST_PATH, 'mock_future_2'),
              (TEST_PATH, 'mock_future_1'),
          ],
          expected_running_futures_len=0,
          expected_instance_running_future_len=0,
      ),
      dict(
          testcase_name='no_op_instance_path_not_found',
          remove_futures=[(BAD_TEST_PATH, 'mock_future_2')],
          expected_running_futures_len=1,
          expected_instance_running_future_len=2,
      ),
      dict(
          testcase_name='no_op_future_not_found',
          remove_futures=[(TEST_PATH, 'bad_future')],
          expected_running_futures_len=1,
          expected_instance_running_future_len=2,
      ),
  ])
  def test_remove_finished_future(
      self,
      remove_futures,
      expected_running_futures_len,
      expected_instance_running_future_len,
  ):
    dicom_web_path = TEST_PATH
    cache = _test_inference_cache()
    cache._running_futures[dicom_web_path.complete_url] = {
        'mock_future_1': [(2, 5)],
        'mock_future_2': [(10, 12)],
    }
    for dicom_instance_path, future in remove_futures:
      cache._remove_finished_future(dicom_instance_path, future)
    self.assertLen(cache._running_futures, expected_running_futures_len)
    self.assertLen(
        cache._running_futures[dicom_web_path.complete_url],
        expected_instance_running_future_len,
    )

  @parameterized.parameters([2, 3, 4, 5, 10, 11, 12])
  def test_is_frame_number_loading_true(self, frame_number):
    cache = _test_inference_cache()
    cache._running_futures[self.test_dicom_instance_path.complete_url] = {
        'mock_future_1': [(2, 5)],
        'mock_future_2': [(10, 12)],
    }
    self.assertTrue(
        cache._is_frame_number_loading(
            self.test_dicom_instance_path, frame_number
        )
    )

  @parameterized.parameters([1, 6, 9, 13, 15])
  def test_is_frame_number_loading_false(self, frame_number):
    cache = _test_inference_cache()
    cache._running_futures[self.test_dicom_instance_path.complete_url] = {
        'mock_future_1': [(2, 5)],
        'mock_future_2': [(10, 12)],
    }
    self.assertFalse(
        cache._is_frame_number_loading(
            self.test_dicom_instance_path, frame_number
        )
    )

  @parameterized.parameters([2, 3, 4, 5, 10, 11, 12])
  def test_is_frame_number_loading_instance_not_found_false(self, frame_number):
    cache = _test_inference_cache()
    cache._running_futures[self.test_dicom_instance_path.complete_url] = {
        'mock_future_1': [(2, 5)],
        'mock_future_2': [(10, 12)],
    }
    self.assertFalse(
        cache._is_frame_number_loading(BAD_TEST_PATH, frame_number)
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='defined_instance_path',
          instance_path=TEST_PATH,
          expected_len=2,
      ),
      dict(
          testcase_name='undefined_instance_path',
          instance_path=BAD_TEST_PATH,
          expected_len=0,
      ),
  ])
  def test_get_instance_future_loading_frame_ranges(
      self, instance_path, expected_len
  ):
    cache = _test_inference_cache()
    cache._running_futures[TEST_PATH.complete_url] = {
        'mock_future_1': [(2, 5)],
        'mock_future_2': [(10, 12)],
    }
    self.assertLen(
        cache._get_instance_future_loading_frame_ranges(instance_path),
        expected_len,
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='range_loaded_across_multiple_futures',
          list_of_list_of_frame_ranges=[
              [(1, 3), (6, 6)],
              [(4, 4)],
              [(7, 7), (8, 8)],
              [(5, 5)],
          ],
          test_range=(1, 8),
          expected=None,
      ),
      dict(
          testcase_name='range_loaded_by_single_future',
          list_of_list_of_frame_ranges=[
              [(1, 3), (6, 6)],
              [(4, 4)],
              [(7, 7), (8, 8)],
              [(5, 5)],
          ],
          test_range=(1, 3),
          expected=None,
      ),
      dict(
          testcase_name='range_loaded_by_single_future_across_mult_range',
          list_of_list_of_frame_ranges=[
              [(1, 3), (6, 6)],
              [(4, 4)],
              [(7, 7), (8, 8)],
              [(5, 5)],
          ],
          test_range=(7, 8),
          expected=None,
      ),
      dict(
          testcase_name='starting_frame_numbers_loading',
          list_of_list_of_frame_ranges=[
              [(1, 3), (6, 6)],
              [(4, 4)],
              [(7, 7), (8, 8)],
              [(5, 5)],
          ],
          test_range=(1, 9),
          expected=(9, 9),
      ),
      dict(
          testcase_name='gap_in_start_frame_number_loading_frames',
          list_of_list_of_frame_ranges=[[(1, 3)], [(4, 4)], [(7, 7), (8, 8)]],
          test_range=(1, 9),
          expected=(5, 9),
      ),
      dict(
          testcase_name='end_frame_number_range_loading',
          list_of_list_of_frame_ranges=[[(3, 3)], [(4, 4)], [(7, 7), (8, 8)]],
          test_range=(1, 8),
          expected=(1, 6),
      ),
      dict(
          testcase_name='clip_start_and_end_starting_frame_numbers',
          list_of_list_of_frame_ranges=[
              [(1, 3), (6, 6)],
              [(4, 4)],
              [(7, 7), (8, 8)],
          ],
          test_range=(1, 8),
          expected=(5, 5),
      ),
      dict(
          testcase_name='no_frames_loading',
          list_of_list_of_frame_ranges=[],
          test_range=(1, 8),
          expected=(1, 8),
      ),
      dict(
          testcase_name='test_range_is_none',
          list_of_list_of_frame_ranges=[],
          test_range=None,
          expected=None,
      ),
  ])
  def test_clip_frame_range_to_loading_frames(
      self, list_of_list_of_frame_ranges, test_range, expected
  ):
    cache = _test_inference_cache()
    with mock.patch.object(
        local_dicom_slide_cache.InMemoryDicomSlideCache,
        '_get_instance_future_loading_frame_ranges',
        return_value=list_of_list_of_frame_ranges,
    ):
      self.assertEqual(
          cache._clip_frame_range_to_loading_frames(
              self.test_dicom_instance_path, test_range
          ),
          expected,
      )

  def test_block_until_frames_are_loaded_waits_for_future_to_finish(self):
    cache = _test_inference_cache()
    with futures.ThreadPoolExecutor(3) as thread_pool:
      future_event = threading.Event()
      future = thread_pool.submit(_future_thread, future_event)
      cache._handle_future(self.test_dicom_instance_path, [(1, 10)], future)
      with self.assertRaises(futures.TimeoutError):
        cache.block_until_frames_are_loaded(self.test_dicom_instance_path, 0.01)
      future_event.set()

  def test_block_until_frames_are_loaded_does_not_wait_undefined_instance(self):
    cache = _test_inference_cache()
    with futures.ThreadPoolExecutor(3) as thread_pool:
      future_event = threading.Event()
      future = thread_pool.submit(_future_thread, future_event)
      cache._handle_future(self.test_dicom_instance_path, [(1, 10)], future)
      with mock.patch.object(futures.Future, 'result', autospec=True) as result:
        cache.block_until_frames_are_loaded(BAD_TEST_PATH)
        result.assert_not_called()
      future_event.set()

  @parameterized.named_parameters([
      dict(
          testcase_name='unbound_range',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 8),
      ),
      dict(
          testcase_name='request_larger_than_frame_number',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=5,
          max_request=20,
          expected_result=(1, 10),
      ),
      dict(
          testcase_name='data_loaded_end_frames',
          preset_indexes=[6, 7, 8, 9],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(2, 5),
      ),
      dict(
          testcase_name='data_loaded_front_frames',
          preset_indexes=[1, 2, 3, 4],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 8),
      ),
      dict(
          testcase_name='range_clipped_by_data',
          preset_indexes=[2, 3, 4, 8, 9],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 7),
      ),
      dict(
          testcase_name='range_shifted_by_preloaded_data',
          preset_indexes=[5, 6, 7, 8, 9],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(2, 5),
      ),
      dict(
          testcase_name='fully_loaded_with_data',
          preset_indexes=list(range(10)),
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 5),
      ),
      dict(
          testcase_name='preloaded_data_in_center',
          preset_indexes=[6, 7],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 8),
      ),
      dict(
          testcase_name='preloaded_data_at_starting_coordinate',
          preset_indexes=[5],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 8),  # Always return the requested
      ),
      dict(
          testcase_name='invalid_max_request',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=5,
          max_request=0,
          expected_result=None,
      ),
      dict(
          testcase_name='invalid_index',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=0,
          max_request=4,
          expected_result=None,
      ),
      dict(
          testcase_name='frame_number_too_large',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=20,
          max_request=4,
          expected_result=None,
      ),
      dict(
          testcase_name='frame_number_too_small',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=0,
          max_request=4,
          expected_result=None,
      ),
      dict(
          testcase_name='request_single_frame',
          preset_indexes=[],
          number_of_frames=10,
          frame_number=5,
          max_request=1,
          expected_result=(5, 5),
      ),
      dict(
          testcase_name='missing_only_requested_frame',
          preset_indexes=[2, 3, 4, 6, 7, 8, 9],
          number_of_frames=10,
          frame_number=5,
          max_request=4,
          expected_result=(5, 5),
      ),
  ])
  def test_get_frame_range_to_load(
      self,
      preset_indexes: List[int],
      number_of_frames: int,
      frame_number: int,
      max_request: int,
      expected_result: Optional[Tuple[int, int]],
  ):
    cache = _test_inference_cache()
    for idx in preset_indexes:
      cache._dicom_instance_frame_bytes[
          local_dicom_slide_cache._frame_number_key(
              self.test_dicom_instance_path,
              idx,
          )
      ] = b'1'
    self.assertEqual(
        cache._get_frame_range_to_load(
            self.test_dicom_instance_path,
            number_of_frames,
            frame_number,
            max_request,
        ),
        expected_result,
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='single_frame',
          frame_list=[b'1'],
          expected=[None, b'1', None],
      ),
      dict(
          testcase_name='multiple_frames',
          frame_list=[b'1', b'2', b'3'],
          expected=[None, b'1', b'2', b'3', None],
      ),
      dict(testcase_name='no_frames', frame_list=[], expected=[None, None]),
  ])
  def test_add_cached_instance_frames(
      self, frame_list: List[bytes], expected: List[Optional[bytes]]
  ):
    cache = _test_inference_cache()
    cache._add_cached_instance_frames(
        self.test_dicom_instance_path, 3, frame_list
    )
    result_lst = []
    for index in range(2, 2 + len(expected)):
      result_lst.append(
          cache._dicom_instance_frame_bytes.get(
              local_dicom_slide_cache._frame_number_key(
                  self.test_dicom_instance_path,
                  index,
              )
          )
      )
    self.assertEqual(result_lst, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='enabled_lru_no_resize',
          max_cache_frame_memory_lru_cache_size_bytes=12000,
          expected_count=2,
      ),
      dict(
          testcase_name='disabled',
          max_cache_frame_memory_lru_cache_size_bytes=None,
          expected_count=15,
      ),
  ])
  def test_lru_cache(
      self, max_cache_frame_memory_lru_cache_size_bytes, expected_count
  ):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          max_cache_frame_memory_lru_cache_size_bytes=max_cache_frame_memory_lru_cache_size_bytes,
      )
      cache.cache_whole_instance_in_memory(
          _get_mock_ez_wsi_dicom_instance(
              self.test_dicom_instance_path,
              self.test_dicom_instance.NumberOfFrames,
          ),
          blocking=True,
      )
      self.assertLen(cache._dicom_instance_frame_bytes, expected_count)
      self.assertEqual(
          cache.lru_caching_enabled,
          max_cache_frame_memory_lru_cache_size_bytes is not None,
      )

  def test_get_frames_returns_none_if_cache_size_less_than_frame_size(self):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          max_cache_frame_memory_lru_cache_size_bytes=1
      )
      self.assertIsNone(
          cache.get_frame(
              self.test_dicom_instance_path,
              self.test_dicom_instance.NumberOfFrames,
              1,
          )
      )
      cache.block_until_frames_are_loaded(self.test_dicom_instance_path)
      self.assertIsNone(
          cache.get_frame(
              self.test_dicom_instance_path,
              self.test_dicom_instance.NumberOfFrames,
              1,
          )
      )

  @mock.patch.object(logging.Logger, 'warning', autospec=True)
  def test_add_frames_larger_than_lru_cache_logs_warning(self, mock_absl_log):
    cache = self._setup_test_slide_cache_with_logger(
        max_cache_frame_memory_lru_cache_size_bytes=1
    )
    cache._add_cached_instance_frames(self.test_dicom_instance_path, 1, [b'23'])
    self.assertRegex(
        mock_absl_log.call_args.args[1],
        r'The maximum size in bytes of the LRU cache.*',
    )

  @parameterized.parameters([None, 3, 10])
  @mock.patch.object(logging.Logger, 'warning', autospec=True)
  def test_add_frames_to_lru_cache_no_warning(self, cache_size, mock_absl_log):
    cache = self._setup_test_slide_cache_with_logger(
        max_cache_frame_memory_lru_cache_size_bytes=cache_size
    )
    cache._add_cached_instance_frames(
        self.test_dicom_instance_path, 1, [b'232']
    )
    mock_absl_log.assert_not_called()

  @parameterized.parameters([([b'12'],), ([b'1', b'2'],)])
  @mock.patch.object(logging.Logger, 'warning', autospec=True)
  def test_total_frame_bytes_read_larger_than_cache_size_triggers_warning(
      self, test_bytes, mock_absl_log
  ):
    cache = self._setup_test_slide_cache_with_logger(
        max_cache_frame_memory_lru_cache_size_bytes=1
    )
    total_bytes_read = cache._total_frame_bytes_read(test_bytes)
    self.assertRegex(
        mock_absl_log.call_args.args[1],
        r'The maximum size in bytes of the LRU cache is smaller than the .*',
    )
    self.assertEqual(total_bytes_read, 2)

  @parameterized.parameters([None, 3, 10])
  @mock.patch.object(logging.Logger, 'warning', autospec=True)
  def test_total_frame_bytes_read_no_warning(self, cache_size, mock_absl_log):
    cache = self._setup_test_slide_cache_with_logger(
        max_cache_frame_memory_lru_cache_size_bytes=cache_size
    )
    total_bytes_read = cache._total_frame_bytes_read([b'12', b'3'])
    mock_absl_log.assert_not_called()
    self.assertEqual(total_bytes_read, 3)

  def test_invalid_lru_cache_size_raises(self):
    with self.assertRaises(
        local_dicom_slide_cache_types.InvalidLRUMaxCacheSizeError
    ):
      self._setup_test_slide_cache_with_logger(
          max_cache_frame_memory_lru_cache_size_bytes=0
      )

  def test_get_frame_bytes_empty(self):
    cache = _test_inference_cache()
    frame_bytes = cache._get_frame_bytes(self.test_dicom_instance_path, 10)
    self.assertIsNone(frame_bytes)

  @parameterized.named_parameters([
      dict(
          testcase_name='found_frame_bytes',
          instance_to_get=TEST_PATH,
          frame_number_to_get=7,
          expected=b'7',
      ),
      dict(
          testcase_name='missing_instance',
          instance_to_get=BAD_TEST_PATH,
          frame_number_to_get=7,
          expected=None,
      ),
      dict(
          testcase_name='missing_frame_number',
          instance_to_get=TEST_PATH,
          frame_number_to_get=8,
          expected=None,
      ),
  ])
  def test_get_frame_bytes_loaded(
      self, instance_to_get, frame_number_to_get, expected
  ):
    cache = _test_inference_cache()
    cache._dicom_instance_frame_bytes = {
        local_dicom_slide_cache._frame_number_key(TEST_PATH, 7): b'7'
    }
    self.assertEqual(
        cache._get_frame_bytes(instance_to_get, frame_number_to_get),
        expected,
    )

  def test_is_frame_number_loaded(self):
    frame_number = 1
    cache = _test_inference_cache()
    self.assertFalse(
        cache._is_frame_number_loaded(
            self.test_dicom_instance_path, frame_number
        )
    )
    cache._dicom_instance_frame_bytes = {
        local_dicom_slide_cache._frame_number_key(
            self.test_dicom_instance_path,
            frame_number,
        ): b'1'
    }
    self.assertTrue(
        cache._is_frame_number_loaded(
            self.test_dicom_instance_path, frame_number
        )
    )

  def _read_uncached_frame_from_dicom_store(
      self,
      cache: local_dicom_slide_cache.InMemoryDicomSlideCache,
      frame_number: int,
      optimization_hint: local_dicom_slide_cache_types.CacheConfigOptimizationHint,
  ) -> Optional[bytes]:
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      return cache.get_frame(
          self.test_dicom_instance_path,
          self.test_dicom_instance.NumberOfFrames,
          frame_number,
          optimization_hint=optimization_hint,
      )

  def test_minimize_dicom_store_qpm_query_optimization(self):
    cache = _test_inference_cache(
        number_of_frames_to_read=1,
        max_instance_number_of_frames_to_prefer_whole_instance_download=0,
    )
    frame_number = 1
    frame_bytes = self._read_uncached_frame_from_dicom_store(
        cache,
        frame_number,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
    )
    self.assertEqual(
        frame_bytes,
        dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
    )
    status = cache.cache_stats
    self.assertEqual(status.frame_cache_miss_count, 1)
    self.assertEqual(status.number_of_frames_downloaded_to_reduce_latency, 0)
    self.assertEqual(status.time_spent_downloading_frames_to_reduce_latency, 0)
    self.assertGreater(
        status.time_spent_blocked_waiting_for_cache_loading_to_complete, 0
    )

  def test_minimize_latency_query_optimization(self):
    cache = _test_inference_cache(
        number_of_frames_to_read=1,
        max_instance_number_of_frames_to_prefer_whole_instance_download=0,
    )
    frame_number = 1
    frame_bytes = self._read_uncached_frame_from_dicom_store(
        cache,
        frame_number,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
    )
    self.assertEqual(
        frame_bytes,
        dicom_test_utils.test_dicom_instance_frame_bytes(frame_number),
    )
    status = cache.cache_stats
    self.assertEqual(status.frame_cache_miss_count, 1)
    self.assertEqual(status.number_of_frames_downloaded_to_reduce_latency, 1)
    self.assertGreater(
        status.time_spent_downloading_frames_to_reduce_latency, 0
    )
    self.assertEqual(
        status.time_spent_blocked_waiting_for_cache_loading_to_complete, 0
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='class_def_single_frame_retrieval_optimization',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
          number_of_frames=1,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
      ),
      dict(
          testcase_name='class_def_single_frame_retrieval_qpm',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
          number_of_frames=1,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
      ),
      dict(
          testcase_name='class_def_single_frame_retrieval_return_none',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE,
          number_of_frames=1,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE,
      ),
      dict(
          testcase_name='class_def_multi_frame_retrieval_latency_optimization',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
          number_of_frames=2,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
      ),
      dict(
          testcase_name='class_def_multi_frame_retrieval_qpm_optimization',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
          number_of_frames=2,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
      ),
      dict(
          testcase_name='class_def_multi_frame_retrieval_none_optimization',
          class_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE,
          number_of_frames=2,
          expected=local_dicom_slide_cache_types.CacheConfigOptimizationHint.DO_NOT_BLOCK_RETURN_NONE_IMMEDIATELY_IF_NOT_IN_CACHE,
      ),
  ])
  def test_get_optimization_hint(self, class_hint, number_of_frames, expected):
    cache = _test_inference_cache(optimization_hint=class_hint)
    self.assertEqual(cache._get_optmization_hint(number_of_frames), expected)

  def test_cache_instance_uid_unchanged_by_pickle(self):
    cache = _test_inference_cache()
    uid = cache.cache_instance_uid
    cache.__setstate__(cache.__getstate__())
    self.assertEqual(uid, cache.cache_instance_uid)

  def test_cache_instance_uid_unchanged_by_reset_cache_stats(self):
    cache = _test_inference_cache()
    uid = cache.cache_instance_uid
    cache.reset_cache_stats()
    self.assertEqual(uid, cache.cache_instance_uid)

  @parameterized.named_parameters([
      dict(
          testcase_name='enabled_lru_cache',
          max_cache_frame_memory_lru_cache_size_bytes=12000,
          expected=12000,
      ),
      dict(
          testcase_name='disable_lru_cache',
          max_cache_frame_memory_lru_cache_size_bytes=None,
          expected=None,
      ),
  ])
  def test_cache_stats_memory_size_limit(
      self, max_cache_frame_memory_lru_cache_size_bytes, expected
  ):
    self.assertEqual(
        _test_inference_cache(
            max_cache_frame_memory_lru_cache_size_bytes=max_cache_frame_memory_lru_cache_size_bytes,
        ).cache_stats.frame_cache_memory_size_limit,
        expected,
    )

  def test_cache_externally_acquired_bytes(self):
    cache_key = 'test_key'
    test_bytes = b'123'
    cache = _test_inference_cache()
    self.assertTrue(
        cache.cache_externally_acquired_bytes(cache_key, test_bytes)
    )
    self.assertEqual(
        test_bytes, cache.get_cached_externally_acquired_bytes(cache_key)
    )

  def test_cache_externally_acquired_bytes_cache_miss(self):
    cache = _test_inference_cache()
    self.assertIsNone(cache.get_cached_externally_acquired_bytes('test_key'))

  def test_get_instance_path_list_from_level(self):
    mock_store_path = f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    with dicom_store_mock.MockDicomStores(mock_store_path) as mock_store:
      mock_store[mock_store_path].add_instance(self.test_dicom_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory_module.CredentialFactory()
          ),
          self.test_dicom_instance_path,
      )
      result = local_dicom_slide_cache._get_instance_path_list(ds.native_level)

      self.assertEqual(
          [(str(path), fc) for path, fc in result],
          [(
              'https://healthcare.googleapis.com/v1/projects/project_name/locations/us-west1/datasets/dataset_name/dicomStores/dicom_store_name/dicomWeb/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634/instances/1.2.276.0.7230010.3.1.4.296485376.89.1688794081.412405',
              15,
          )],
      )

  def test_get_instance_path_list_from_instance(self):
    mock_store_path = f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    with dicom_store_mock.MockDicomStores(mock_store_path) as mock_store:
      mock_store[mock_store_path].add_instance(self.test_dicom_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory_module.CredentialFactory()
          ),
          self.test_dicom_instance_path,
      )
      result = local_dicom_slide_cache._get_instance_path_list(
          next(ds.native_level.instance_iterator)
      )

      self.assertEqual(
          [(str(path), fc) for path, fc in result],
          [(
              'https://healthcare.googleapis.com/v1/projects/project_name/locations/us-west1/datasets/dataset_name/dicomStores/dicom_store_name/dicomWeb/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634/instances/1.2.276.0.7230010.3.1.4.296485376.89.1688794081.412405',
              15,
          )],
      )

  def test_get_instance_path_list_from_unexpected_type_raises(self):
    with self.assertRaises(local_dicom_slide_cache_types.UnexpectedTypeError):
      local_dicom_slide_cache._get_instance_path_list('bad')

  def test_optimization_hint_accessor(self):
    cache = _test_inference_cache()
    self.assertEqual(
        cache._optimization_hint,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
    )
    self.assertEqual(
        cache.optimization_hint,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
    )

  def test_optimization_hint_setter(self):
    cache = _test_inference_cache()
    cache.optimization_hint = (
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
    )
    self.assertEqual(
        cache.optimization_hint,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
    )
    self.assertEqual(
        cache._optimization_hint,
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY,
    )

  def test_get_cached_frame_zero(self):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          number_of_frames_to_read=1,
          max_instance_number_of_frames_to_prefer_whole_instance_download=0,
      )
      self.assertIsNone(
          cache.get_cached_frame(
              self.test_dicom_instance_path,
              0,
          )
      )

  def test_get_cached_frame_not_found(self):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          number_of_frames_to_read=1,
          max_instance_number_of_frames_to_prefer_whole_instance_download=0,
      )
      self.assertIsNone(
          cache.get_cached_frame(
              self.test_dicom_instance_path,
              1,
          )
      )

  def test_get_cached_frame_found(self):
    optimization_hint = (
        local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
    )
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      cache = _test_inference_cache(
          number_of_frames_to_read=1,
          max_instance_number_of_frames_to_prefer_whole_instance_download=0,
      )
      frame = cache.get_frame(
          self.test_dicom_instance_path,
          self.test_dicom_instance.NumberOfFrames,
          1,
          optimization_hint=optimization_hint,
      )
      self.assertIsNotNone(
          cache.get_cached_frame(
              self.test_dicom_instance_path,
              1,
          )
      )
      self.assertIs(
          cache.get_cached_frame(
              self.test_dicom_instance_path,
              1,
          ),
          frame,
      )

  def test_preload_cache_from_dicom_store_with_copy_from_pre_existing_cache(
      self,
  ):
    with dicom_store_mock.MockDicomStores(self.dicom_store_path) as mock_store:
      mock_store[self.dicom_store_path].add_instance(self.test_dicom_instance)
      copy_from_cache = _test_inference_cache()
      copy_from_cache.preload_instance_frame_numbers(
          {self.test_dicom_instance_path.complete_url: list(range(1, 16))},
          copy_from_cache,
      )
      copy_from_cache.block_until_frames_are_loaded(
          self.test_dicom_instance_path
      )
      self.assertEqual(
          copy_from_cache.cache_stats.number_of_frame_blocks_read, 1
      )
      self.assertEqual(
          copy_from_cache.cache_stats.number_of_frames_read_in_frame_blocks, 15
      )

      cache = _test_inference_cache(number_of_frames_to_read=1)
      cache.preload_instance_frame_numbers(
          {self.test_dicom_instance_path.complete_url: list(range(1, 16))},
          copy_from_cache,
      )
      cache.block_until_frames_are_loaded(self.test_dicom_instance_path)
      self.assertEqual(cache.cache_stats.number_of_frame_blocks_read, 0)
      self.assertEqual(
          cache.cache_stats.number_of_frames_read_in_frame_blocks, 0
      )
      for frame_number in range(1, 16):
        self.assertIsNotNone(
            cache.get_cached_frame(
                self.test_dicom_instance_path,
                frame_number,
            )
        )


if __name__ == '__main__':
  absltest.main()
