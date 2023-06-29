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
"""Abstract class for DICOM Slide frame cache."""
import abc
import dataclasses
from typing import List, Mapping, Optional, Union

from ez_wsi_dicomweb import slide_level_map


@dataclasses.dataclass
class AbstractSlideFrameCacheStats:
  """Abstract base class for SlideFrameCacheStats."""


class AbstractSlideFrameCache(metaclass=abc.ABCMeta):
  """Abstract class for DICOM Slide frame cache."""

  @abc.abstractmethod
  def get_frame(
      self, instance_path: str, number_of_frames: int, frame_number: int
  ) -> Optional[bytes]:
    """Returns DICOM instance frame bytes or None.

    Args:
      instance_path: DICOMweb path to instance.
      number_of_frames: Total number of frames in instance.
      frame_number: Frame index within instance.  The frames are referenced in
        with 1-based indexing.

    Returns:
      Frame bytes or None.
    """

  @abc.abstractmethod
  def preload_instance_frame_numbers(
      self,
      instance_frame_numbers: Mapping[str, List[int]],
  ) -> None:
    """Preloads select instance frames from DICOM Store into cache.

    Args:
      instance_frame_numbers: Map of instance path to frame numbers.

    Returns:
      None.
    """

  @abc.abstractmethod
  def cache_whole_instance_in_memory(
      self,
      instance_paths: Union[slide_level_map.Level, slide_level_map.Instance],
      blocking: bool,
  ) -> None:
    """Caches whole DICOM instance in memory.

    Args:
      instance_paths: DICOMweb path to instance.
      blocking: Load cache as blocking operation.
    """

  @abc.abstractmethod
  def reset_cache_stats(self) -> None:
    """Resets cache status metrics."""

  @property
  @abc.abstractmethod
  def cache_stats(self) -> AbstractSlideFrameCacheStats:
    """Returns dataclass which encodes cache stats."""
