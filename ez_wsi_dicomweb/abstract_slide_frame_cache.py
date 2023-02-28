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

from typing import Optional


class AbstractSlideFrameCache(metaclass=abc.ABCMeta):
  """Abstract class for DICOM Slide frame cache."""

  @abc.abstractmethod
  def is_supported_transfer_syntax(self, transfer_syntax: str) -> bool:
    """Returns True if cache supports operation on DICOM encoding.

    Args:
      transfer_syntax: DICOM transfer syntax uid.

    Returns:
      True if cache supports operation on instances with encoding.
    """

  @abc.abstractmethod
  def get_frame(self, instance_path: str, frame_index: int) -> Optional[bytes]:
    """Returns DICOM instance frame bytes or None.

    Args:
      instance_path: DICOMweb path to instance.
      frame_index: Frame index within instance.  The frames are referenced in
        with 1-based indexing.

    Returns:
      Frame bytes or None.
    """
