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


class EZWsiError(Exception):
  pass


class CoordinateOutofImageDimensionsError(EZWsiError):
  pass


class DicomImageMissingRegionError(EZWsiError):
  pass


class DicomPathError(EZWsiError):
  pass


class DicomSlideMissingError(EZWsiError):
  pass


class DicomTagNotFoundError(EZWsiError):
  pass


class FrameNumberOutofBoundsError(EZWsiError):
  pass


class InputFrameNumberOutOfRangeError(EZWsiError):
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


class NonSquarePixelError(EZWsiError):
  pass


class PatchIntersectionNotFoundError(EZWsiError):
  pass


class PixelSpacingLevelNotFoundError(EZWsiError):
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


class UnsupportedPixelFormatError(EZWsiError):
  pass
