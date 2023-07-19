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
"""SlideLevelMap defines the relationship between frames and DICOM instances."""
import collections
import dataclasses
import math
from typing import Collection, Dict, Iterator, Mapping, Optional, Tuple

from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import tags

# Since the level distance is a log function, this max distance value allows
# a level that has a pixel spacing ratio within range of [2^-0.3, 2^0.3] to be
# matched as the closest level by get_level_by_pixel_spacing() function.
_MAX_LEVEL_DIST = 0.3


@dataclasses.dataclass(frozen=True)
class Instance:
  """Wrapper of a DICOM instance object, with details for frame lookup.

  Used by the Level class to represent an instance in a specific
  magnification level.

  Attributes:
    frame_offset: The frame index offset of the first possible frame from the
      instance on the parent magnification level.
    frame_count: Number of frames contained inside the instance.
    dicom_object: The DicomObject of the instance.
  """

  frame_offset: int
  frame_count: int
  dicom_object: dicom_web_interface.DicomObject

  def frame_index_from_frame_number(self, frame_number: int) -> int:
    """Converts a frame_number to a frame_index within this instance.

    The frame_number and frame_index are two different concepts:
      - The frame_number is global to a Level. The values of frame_number are
        within [level.frame_number_min and level.frame_number_max].
      - The frame_index is local to the host instance. The frame_index starts
        with 1. The following equation converts a frame_number to frame_index:
        frame_index = frame_number - instance.frame_offset + 1

    Args:
     frame_number: The frame_number to be converted from.

    Returns:
     The index of the frame within this instance.
    """
    return frame_number - self.frame_offset + 1


@dataclasses.dataclass
class Level:
  """Represents the dimensions and instances of a specific magnification level.

  Attributes:
    level_index: The index of the level using 1-based indexing. A level with a
      higher index corresponding to a lower magnification level.
    width: The width, in number of pixels, of the entire level.
    height: The height, in number of pixels, of the entire level.
    samples_per_pixel: The number of samples per pixel. i.e. 3 for RGB format.
    bits_allocated: The number of bits allocated for storing one pixel.
    high_bit: The index of the highest bit in each sample of a pixel.
    pixel_spacing_x_mm: The pixel spacing, in millimeters, in x direction of the
      level.
    pixel_spacing_y_mm: The pixel spacing, in millimeters, in y direction of the
      level.
    frame_width: The width, in pixels, of all frames on this level.
    frame_height: The width, in pixels, of all frames on this level.
    frame_number_min: The minimum frame number at this level.
    frame_number_max: The maximum frame number at this level.
    instances: All instances on this level, keyed by the frame_offset of the
      instance. The instances should be ordered in the dictionary by the key.
    transfer_syntax_uid: DICOM transfer syntax instances are encoded as.
  """

  level_index: int
  width: int
  height: int
  samples_per_pixel: int
  bits_allocated: int
  high_bit: int
  pixel_spacing_x_mm: float
  pixel_spacing_y_mm: float
  frame_width: int
  frame_height: int
  frame_number_min: int
  frame_number_max: int
  instances: Dict[int, Instance]
  transfer_syntax_uid: str

  def get_instance_by_frame(self, frame_number: int) -> Optional[Instance]:
    """Gets the instance that contains the requested frame.

    Args:
      frame_number: The frame number of the frame to request. The frame number
        of a frame is stored in the DICOM tag
        CONCATENATION_FRAME_OFFSET_NUMBER('00209228').

    Returns:
      The instance that contains the requested frame, or None if the input frame
      is out of range of any instance.
    """
    for frame_offset, instance in self.instances.items():
      if frame_number < frame_offset:
        break
      if frame_number < frame_offset + instance.frame_count:
        return instance
    return None

  def get_frame_number_by_point(self, x: int, y: int) -> int:
    """Gets the frame number corresponding to the input coordinate.

    Args:
      x: The x coordinate at the level the instance belongs to.
      y: The y coordinate at the level the instance belongs to.

    Returns:
      The frame number that contains the input coordinate.

    Raises:
      CoordinateOutofImageDimensionsError if the input coordinate is out of the
      range.
    """
    if x < 0 or y < 0 or x >= self.width or y >= self.height:
      raise ez_wsi_errors.CoordinateOutofImageDimensionsError(
          f'The input coordinate {x, y} is out of the range: '
          f'{0, 0, self.width - 1, self.height - 1}'
      )
    frame_x = int(x / self.frame_width)
    frame_y = int(y / self.frame_height)
    frames_per_row = int(math.ceil(float(self.width) / float(self.frame_width)))
    return frame_y * frames_per_row + frame_x

  def get_frame_position(self, frame_number: int) -> Tuple[int, int]:
    """Gets the coordinate of the upper-left corner of the input frame.

    Args:
      frame_number: The input frame number to get the position of.

    Returns:
      The X and Y coordinates of the upper-left corner of the input frame.

    Raises:
      FrameNumberOutofBounds if the input frame number is out of the
      range.
    """
    if (
        frame_number < self.frame_number_min
        or frame_number > self.frame_number_max
    ):
      raise ez_wsi_errors.FrameNumberOutofBoundsError(
          f'The input frame number ({frame_number}) is out of the range: '
          f'{self.frame_number_min, self.frame_number_max}'
      )
    frame_number = int(frame_number)
    frames_per_row = int(math.ceil(float(self.width) / float(self.frame_width)))
    frame_x = frame_number % frames_per_row
    frame_y = int(frame_number / frames_per_row)
    return frame_x * self.frame_width, frame_y * self.frame_height

  def get_instance_by_point(self, x: int, y: int) -> Optional[Instance]:
    """Gets the instance that contains the input point.

    Args:
      x: The x coordinate at the level the instance belongs to.
      y: The y coordinate at the level the instance belongs to.

    Returns:
      The instance that contains the input point, or None if the point is out of
      range of the level or is not within any instance.
    """
    return self.get_instance_by_frame(self.get_frame_number_by_point(x, y))

  @property
  def instance_iterator(self) -> Iterator[Instance]:
    """Returns iterator of level's DICOM instances."""
    return iter(self.instances.values())


def _build_level_map(
    dicom_objects: Collection[dicom_web_interface.DicomObject],
) -> Mapping[int, Level]:
  """Returns a slide level map built from a set of input DICOM instances.

  The algorithm groups instances in multiple sets by level. The result is stored
  in a ordered dictionary, using the level index as the key

  Within each level, all instances belonging to that level are stored into a
  dictionary, using the frame offset of the instance as the key:
    Dict[frame_offset, Instance]

  Both the level dictionary and the instance dictionary need to be accessed in
  order by the key values. To avoid sorting the keys on-the-fly, we sort
  both dictionaries beforehand.

  Args:
    dicom_objects: DICOM objects to use to build the SlideLevelMap.

  Returns:
    A sorted dict that contains all levels of the DICOM object.

  Raises:
    UnexpectedDicomObjectInstanceError: If any of the input objects is not a
    DICOM instance object.
    NoDicomLevelsDetectedError: No level was detected in the Dicom object.
    DicomTagNotFoundError: If any of the required tags is missing from any DICOM
    object.
    InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
    value.
    SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide level
      contains instances with different transfer syntaxes.
  """
  levels = {}
  # https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.32.html#sect_A.32.8
  vl_micro_sop_class_id = "1.2.840.10008.5.1.4.1.1.77.1.6"
  level_instances = collections.defaultdict(dict)
  for dicom_object in dicom_objects:
    if dicom_object.type() != dicom_path.Type.INSTANCE:
      raise ez_wsi_errors.UnexpectedDicomObjectInstanceError(
          'SlideLevelMap expects all input DicomObject to have a type of '
          f'INSTANCE. Actual: {dicom_object.type()}'
      )
    _check_for_tag_or_raise(dicom_object, tags.SOP_CLASS_UID)
    if dicom_object.get_value(tags.SOP_CLASS_UID) != vl_micro_sop_class_id:
      continue

    # Some clincal viewer (sectra) appears to require label, thumbnail,
    # and macro images (overviews) to be encoded using the
    # VL Whole Slide Microscopy Image Storage IOD to be visible in the software.
    image_type = dicom_object.get_list_value(tags.IMAGE_TYPE)
    if image_type:
      if frozenset(image_type) & frozenset(('LABEL', 'THUMBNAIL', 'OVERVIEW')):
        continue

    # Makes sure the following tags have defined values in dicom_object.
    _check_for_tag_or_raise(dicom_object, tags.INSTANCE_NUMBER)
    _check_for_tag_or_raise(dicom_object, tags.TOTAL_PIXEL_MATRIX_COLUMNS, True)
    _check_for_tag_or_raise(dicom_object, tags.TOTAL_PIXEL_MATRIX_ROWS, True)
    _check_for_tag_or_raise(dicom_object, tags.COLUMNS, True)
    _check_for_tag_or_raise(dicom_object, tags.ROWS, True)
    _check_for_tag_or_raise(dicom_object, tags.SAMPLES_PER_PIXEL, True)
    _check_for_tag_or_raise(dicom_object, tags.BITS_ALLOCATED, True)
    _check_for_tag_or_raise(dicom_object, tags.HIGH_BIT, True)
    _check_for_tag_or_raise(dicom_object, tags.IMAGE_VOLUME_WIDTH, True)
    _check_for_tag_or_raise(dicom_object, tags.IMAGE_VOLUME_HEIGHT, True)
    _check_for_tag_or_raise(dicom_object, tags.TRANSFER_SYNTAX_UID, True)

    level_index = dicom_object.get_value(tags.INSTANCE_NUMBER)

    level_index_data = levels.get(level_index)
    if level_index_data is not None:
      if level_index_data.transfer_syntax_uid != dicom_object.get_value(
          tags.TRANSFER_SYNTAX_UID
      ):
        raise ez_wsi_errors.SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError()
    else:
      level_index_data = Level(
          level_index=level_index,
          width=dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS),
          height=dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_ROWS),
          samples_per_pixel=dicom_object.get_value(tags.SAMPLES_PER_PIXEL),
          bits_allocated=dicom_object.get_value(tags.BITS_ALLOCATED),
          high_bit=dicom_object.get_value(tags.HIGH_BIT),
          pixel_spacing_x_mm=dicom_object.get_value(tags.IMAGE_VOLUME_WIDTH)
          / dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_COLUMNS),
          pixel_spacing_y_mm=dicom_object.get_value(tags.IMAGE_VOLUME_HEIGHT)
          / dicom_object.get_value(tags.TOTAL_PIXEL_MATRIX_ROWS),
          frame_width=dicom_object.get_value(tags.COLUMNS),
          frame_height=dicom_object.get_value(tags.ROWS),
          frame_number_min=-1,
          frame_number_max=-1,
          instances={},
          transfer_syntax_uid=dicom_object.get_value(tags.TRANSFER_SYNTAX_UID),
      )
      levels[level_index] = level_index_data

    frame_offset = (
        dicom_object.get_value(tags.CONCATENATION_FRAME_OFFSET_NUMBER) or 0
    )
    level_instances[level_index][frame_offset] = Instance(
        frame_offset=frame_offset,
        frame_count=dicom_object.get_value(tags.NUMBER_OF_FRAMES),
        dicom_object=dicom_object,
    )
  if not levels:
    raise ez_wsi_errors.NoDicomLevelsDetectedError(
        f'No level detected from the input DICOM objects {dicom_objects}'
    )

  # Sort levels based on level index.
  sorted_levels = collections.OrderedDict()
  for level_index in sorted(levels):
    # Sort instances in each level based on frame_offset
    sorted_level_index_data = levels[level_index]
    instances = level_instances[level_index]
    sorted_instances = collections.OrderedDict()
    for frame_offset in sorted(instances):
      sorted_instances[frame_offset] = instances[frame_offset]
    frame_number_min = next(iter(sorted_instances.keys()))
    max_frame_offset = next(reversed(sorted_instances.keys()))
    frame_number_max = (
        max_frame_offset + instances[max_frame_offset].frame_count
    )
    sorted_levels[level_index] = dataclasses.replace(
        sorted_level_index_data,
        frame_number_max=frame_number_max,
        frame_number_min=frame_number_min,
        instances=sorted_instances,
    )

  return sorted_levels


def _check_for_tag_or_raise(
    dicom_object: dicom_web_interface.DicomObject,
    tag: tags.DicomTag,
    check_no_zero: bool = False,
):
  """Check for the existence of a DICOM tag in a DICOM object.

  Args:
    dicom_object: The host DICOM object to check against.
    tag: The DICOM tag to check for.
    check_no_zero: If enabled (true), this method will check if the value of the
      tag is non-zero.

  Raises:
    DicomTagNotFoundError: If the tag does not exist in the object.
    InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
    value.
  """
  value = dicom_object.get_value(tag)
  if value is None:
    raise ez_wsi_errors.DicomTagNotFoundError(
        f'DICOM tag {tag.number} is missing from the DICOM object: '
        f'{str(dicom_object.path)}'
    )
  if check_no_zero and value == 0:
    raise ez_wsi_errors.InvalidDicomTagError(
        f'DICOM tag {tag.number} cannot have zero value.'
    )


class SlideLevelMap(object):
  """A class that builds and maintains the level-instance-frame mapping.

  When converting a WSI slide to DICOM format, a single WSI slide gets mapped to
  one DICOM series, which stores images at multiple levels of magnification with
  multiple DICOM instances.
  At each level of magnification, the image is divided into multiple frames with
  a fixed frame size(i.e. 500).
  DICOM tries to store all frames at a magnification level into one DICOM
  instance. If the number of frames at that level exceeds the limit, which is
  typically 2048, multiple instances are used. In that case, the index of the
  first possible frame within an instance is tagged as the frame_offset of that
  instance.
  """

  def __init__(
      self, dicom_objects: Collection[dicom_web_interface.DicomObject]
  ):
    """Constructor.

    Args:
      dicom_objects: DICOM objects of all instances within a DICOM series.

    Raises:
      UnexpectedDicomObjectInstanceError: If any of the input objects is not a
      DICOM instance object
      NoDicomLevelsDetectedError: No level was detected in the Dicom object.
      DicomTagNotFoundError: If any of the required tags is missing from any
      DICOM object.
      InvalidDicomTagError: If check_no_zero is enabled and the tag has zero
      value.
      SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError: Slide
      level contains instances with different transfer syntaxes.
    """
    self._level_map = _build_level_map(dicom_objects)
    level_index_list = list(self._level_map)
    self.level_index_min = level_index_list[0]
    self.level_index_max = level_index_list[-1]

  @property
  def level_map(self) -> Mapping[int, Level]:
    """Returns an ordered level map built from a set of input DICOM instances.

    The algorithm groups instances in multiple sets by level. The result is
    stored in a dictionary, using the level index as the key:
      Dict[level_index, Level]
    """
    return self._level_map

  def get_level(self, level_index: int) -> Optional[Level]:
    """Returns the level by requested level index.

    Args:
      level_index: The level index to be required. Level index usually starts at
        1.
    """
    return self._level_map.get(level_index)

  def get_level_by_pixel_spacing(
      self, pixel_spacing_mm: float
  ) -> Optional[Level]:
    """Returns the level that has the closest pixel spacing as the input.

    NOTE: This code assumes all images will have square pixels. It also assumes
    magnifications are powers of 2.
    Args:
      pixel_spacing_mm: the requested pixel spacing in millimeters.

    Returns:
      The level that closely matches with the requestd pixel spacing, or None
      if the requested pixel spacing is out range of any existing levels.
    """

    def _compute_distance_from_level(index: int) -> float:
      """Returns the distance between requested pixel spacing and a level.

      The distance is defined as the log2 of the pixel spacing ratio, which is
      defined as:
        pixel_spacing_ratio = requested_pixel_spacing/level_pixel_spacing

      Args:
        index: The index of the level to be calculated against.
      """
      return abs(
          math.log2(
              pixel_spacing_mm / self._level_map[index].pixel_spacing_x_mm
          )
      )
    min_level = None
    min_distance = 1e10
    # Finds the min_distance in level_map.
    for level in self._level_map:
      distance = _compute_distance_from_level(level)
      if distance < min_distance:
        min_distance = distance
        min_level = level

    if min_distance < _MAX_LEVEL_DIST:
      return self._level_map[min_level]

    return None
