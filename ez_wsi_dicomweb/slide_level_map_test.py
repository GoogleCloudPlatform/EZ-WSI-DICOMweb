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
"""Tests for ez_wsi_dicomweb.slide_level_map."""
import json
import random
from typing import Iterator, Mapping

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import slide_level_map
from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import tags
from ez_wsi_dicomweb.test_utils import dicom_test_utils


class SlideLevelMapTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.dicom_objects = []
    with open(dicom_test_utils.sample_instances_path()) as json_file:
      data = json.load(json_file)
      data = data['test_data']
      for x in data:
        dicom_object = dicom_web_interface.DicomObject(
            dicom_path.FromString(x['path']), x['dicom_tags']
        )
        self.dicom_objects.append(dicom_object)
      # Shuffle the order of the DICOM objects.
      random.shuffle(self.dicom_objects)

  def test_level_map_property(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)

    self.assertIsInstance(level_map.level_map, Mapping)

  def test_constructor_with_normal_input(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertEqual(1, level_map.level_index_min)
    self.assertEqual(10, level_map.level_index_max)
    level = level_map.get_level(1)
    self.assertIsNotNone(level, 'The native level cannot be missing.')
    if level is not None:
      self.assertEqual(0, level.frame_number_min)
      self.assertEqual(79002, level.frame_number_max)

  def test_raises_if_instances_have_different_transfer_sytax(self):
    with open(
        dicom_test_utils.testdata_path('error_multi_transfer_syntax.json')
    ) as json_file:
      data = json.load(json_file)
      data = data['test_data']
      dicom_objects = [
          dicom_web_interface.DicomObject(
              dicom_path.FromString(x['path']), x['dicom_tags']
          )
          for x in data
      ]
    with self.assertRaises(
        ez_wsi_errors.SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError
    ):
      slide_level_map.SlideLevelMap(dicom_objects)

  def test_constructor_with_invalid_input_raises_error(self):
    empty_object_dict = []
    with self.assertRaises(ez_wsi_errors.NoDicomLevelsDetectedError):
      slide_level_map.SlideLevelMap(empty_object_dict)

  def test_get_level_with_invalid_input_return_none(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertIsNone(
        level_map.get_level(0), 'Level 0 should not exist in the testing map.'
    )

  def test_get_level_with_valid_input(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(1)
    self.assertIsNotNone(level, 'Level 1 should exist in the testing map.')
    if level is not None:  # level is Optional[Level]
      self.assertEqual(level.width, 98816)
      self.assertEqual(level.height, 199168)
      self.assertEqual(level.samples_per_pixel, 3)
      self.assertEqual(level.bits_allocated, 8)
      self.assertEqual(level.high_bit, 7)
      self.assertAlmostEqual(level.pixel_spacing_x_mm, 0.00024309, 7)
      self.assertAlmostEqual(level.pixel_spacing_y_mm, 0.00024309, 7)
      self.assertEqual(level.frame_width, 500)
      self.assertEqual(level.frame_height, 500)
      self.assertLen(level.instances, 39)

  @parameterized.named_parameters([
      dict(testcase_name='missing_spacing_1', pixel_spacing=0.0001),
      dict(testcase_name='missing_spacing_2', pixel_spacing=0.0026),
      dict(testcase_name='missing_spacing_3', pixel_spacing=0.000001),
  ])
  def test_get_level_by_pixel_spacing_with_out_of_range_pixel_spacing(
      self, pixel_spacing: float
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level_by_pixel_spacing(pixel_spacing)
    self.assertIsNone(
        level,
        (
            'There should be no level that has pixel spacing close to '
            f'{pixel_spacing}. Actual: matched to '
            f'{level.pixel_spacing_x_mm if level is not None else None}'
        ),
    )

  @parameterized.named_parameters([
      dict(testcase_name='level1', pixel_spacing=0.00024, expected_level=1),
      dict(
          testcase_name='close_to_level1',
          pixel_spacing=0.00028,
          expected_level=1,
      ),
      dict(testcase_name='level2', pixel_spacing=0.00049, expected_level=2),
      dict(testcase_name='level3', pixel_spacing=0.00095, expected_level=3),
      dict(testcase_name='level10', pixel_spacing=0.1244, expected_level=10),
  ])
  def test_get_level_by_pixel_spacing_with_in_range_pixel_spacing(
      self, pixel_spacing: float, expected_level: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level_by_pixel_spacing(pixel_spacing)
    self.assertIsNotNone(
        level,
        (
            'There should have a level that has pixel spacing '
            f'close to {pixel_spacing}.'
        ),
    )
    self.assertEqual(
        level.level_index if level is not None else 0,
        expected_level,
        'The found level should have an index of {expected_level}.',
    )

  @parameterized.parameters((1, 0), (1, 2047), (1, 40960), (2, 2048))
  def test_get_instance_by_frame_with_existing_frames(
      self, level_index: int, frame_index: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      instance = level.get_instance_by_frame(frame_index)
      self.assertIsNotNone(
          instance,
          (
              'There should exist an instance containing frame: '
              f'{frame_index} at level {level_index}.'
          ),
      )

  @parameterized.parameters((1, -1), (1, 1000000), (10, 40960))
  def test_get_instance_by_frame_with_nonexisting_frames(
      self, level_index: int, frame_index: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      instance = level.get_instance_by_frame(frame_index)
      self.assertIsNone(
          instance,
          f'Frame {frame_index} should not exist at level {level_index}.',
      )

  @parameterized.named_parameters([
      dict(testcase_name='left_edge_level_1', level_index=1, x=0, y=0),
      dict(testcase_name='middle_level_1', level_index=1, x=2047, y=2047),
      dict(testcase_name='first_row_level_1', level_index=1, x=0, y=40960),
      dict(testcase_name='first_row_level_2', level_index=2, x=1, y=2048),
      dict(testcase_name='inside_level_10', level_index=10, x=192, y=388),
  ])
  def test_get_instance_by_point_with_valid_point(
      self, level_index: int, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      instance = level.get_instance_by_point(x, y)
      self.assertIsNotNone(
          instance,
          (
              'There should exist an instance containing point: ('
              f'{x, y}) at level {level_index}.'
          ),
      )

  @parameterized.named_parameters([
      dict(testcase_name='off_left_level_1', level_index=1, x=-1, y=0),
      dict(testcase_name='below_level_1', level_index=1, x=0, y=409600),
      dict(testcase_name='outside_level_10', level_index=10, x=193, y=389),
  ])
  def test_get_instance_by_point_with_out_of_range_point(
      self, level_index: int, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      with self.assertRaises(ez_wsi_errors.CoordinateOutofImageDimensionsError):
        level.get_instance_by_point(x, y)

  @parameterized.parameters(
      (1, 0, 0, 0),
      (1, 2047, 2047, 796),
      (1, 0, 40960, 16038),
      (2, 1, 2048, 396),
      (10, 192, 388, 0),
  )
  def test_get_frame_by_point_with_valid_point(
      self, level_index: int, x: int, y: int, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      calculated_frame_number = level.get_frame_number_by_point(x, y)
      self.assertEqual(
          frame_number,
          calculated_frame_number,
          (
              f'There calculated frame number({calculated_frame_number}) '
              f'does not match with expected value: {frame_number}'
          ),
      )

  @parameterized.named_parameters([
      dict(testcase_name='off_left_side_level_1', level_index=1, x=-1, y=0),
      dict(testcase_name='off_bottom_level_1', level_index=1, x=0, y=409600),
      dict(testcase_name='outside_level_10', level_index=10, x=193, y=389),
  ])
  def test_get_frame_by_point_with_out_of_range_point(
      self, level_index: int, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      with self.assertRaises(ez_wsi_errors.CoordinateOutofImageDimensionsError):
        level.get_instance_by_point(x, y)

  @parameterized.parameters((1, -1), (1, 6553600), (10, 20))
  def test_get_frame_position_with_out_of_range_input(
      self, level_index: int, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      with self.assertRaises(ez_wsi_errors.FrameNumberOutofBoundsError):
        level.get_frame_position(frame_number)

  @parameterized.parameters(
      (1, 0, 0, 0),
      (1, 7, 3500, 0),
      (1, 256, 29000, 500),
      (2, 5, 2500, 0),
      (10, 0, 0, 0),
  )
  def test_get_frame_position_with_valid_input(
      self, level_index: int, frame_number: int, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(level_index)
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    if level is not None:
      pos_x, pos_y = level.get_frame_position(frame_number)
      self.assertEqual(
          (x, y),
          (pos_x, pos_y),
          (
              f'The returned corner ({pos_x, pos_y}) do not match expectation:'
              f' ({x, y}) at level {level_index}.'
          ),
      )

  @parameterized.parameters((0, 1), (1, 2), (2047, 2048), (2048, 1))
  def test_index_from_frame_number(self, frame_number: int, frame_index: int):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level(1)
    self.assertIsNotNone(level)
    if level is not None:
      instance = level.get_instance_by_frame(frame_number)
      self.assertIsNotNone(instance)
      if instance is not None:
        self.assertEqual(
            frame_index, instance.frame_index_from_frame_number(frame_number)
        )

  def test_wsi_sop_class_id_detection(self):
    vl_micro_sop_class_id = "1.2.840.10008.5.1.4.1.1.77.1.6"
    self.assertEqual(vl_micro_sop_class_id, '1.2.840.10008.5.1.4.1.1.77.1.6')

  def test_slide_level_map_excludes_thumbnail_macro_and_label_images(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    for level in level_map._level_map.values():
      for instance in level.instances.values():
        image_type = instance.dicom_object.get_list_value(tags.IMAGE_TYPE)
        if image_type:
          # Validate that Thumbnail, label, and overview images are not being
          # stored.
          self.assertFalse(
              frozenset(image_type)
              & frozenset(('LABEL', 'THUMBNAIL', 'OVERVIEW'))
          )
        else:
          self.assertIsNone(image_type)

  def test_instance_iterator(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertIsInstance(level_map._level_map[1].instance_iterator, Iterator)
    self.assertLen(list(level_map._level_map[1].instance_iterator), 39)


if __name__ == '__main__':
  absltest.main()
