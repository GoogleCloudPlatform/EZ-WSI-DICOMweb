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
"""Tests for slide level map."""

import dataclasses
import json
import random
import typing
from typing import Iterator, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import pixel_spacing as pixel_spacing_module
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import pydicom

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


def _round(l):
  if isinstance(l, list) or isinstance(l, tuple):
    return [round(x, 4) for x in l]
  return round(l, 4)


class SlideLevelMapTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dicom_objects = []
    self.concat_dicom_objects = []
    with open(dicom_test_utils.sample_instances_path()) as json_file:
      data = json.load(json_file)
      data = data['test_data']
      for x in data:
        dicom_object = dicom_web_interface.DicomObject(
            dicom_path.FromString(x['path']), x['dicom_tags'], ''
        )
        self.dicom_objects.append(dicom_object)
      # Shuffle the order of the DICOM objects.
      random.shuffle(self.dicom_objects)

    with open(
        dicom_test_utils.instance_concatenation_test_data_path()
    ) as json_file:
      data = json.load(json_file)
      data = data['test_data']
      for x in data:
        dicom_object = dicom_web_interface.DicomObject(
            dicom_path.FromString(x['path']), x['dicom_tags'], ''
        )
        self.concat_dicom_objects.append(dicom_object)
      # Shuffle the order of the DICOM objects.
      random.shuffle(self.concat_dicom_objects)

  def test_level_map_property(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)

    self.assertIsInstance(level_map.level_map, Mapping)

  def test_constructor_with_normal_input(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertEqual(1, level_map.level_index_min)
    self.assertEqual(10, level_map.level_index_max)
    level = typing.cast(slide_level_map.Level, level_map.get_level(1))
    self.assertIsNotNone(level, 'The native level cannot be missing.')
    self.assertEqual(1, level.frame_number_min)
    self.assertEqual(2048, level.frame_number_max)

  def test_level_scale_factor_from_level(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(1))
    level2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    self.assertEqual(level.scale_factors(level2), (2.0, 2.0))

  def test_level_scale_factor_from_resized_level(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(1))
    level2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    ds_level = level.resize(level2)
    self.assertEqual(level.scale_factors(ds_level), (2.0, 2.0))
    self.assertEqual((ds_level.width, ds_level.height), (49408, 99584))

  def test_get_resized_from_resized_level(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(1))
    level2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    ds_level = level.resize(level.resize(level2))
    self.assertEqual((ds_level.width, ds_level.height), (49408, 99584))

  def test_raises_if_instances_have_different_transfer_sytax(self):
    with open(
        dicom_test_utils.testdata_path('error_multi_transfer_syntax.json')
    ) as json_file:
      data = json.load(json_file)
      data = data['test_data']
      dicom_objects = [
          dicom_web_interface.DicomObject(
              dicom_path.FromString(x['path']), x['dicom_tags'], ''
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
    level = typing.cast(slide_level_map.Level, level_map.get_level(1))
    self.assertIsNotNone(level, 'Level 1 should exist in the testing map.')
    self.assertEqual(level.width, 98816)
    self.assertEqual(level.height, 199168)
    self.assertEqual(level.samples_per_pixel, 3)
    self.assertEqual(level.bits_allocated, 8)
    self.assertEqual(level.high_bit, 7)
    self.assertAlmostEqual(level.pixel_spacing.column_spacing_mm, 0.00024309, 7)
    self.assertAlmostEqual(level.pixel_spacing.row_spacing_mm, 0.00024309, 7)
    self.assertEqual(level.frame_width, 500)
    self.assertEqual(level.frame_height, 500)
    self.assertLen(level.instances, 1)

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
            f'{level.pixel_spacing.pixel_spacing_mm if level is not None else None}'
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

  def test_get_resized_level_by_pixel_spacing(self):
    pixel_spacing = 0.0026
    expected_level = 4
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = level_map.get_level_by_pixel_spacing(
        pixel_spacing, maximum_downsample=8.0
    )
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

  @parameterized.named_parameters([
      dict(testcase_name='first_frame_level_1', level_index=1, frame_number=1),
      dict(
          testcase_name='middle_frame_level_1', level_index=1, frame_number=2048
      ),
      dict(testcase_name='end_frame_level_2', level_index=2, frame_number=2048),
  ])
  def test_get_instance_by_frame_with_existing_frames(
      self, level_index, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    instance = level.get_instance_by_frame(frame_number)
    self.assertIsNotNone(
        instance,
        (
            'There should exist an instance containing frame: '
            f'{frame_number} at level {level_index}.'
        ),
    )

  @parameterized.named_parameters([
      dict(testcase_name='frame_zero', level_index=1, frame_number=0),
      dict(
          testcase_name='beyond_last_frame_level_1',
          level_index=1,
          frame_number=1000001,
      ),
      dict(
          testcase_name='beyond_last_frame_level_10',
          level_index=10,
          frame_number=40961,
      ),
  ])
  def test_get_instance_by_frame_with_nonexisting_frames(
      self, level_index, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    instance = level.get_instance_by_frame(frame_number)
    self.assertIsNone(
        instance,
        f'Frame {frame_number} should not exist at level {level_index}.',
    )

  @parameterized.named_parameters([
      dict(testcase_name='left_edge_level_1', level_index=1, x=0, y=0),
      dict(testcase_name='middle_level_1', level_index=1, x=2047, y=2047),
      dict(testcase_name='inside_level_10', level_index=10, x=192, y=388),
  ])
  def test_get_instance_by_point_with_valid_point(
      self, level_index, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
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
      self, level_index, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    with self.assertRaises(ez_wsi_errors.CoordinateOutofImageDimensionsError):
      level.get_instance_by_point(x, y)

  @parameterized.named_parameters([
      dict(
          testcase_name='edge_first_frame_level_1',
          level_index=1,
          x=0,
          y=0,
          frame_number=1,
      ),
      dict(
          testcase_name='middle_level_1',
          level_index=1,
          x=2047,
          y=2047,
          frame_number=797,
      ),
      dict(
          testcase_name='first_row_level_1',
          level_index=1,
          x=0,
          y=40960,
          frame_number=16039,
      ),
      dict(
          testcase_name='first_row_level_2',
          level_index=2,
          x=1,
          y=2048,
          frame_number=397,
      ),
      dict(
          testcase_name='inside_first_frame_level_10',
          level_index=10,
          x=192,
          y=388,
          frame_number=1,
      ),
  ])
  def test_get_frame_by_point_with_valid_point(
      self, level_index, x: int, y: int, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
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
      self, level_index, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    with self.assertRaises(ez_wsi_errors.CoordinateOutofImageDimensionsError):
      level.get_instance_by_point(x, y)

  @parameterized.named_parameters([
      dict(testcase_name='frame_num_zero', level_index=1, frame_number=0),
      dict(
          testcase_name='frame_num_beyond_last_frame_large_image',
          level_index=1,
          frame_number=6553601,
      ),
      dict(
          testcase_name='frame_num_beyond_last_frame_small_image',
          level_index=10,
          frame_number=21,
      ),
  ])
  def test_get_frame_position_with_out_of_range_input(
      self, level_index, frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    with self.assertRaises(ez_wsi_errors.FrameNumberOutofBoundsError):
      level.get_frame_position(frame_number)

  @parameterized.named_parameters([
      dict(
          testcase_name='first_frame', level_index=1, frame_number=1, x=0, y=0
      ),
      dict(
          testcase_name='frame_in_first_row',
          level_index=1,
          frame_number=8,
          x=3500,
          y=0,
      ),
      dict(
          testcase_name='frame_inside_image',
          level_index=1,
          frame_number=257,
          x=29000,
          y=500,
      ),
      dict(
          testcase_name='frame_in_first_row_resized_image',
          level_index=2,
          frame_number=6,
          x=2500,
          y=0,
      ),
      dict(
          testcase_name='first_frame_highly_resized_image',
          level_index=10,
          frame_number=1,
          x=0,
          y=0,
      ),
  ])
  def test_get_frame_position_with_valid_input(
      self, level_index, frame_number: int, x: int, y: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(level_index))
    self.assertIsNotNone(
        level, f'There should exist a level at index {level_index}.'
    )
    pos_x, pos_y = level.get_frame_position(frame_number)
    self.assertEqual(
        (x, y),
        (pos_x, pos_y),
        (
            f'The returned corner ({pos_x, pos_y}) do not match expectation:'
            f' ({x, y}) at level {level_index}.'
        ),
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='first_instance_frame',
          level_number=1,
          frame_number=1,
          instance_frame_number=1,
      ),
      dict(
          testcase_name='last_instance_frame',
          level_number=1,
          frame_number=2048,
          instance_frame_number=2048,
      ),
      dict(
          testcase_name='second_instance_frame',
          level_number=3,
          frame_number=3,
          instance_frame_number=3,
      ),
      dict(
          testcase_name='first_instance_frame_in_concat_instance',
          level_number=3,
          frame_number=2049,
          instance_frame_number=1,
      ),
  ])
  def test_index_from_frame_number(
      self, level_number, frame_number: int, instance_frame_number: int
  ):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(
        slide_level_map.Level, level_map.get_level(level_number)
    )
    self.assertIsNotNone(level)
    instance = level.get_instance_by_frame(frame_number)
    self.assertIsNotNone(instance)
    if instance is not None:
      self.assertEqual(
          instance_frame_number,
          instance.instance_frame_number_from_wholes_slide_frame_number(
              frame_number
          ),
      )

  def test_wsi_sop_class_id_detection(self):
    vl_micro_sop_class_id = '1.2.840.10008.5.1.4.1.1.77.1.6'
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

  @parameterized.named_parameters([
      dict(
          testcase_name='level_with_one_instance',
          level=1,
          expected_length=1,
      ),
      dict(
          testcase_name='level_with_two_instances',
          level=3,
          expected_length=2,
      ),
  ])
  def test_instance_iterator(self, level, expected_length):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertIsInstance(
        level_map._level_map[level].instance_iterator, Iterator
    )
    self.assertLen(
        list(level_map._level_map[level].instance_iterator), expected_length
    )

  def test_get_dicom_instance_frames_across_concat_instances_empty(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertFalse(level_map.are_instances_concatenated([]))

  def test_get_dicom_instance_frames_across_concat_single_instance(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertFalse(level_map.are_instances_concatenated(['1.2.3.4']))

  def test_get_dicom_instance_frames_across_concat_instances(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertTrue(
        level_map.are_instances_concatenated([
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
        ])
    )

  def test_get_dicom_instance_frames_across_concat_instances_extra_false(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertFalse(
        level_map.are_instances_concatenated([
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791299',
        ])
    )

  def test_get_dicom_instance_frames_across_concat_instances_false(self):
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    self.assertFalse(
        level_map.are_instances_concatenated([
            '1.2.276.0.7230010.3.1.4.2148154112.13.1559583787.823504',
            '1.2.276.0.7230010.3.1.4.2148154112.13.1559584938.823592',
            '1.2.276.0.7230010.3.1.4.2148154112.13.1559584964.823594',
            '1.2.276.0.7230010.3.1.4.2148154112.13.1559583814.823506',
            '1.2.276.0.7230010.3.1.4.2148154112.13.1559583892.823512',
        ])
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='instance_1',
          instance_uid='1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
          pixel_spacing=pixel_spacing_module.PixelSpacing(0.256, 0.256),
      ),
      dict(
          testcase_name='instance_2',
          instance_uid='1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
          pixel_spacing=pixel_spacing_module.PixelSpacing(0.256, 0.255625),
      ),
  ])
  def test_get_instance_pixel_spacing(
      self, instance_uid: str, pixel_spacing: pixel_spacing_module.PixelSpacing
  ):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    ps = level_map.get_instance_pixel_spacing(instance_uid)

    self.assertIsNotNone(ps)
    self.assertTrue(ps.__eq__(pixel_spacing))

  def test_get_instance_pixel_spacing_raises(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    with self.assertRaises(ez_wsi_errors.PixelSpacingNotFoundForInstanceError):
      level_map.get_instance_pixel_spacing(
          '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791299'
      )

  def test_init_slide_level_map_from_dicom_and_json_raises_not_found(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
      slide_level_map.SlideLevelMap.create_from_json(
          level_map.to_json([
              slide_level_map.Level(
                  0,
                  0,
                  0,
                  0,
                  0,
                  0,
                  pixel_spacing_module.PixelSpacing(0.0, 0.0),
                  0,
                  0,
                  0,
                  0,
                  {},
                  '',
              )
          ])
      )

  def test_init_slide_level_map_from_json(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    json_metadata = level_map.to_json()
    init_from_json = slide_level_map.SlideLevelMap.create_from_json(
        json_metadata
    )
    self.assertEqual(level_map.level_index_min, init_from_json.level_index_min)
    self.assertEqual(level_map.level_index_max, init_from_json.level_index_max)
    self.assertEqual(level_map.level_map, init_from_json.level_map)
    self.assertEqual(
        str(level_map._smallest_level_path),
        str(init_from_json._smallest_level_path),
    )

  def test_init_slide_level_map_json_from_defined_level(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.set_icc_profile_bytes(b'badf00d')
    json_metadata = level_map.to_json([level_map.level_map[1]])
    init_from_json = slide_level_map.SlideLevelMap.create_from_json(
        json_metadata
    )
    self.assertEqual(level_map.level_index_min, init_from_json.level_index_min)
    self.assertEqual(level_map.level_index_max, init_from_json.level_index_max)
    self.assertEqual(level_map.level_map, init_from_json.level_map)
    self.assertEqual(
        str(level_map._smallest_level_path),
        str(init_from_json._smallest_level_path),
    )
    self.assertEqual(
        init_from_json.get_icc_profile_bytes(
            dicom_web_interface.DicomWebInterface(
                credential_factory.NoAuthCredentialsFactory()
            )
        ),
        b'badf00d',
    )

  def test_icc_profile_not_saved_in_json_if_exceeds_size_limit(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.set_icc_profile_bytes(
        b'9'
        * (
            slide_level_map.DEFAULT_MAX_JSON_ENCODED_ICC_PROFILE_SIZE_IN_BYTES
            + 1
        )
    )
    json_metadata = level_map.to_json([level_map.level_map[1]])
    init_from_json = slide_level_map.SlideLevelMap.create_from_json(
        json_metadata
    )
    self.assertFalse(init_from_json.is_icc_profile_initialized())

  def test_get_level_sop_instance_uid(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertEqual(
        level_map.level_map[1].get_level_sop_instance_uids(),
        [
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
        ],
    )

  def test_slide_level_init_with_icc_profile_not_initalized(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertFalse(level_map.is_icc_profile_initialized())

  def test_slide_level_init_with_icc_profile_initalized_after_data_set(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.set_icc_profile_bytes(b'badf00d')
    self.assertTrue(level_map.is_icc_profile_initialized())

  def test_slide_level_get_iccprofile_bytes(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.set_icc_profile_bytes(b'badf00d')
    self.assertEqual(
        level_map.get_icc_profile_bytes(
            dicom_web_interface.DicomWebInterface(
                credential_factory.NoAuthCredentialsFactory()
            )
        ),
        b'badf00d',
    )

  def test_icc_profile_bulkdata_uri_empty_if_not_set(self):
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    self.assertEmpty(level_map.level_map[1].icc_profile_bulkdata_uri())

  def test_icc_profile_bulkdata_uri_set(self):
    test_uri = 'http://mock'
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.level_map[1].instances[0] = dataclasses.replace(
        level_map.level_map[1].instances[0],
        dicom_object=dataclasses.replace(
            level_map.level_map[1].instances[0].dicom_object,
            icc_profile_bulkdata_url=test_uri,
        ),
    )
    self.assertEqual(
        level_map.level_map[1].icc_profile_bulkdata_uri(), test_uri
    )

  def test_get_icc_profile_bytes_from_pre_initalized_level(self):
    mock_bytes = b'badf00d'
    level_map = slide_level_map.SlideLevelMap(self.concat_dicom_objects)
    level_map.set_icc_profile_bytes(mock_bytes)
    self.assertEqual(
        level_map.get_icc_profile_bytes(
            dicom_web_interface.DicomWebInterface(
                credential_factory.NoAuthCredentialsFactory()
            )
        ),
        mock_bytes,
    )

  def test_get_root_level_icc_profile_bytes_via_instance_download(self):
    test_icc_profile_bytes = b'badf01dd'
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance.ICCProfile = test_icc_profile_bytes
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=False
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.NoAuthCredentialsFactory()
      )
      level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
      profile_bytes = level_map.get_icc_profile_bytes(dwi)
      self.assertTrue(level_map.is_icc_profile_initialized())
      self.assertEqual(profile_bytes, test_icc_profile_bytes)

  def test_get_root_level_icc_profile_bytes_via_bulkdata(self):
    test_icc_profile_bytes = b'badf01dd'
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance.ICCProfile = test_icc_profile_bytes
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=True
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.NoAuthCredentialsFactory()
      )
      level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
      with mock.patch.object(
          dicom_web_interface.DicomWebInterface,
          'download_instance_untranscoded',
      ) as mock_instance_download:
        profile_bytes = level_map.get_icc_profile_bytes(dwi)
        mock_instance_download.assert_not_called()
      self.assertTrue(level_map.is_icc_profile_initialized())
      self.assertEqual(profile_bytes, test_icc_profile_bytes)

  @parameterized.parameters([True, False])
  def test_get_icc_profile_bytes_returns_empty_if_no_profile_found(
      self, bulkdata_uri_enabled
  ):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=bulkdata_uri_enabled
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.NoAuthCredentialsFactory()
      )
      level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
      profile_bytes = level_map.get_icc_profile_bytes(dwi)
      self.assertTrue(level_map.is_icc_profile_initialized())
      self.assertEqual(profile_bytes, b'')

  @parameterized.parameters([True, False])
  def test_get_icc_profile_bytes_optical_path_sequence(
      self, bulkdata_uri_enabled
  ):
    test_icc_profile_bytes = b'badf01dd'
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    ds = pydicom.Dataset()
    ds.ICCProfile = test_icc_profile_bytes
    test_instance.OpticalPathSequence = [ds]
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=bulkdata_uri_enabled
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.NoAuthCredentialsFactory()
      )
      level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
      profile_bytes = level_map.get_icc_profile_bytes(dwi)
      self.assertTrue(level_map.is_icc_profile_initialized())
      self.assertEqual(profile_bytes, test_icc_profile_bytes)

  @parameterized.parameters(['THUMBNAIL', 'LABEL', 'OVERVIEW'])
  def test_thumbnail_level_icc_profile_bytes(self, image_type):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    test_instance.ImageType = ['ORIGINAL', 'PRIMARY', image_type]
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=True
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.CredentialFactory()
      )
      level_map = slide_level_map.SlideLevelMap(dwi.get_instances(path))
      if image_type == 'THUMBNAIL':
        level = level_map.thumbnail
      elif image_type == 'LABEL':
        level = level_map.label
      else:
        level = level_map.overview
      self.assertEqual(level.width, 234)
      self.assertEqual(level.height, 117)
      self.assertIs(level_map.get_level(level.level_index), level)

  def test_label_overview_thumbnail_images_has_no_images(self):
    images = slide_level_map._LabelOverviewThumbnailImages()
    self.assertFalse(images.has_image())

  @parameterized.parameters(['THUMBNAIL', 'LABEL', 'OVERVIEW'])
  def test_label_overview_thumbnail_images_has_add_image(self, image_type):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    images = slide_level_map._LabelOverviewThumbnailImages()
    images.add_image(
        image_type,
        dicom_object,
        pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
    )
    self.assertTrue(images.has_image())

  @parameterized.parameters(['THUMBNAIL', 'LABEL', 'OVERVIEW'])
  def test_label_overview_thumbnail_images_add_duplicate_image_raises(
      self, image_type
  ):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    images = slide_level_map._LabelOverviewThumbnailImages()
    pixel_spacing_diff_tolerance = (
        pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
    )
    images.add_image(image_type, dicom_object, pixel_spacing_diff_tolerance)
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      images.add_image(image_type, dicom_object, pixel_spacing_diff_tolerance)

  def test_label_overview_thumbnail_images_add_invalid_image_typeraises(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    images = slide_level_map._LabelOverviewThumbnailImages()
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      images.add_image(
          'invalid',
          dicom_object,
          pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
      )

  @parameterized.named_parameters([
      dict(testcase_name='none', json_dict=None),
      dict(testcase_name='empty_dict', json_dict={}),
  ])
  def test_init_label_overview_thumbnail_images_from_empy_json(self, json_dict):
    self.assertFalse(
        slide_level_map._LabelOverviewThumbnailImages.from_dict(
            json_dict
        ).has_image()
    )

  def test_label_overview_thumbnail_images_add_invalid_image_type_raises(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    images = slide_level_map._LabelOverviewThumbnailImages()
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      images.add_image(
          'invalid',
          dicom_object,
          pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
      )

  def test_label_overview_thumbnail_images_add_level_invalid_image_type_raises(
      self,
  ):
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      slide_level_map._LabelOverviewThumbnailImages._add_level(
          slide_level_map._LabelOverviewThumbnailImages(),
          'invalid',
          mock.create_autospec(slide_level_map.Level, instance=True),
      )

  @parameterized.parameters(['THUMBNAIL', 'LABEL', 'OVERVIEW'])
  def test_to_from_json(self, image_type):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 10.0
    test_instance.ImagedVolumeHeight = 5.0
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    images = slide_level_map._LabelOverviewThumbnailImages()
    images.add_image(
        image_type,
        dicom_object,
        pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE,
    )
    new_images = slide_level_map._LabelOverviewThumbnailImages.from_dict(
        images.to_dict()
    )
    self.assertEqual(str(images.label), str(new_images.label))
    self.assertEqual(str(images.thumbnail), str(new_images.thumbnail))
    self.assertEqual(str(images.overview), str(new_images.overview))
    self.assertEqual(new_images.label is None, image_type != 'LABEL')
    self.assertEqual(new_images.overview is None, image_type != 'OVERVIEW')
    self.assertEqual(new_images.thumbnail is None, image_type != 'THUMBNAIL')

  def test_level_defined_for_all_indexs(self):
    images = slide_level_map._LabelOverviewThumbnailImages()
    self.assertEqual(
        set(images._get_key_val_map()),
        set(slide_level_map._LABEL_OVERVIEW_THUMBNAIL_LEVEL_SET),
    )

  def test_create_wsi_label_dicom_number_of_frames_not_1_raises(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.NumberOfFrames = 2
    obj = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      slide_level_map._create_wsi_label_thumbnail_or_overview_level(
          '1', obj, pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
      )

  def test_create_wsi_label_dicom_concatenated_frame_offset_not_0_raises(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ConcatenationFrameOffsetNumber = 2
    obj = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    with self.assertRaises(ez_wsi_errors.DicomSlideInitError):
      slide_level_map._create_wsi_label_thumbnail_or_overview_level(
          '1', obj, pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
      )

  def test_get_pixel_spacing_default_values(self):
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        {},
        '',
    )
    ps = slide_level_map._get_pixel_spacing(
        dicom_object, pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
    )
    self.assertFalse(ps.is_defined)

  def test_get_pixel_spacing_computed_spacing(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.ImagedVolumeWidth = 117
    test_instance.ImagedVolumeHeight = 117
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    ps = slide_level_map._get_pixel_spacing(
        dicom_object, pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
    )
    self.assertEqual((ps.column_spacing_mm, ps.row_spacing_mm), (0.5, 1.0))

  def test_get_pixel_spacing_metadata(self):
    test_instance = pydicom.dcmread(dicom_test_utils.test_dicominstance_path())
    test_instance.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[
        0
    ].PixelSpacing = [0.2, 0.1]
    dicom_object = dicom_web_interface.DicomObject(
        dicom_path.FromString(
            f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/1.2.3/series/1.2.3.4'
        ),
        test_instance.to_json_dict(),
        '',
    )
    ps = slide_level_map._get_pixel_spacing(
        dicom_object, pixel_spacing_module.PIXEL_SPACING_DIFF_TOLERANCE
    )
    self.assertEqual((ps.column_spacing_mm, ps.row_spacing_mm), (0.1, 0.2))

  def _get_test_slide_level_map(self) -> slide_level_map.SlideLevelMap:
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    dicom_store_path_instance_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_store_path_instance_path}'
    )
    path = dicom_path.FromString(
        f'{dicom_store_path_instance_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=True
    ) as mock_dicom_stores:
      mock_dicom_stores[dicom_store_path].add_instance(test_instance)
      dwi = dicom_web_interface.DicomWebInterface(
          credential_factory.CredentialFactory()
      )
      return slide_level_map.SlideLevelMap(dwi.get_instances(path))

  def test_resize_level_by_pixel_spacing(self):
    level = self._get_test_slide_level_map().get_level(1)
    ps = pixel_spacing_module.PixelSpacing(0.5, 0.25)
    self.assertEqual(_round(level.scale_factors(ps)), [17.2598, 8.6301])
    ds = level.resize(ps)
    self.assertEqual((level.width, level.height), (1152, 700))
    self.assertEqual(
        _round([
            level.pixel_spacing.column_spacing_mm,
            level.pixel_spacing.row_spacing_mm,
        ]),
        [0.029, 0.029],
    )
    self.assertEqual((ds.width, ds.height), (66, 81))
    self.assertEqual(
        _round([
            ds.pixel_spacing.column_spacing_mm,
            ds.pixel_spacing.row_spacing_mm,
        ]),
        [0.5, 0.25],
    )
    self.assertEqual(_round(ds.scale_factors()), [17.2598, 8.6301])

  def test_resize_level_by_dim(self):
    level = self._get_test_slide_level_map().get_level(1)
    target_dim = slide_level_map.ImageDimensions(66, 81)
    self.assertEqual(_round(level.scale_factors(target_dim)), [17.4545, 8.642])
    ds = level.resize(target_dim)
    self.assertEqual((level.width, level.height), (1152, 700))
    self.assertEqual(
        _round([
            level.pixel_spacing.column_spacing_mm,
            level.pixel_spacing.row_spacing_mm,
        ]),
        [0.029, 0.029],
    )
    self.assertEqual((ds.width, ds.height), (66, 81))
    self.assertEqual(
        _round([
            ds.pixel_spacing.column_spacing_mm,
            ds.pixel_spacing.row_spacing_mm,
        ]),
        [0.5056, 0.2503],
    )
    self.assertEqual(_round(ds.scale_factors()), [17.4545, 8.642])

  def test_resize_level_by_level_without_ps_produces_undefined_dim(self):
    level = self._get_test_slide_level_map().get_level(1)
    level_with_undefined_ps = dataclasses.replace(
        level, pixel_spacing=pixel_spacing_module.UndefinedPixelSpacing()
    )
    target_dim = slide_level_map.ImageDimensions(66, 81)
    self.assertFalse(
        level_with_undefined_ps.resize(target_dim).pixel_spacing.is_defined
    )

  def test_get_level_sf_for_level_with_no_ps_from_level(self):
    level = self._get_test_slide_level_map().get_level(1)
    level_with_undefined_ps = dataclasses.replace(
        level, pixel_spacing=pixel_spacing_module.UndefinedPixelSpacing()
    )
    # Another set of levels diffent pyramid.
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level_2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    self.assertEqual(
        level_with_undefined_ps.scale_factors(level_2),
        (0.023316062176165803, 0.007029241645244216),
    )

  def test_get_level_sf_for_level_with_no_ps_from_ds_level(self):
    level = self._get_test_slide_level_map().get_level(1)
    level_with_undefined_ps = dataclasses.replace(
        level, pixel_spacing=pixel_spacing_module.UndefinedPixelSpacing()
    )
    # Another set of levels diffent pyramid.
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level_2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    ds = level_2.resize(
        slide_level_map.ImageDimensions(
            level_with_undefined_ps.width / 2,
            level_with_undefined_ps.height / 3,
        )
    )
    self.assertEqual(
        level_with_undefined_ps.scale_factors(ds), (2.0, 3.004291845493562)
    )

  def test_get_level_resize_for_level_with_no_ps_from_ds_level(self):
    level = self._get_test_slide_level_map().get_level(1)
    level_with_undefined_ps = dataclasses.replace(
        level, pixel_spacing=pixel_spacing_module.UndefinedPixelSpacing()
    )
    # Another set of levels different pyramid.
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level_2 = typing.cast(slide_level_map.Level, level_map.get_level(2))
    ds = level_2.resize(
        slide_level_map.ImageDimensions(
            level_with_undefined_ps.width / 2,
            level_with_undefined_ps.height / 3,
        )
    )
    result = level_with_undefined_ps.resize(ds)
    self.assertEqual((result.width, result.height), (576, 233))

  def test_resize_level_equal(self):
    level = self._get_test_slide_level_map().get_level(1)
    resize_level_1 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 2,
            level.height // 3,
        )
    )
    resize_level_2 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 2,
            level.height // 3,
        )
    )
    self.assertEqual(resize_level_1, resize_level_2)

  def test_resize_level_not_equal_different_dim(self):
    level = self._get_test_slide_level_map().get_level(1)
    resize_level_1 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 2,
            level.height // 3,
        )
    )
    resize_level_2 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 3,
            level.height // 5,
        )
    )
    self.assertNotEqual(resize_level_1, resize_level_2)

  def test_resize_level_not_equal_different_source(self):
    level = self._get_test_slide_level_map().get_level(1)
    resize_level_1 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 2,
            level.height // 3,
        )
    )
    level_map = slide_level_map.SlideLevelMap(self.dicom_objects)
    level = typing.cast(slide_level_map.Level, level_map.get_level(2))
    resize_level_2 = level.resize(
        slide_level_map.ImageDimensions(
            resize_level_1.width,
            resize_level_1.height,
        )
    )
    self.assertNotEqual(resize_level_1, resize_level_2)

  def test_resize_level_not_equal_not_level(self):
    level = self._get_test_slide_level_map().get_level(1)
    resize_level_1 = level.resize(
        slide_level_map.ImageDimensions(
            level.width // 2,
            level.height // 3,
        )
    )
    self.assertNotEqual(resize_level_1, 'A')

  def test_slide_level_map_image_map_get_state_re_init_lock(self):
    instance = slide_level_map.SlideLevelMap(self.dicom_objects)
    orig = instance._slide_metadata_lock
    instance_state = instance.__getstate__()
    self.assertNotIn('_slide_metadata_lock', instance_state)
    instance.__setstate__(instance_state)
    self.assertIsNotNone(instance._slide_metadata_lock)
    self.assertIsNot(instance._slide_metadata_lock, orig)

  @parameterized.parameters([
      '',
      ({},),
  ])
  def test_create_untiled_image_map_empty_from_json(self, input_json):
    with self.assertRaises(ez_wsi_errors.NoDicomLevelsDetectedError):
      slide_level_map.UntiledImageMap.create_from_json(input_json)

  @parameterized.parameters([
      '',
      ({},),
  ])
  def test_create_slide_level_map_from_empty_json(self, input_json):
    with self.assertRaises(ez_wsi_errors.NoDicomLevelsDetectedError):
      slide_level_map.SlideLevelMap.create_from_json(input_json)


if __name__ == '__main__':
  absltest.main()
