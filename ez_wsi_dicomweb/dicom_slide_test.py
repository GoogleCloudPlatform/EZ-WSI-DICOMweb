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
"""Tests for ez_wsi_dicomweb.dicom_slide."""
from typing import List, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import abstract_slide_frame_cache
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_test_utils
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import magnification
import numpy as np
import PIL.Image

from hcls_imaging_ml_toolkit import dicom_path


def _fake_get_frame_raw_image(
    unused_param: dicom_path.Path,
    x: int,
    transcode_frame: dicom_web_interface.TranscodeDicomFrame,
) -> bytes:
  """Mocks dicom_web_interface get_frame_image for uncompressed transfer syntax.

  uncompressed transfer syntax = 1.2.840.10008.1.2.1

  Args:
    unused_param: Path to DICOM instance
    x: Frame number.
    transcode_frame: How DICOM frame should be transcoded.

  Returns:
    Mocked image bytes.
  """
  assert (
      transcode_frame
      == dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN
  )
  return bytes(range((x - 1) * 4, x * 4))


class MockFrameCache(abstract_slide_frame_cache.AbstractSlideFrameCache):

  def is_supported_transfer_syntax(self, transfer_syntax: str) -> bool:
    return (
        transfer_syntax
        == dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN.value
    )

  def get_frame(self, instance_path: str, frame_index: int) -> Optional[bytes]:
    return _fake_get_frame_raw_image(
        mock.ANY,
        frame_index,
        dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
    )


def _fake_get_frame_image_jpeg(
    unused_param: dicom_path.Path,
    unused_x: int,
    transcode_frame: dicom_web_interface.TranscodeDicomFrame,
) -> bytes:
  """Mocks dicom_web_interface get_frame_image for jpeg transfer syntax.

  jpeg_transfer_syntax = 1.2.840.10008.1.2.4.50

  Args:
    unused_param: Path to DICOM instance
    unused_x: Frame number.
    transcode_frame: How DICOM frame should be transcoded.

  Returns:
    Mocked image bytes.
  """
  assert (
      transcode_frame
      == dicom_web_interface.TranscodeDicomFrame.DO_NOT_TRANSCODE
  )
  with open(dicom_test_utils.TEST_JPEG_PATH, 'rb') as infile:
    return infile.read()


class DicomSlideTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Load DICOM objects for the testing slide.
    self.mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.SAMPLE_INSTANCES_PATH
    )
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )

  def test_getstate(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )

    self.assertIn('_dwi', slide.__dict__)
    self.assertIn('_server_request_frame_cache', slide.__dict__)
    state = slide.__getstate__()
    self.assertNotIn('_dwi', state)
    self.assertNotIn('_server_request_frame_cache', state)

  def test_dicom_slide_equal(self):
    self.assertEqual(
        dicom_slide.DicomSlide(
            self.mock_dwi,
            self.dicom_series_path,
            enable_client_slide_frame_decompression=True,
        ),
        dicom_slide.DicomSlide(
            self.mock_dwi,
            self.dicom_series_path,
            enable_client_slide_frame_decompression=True,
        ),
    )

  def test_dicom_slide_not_equal_dicom_slide_object(self):
    self.assertNotEqual(
        dicom_slide.DicomSlide(
            self.mock_dwi,
            self.dicom_series_path,
            enable_client_slide_frame_decompression=True,
        ),
        dicom_slide.DicomSlide(
            self.mock_dwi,
            dicom_path.FromString(dicom_test_utils.TEST_DICOM_SERIES_2),
            enable_client_slide_frame_decompression=True,
        ),
    )

  def test_dicom_slide_not_equal_other_object(self):
    self.assertFalse(
        dicom_slide.DicomSlide(
            self.mock_dwi,
            self.dicom_series_path,
            enable_client_slide_frame_decompression=True,
        ).__eq__('foo')
    )

  def test_setstate(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    state = slide.__getstate__()
    origional_frame_cache_instance = slide._server_request_frame_cache
    self.assertIsNotNone(slide.dwi)

    slide.__setstate__(state)
    self.assertIsNone(slide.dwi)
    self.assertIsNot(
        slide._server_request_frame_cache, origional_frame_cache_instance
    )

  def test_dwi_setter_getter(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )

    val = mock.MagicMock()
    slide.dwi = val
    self.assertIs(slide.dwi, val)
    self.assertIs(slide._dwi, val)

  def test_get_magnifications(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertListEqual(
        slide.magnifications,
        [
            magnification.Magnification.FromString('40X'),
            magnification.Magnification.FromString('20X'),
            magnification.Magnification.FromString('10X'),
            magnification.Magnification.FromString('5X'),
            magnification.Magnification.FromString('2.5X'),
            magnification.Magnification.FromString('1.25X'),
            magnification.Magnification.FromString('0.625X'),
            magnification.Magnification.FromString('0.3125X'),
            magnification.Magnification.FromString('0.15625X'),
            magnification.Magnification.FromString('0.078125X'),
        ],
    )

  def test_constructor(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertIsNotNone(slide._level_map)
    self.assertEqual(
        '40X',
        slide.native_magnification.as_string,
        'The native mangification of the test slide must be 40X.',
    )
    self.assertEqual(199168, slide.total_pixel_matrix_rows)  # Tag: 00480007
    self.assertEqual(98816, slide.total_pixel_matrix_columns)  # Tag: 00480006
    self.assertEqual(
        np.uint8, slide.pixel_format, 'The pixel format is incorrect.'
    )

  def test_get_pixel_format_with_valid_input(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_magnification(slide.native_magnification)
    self.assertEqual(
        np.uint8,
        dicom_slide._get_pixel_format(level),
        'The pixel format is not correct.',
    )

  def test_get_pixel_format_with_unsupported_pixel_format_raises_error(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_magnification(slide.native_magnification)
    level.bits_allocated = 32
    with self.assertRaises(ez_wsi_errors.UnsupportedPixelFormatError):
      dicom_slide._get_pixel_format(level)

  def test_get_native_level_with_valid_input(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = dicom_slide._get_native_level(slide._level_map)
    self.assertEqual(
        1, level.level_index, 'The index of the native level is incorrect.'
    )

  def test_get_native_level_with_missing_min_level_raises_error(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide._level_map._level_map[slide._level_map.level_index_min] = None
    with self.assertRaises(ez_wsi_errors.MagnificationLevelNotFoundError):
      dicom_slide._get_native_level(slide._level_map)

  def test_get_level_with_nonexsting_mag_returns_none(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_magnification(
        magnification.Magnification.FromString('80X')
    )
    self.assertIsNone(level, 'There should be no level at magnification 80X.')

  def test_get_level_with_existing_mag_returns_valid_level(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_magnification(
        magnification.Magnification.FromString('10X')
    )
    self.assertIsNotNone(
        level, 'The testing slide should contain a level at magnification 10X.'
    )
    if level is not None:
      self.assertEqual(24704, level.width)
      self.assertEqual(49792, level.height)
      self.assertEqual(500, level.frame_width)
      self.assertEqual(500, level.frame_height)

  @parameterized.parameters(('40X', -1), ('40X', 655360), ('5X', 2047))
  def test_get_frame_with_out_of_range_frame_number_raise_error(
      self, mag: str, frame_number: int
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    with self.assertRaises(ez_wsi_errors.InputFrameNumberOutOfRangeError):
      slide.get_frame(
          magnification.Magnification.FromString(mag),
          frame_number,
      )

  def test_get_frame_normal(self):
    self.mock_dwi.get_frame_image.return_value = b'abc123abc123'
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Set frame size to 2x2 for the first level.
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    frame = slide.get_frame(slide.native_magnification, 1)
    self.assertIsNotNone(
        frame,
        (
            'The testing slide must contain a frame at index 1 on the native '
            'magnification level.'
        ),
    )
    if frame is not None:
      self.assertEqual(
          (2, 0, 2, 2),
          (frame.x_origin, frame.y_origin, frame.width, frame.height),
          'The location and dimension of the frame are not correct.',
      )
      self.assertTrue(
          np.array_equal(
              [[[97, 98, 99], [49, 50, 51]], [[97, 98, 99], [49, 50, 51]]],
              frame.image_np,
          ),
          'The returned frame is not as expected.',
      )

  def test_get_frame_from_cache(self):
    self.mock_dwi.get_frame_image.return_value = b'abc123abc123'
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Set frame size to 2x2 for the first level.
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    frame_1 = slide.get_frame(slide.native_magnification, 1)
    frame_2 = slide.get_frame(slide.native_magnification, 1)
    self.assertIsNotNone(frame_1)
    self.assertIsNotNone(frame_2)
    self.assertTrue(
        np.array_equal(frame_1.image_np, frame_2.image_np),
        'Cached frame not equal to original frame.',
    )
    self.mock_dwi.get_frame_image.assert_called_once()

  def test_get_patch_with_invalid_magnification_raise_error(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    with self.assertRaises(ez_wsi_errors.MagnificationLevelNotFoundError):
      slide.get_patch(magnification.Magnification.FromString('80X'), 0, 0, 1, 1)

  @parameterized.parameters(
      (-4, -4, 4, 4),
      (0, 7, 4, 4),
      (6, 7, 1, 2),
  )
  def test_get_patch_with_out_of_scope_patch_raise_error(
      self, x: int, y: int, width: int, height: int
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values, with frame boundaries, are shown below:
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    with self.assertRaises(ez_wsi_errors.SectionOutOfImageBoundsError):
      slide.get_patch(
          slide.native_magnification, x, y, width, height
      ).image_bytes()

  @parameterized.named_parameters(
      (
          'Align with frame boundary',
          0,
          0,
          4,
          4,
          [0, 1, 4, 5, 2, 3, 6, 7, 12, 13, 16, 17, 14, 15, 18, 19],
      ),
      (
          'Across more than 2 frames in both x and y direction.',
          1,
          1,
          4,
          4,
          [3, 6, 7, 10, 13, 16, 17, 20, 15, 18, 19, 22, 25, 28, 29, 32],
      ),
      ('Single frame', 2, 2, 2, 2, [16, 17, 18, 19]),
      ('Single pixel', 3, 3, 1, 1, [19]),
      ('Single row across multiple frames', 1, 4, 4, 1, [25, 28, 29, 32]),
      (
          'Extends beyond the width of the image.',
          5,
          1,
          2,
          3,
          [11, 0, 21, 0, 23, 0],
      ),
      (
          'Extends beyond the height of the image.',
          0,
          5,
          4,
          2,
          [26, 27, 30, 31, 0, 0, 0, 0],
      ),
      (
          'Extends beyond both the width and the height of the image.',
          4,
          4,
          3,
          3,
          [32, 33, 0, 34, 35, 0, 0, 0, 0],
      ),
      (
          'Starts outside the scope of the image.',
          -1,
          -1,
          3,
          3,
          [0, 0, 0, 0, 0, 1, 0, 2, 3],
      ),
      (
          (
              'Starts outside the scope of the image, and extends beyond the'
              ' scope.'
          ),
          -1,
          4,
          8,
          3,
          [
              0,
              24,
              25,
              28,
              29,
              32,
              33,
              0,
              0,
              26,
              27,
              30,
              31,
              34,
              35,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
          ],
      ),
  )
  def test_get_patch_with_valid_input(
      self, x: int, y: int, width: int, height: int, expected_array: List[int]
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # --------+---------+-------
    #  0   1  |  4   5  |  8  9
    #  2   3  |  6   7  | 10 11
    # --------+---------+-------
    # 12  13  | 16  17  | 20 21
    # 14  15  | 18  19  | 22 23
    # --------+---------+-------
    # 24  25  | 28  29  | 32 33
    # 26  27  | 30  31  | 34 35
    # --------+---------+-------
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.1'
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch = slide.get_patch(slide.native_magnification, x, y, width, height)
    self.assertEqual(patch.magnification, slide.native_magnification)
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    self.assertEqual(
        np.asarray(expected_array, np.uint8).reshape(height, width, 1).tolist(),
        patch.image_bytes().tolist(),
    )

  @parameterized.named_parameters([
      (
          'All frames described at least once',
          [(0, 0, 4, 4), (-1, 4, 8, 3), (-1, -1, 3, 3)],
          [1, 2, 4, 5, 7, 8, 9],
      ),
      (
          'A single patch bounds region describes all frames',
          [(1, 1, 4, 4)],
          [1, 2, 3, 4, 5, 6, 7, 8, 9],
      ),
      (
          'Patch bounds describe frames [5], [5], & [7, 8,9]',
          [(2, 2, 2, 2), (3, 3, 1, 1), (1, 4, 4, 1)],
          [5, 7, 8, 9],
      ),
      (
          'Patches describes subset of image and extend beyond image bounds',
          [(4, 4, 3, 3), (0, 5, 4, 2), (0, 5, 4, 2), (5, 1, 2, 3)],
          [3, 6, 7, 8, 9],
      ),
      (
          'Patches describes subset of image frames with overlap',
          [(-1, 4, 8, 3), (-1, -1, 3, 3)],
          [1, 7, 8, 9],
      ),
  ])
  def test_get_patch_dicom_instances_and_frame_indexes(
      self,
      patch_pos_dim_list: List[Tuple[int, int, int, int]],
      expected_array: List[int],
  ):
    # Tests takes list of frame patch bounding regions and tests that the
    # checks that the returned frame numbers match those in the list.
    # Some of the tested patch bounding regions overlap (i.e. would share common
    # frames. The list retruned should be sorted and not have duplicates.

    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # --------+---------+-------
    #  0   1  |  4   5  |  8  9
    #  2   3  |  6   7  | 10 11
    # --------+---------+-------
    # 12  13  | 16  17  | 20 21
    # 14  15  | 18  19  | 22 23
    # --------+---------+-------
    # 24  25  | 28  29  | 32 33
    # 26  27  | 30  31  | 34 35
    # --------+---------+-------
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.1'
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    count = 1
    for instance in slide._level_map._level_map[1].instances.values():
      path = dicom_path.FromPath(slide.path, instance_uid=str(count))
      instance.dicom_object.path = path
      count += 1
    expected = {
        str(dicom_path.FromPath(slide.path, instance_uid='1')): expected_array
    }
    patch_list = [
        slide.get_patch(slide.native_magnification, x, y, width, height)
        for x, y, width, height in patch_pos_dim_list
    ]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_indexes(
        patch_list[0].magnification,
        [patch.patch_bounds for patch in patch_list],
    )

    self.assertEqual(instance_frame_map, expected)

  def test_get_dicom_instance_frames_across_concat_instances(self):
    mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.INSTANCE_CONCATENATION_TEST_DATA_PATH
    )
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )
    slide = dicom_slide.DicomSlide(
        mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide._level_map._level_map[1].samples_per_pixel = 1
    mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    count = 1
    for instance in slide._level_map._level_map[1].instances.values():
      path = dicom_path.FromPath(slide.path, instance_uid=str(count))
      instance.dicom_object.path = path
      count += 1
    patch_list = [slide.get_patch(slide.native_magnification, 0, 0, 8, 6)]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_indexes(
        patch_list[0].magnification,
        [patch.patch_bounds for patch in patch_list],
    )
    expected = {
        str(dicom_path.FromPath(slide.path, instance_uid='1')): [
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        str(dicom_path.FromPath(slide.path, instance_uid='2')): [
            1,
            2,
            3,
            4,
            5,
            6,
        ],
    }
    self.assertEqual(instance_frame_map, expected)

  def test_get_dicom_instance_invalid_mag(self):
    mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.INSTANCE_CONCATENATION_TEST_DATA_PATH
    )
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )
    slide = dicom_slide.DicomSlide(
        mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide._level_map._level_map[1].samples_per_pixel = 1
    mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    count = 1
    for instance in slide._level_map._level_map[1].instances.values():
      path = dicom_path.FromPath(slide.path, instance_uid=str(count))
      instance.dicom_object.path = path
      count += 1
    patch_list = [slide.get_patch(slide.native_magnification, 0, 0, 8, 6)]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_indexes(
        patch_list[0].magnification.next_higher_magnification,
        [patch.patch_bounds for patch in patch_list],
    )
    self.assertEmpty(instance_frame_map)

  @parameterized.named_parameters(
      ('Align with frame boundary', 0, 0, 4, 4, [0, 1, 3, 4]),
      (
          'Across more than 2 frames in both x and y direction.',
          1,
          1,
          4,
          4,
          [0, 1, 2, 3, 4, 5, 6, 7, 8],
      ),
      ('Single frame.', 2, 2, 2, 2, [4]),
      ('Single pixel', 3, 3, 1, 1, [4]),
      ('Single row across multiple frames', 1, 4, 4, 1, [6, 7, 8]),
      ('Extends beyond the width of the image.', 5, 1, 2, 3, [2, 5]),
      ('Extends beyond the height of the image.', 0, 5, 4, 2, [6, 7]),
      (
          'Extends beyond both the width and the height of the image.',
          4,
          4,
          3,
          3,
          [8],
      ),
      ('Starts outside the scope of the image.', -1, -1, 3, 3, [0]),
      (
          (
              'Starts outside the scope of the image, and extends beyond the'
              ' scope.'
          ),
          -1,
          4,
          8,
          3,
          [6, 7, 8],
      ),
  )
  def test_get_patch_frame_numbers_with_valid_input(
      self, x: int, y: int, width: int, height: int, expected_array: List[int]
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # ----+----+-----
    #  1  |  2 |  3
    # ----+----+-----
    #  4  |  5 |  6
    # ----+----+-----
    #  7  |  8 |  9
    # ----+----+-----
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.1'
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch = slide.get_patch(slide.native_magnification, x, y, width, height)
    self.assertEqual(patch.magnification, slide.native_magnification)
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    # sort order is expected
    self.assertEqual(expected_array, list(patch.frame_number()))

  def test_get_image(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # --------+---------+-------
    #  0   1  |  4   5  |  8  9
    #  2   3  |  6   7  | 10 11
    # --------+---------+-------
    # 12  13  | 16  17  | 20 21
    # 14  15  | 18  19  | 22 23
    # --------+---------+-------
    # 24  25  | 28  29  | 32 33
    # 26  27  | 30  31  | 34 35
    # --------+---------+-------
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.1'
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    image = slide.get_image(slide.native_magnification)
    self.assertEqual(image.magnification, slide.native_magnification)
    self.assertEqual([image.width, image.height], [6, 6])
    self.assertEqual(
        np.asarray(
            [
                0,
                1,
                4,
                5,
                8,
                9,
                2,
                3,
                6,
                7,
                10,
                11,
                12,
                13,
                16,
                17,
                20,
                21,
                14,
                15,
                18,
                19,
                22,
                23,
                24,
                25,
                28,
                29,
                32,
                33,
                26,
                27,
                30,
                31,
                34,
                35,
            ],
            np.uint8,
        )
        .reshape(6, 6, 1)
        .tolist(),
        image.image_bytes().tolist(),
    )
    self.assertEqual(self.mock_dwi.get_frame_image.call_count, 9)

  def test_set_dicom_slide_frame_cache(self):
    mock_frame_cache = MockFrameCache()
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = mock_frame_cache
    self.assertIs(slide.slide_frame_cache, mock_frame_cache)

  def test_get_image_from_slide_frame_cache_by_instance(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = MockFrameCache()
    level = slide._level_map._level_map[1]
    self.assertEqual(
        slide._get_cached_frame_bytes(level, level.instances[0], 1),
        b'\x00\x01\x02\x03',
    )

  def test_get_image_from_slide_frame_cache_by_path(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = MockFrameCache()
    level = slide._level_map._level_map[1]
    self.assertEqual(
        slide._get_cached_frame_bytes(
            level, level.instances[0].dicom_object.path, 2
        ),
        b'\x04\x05\x06\x07',
    )

  def test_get_image_from_slide_frame_cache_by_invalid_path_type(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = MockFrameCache()
    level = slide._level_map._level_map[1]
    self.assertIsNone(slide._get_cached_frame_bytes(level, slide, 1))  # pytype: disable=wrong-arg-types

  def test_get_image_from_slide_frame_cache(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = MockFrameCache()
    # Create an image at the native level, with the following properties:
    # frame size = 2 x 2
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # --------+---------+-------
    #  0   1  |  4   5  |  8  9
    #  2   3  |  6   7  | 10 11
    # --------+---------+-------
    # 12  13  | 16  17  | 20 21
    # 14  15  | 18  19  | 22 23
    # --------+---------+-------
    # 24  25  | 28  29  | 32 33
    # 26  27  | 30  31  | 34 35
    # --------+---------+-------
    slide._level_map._level_map[1].width = 6
    slide._level_map._level_map[1].height = 6
    slide._level_map._level_map[1].frame_width = 2
    slide._level_map._level_map[1].frame_height = 2
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 8
    slide._level_map._level_map[1].samples_per_pixel = 1
    slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.1'
    image = slide.get_image(slide.native_magnification)
    self.assertEqual(image.magnification, slide.native_magnification)
    self.assertEqual([image.width, image.height], [6, 6])
    self.assertEqual(
        np.asarray(
            [
                0,
                1,
                4,
                5,
                8,
                9,
                2,
                3,
                6,
                7,
                10,
                11,
                12,
                13,
                16,
                17,
                20,
                21,
                14,
                15,
                18,
                19,
                22,
                23,
                24,
                25,
                28,
                29,
                32,
                33,
                26,
                27,
                30,
                31,
                34,
                35,
            ],
            np.uint8,
        )
        .reshape(6, 6, 1)
        .tolist(),
        image.image_bytes().tolist(),
    )
    self.assertEqual(self.mock_dwi.get_frame_image.call_count, 0)

  @parameterized.parameters((3, 3, 2, 2), (4, 5, 3, 3), (-1, -1, 1, 1))
  def test_copy_overlapped_region_with_invalid_input_raise_error(
      self, dst_x: int, dst_y: int, dst_width: int, dst_height: int
  ):
    mag = magnification.Magnification.FromString('40X')
    src_frame = dicom_slide.Frame(0, 0, 3, 3, np.ndarray((3, 3, 3), np.uint8))
    dst_patch = dicom_slide.Patch(mag, dst_x, dst_y, dst_width, dst_height)
    with self.assertRaises(ez_wsi_errors.PatchIntersectionNotFoundError):
      dst_np = np.ndarray((dst_height, dst_width, 3), np.uint8)
      dst_patch._copy_overlapped_region(src_frame, dst_np)

  @parameterized.parameters(
      (1, 0, 2, 1, [[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [2, 2, 2]]]),
      (1, 1, 2, 2, [[[1, 1, 1], [2, 2, 2]], [[4, 4, 4], [5, 5, 5]]]),
      (3, 2, 1, 2, [[[6, 6, 6], [0, 0, 0]], [[9, 9, 9], [0, 0, 0]]]),
  )
  def test_copy_overlapped_region_with_valid(
      self,
      dst_x: int,
      dst_y: int,
      expected_region_width: int,
      expected_region_height: int,
      expected_array: np.ndarray,
  ):
    mag = magnification.Magnification.FromString('40X')
    src_frame = dicom_slide.Frame(
        1,
        1,
        3,
        3,
        np.array(
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            np.uint8,
        ),
    )
    dst_patch = dicom_slide.Patch(mag, dst_x, dst_y, 2, 2)
    dst_np = np.zeros((2, 2, 3), np.uint8)

    region_width, region_height = dst_patch._copy_overlapped_region(
        src_frame, dst_np
    )

    self.assertEqual(
        (expected_region_width, expected_region_height),
        (region_width, region_height),
    )
    self.assertTrue(np.array_equal(expected_array, dst_np))

  @parameterized.parameters(
      (1, 1, 3, 3, 0, 0), (1, 1, 2, 2, 1, 1), (1, 1, 3, 3, 0, 0)
  )
  def test_copy_ndarray_with_invalid_input_raise_error(
      self,
      src_x: int,
      src_y: int,
      width: int,
      height: int,
      dst_x: int,
      dst_y: int,
  ):
    src_array = np.ndarray((3, 3, 3), np.uint8)
    dst_array = np.ndarray((2, 2, 3), np.uint8)
    with self.assertRaises(ez_wsi_errors.SectionOutOfImageBoundsError):
      dicom_slide._copy_ndarray(
          src_array, src_x, src_y, width, height, dst_array, dst_x, dst_y
      )

  @parameterized.parameters(
      (1, 1, 2, 2, 0, 0, [[[5, 5, 5], [6, 6, 6]], [[8, 8, 8], [9, 9, 9]]]),
      (1, 1, 1, 2, 1, 0, [[[0, 0, 0], [5, 5, 5]], [[0, 0, 0], [8, 8, 8]]]),
  )
  def test_copy_ndarray_with_valid_input(
      self,
      src_x: int,
      src_y: int,
      width: int,
      height: int,
      dst_x: int,
      dst_y: int,
      expected_array: np.ndarray,
  ):
    src_array = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
        ],
        np.uint8,
    )
    dst_array = np.zeros((2, 2, 3), np.uint8)
    dicom_slide._copy_ndarray(
        src_array, src_x, src_y, width, height, dst_array, dst_x, dst_y
    )
    self.assertTrue(np.array_equal(expected_array, dst_array))

  def test_get_slide_patch_id(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        accession_number='slideid',
        enable_client_slide_frame_decompression=True,
    )
    self.assertEqual('slideid', slide.accession_number)
    patch = slide.get_patch(
        magnification.Magnification.FromString('10X'),
        x=10,
        y=20,
        width=100,
        height=200,
    )
    self.assertEqual('slideid:M_10X:000100x000200+000010+000020', patch.id)

  def test_get_patch_from_jpeg(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    # Create an image at the native level, with the following properties:
    # frame size = 156 x 455
    # image size = 6 x 6
    # frame numbers: 0~8
    # samples per pixel = 1
    # The image pixel values are generated on-the-fly by fake_get_frame_image().
    # The actual pixel values, with frame boundaries, are shown below:
    #
    # ---+----+----
    #  0 |  2 |  3
    # ---+----+----
    #  4 |  5 |  6
    # ---+----+----
    #  7 |  8 |  9
    # ---+----+----
    slide._level_map._level_map[1].width = 1365
    slide._level_map._level_map[1].height = 468
    slide._level_map._level_map[1].frame_width = 455
    slide._level_map._level_map[1].frame_height = 156
    slide._level_map._level_map[1].frame_number_min = 0
    slide._level_map._level_map[1].frame_number_max = 3
    slide._level_map._level_map[1].samples_per_pixel = 3
    slide._level_map._level_map[1].transfer_syntax_uid = (
        '1.2.840.10008.1.2.4.50'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_image_jpeg
    x = 100
    y = 50
    width = 300
    height = 100
    expected_array = np.asarray(PIL.Image.open(dicom_test_utils.TEST_JPEG_PATH))
    expected_array = expected_array[y : y + height, x : x + width, :]

    patch = slide.get_patch(slide.native_magnification, x, y, width, height)
    self.assertEqual(patch.magnification, slide.native_magnification)
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    self.assertTrue(np.array_equal(expected_array, patch.image_bytes()))
    self.mock_dwi.get_frame_image.assert_called_once()


def test_get_patch_from_jpeg_locally_fails_over_server_side_decoding(self):
  slide = dicom_slide.DicomSlide(
      self.mock_dwi,
      self.dicom_series_path,
      enable_client_slide_frame_decompression=True,
  )
  # Create an image at the native level, with the following properties:
  # frame size = 156 x 455
  # image size = 6 x 6
  # frame numbers: 0~8
  # samples per pixel = 1
  # The image pixel values are generated on-the-fly by fake_get_frame_image().
  # The actual pixel values, with frame boundaries, are shown below:
  #
  # --------+---------+-------
  #  0   1  |  4   5  |  8  9
  #  2   3  |  6   7  | 10 11
  # --------+---------+-------
  # 12  13  | 16  17  | 20 21
  # 14  15  | 18  19  | 22 23
  # --------+---------+-------
  # 24  25  | 28  29  | 32 33
  # 26  27  | 30  31  | 34 35
  # --------+---------+-------
  slide._level_map._level_map[1].width = 6
  slide._level_map._level_map[1].height = 6
  slide._level_map._level_map[1].frame_width = 2
  slide._level_map._level_map[1].frame_height = 2
  slide._level_map._level_map[1].frame_number_min = 0
  slide._level_map._level_map[1].frame_number_max = 8
  slide._level_map._level_map[1].samples_per_pixel = 1
  slide._level_map._level_map[1].transfer_syntax_uid = '1.2.840.10008.1.2.4.50'
  self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
  x = 1
  y = 2
  width = 4
  height = 2
  expected_array = np.asarray(
      [13, 16, 17, 20, 15, 18, 19, 22], np.uint8
  ).reshape(height, width, 1)

  patch = slide.get_patch(slide.native_magnification, x, y, width, height)
  self.assertEqual(patch.magnification, slide.native_magnification)
  self.assertEqual(
      [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
  )
  self.assertTrue(np.array_equal(expected_array, patch.image_bytes()))
  self.assertEqual(self.mock_dwi.get_frame_image.call_count, 2)


if __name__ == '__main__':
  absltest.main()
