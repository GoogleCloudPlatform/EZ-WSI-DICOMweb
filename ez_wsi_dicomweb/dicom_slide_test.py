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
import collections
import dataclasses
import typing
from typing import Iterator, List, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import dicomweb_credential_factory
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
from hcls_imaging_ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
import PIL.Image


def _init_test_slide_level_map(
    slide: dicom_slide.DicomSlide,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
    frame_number_min: int,
    frame_number_max: int,
    samples_per_pixel: int,
    transfer_syntax_uid: str,
    mock_path: bool = False,
):
  new_map = collections.OrderedDict()
  if 1 not in slide._level_map._level_map:
    spacing_placeholder = 0.01
    instances = {}
    new_map[1] = slide_level_map.Level(
        1,
        width,
        height,
        samples_per_pixel,
        8,
        7,
        spacing_placeholder,
        spacing_placeholder,
        frame_width,
        frame_height,
        frame_number_min,
        frame_number_max,
        instances,
        transfer_syntax_uid,
    )
  else:
    for key, value in slide._level_map._level_map.items():
      if key == 1:
        new_map[key] = dataclasses.replace(
            slide._level_map._level_map[key],
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_number_min=frame_number_min,
            frame_number_max=frame_number_max,
            samples_per_pixel=samples_per_pixel,
            transfer_syntax_uid=transfer_syntax_uid,
        )
      else:
        new_map[key] = value
  count = 1
  updated_instances = collections.OrderedDict()
  if new_map[1].instances is not None:  # pytype: disable=attribute-error
    for key, instance in new_map[1].instances.items():  # pytype: disable=attribute-error
      if mock_path:
        dicom_object = dataclasses.replace(
            instance.dicom_object,
            path=dicom_path.FromPath(slide.path, instance_uid=str(count)),
        )
      else:
        dicom_object = instance.dicom_object
      updated_instances[key] = dataclasses.replace(
          instance,
          dicom_object=dicom_object,
      )
      count += 1
    new_map[1] = dataclasses.replace(new_map[1], instances=updated_instances)
  slide._level_map._level_map = new_map
  return slide


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


class MockFrameCache:

  def is_supported_transfer_syntax(self, transfer_syntax: str) -> bool:
    return (
        transfer_syntax
        == dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN.value
    )

  def get_frame(
      self, instance_path: str, frame_count: int, frame_number: int
  ) -> Optional[bytes]:
    del instance_path, frame_count
    return _fake_get_frame_raw_image(
        mock.ANY,
        frame_number,
        dicom_web_interface.TranscodeDicomFrame.UNCOMPRESSED_LITTLE_ENDIAN,
    )


def _create_mock_frame_cache() -> (
    local_dicom_slide_cache.InMemoryDicomSlideCache
):
  return typing.cast(
      local_dicom_slide_cache.InMemoryDicomSlideCache, MockFrameCache()
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
  with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
    return infile.read()


class DicomSlideTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Load DICOM objects for the testing slide.
    self.mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.sample_instances_path()
    )
    self.mock_dwi.get_frame_image = mock.MagicMock()
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )

  @parameterized.named_parameters(
      [
          dict(
              testcase_name='jpeg_baseline',
              transfer_syntax='1.2.840.10008.1.2.4.50',
              expected=True,
          ),
          dict(
              testcase_name='jpeg_2000_lossless',
              transfer_syntax='1.2.840.10008.1.2.4.90',
              expected=True,
          ),
          dict(
              testcase_name='jpeg_2000',
              transfer_syntax='1.2.840.10008.1.2.4.91',
              expected=True,
          ),
          dict(
              testcase_name='explicit_vr_little_endian',
              transfer_syntax='1.2.840.10008.1.2.1',
              expected=True,
          ),
          dict(
              testcase_name='implicit_vr_endian_default',
              transfer_syntax='1.2.840.10008.1.2',
              expected=True,
          ),
          dict(
              testcase_name='private_syntax',
              transfer_syntax='1.2.3',
              expected=False,
          ),
          dict(
              testcase_name='deflated_explicit_vr_little_endian',
              transfer_syntax='1.2.840.10008.1.2.1.99',
              expected=False,
          ),
          dict(
              testcase_name='explicit_vr_big_endian',
              transfer_syntax='1.2.840.10008.1.2.2',
              expected=False,
          ),
          dict(
              testcase_name='jpeg_baseline_process_2_and_4',
              transfer_syntax='1.2.840.10008.1.2.4.51',
              expected=False,
          ),
          dict(
              testcase_name='jpeg_lossless_nonhierarchical',
              transfer_syntax='1.2.840.10008.1.2.4.70',
              expected=False,
          ),
          dict(
              testcase_name='jpeg-ls_lossless_image_compression',
              transfer_syntax='1.2.840.10008.1.2.4.80',
              expected=False,
          ),
          dict(
              testcase_name='jpeg-ls_lossy_near-lossless_image_compression',
              transfer_syntax='1.2.840.10008.1.2.4.81',
              expected=False,
          ),
          dict(
              testcase_name='jpeg_2000_part_2_multicomponent_image_compression_lossless_only',
              transfer_syntax='1.2.840.10008.1.2.4.92',
              expected=False,
          ),
          dict(
              testcase_name='jpeg_2000_part_2_multicomponent_image_compression',
              transfer_syntax='1.2.840.10008.1.2.4.93',
              expected=False,
          ),
          dict(
              testcase_name='jpip_referenced',
              transfer_syntax='1.2.840.10008.1.2.4.94',
              expected=False,
          ),
          dict(
              testcase_name='jpip_referenced_deflate',
              transfer_syntax='1.2.840.10008.1.2.4.95',
              expected=False,
          ),
          dict(
              testcase_name='rle_lossless',
              transfer_syntax='1.2.840.10008.1.2.5',
              expected=False,
          ),
          dict(
              testcase_name='rfc_2557_mime_encapsulation',
              transfer_syntax='1.2.840.10008.1.2.6.1',
              expected=False,
          ),
      ],
  )
  def test_is_client_side_pixel_decoding_supported(
      self, transfer_syntax: str, expected: bool
  ):
    self.assertEqual(
        dicom_slide.is_client_side_pixel_decoding_supported(transfer_syntax),
        expected,
    )

  def test_dicom_slide_logger_default_initalization(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertIsNotNone(slide.logger)

  def test_dicom_slide_logger_default_init_once(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    instance = slide.logger
    self.assertIs(instance, slide.logger)

  def test_dicom_slide_logger_logging_factory(self):
    mock_factory = mock.create_autospec(
        ez_wsi_logging_factory.BasePythonLoggerFactory
    )
    mock_factory.create_logger.return_value = None
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
        logging_factory=mock_factory,
    )
    _ = slide.logger
    mock_factory.create_logger.assert_called_once()

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

  def test_get_ps(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertListEqual(
        slide.all_pixel_spacing_mms,
        [
            pixel_spacing.PixelSpacing.FromDouble(
                0.00024309399214433264
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.0004861879842886653
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.0009723759685773306
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.0019447519371546612
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.0038895038743093223
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.007779007748618645
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.01555801549723729
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.03111603099447458
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.06223206198894916
            ).pixel_spacing_mm,
            pixel_spacing.PixelSpacing.FromDouble(
                0.12446412397789831
            ).pixel_spacing_mm,
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
        '41.13635117994051X',
        slide.native_pixel_spacing.as_magnification_string,
        'The native pixel spacing of the test slide must be 40X.',
    )
    self.assertEqual(199168, slide.total_pixel_matrix_rows)  # Tag: 00480007
    self.assertEqual(98816, slide.total_pixel_matrix_columns)  # Tag: 00480006
    self.assertEqual(
        np.uint8, slide.pixel_format, 'The pixel format is incorrect.'
    )

  def test_unsupported_pixel_spacing_raises(self):
    with self.assertRaises(ez_wsi_errors.NonSquarePixelError):
      dicom_slide.DicomSlide(
          self.mock_dwi,
          self.dicom_series_path,
          enable_client_slide_frame_decompression=True,
          pixel_spacing_diff_tolerance=0.0,
      )

  def test_get_pixel_format_with_valid_input(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
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
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    level = dataclasses.replace(level, bits_allocated=32)
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
    slide._level_map._level_map[slide._level_map.level_index_min] = (
        None  # pytype: disable=unsupported-operands  # always-use-return-annotations
    )
    with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
      dicom_slide._get_native_level(slide._level_map)

  def test_get_level_with_nonexsting_ps_returns_none(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_pixel_spacing(
        pixel_spacing.PixelSpacing.FromMagnificationString('80X')
    )
    self.assertIsNone(level, 'There should be no level at pixel spacing 80X.')

  def test_get_level_with_existing_ps_returns_valid_level(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_level_by_pixel_spacing(
        pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    )
    self.assertIsNotNone(
        level, 'The testing slide should contain a level at ps 10X.'
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
          pixel_spacing.PixelSpacing.FromMagnificationString(mag),
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
    level_1 = slide._level_map._level_map[1]
    _init_test_slide_level_map(
        slide,
        level_1.width,
        level_1.height,
        2,
        2,
        level_1.frame_number_min,
        level_1.frame_number_max,
        level_1.samples_per_pixel,
        level_1.transfer_syntax_uid,
    )
    frame = slide.get_frame(slide.native_pixel_spacing, 1)
    self.assertIsNotNone(
        frame,
        (
            'The testing slide must contain a frame at index 1 on the native '
            'pixel spacing.'
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

  @parameterized.named_parameters([
      dict(
          testcase_name='explicit_vr_little_endian',
          transfer_syntax_uid='1.2.840.10008.1.2.1',
          expected_mock_get_frame_from_cache_call_count=1,
          expected_frame_bytes=49,
      ),
      dict(
          testcase_name='implicit_vr_endian',
          transfer_syntax_uid='1.2.840.10008.1.2',
          expected_mock_get_frame_from_cache_call_count=1,
          expected_frame_bytes=49,
      ),
      dict(
          testcase_name='deflated_explicit_vr_little_endian',
          transfer_syntax_uid='1.2.840.10008.1.2.1.99',
          expected_mock_get_frame_from_cache_call_count=0,
          expected_frame_bytes=50,
      ),
      dict(
          testcase_name='explicit_vr_big_endian',
          transfer_syntax_uid='1.2.840.10008.1.2.2',
          expected_mock_get_frame_from_cache_call_count=0,
          expected_frame_bytes=50,
      ),
      dict(
          testcase_name='jpeg_baseline_process_1',
          transfer_syntax_uid='1.2.840.10008.1.2.4.50',
          expected_mock_get_frame_from_cache_call_count=0,
          expected_frame_bytes=50,
      ),
      dict(
          testcase_name='unknown_transfer_syntax',
          transfer_syntax_uid='1.2.3',
          expected_mock_get_frame_from_cache_call_count=0,
          expected_frame_bytes=50,
      ),
  ])
  def test_get_frame_server_side_use_cache_only_if_image_in_raw_transfer_syntax(
      self,
      transfer_syntax_uid,
      expected_mock_get_frame_from_cache_call_count,
      expected_frame_bytes,
  ):
    with mock.patch.object(
        dicom_slide.DicomSlide,
        '_get_frame_bytes_from_server',
        return_value=b'2',
    ) as mock_get_frame_from_server:
      with mock.patch.object(
          dicom_slide.DicomSlide,
          '_get_cached_frame_bytes',
          return_value=b'1',
      ) as mock_get_frame_from_cache:
        slide = dicom_slide.DicomSlide(
            self.mock_dwi,
            self.dicom_series_path,
            enable_client_slide_frame_decompression=True,
        )
        _init_test_slide_level_map(
            slide, 6, 6, 1, 1, 0, 8, 1, transfer_syntax_uid
        )
        level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
        result = slide._get_frame_server_transcoding(
            level, level.instances[0], 1  # pytype: disable=attribute-error
        )
        self.assertEqual(
            mock_get_frame_from_cache.call_count,
            expected_mock_get_frame_from_cache_call_count,
        )
        self.assertEqual(
            mock_get_frame_from_server.call_count,
            1 - expected_mock_get_frame_from_cache_call_count,
        )
    self.assertEqual(result.tolist(), [[[expected_frame_bytes]]])

  def test_server_side_transcoding_frame_cache_unsupported_transfer_syntax(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=False,
    )
    transfer_syntax_which_requires_server_side_decoding = (
        '1.2.840.10008.1.2.​4.​106'
    )
    self.mock_dwi.get_frame_image.return_value = b'abc123abc123'
    slide.slide_frame_cache = local_dicom_slide_cache.InMemoryDicomSlideCache(
        dicomweb_credential_factory.CredentialFactory()
    )
    # Set frame size to 2x2 for the first level.
    level_1 = slide._level_map._level_map[1]
    _init_test_slide_level_map(
        slide,
        level_1.width,  # pytype: disable=attribute-error
        level_1.height,  # pytype: disable=attribute-error
        2,
        2,
        level_1.frame_number_min,  # pytype: disable=attribute-error
        level_1.frame_number_max,  # pytype: disable=attribute-error
        level_1.samples_per_pixel,  # pytype: disable=attribute-error
        transfer_syntax_which_requires_server_side_decoding,
    )
    frame_1 = slide.get_frame(slide.native_pixel_spacing, 1)
    frame_2 = slide.get_frame(slide.native_pixel_spacing, 1)
    self.assertIsNotNone(frame_1)
    self.assertIsNotNone(frame_2)
    self.assertTrue(
        np.array_equal(frame_1.image_np, frame_2.image_np),  # pytype: disable=attribute-error
        'Cached frame not equal to original frame.',
    )
    self.mock_dwi.get_frame_image.assert_called_once()

  def test_get_patch_with_invalid_pixel_spacing_raise_error(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    with self.assertRaises(ez_wsi_errors.PixelSpacingLevelNotFoundError):
      slide.get_patch(
          pixel_spacing.PixelSpacing.FromMagnificationString('80X'), 0, 0, 1, 1
      )

  @parameterized.named_parameters([
      dict(testcase_name='beyond_upper_left', x=-4, y=-4, width=4, height=4),
      dict(testcase_name='below_image_1', x=0, y=7, width=4, height=4),
      dict(testcase_name='below_image_2', x=6, y=7, width=1, height=2),
  ])
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    with self.assertRaises(ez_wsi_errors.SectionOutOfImageBoundsError):
      slide.get_patch(
          slide.native_pixel_spacing, x, y, width, height
      ).image_bytes()

  @parameterized.named_parameters(
      dict(
          testcase_name='align_with_frame_boundary',
          x=0,
          y=0,
          width=4,
          height=4,
          expected_array=[
              0,
              1,
              4,
              5,
              2,
              3,
              6,
              7,
              12,
              13,
              16,
              17,
              14,
              15,
              18,
              19,
          ],
      ),
      dict(
          testcase_name='across_more_than_2_frames_in_both_x_and_y_direction',
          x=1,
          y=1,
          width=4,
          height=4,
          expected_array=[
              3,
              6,
              7,
              10,
              13,
              16,
              17,
              20,
              15,
              18,
              19,
              22,
              25,
              28,
              29,
              32,
          ],
      ),
      dict(
          testcase_name='single_frame',
          x=2,
          y=2,
          width=2,
          height=2,
          expected_array=[16, 17, 18, 19],
      ),
      dict(
          testcase_name='single_pixel',
          x=3,
          y=3,
          width=1,
          height=1,
          expected_array=[19],
      ),
      dict(
          testcase_name='single_row_across_multiple_frames',
          x=1,
          y=4,
          width=4,
          height=1,
          expected_array=[25, 28, 29, 32],
      ),
      dict(
          testcase_name='extends_beyond_the_width_of_the_image',
          x=5,
          y=1,
          width=2,
          height=3,
          expected_array=[11, 0, 21, 0, 23, 0],
      ),
      dict(
          testcase_name='extends_beyond_the_height_of_the_image',
          x=0,
          y=5,
          width=4,
          height=2,
          expected_array=[26, 27, 30, 31, 0, 0, 0, 0],
      ),
      dict(
          testcase_name=(
              'extends_beyond_both_the_width_and_the_height_of_the_image'
          ),
          x=4,
          y=4,
          width=3,
          height=3,
          expected_array=[32, 33, 0, 34, 35, 0, 0, 0, 0],
      ),
      dict(
          testcase_name='starts_outside_the_scope_of_the_image',
          x=-1,
          y=-1,
          width=3,
          height=3,
          expected_array=[0, 0, 0, 0, 0, 1, 0, 2, 3],
      ),
      dict(
          testcase_name=(
              'starts_outside_the_scope_of_the_image_and_extends_beyond_the_'
              'scope'
          ),
          x=-1,
          y=4,
          width=8,
          height=3,
          expected_array=[
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch = slide.get_patch(slide.native_pixel_spacing, x, y, width, height)
    self.assertEqual(
        patch.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    self.assertEqual(
        np.asarray(expected_array, np.uint8).reshape(height, width, 1).tolist(),
        patch.image_bytes().tolist(),
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='all_frames_described_at_least_once',
          patch_pos_dim_list=[(0, 0, 4, 4), (-1, 4, 8, 3), (-1, -1, 3, 3)],
          expected_array=[1, 2, 4, 5, 7, 8, 9],
      ),
      dict(
          testcase_name='a_single_patch_bounds_region_describes_all_frames',
          patch_pos_dim_list=[(1, 1, 4, 4)],
          expected_array=[1, 2, 3, 4, 5, 6, 7, 8, 9],
      ),
      dict(
          testcase_name='patch_bounds_describe_frames_[5]_[5]_[7,8,9]',
          patch_pos_dim_list=[(2, 2, 2, 2), (3, 3, 1, 1), (1, 4, 4, 1)],
          expected_array=[5, 7, 8, 9],
      ),
      dict(
          testcase_name=(
              'patches_describes_subset_of_image_and_extend_beyond_image_bounds'
          ),
          patch_pos_dim_list=[
              (4, 4, 3, 3),
              (0, 5, 4, 2),
              (0, 5, 4, 2),
              (5, 1, 2, 3),
          ],
          expected_array=[3, 6, 7, 8, 9],
      ),
      dict(
          testcase_name='patches_describes_subset_of_image_frames_with_overlap',
          patch_pos_dim_list=[(-1, 4, 8, 3), (-1, -1, 3, 3)],
          expected_array=[1, 7, 8, 9],
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1', mock_path=True
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    expected = {
        str(dicom_path.FromPath(slide.path, instance_uid='1')): expected_array
    }
    patch_list = [
        slide.get_patch(slide.native_pixel_spacing, x, y, width, height)
        for x, y, width, height in patch_pos_dim_list
    ]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_numbers(
        patch_list[0].pixel_spacing,
        [patch.patch_bounds for patch in patch_list],
    )

    self.assertEqual(instance_frame_map, expected)

  def test_get_dicom_instance_frames_across_concat_instances(self):
    mock_dwi = dicom_test_utils.create_mock_dicom_web_interface(
        dicom_test_utils.instance_concatenation_test_data_path()
    )
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )
    slide = dicom_slide.DicomSlide(
        mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level_1 = slide._level_map._level_map[1]
    _init_test_slide_level_map(
        slide,
        level_1.width,
        level_1.height,
        level_1.frame_width,
        level_1.frame_height,
        level_1.frame_number_min,
        level_1.frame_number_max,
        1,
        level_1.transfer_syntax_uid,
        mock_path=True,
    )
    mock_dwi.get_frame_image = mock.MagicMock()
    mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch_list = [slide.get_patch(slide.native_pixel_spacing, 0, 0, 8, 6)]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_numbers(
        patch_list[0].pixel_spacing,
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
        dicom_test_utils.instance_concatenation_test_data_path()
    )
    self.dicom_series_path = dicom_path.FromString(
        dicom_test_utils.TEST_DICOM_SERIES
    )
    slide = dicom_slide.DicomSlide(
        mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level_1 = slide._level_map._level_map[1]
    _init_test_slide_level_map(
        slide,
        level_1.width,
        level_1.height,
        level_1.frame_width,
        level_1.frame_height,
        level_1.frame_number_min,
        level_1.frame_number_max,
        1,
        level_1.transfer_syntax_uid,
        mock_path=True,
    )
    mock_dwi.get_frame_image = mock.MagicMock()
    mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch_list = [slide.get_patch(slide.native_pixel_spacing, 0, 0, 8, 6)]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_numbers(
        pixel_spacing.PixelSpacing.FromMagnificationString('500X'),
        [patch.patch_bounds for patch in patch_list],
    )
    self.assertEmpty(instance_frame_map)

  @parameterized.named_parameters(
      dict(
          testcase_name='align_with_frame_boundary',
          x=0,
          y=0,
          width=4,
          height=4,
          expected_array=[0, 1, 3, 4],
      ),
      dict(
          testcase_name='across_more_than_2_frames_in_both_x_and_y_direction',
          x=1,
          y=1,
          width=4,
          height=4,
          expected_array=[0, 1, 2, 3, 4, 5, 6, 7, 8],
      ),
      dict(
          testcase_name='single_frame',
          x=2,
          y=2,
          width=2,
          height=2,
          expected_array=[4],
      ),
      dict(
          testcase_name='single_ pixel',
          x=3,
          y=3,
          width=1,
          height=1,
          expected_array=[4],
      ),
      dict(
          testcase_name='single_row_across_multiple_frames',
          x=1,
          y=4,
          width=4,
          height=1,
          expected_array=[6, 7, 8],
      ),
      dict(
          testcase_name='extends_beyond_the_width_of_the_image',
          x=5,
          y=1,
          width=2,
          height=3,
          expected_array=[2, 5],
      ),
      dict(
          testcase_name='extends_beyond_the_height_of_the_image',
          x=0,
          y=5,
          width=4,
          height=2,
          expected_array=[6, 7],
      ),
      dict(
          testcase_name=(
              'extends_beyond_both_the_width_and_the_height_of_the_image'
          ),
          x=4,
          y=4,
          width=3,
          height=3,
          expected_array=[8],
      ),
      dict(
          testcase_name='starts_outside_the_scope_of_the_image',
          x=-1,
          y=-1,
          width=3,
          height=3,
          expected_array=[0],
      ),
      dict(
          testcase_name=(
              'Starts outside the scope of the image, and extends beyond the'
              ' scope.'
          ),
          x=-1,
          y=4,
          width=8,
          height=3,
          expected_array=[6, 7, 8],
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    patch = slide.get_patch(slide.native_pixel_spacing, x, y, width, height)
    self.assertEqual(
        patch.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    image = slide.get_image(slide.native_pixel_spacing)
    self.assertEqual(
        image.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
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
    mock_frame_cache = _create_mock_frame_cache()
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = mock_frame_cache
    self.assertIs(slide.slide_frame_cache, mock_frame_cache)

  def test_remove_slide_frame_cache(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = _create_mock_frame_cache()
    slide.remove_slide_frame_cache()
    self.assertIsNone(slide.slide_frame_cache)

  def test_get_image_from_slide_frame_cache_by_instance(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = _create_mock_frame_cache()
    level = slide._level_map._level_map[1]
    self.assertEqual(
        slide._get_cached_frame_bytes(level.instances[0], 1),
        b'\x00\x01\x02\x03',
    )

  def test_init_slide_frame_cache(self) -> None:
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    val = slide.init_slide_frame_cache()
    self.assertIsNotNone(slide.slide_frame_cache)
    self.assertIs(val, slide.slide_frame_cache)

  def test_init_slide_frame_cache_constructor(self) -> None:
    mock_cache = _create_mock_frame_cache()
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
        slide_frame_cache=mock_cache,
    )
    self.assertIs(slide.slide_frame_cache, mock_cache)

  def test_get_image_from_slide_frame_cache_by_path(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = _create_mock_frame_cache()
    level = slide._level_map._level_map[1]
    self.assertEqual(
        slide._get_cached_frame_bytes(level.instances[0], 2),
        b'\x04\x05\x06\x07',
    )

  def test_get_image_from_slide_frame_cache(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide.slide_frame_cache = _create_mock_frame_cache()
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
    image = slide.get_image(slide.native_pixel_spacing)
    self.assertEqual(image.pixel_spacing, slide.native_pixel_spacing)
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

  @parameterized.named_parameters([
      dict(
          testcase_name='missing_overlap_1',
          dst_x=3,
          dst_y=3,
          dst_width=2,
          dst_height=2,
      ),
      dict(
          testcase_name='missing_overlap_2',
          dst_x=4,
          dst_y=5,
          dst_width=3,
          dst_height=3,
      ),
      dict(
          testcase_name='missing_overlap_3',
          dst_x=-1,
          dst_y=-1,
          dst_width=1,
          dst_height=1,
      ),
  ])
  def test_copy_overlapped_region_with_invalid_input_raise_error(
      self, dst_x: int, dst_y: int, dst_width: int, dst_height: int
  ):
    ps = pixel_spacing.PixelSpacing.FromMagnificationString('40X')
    src_frame = dicom_slide.Frame(0, 0, 3, 3, np.ndarray((3, 3, 3), np.uint8))
    dst_patch = dicom_slide.Patch(ps, dst_x, dst_y, dst_width, dst_height)
    with self.assertRaises(ez_wsi_errors.PatchIntersectionNotFoundError):
      dst_np = np.ndarray((dst_height, dst_width, 3), np.uint8)
      dst_patch._copy_overlapped_region(src_frame, dst_np)

  @parameterized.named_parameters([
      dict(
          testcase_name='region1',
          dst_x=1,
          dst_y=0,
          expected_region_width=2,
          expected_region_height=1,
          expected_array=[[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [2, 2, 2]]],
      ),
      dict(
          testcase_name='region2',
          dst_x=1,
          dst_y=1,
          expected_region_width=2,
          expected_region_height=2,
          expected_array=[[[1, 1, 1], [2, 2, 2]], [[4, 4, 4], [5, 5, 5]]],
      ),
      dict(
          testcase_name='region3',
          dst_x=3,
          dst_y=2,
          expected_region_width=1,
          expected_region_height=2,
          expected_array=[[[6, 6, 6], [0, 0, 0]], [[9, 9, 9], [0, 0, 0]]],
      ),
  ])
  def test_copy_overlapped_region_with_valid(
      self,
      dst_x: int,
      dst_y: int,
      expected_region_width: int,
      expected_region_height: int,
      expected_array: np.ndarray,
  ):
    ps = pixel_spacing.PixelSpacing.FromMagnificationString('40X')
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
    dst_patch = dicom_slide.Patch(ps, dst_x, dst_y, 2, 2)
    dst_np = np.zeros((2, 2, 3), np.uint8)

    region_width, region_height = dst_patch._copy_overlapped_region(
        src_frame, dst_np
    )

    self.assertEqual(
        (expected_region_width, expected_region_height),
        (region_width, region_height),
    )
    self.assertTrue(np.array_equal(expected_array, dst_np))

  @parameterized.named_parameters([
      dict(
          testcase_name='region1',
          src_x=1,
          src_y=1,
          width=3,
          height=3,
          dst_x=0,
          dst_y=0,
      ),
      dict(
          testcase_name='region2',
          src_x=1,
          src_y=1,
          width=2,
          height=2,
          dst_x=1,
          dst_y=1,
      ),
      dict(
          testcase_name='region3',
          src_x=1,
          src_y=1,
          width=3,
          height=3,
          dst_x=0,
          dst_y=0,
      ),
  ])
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

  @parameterized.named_parameters([
      dict(
          testcase_name='region1',
          src_x=1,
          src_y=1,
          width=2,
          height=2,
          dst_x=0,
          dst_y=0,
          expected_array=[[[5, 5, 5], [6, 6, 6]], [[8, 8, 8], [9, 9, 9]]],
      ),
      dict(
          testcase_name='region2',
          src_x=1,
          src_y=1,
          width=1,
          height=2,
          dst_x=1,
          dst_y=0,
          expected_array=[[[0, 0, 0], [5, 5, 5]], [[0, 0, 0], [8, 8, 8]]],
      ),
  ])
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
        pixel_spacing=pixel_spacing.PixelSpacing.FromMagnificationString('10X'),
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
    _init_test_slide_level_map(
        slide, 1365, 468, 455, 156, 0, 3, 3, '1.2.840.10008.1.2.4.50'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_image_jpeg
    x = 100
    y = 50
    width = 300
    height = 100
    expected_array = np.asarray(
        PIL.Image.open(dicom_test_utils.test_jpeg_path())
    )
    expected_array = expected_array[y : y + height, x : x + width, :]

    patch = slide.get_patch(slide.native_pixel_spacing, x, y, width, height)
    self.assertEqual(
        patch.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    x = 1
    y = 2
    width = 4
    height = 2
    expected_array = np.asarray(
        [13, 16, 17, 20, 15, 18, 19, 22], np.uint8
    ).reshape(height, width, 1)

    patch = slide.get_patch(slide.native_pixel_spacing, x, y, width, height)
    self.assertEqual(patch.pixel_spacing, slide.native_pixel_spacing)
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    self.assertTrue(np.array_equal(expected_array, patch.image_bytes()))
    self.assertEqual(self.mock_dwi.get_frame_image.call_count, 3)

  def test_slide_levels(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 0, 8, 1, '1.2.840.10008.1.2.1'
    )
    self.assertIsInstance(slide.levels, Iterator)
    self.assertLen(list(slide.levels), 10)


if __name__ == '__main__':
  absltest.main()
