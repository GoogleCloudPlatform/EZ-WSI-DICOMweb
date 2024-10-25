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
"""Tests for dicom slide."""

import collections
import copy
import dataclasses
import json
import typing
from typing import Iterator, List, Optional, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
from PIL import ImageCms
import PIL.Image
import pydicom

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock

MOCK_USER_ID = 'mockuserid'


def _mock_dicom_ann_instance() -> pydicom.FileDataset:
  """Returns Mock WSI DICOM."""
  sop_class_uid = '1.2.840.10008.5.1.4.1.1.77.1.6'
  sop_instance_uid = '1.2.3.4.5'
  file_meta = pydicom.dataset.FileMetaDataset()
  file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
  file_meta.MediaStorageSOPClassUID = sop_class_uid
  file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
  file_meta.ImplementationClassUID = '1.2.4'
  mk_instance = pydicom.FileDataset(
      '', {}, file_meta=file_meta, preamble=b'\0' * 128
  )
  mk_instance.StudyInstanceUID = '1.2.3'
  mk_instance.SOPClassUID = sop_class_uid
  mk_instance.SeriesInstanceUID = '1.2.3.4'
  mk_instance.SOPInstanceUID = sop_instance_uid

  mk_instance.SOPClassUID = '1.2.840.10008.5.1.4.1.1.91.1'  # Annotation IOD
  mk_instance.OperatorIdentificationSequence = [pydicom.Dataset()]
  (
      mk_instance.OperatorIdentificationSequence[
          0
      ].PersonIdentificationCodeSequence
  ) = [pydicom.Dataset()]
  (
      mk_instance.OperatorIdentificationSequence[0]
      .PersonIdentificationCodeSequence[0]
      .LongCodeValue
  ) = MOCK_USER_ID

  return mk_instance


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
    self.enter_context(
        mock.patch.object(
            credential_factory,
            'get_default_gcp_project',
            return_value='MOCK_PROJECT',
        )
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

  def test_dicom_slide_logger_default_initialization(self):
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
        '41.13635187686038X',
        slide.native_pixel_spacing.as_magnification_string,
        'The native pixel spacing of the test slide must be 40X.',
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
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    self.assertEqual(
        np.uint8,
        level.pixel_format,
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
      _ = level.pixel_format

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

  def test_get_native_level_with_missing_min_level_returns_none(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide._level_map._level_map[slide._level_map.level_index_min] = None  # pytype: disable=unsupported-operands  # always-use-return-annotations
    self.assertIsNone(dicom_slide._get_native_level(slide._level_map))
    slide._native_level = None
    with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
      _ = slide.native_level

  @parameterized.named_parameters([
      dict(testcase_name='above_overlap', x=0, y=-1, w=10, h=10),
      dict(testcase_name='left_overlap', x=-1, y=0, w=10, h=10),
      dict(testcase_name='above_left_overlap', x=-1, y=-1, w=10, h=10),
      dict(testcase_name='fully_above', x=0, y=-20, w=10, h=10),
      dict(testcase_name='fully_left', x=-20, y=0, w=10, h=10),
      dict(testcase_name='fully_outside', x=-20, y=-20, w=10, h=10),
      dict(testcase_name='fully_wrap', x=-1, y=-1, w=200, h=400),
      dict(testcase_name='below_overlap', x=0, y=380, w=10, h=10),
      dict(testcase_name='right_overlap', x=184, y=0, w=10, h=10),
      dict(testcase_name='below_right_overlap', x=184, y=380, w=10, h=10),
      dict(testcase_name='fully_below_overlap', x=0, y=389, w=10, h=10),
      dict(testcase_name='fully_right_overlap', x=193, y=0, w=10, h=10),
      dict(testcase_name='fully_below_right_overlap', x=193, y=389, w=10, h=10),
  ])
  def test_get_patch_raises_out_of_image_bounds_error(self, x, y, w, h):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    smallest_level = list(slide.levels)[-1]
    # validate level dims. If level dim change then unit test will need to
    # change.
    self.assertEqual((smallest_level.width, smallest_level.height), (193, 389))
    with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
      slide.get_patch(smallest_level, x, y, w, h, True)

  @parameterized.named_parameters([
      dict(testcase_name='above_overlap', x=0, y=-1, w=10, h=10),
      dict(testcase_name='left_overlap', x=-1, y=0, w=10, h=10),
      dict(testcase_name='above_left_overlap', x=-1, y=-1, w=10, h=10),
      dict(testcase_name='fully_above', x=0, y=-20, w=10, h=10),
      dict(testcase_name='fully_left', x=-20, y=0, w=10, h=10),
      dict(testcase_name='fully_outside', x=-20, y=-20, w=10, h=10),
      dict(testcase_name='fully_wrap', x=-1, y=-1, w=22, h=22),
      dict(testcase_name='below_overlap', x=0, y=11, w=10, h=10),
      dict(testcase_name='right_overlap', x=11, y=0, w=10, h=10),
      dict(testcase_name='below_right_overlap', x=11, y=11, w=10, h=10),
      dict(testcase_name='fully_below_overlap', x=0, y=20, w=10, h=10),
      dict(testcase_name='fully_right_overlap', x=20, y=0, w=10, h=10),
      dict(testcase_name='fully_below_right_overlap', x=20, y=20, w=10, h=10),
  ])
  def test_get_patch_raises_out_of_resized_image_bounds_error(self, x, y, w, h):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    smallest_level = list(slide.levels)[-1]
    # validate level dims. If level dim change then unit test will need to
    # change.
    resized_level = smallest_level.resize(dicom_slide.ImageDimensions(20, 20))
    self.assertEqual((resized_level.width, resized_level.height), (20, 20))
    with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
      slide.get_patch(resized_level, x, y, w, h, True)

  def test_get_patch_override_raises_image_bounds_error(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    smallest_level = list(slide.levels)[-1]
    resized_level = smallest_level.resize(dicom_slide.ImageDimensions(20, 20))
    patch = slide.get_patch(resized_level, 0, 0, 10, 10, False)
    with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
      patch.get_patch(0, 0, 50, 50, True)

  def test_get_patch_default_raises_image_bounds_error(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    smallest_level = list(slide.levels)[-1]
    resized_level = smallest_level.resize(dicom_slide.ImageDimensions(20, 20))
    patch = slide.get_patch(resized_level, 0, 0, 10, 10, True)
    with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
      patch.get_patch(0, 0, 50, 50)

  def test_get_native_level_with_no_init_native_level_raises(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide._native_level = None
    with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
      _ = slide.native_level

  def test_cannot_generate_dicom_patches_from_sparse_tiled_levels_w_mult_frames(
      self,
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = mock.create_autospec(slide_level_map.Level, instance=True)
    type(level).tiled_full = mock.PropertyMock(return_value=False)
    type(level).number_of_frames = mock.PropertyMock(return_value=10)
    with self.assertRaises(ez_wsi_errors.DicomPatchGenerationError):
      dicom_slide.DicomPatch(level, 0, 0, 10, 10, slide)

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

  @parameterized.named_parameters([
      dict(
          testcase_name='10x',
          scale_factor=1.0,
          expected_dim=(24704, 49792),
      ),
      dict(
          testcase_name='8x',
          scale_factor=10.0 / 8.0,
          expected_dim=(24704, 49792),
      ),
      dict(
          testcase_name='5x',
          scale_factor=2.0,
          expected_dim=(12352, 24896),
      ),
      dict(
          testcase_name='below_bottom',
          scale_factor=24896,
          expected_dim=(193, 389),
      ),
  ])
  def test_get_closest_level_with_pixel_spacing_equal_or_less_than_target(
      self, scale_factor, expected_dim
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    base_ps = pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    ps = pixel_spacing.PixelSpacing(
        base_ps.column_spacing_mm * scale_factor,
        base_ps.row_spacing_mm * scale_factor,
    )
    level = (
        slide.get_closest_level_with_pixel_spacing_equal_or_less_than_target(ps)
    )
    self.assertEqual((level.width, level.height), expected_dim)  # pytype: disable=attribute-error

  def test_get_closest_level_with_pixel_less_than_target_above_pyramuid(self):
    scale_factor = 0.01
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    base_ps = pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    ps = pixel_spacing.PixelSpacing(
        base_ps.column_spacing_mm * scale_factor,
        base_ps.row_spacing_mm * scale_factor,
    )
    self.assertIsNone(
        slide.get_closest_level_with_pixel_spacing_equal_or_less_than_target(ps)
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='10x',
          scale_factor=1.0,
          expected_dim=(12352, 24896),
      ),
      dict(
          testcase_name='8x',
          scale_factor=10.0 / 8.0,
          expected_dim=(12352, 24896),
      ),
      dict(
          testcase_name='5x',
          scale_factor=2.0,
          expected_dim=(6176, 12448),
      ),
      dict(
          testcase_name='above_pyramid',
          scale_factor=0.01,
          expected_dim=(98816, 199168),
      ),
  ])
  def test_get_closest_level_with_pixel_spacing_greater_than_target(
      self, scale_factor, expected_dim
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    base_ps = pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    ps = pixel_spacing.PixelSpacing(
        base_ps.column_spacing_mm * scale_factor,
        base_ps.row_spacing_mm * scale_factor,
    )
    level = slide.get_closest_level_with_pixel_spacing_greater_than_target(ps)
    self.assertEqual((level.width, level.height), expected_dim)  # pytype: disable=attribute-error

  def test_get_closest_level_with_pixel_spacing_greater_below_pyramid(self):
    scale_factor = 24896
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    base_ps = pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    ps = pixel_spacing.PixelSpacing(
        base_ps.column_spacing_mm * scale_factor,
        base_ps.row_spacing_mm * scale_factor,
    )
    self.assertIsNone(
        slide.get_closest_level_with_pixel_spacing_greater_than_target(ps)
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='instance',
          instance='1.2.276.0.7230010.3.1.4.2148154112.13.1559585117.823618',
          expected=(386, 778),
      ),
      dict(
          testcase_name='concatenated_level_1',
          instance='1.2.276.0.7230010.3.1.4.2148154112.13.1559585058.823602',
          expected=(24704, 49792),
      ),
      dict(
          testcase_name='concatenated_level_2',
          instance='1.2.276.0.7230010.3.1.4.2148154112.13.1559585084.823604',
          expected=(24704, 49792),
      ),
  ])
  def test_instance_level(self, instance, expected):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    level = slide.get_instance_level(instance)
    self.assertEqual(
        (level.width, level.height), expected  # pytype: disable=attribute-error
    )

  def test_instance_level_not_found(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertIsNone(slide.get_instance_level('1.2.3'))

  @parameterized.named_parameters([
      dict(testcase_name='before_first_frame', mag='40X', frame_number=0),
      dict(
          testcase_name='after_last_frame_40x', mag='40X', frame_number=655360
      ),
      dict(testcase_name='after_last_frame_5x', mag='5X', frame_number=2047),
  ])
  def test_get_frame_with_out_of_range_frame_number_raise_error(
      self, mag: str, frame_number: int
  ):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    with self.assertRaises(ez_wsi_errors.InputFrameNumberOutOfRangeError):
      level = slide.get_level_by_pixel_spacing(
          pixel_spacing.PixelSpacing.FromMagnificationString(mag)
      )
      slide.get_frame(level, frame_number)

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
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    frame = slide.get_frame(level, 2)
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
            slide, 6, 6, 1, 1, 1, 9, 1, transfer_syntax_uid
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
        credential_factory.CredentialFactory()
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
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    frame_1 = slide.get_frame(level, 1)
    frame_2 = slide.get_frame(level, 1)
    self.assertIsNotNone(frame_1)
    self.assertIsNotNone(frame_2)
    self.assertTrue(
        np.array_equal(frame_1.image_np, frame_2.image_np),  # pytype: disable=attribute-error
        'Cached frame not equal to original frame.',
    )
    self.mock_dwi.get_frame_image.assert_called_once()

  def test_server_side_transcoding_frame_cache_supported_transfer_syntax(
      self,
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    study_uid = '1.2'
    series_uid = f'{study_uid}.3'
    instance_uid = f'{series_uid}.4'
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    test_frame_data = b'abc123abc123'
    test_instance = dicom_test_utils.create_test_dicom_instance(
        study_uid, series_uid, instance_uid, frame_data=test_frame_data
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=False,
      )
      slide.slide_frame_cache = local_dicom_slide_cache.InMemoryDicomSlideCache(
          credential_factory.CredentialFactory()
      )
      # Set frame size to 2x2 for the first level.
      level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      frame_1 = slide.get_frame(level, 1)
      self.assertEqual(
          slide.slide_frame_cache.cache_stats.frame_cache_hit_count, 0
      )
      self.assertEqual(frame_1.image_np.tobytes(), test_frame_data)  # pytype: disable=attribute-error
      frame_2 = slide.get_frame(level, 1)
      self.assertEqual(
          slide.slide_frame_cache.cache_stats.frame_cache_hit_count, 1
      )
      self.assertEqual(frame_2.image_np.tobytes(), test_frame_data)  # pytype: disable=attribute-error

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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    with self.assertRaises(ez_wsi_errors.SectionOutOfImageBoundsError):
      level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      slide.get_patch(level, x, y, width, height).image_bytes()

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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch = slide.get_patch(level, x, y, width, height)
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
    # frames. The list returned should be sorted and not have duplicates.

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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1', mock_path=True
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    expected = {
        str(dicom_path.FromPath(slide.path, instance_uid='1')): expected_array
    }
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch_list = [
        slide.get_patch(level, x, y, width, height)
        for x, y, width, height in patch_pos_dim_list
    ]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_numbers(
        level,
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
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch_list = [slide.get_patch(level, 0, 0, 8, 6)]
    instance_frame_map = slide.get_patch_bounds_dicom_instance_frame_numbers(
        level,
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

  def test_are_instances_concatenated(self):
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

    self.assertTrue(
        slide.are_instances_concatenated([
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
        ])
    )

  def test_get_instance_pixel_spacing(self):
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
    ps = slide.get_instance_pixel_spacing(
        '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291'
    )

    self.assertIsNotNone(ps)
    self.assertTrue(ps.__eq__(pixel_spacing.PixelSpacing(0.256, 0.255625)))

  def test_are_instances_concatenated_false(self):
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

    self.assertFalse(
        slide.are_instances_concatenated([
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791293',
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791291',
            # fake instance
            '1.2.276.0.7230010.3.1.4.296485376.35.1674232412.791295',
        ])
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='align_with_frame_boundary',
          x=0,
          y=0,
          width=4,
          height=4,
          expected_array=[1, 2, 4, 5],
      ),
      dict(
          testcase_name='across_more_than_2_frames_in_both_x_and_y_direction',
          x=1,
          y=1,
          width=4,
          height=4,
          expected_array=[1, 2, 3, 4, 5, 6, 7, 8, 9],
      ),
      dict(
          testcase_name='single_frame',
          x=2,
          y=2,
          width=2,
          height=2,
          expected_array=[5],
      ),
      dict(
          testcase_name='single_ pixel',
          x=3,
          y=3,
          width=1,
          height=1,
          expected_array=[5],
      ),
      dict(
          testcase_name='single_row_across_multiple_frames',
          x=1,
          y=4,
          width=4,
          height=1,
          expected_array=[7, 8, 9],
      ),
      dict(
          testcase_name='extends_beyond_the_width_of_the_image',
          x=5,
          y=1,
          width=2,
          height=3,
          expected_array=[3, 6],
      ),
      dict(
          testcase_name='extends_beyond_the_height_of_the_image',
          x=0,
          y=5,
          width=4,
          height=2,
          expected_array=[7, 8],
      ),
      dict(
          testcase_name=(
              'extends_beyond_both_the_width_and_the_height_of_the_image'
          ),
          x=4,
          y=4,
          width=3,
          height=3,
          expected_array=[9],
      ),
      dict(
          testcase_name='starts_outside_the_scope_of_the_image',
          x=-1,
          y=-1,
          width=3,
          height=3,
          expected_array=[1],
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
          expected_array=[7, 8, 9],
      ),
  ])
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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch = slide.get_patch(level, x, y, width, height)
    self.assertEqual(
        patch.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
    self.assertEqual(
        [patch.x, patch.y, patch.width, patch.height], [x, y, width, height]
    )
    # sort order is expected
    self.assertEqual(expected_array, list(patch.frame_numbers()))

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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    image = slide.get_image(
        slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    )
    self.assertEqual(
        image.pixel_spacing.pixel_spacing_mm,
        slide.native_pixel_spacing.pixel_spacing_mm,
    )
    self.assertEqual([image.width, image.height], [6, 6])
    self.assertEqual(image.source.path, slide.path)
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
    self.assertIsNone(slide.slide_frame_cache)
    self.assertIs(slide.init_slide_frame_cache(), slide.slide_frame_cache)
    self.assertIsNotNone(slide.slide_frame_cache)

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
    _init_test_slide_level_map(
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    image = slide.get_image(
        slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    )
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
    mk_slide = mock.create_autospec(dicom_slide.DicomSlide, instsance=True)
    mk_level = mock.create_autospec(slide_level_map.Level)
    type(mk_level).pixel_spacing = mock.PropertyMock(return_value=ps)
    type(mk_level).width = mock.PropertyMock(return_value=10)
    type(mk_level).height = mock.PropertyMock(return_value=10)
    type(mk_level).level_index = mock.PropertyMock(return_value=1)
    type(mk_level).tiled_full = mock.PropertyMock(return_value=True)
    mk_slide.has_level.return_value = True
    dst_patch = dicom_slide.DicomPatch(
        mk_level,
        dst_x,
        dst_y,
        dst_width,
        dst_height,
        mk_slide,
        mk_level,
    )
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
    mk_slide = mock.create_autospec(dicom_slide.DicomSlide)
    mk_level = mock.create_autospec(slide_level_map.Level)
    type(mk_level).pixel_spacing = mock.PropertyMock(return_value=ps)
    type(mk_level).width = mock.PropertyMock(return_value=10)
    type(mk_level).height = mock.PropertyMock(return_value=10)
    type(mk_level).level_index = mock.PropertyMock(return_value=1)
    type(mk_level).tiled_full = mock.PropertyMock(return_value=True)
    mk_slide.has_level.return_value = True
    dst_patch = dicom_slide.DicomPatch(
        mk_level, dst_x, dst_y, 2, 2, mk_slide, mk_level
    )
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
    level = slide.get_level_by_pixel_spacing(
        pixel_spacing.PixelSpacing.FromMagnificationString('10X')
    )
    patch = slide.get_patch(
        level,
        x=10,
        y=20,
        width=100,
        height=200,
    )
    self.assertEqual(
        'slideid:M_10.284087969215095X:000100x000200+000010+000020', patch.id
    )

  def test_get_patch_from_jpeg(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    _init_test_slide_level_map(
        slide, 1365, 468, 455, 156, 1, 4, 3, '1.2.840.10008.1.2.4.50'
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

    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch = slide.get_patch(level, x, y, width, height)
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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    self.mock_dwi.get_frame_image.side_effect = _fake_get_frame_raw_image
    x = 1
    y = 2
    width = 4
    height = 2
    expected_array = np.asarray(
        [13, 16, 17, 20, 15, 18, 19, 22], np.uint8
    ).reshape(height, width, 1)
    level = slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
    patch = slide.get_patch(level, x, y, width, height)
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
        slide, 6, 6, 2, 2, 1, 9, 1, '1.2.840.10008.1.2.1'
    )
    self.assertIsInstance(slide.levels, Iterator)
    self.assertLen(list(slide.levels), 10)

  def test_init_slide_levels_from_json(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    slide_json_init = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
        json_metadata=slide.json_metadata(),
    )
    self.assertEqual(
        slide._level_map._level_map,
        slide_json_init._level_map._level_map,
    )

  def test_init_slide_levels_from_json_slide_not_equal_raises(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    json_metadata = slide.json_metadata()
    metadata = json.loads(json_metadata)
    metadata[dicom_slide._SLIDE_PATH] = dicom_path.Path(
        'http://A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
    ).to_dict()
    with self.assertRaises(
        ez_wsi_errors.SlidePathDoesNotMatchJsonMetadataError
    ):
      dicom_slide.DicomSlide(
          self.mock_dwi,
          self.dicom_series_path,
          enable_client_slide_frame_decompression=True,
          json_metadata=json.dumps(metadata),
      )

  def test_init_slide_levels_from_empty_json_metadata_succeeds(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
        json_metadata='{}',
    )
    self.assertLen(list(slide.levels), 10)

  @parameterized.parameters(['{1:2}', 'abc'])
  def test_init_slide_levels_from_invalid_json_metadata_raises(self, metadata):
    with self.assertRaises(ez_wsi_errors.InvalidSlideJsonMetadataError):
      dicom_slide.DicomSlide(
          self.mock_dwi,
          self.dicom_series_path,
          enable_client_slide_frame_decompression=True,
          json_metadata=metadata,
      )

  @parameterized.parameters([True, False])
  def test_get_icc_profile_dicom_missing_icc_profile(
      self, bulkdata_uri_enabled
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=bulkdata_uri_enabled
    ) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertEqual(slide.get_icc_profile_bytes(), b'')

  @parameterized.parameters([True, False])
  def test_get_icc_profile_dicom_has_icc_profile(self, bulkdata_uri_enabled):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance.ICCProfile = b'1234'
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(
        dicom_store_path, bulkdata_uri_enabled=bulkdata_uri_enabled
    ) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertEqual(slide.get_icc_profile_bytes(), b'1234')

  def test_get_credential_header(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertEqual(
          slide.get_credential_header(),
          {'authorization': 'Bearer MOCK_BEARER_TOKEN'},
      )

  def test_get_srgb_icc_profile(self):
    self.assertIsNotNone(dicom_slide.get_srgb_icc_profile())

  def test_get_patch_image_bytes(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      patch = slide.get_patch(slide.native_pixel_spacing, 10, 10, 500, 500)
      with mock.patch.object(ImageCms, 'applyTransform', autospec=True) as mk:
        self.assertEqual(
            patch.image_bytes().tobytes(),
            patch.image_bytes(
                mock.create_autospec(ImageCms.ImageCmsTransform, instance=True)
            ).tobytes(),
        )
        mk.assert_called_once()

  def test_create_icc_profile_transformation_for_level_with_no_icc_profile_returns_none(
      self,
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertIsNone(
          slide.create_icc_profile_transformation(b'MOCK_ICC_PROFILE')
      )

  @parameterized.parameters([b'', None])
  def test_create_icc_profile_transformation_for_level_with_no_target_profile_returns_none(
      self, target_profile
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      slide.set_icc_profile_bytes(b'MOCK_ICC_PROFILE')
      self.assertIsNone(slide.create_icc_profile_transformation(target_profile))

  def test_create_icc_profile_transformation_for_level_with_srgb(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      slide.set_icc_profile_bytes(dicom_slide.get_srgb_icc_profile_bytes())
      self.assertIsNotNone(
          slide.create_icc_profile_transformation(
              dicom_slide.get_srgb_icc_profile()
          )
      )

  def test_get_json_encoded_icc_profile_size(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      slide.set_icc_profile_bytes(b'MOCK_ICC_PROFILE')
      self.assertEqual(slide.get_json_encoded_icc_profile_size(), 89)

  @parameterized.parameters([0, 1, 2, 3, 4])
  def test_patch_downsample_resizeing(self, offset):
    scale_factor = 4.0
    interpolation = cv2.INTER_AREA
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      patch_1 = slide.get_image(
          slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      ).image_bytes()
      patch_1_expected = cv2.resize(
          patch_1,
          (
              int(patch_1.shape[1] / scale_factor),
              int(patch_1.shape[0] / scale_factor),
          ),
          interpolation=interpolation,
      )
      patch_1_expected = patch_1_expected[
          (100 + offset) : (200 + offset), (100 + offset) : (200 + offset)
      ]
      patch_1_expected = np.pad(
          patch_1_expected,
          (
              (0, 100 - patch_1_expected.shape[0]),
              (0, 100 - patch_1_expected.shape[1]),
              (0, 0),
          ),
          constant_values=0,
      )
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      patch_2 = slide.get_patch(
          slide_level_map.ResizedLevel(slide.native_level, resized_level_dim),
          100 + offset,
          100 + offset,
          100,
          100,
      ).image_bytes()
      np.testing.assert_allclose(patch_1_expected, patch_2, atol=6)

  @parameterized.parameters([0, 1, 2, 3, 4])
  def test_patch_upsample_resizeing(self, offset):
    scale_factor = 0.25
    interpolation = cv2.INTER_CUBIC
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      patch_1 = slide.get_image(
          slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      ).image_bytes()
      patch_1_expected = cv2.resize(
          patch_1,
          (
              int(patch_1.shape[1] / scale_factor),
              int(patch_1.shape[0] / scale_factor),
          ),
          interpolation=interpolation,
      )
      patch_1_expected = patch_1_expected[
          (100 + offset) : (500 + offset), (100 + offset) : (500 + offset)
      ]
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      patch_2 = slide.get_patch(
          slide_level_map.ResizedLevel(slide.native_level, resized_level_dim),
          100 + offset,
          100 + offset,
          400,
          400,
      ).image_bytes()
      np.testing.assert_allclose(patch_1_expected, patch_2, atol=15)

  def test_patch_upsample_resizeing_low_right_edge(self):
    scale_factor = 0.25
    interpolation = cv2.INTER_CUBIC
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      patch_1 = slide.get_image(
          slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      ).image_bytes()
      patch_1_expected = cv2.resize(
          patch_1,
          (
              int(patch_1.shape[1] / scale_factor),
              int(patch_1.shape[0] / scale_factor),
          ),
          interpolation=interpolation,
      )
      patch_1_expected = patch_1_expected[-400:, -400:]
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      patch_2 = slide.get_patch(
          slide_level_map.ResizedLevel(slide.native_level, resized_level_dim),
          resized_level_dim.width_px - 400,
          resized_level_dim.height_px - 400,
          400,
          400,
      ).image_bytes()
      np.testing.assert_allclose(patch_1_expected, patch_2, atol=4)

  def test_patch_upsample_resizeing_upper_left_edge(self):
    scale_factor = 0.25
    interpolation = cv2.INTER_CUBIC
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      patch_1 = slide.get_image(
          slide.get_level_by_pixel_spacing(slide.native_pixel_spacing)
      ).image_bytes()
      patch_1_expected = cv2.resize(
          patch_1,
          (
              int(patch_1.shape[1] / scale_factor),
              int(patch_1.shape[0] / scale_factor),
          ),
          interpolation=interpolation,
      )
      patch_1_expected = patch_1_expected[:400, :400]
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      patch_2 = slide.get_patch(
          slide_level_map.ResizedLevel(slide.native_level, resized_level_dim),
          0,
          0,
          400,
          400,
      ).image_bytes()
      np.testing.assert_allclose(patch_1_expected, patch_2, atol=9)

  @parameterized.named_parameters([
      dict(
          testcase_name='64x',
          scale_factor=64.0,
          expected=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      ),
      dict(
          testcase_name='16x',
          scale_factor=16.0,
          expected=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      ),
      dict(
          testcase_name='4x',
          scale_factor=4.0,
          expected=[1, 2, 6, 7],
      ),
      dict(
          testcase_name='1x',
          scale_factor=1.0,
          expected=[1],
      ),
      dict(
          testcase_name='0.5x',
          scale_factor=0.5,
          expected=[1],
      ),
      dict(
          testcase_name='0.25x',
          scale_factor=0.25,
          expected=[1],
      ),
      dict(
          testcase_name='0.125x',
          scale_factor=0.125,
          expected=[1],
      ),
  ])
  def test_get_resized_patch_bounds_soure_dicom_instance_frame_numbers(
      self, scale_factor, expected
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      result = slide.get_patch_bounds_dicom_instance_frame_numbers(
          slide_level_map.ResizedLevel(slide.native_level, resized_level_dim),
          [dicom_slide.PatchBounds(2, 3, 100, 100)],
      )
      self.assertEqual(list(result.values()), [expected])

  @parameterized.named_parameters([
      dict(
          testcase_name='resize_height_more_than_width',
          width_scale_factor=1.0,
          height_scale_factor=2.0,
      ),
      dict(
          testcase_name='resize_width_more_than_height',
          width_scale_factor=2.0,
          height_scale_factor=1.0,
      ),
      dict(
          testcase_name='expand_height_more_than_width',
          width_scale_factor=1.0,
          height_scale_factor=0.5,
      ),
      dict(
          testcase_name='expand_width_more_than_height',
          width_scale_factor=0.5,
          height_scale_factor=1.0,
      ),
  ])
  def test_resize_level(self, width_scale_factor, height_scale_factor):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      resized_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / width_scale_factor),
          int(slide.native_level.height / height_scale_factor),
      )
      dl = slide_level_map.ResizedLevel(slide.native_level, resized_level_dim)
      self.assertIs(dl.source_level, slide.native_level)
      self.assertEqual(
          dl.width, int(slide.native_level.width / width_scale_factor)
      )
      self.assertEqual(
          dl.height, int(slide.native_level.height / height_scale_factor)
      )  # pytype: disable=attribute-error

  def test_upsample_level(self):
    scale_factor = 0.5
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    dcm = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    study_uid = dcm.StudyInstanceUID
    series_uid = dcm.SeriesInstanceUID
    series_path = (
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{study_uid}/'
        f'series/{series_uid}'
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(dcm)
      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
          enable_client_slide_frame_decompression=True,
      )
      resize_level_dim = slide_level_map.ImageDimensions(
          int(slide.native_level.width / scale_factor),
          int(slide.native_level.height / scale_factor),
      )
      dl = slide_level_map.ResizedLevel(slide.native_level, resize_level_dim)
      self.assertIs(dl.source_level, slide.native_level)
      self.assertEqual(dl.width, int(slide.native_level.width / scale_factor))
      self.assertEqual(dl.height, int(slide.native_level.height / scale_factor))  # pytype: disable=attribute-error

  def test_find_annotation_instances(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      # It should be returned because it's from the same series as the images.
      ann_ds = _mock_dicom_ann_instance()
      ann_ds.StudyInstanceUID = test_instance.StudyInstanceUID
      ann_ds.SeriesInstanceUID = test_instance.SeriesInstanceUID
      ann_ds.SOPInstanceUID = '1.2.3.4'
      mock_store[dicom_store_path].add_instance(ann_ds)

      # It should be returned because it references the image.
      ann_ds = _mock_dicom_ann_instance()
      ann_ds.StudyInstanceUID = test_instance.StudyInstanceUID
      ann_ds.SeriesInstanceUID = '5.5.5.5'
      ann_ds.SOPInstanceUID = '1.2.3.5'
      ann_ds.ReferencedImageSequence = [pydicom.Dataset()]
      ann_ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID = (
          test_instance.SOPInstanceUID
      )
      mock_store[dicom_store_path].add_instance(ann_ds)

      # It should not be returned because it references the image wrong image.
      ann_ds = _mock_dicom_ann_instance()
      ann_ds.StudyInstanceUID = test_instance.StudyInstanceUID
      ann_ds.SeriesInstanceUID = '5.5.5.5'
      ann_ds.SOPInstanceUID = '1.2.3.6'
      ann_ds.ReferencedImageSequence = [pydicom.Dataset()]
      ann_ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID = '9.8.7.6'
      mock_store[dicom_store_path].add_instance(ann_ds)

      # It should not be returned because operator id is missing.
      ann_ds = _mock_dicom_ann_instance()
      ann_ds.StudyInstanceUID = test_instance.StudyInstanceUID
      ann_ds.SeriesInstanceUID = '5.5.5.5'
      ann_ds.SOPInstanceUID = '1.2.3.7'
      ann_ds.OperatorIdentificationSequence = []
      mock_store[dicom_store_path].add_instance(ann_ds)

      # It should not be returned because SOPClassUID is not annotation.
      ann_ds = _mock_dicom_ann_instance()
      ann_ds.StudyInstanceUID = test_instance.StudyInstanceUID
      ann_ds.SeriesInstanceUID = '5.5.5.5'
      ann_ds.SOPInstanceUID = '1.2.3.8'
      ann_ds.SOPClassUID = '10.9.8.7'
      mock_store[dicom_store_path].add_instance(ann_ds)

      slide = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      ann_iter = slide.find_annotation_instances(
          filter_by_operator_id=MOCK_USER_ID
      )
      self.assertEqual(
          list(ann_iter),
          [
              dicom_path.FromPath(
                  slide.path,
                  instance_uid='1.2.3.4',
              ),
              dicom_path.FromPath(
                  slide.path,
                  instance_uid='1.2.3.5',
                  series_uid='5.5.5.5',
              ),
          ],
      )

  def test_dicom_slide_returns_label_overview_and_thumnail_when_not_set(self):
    slide = dicom_slide.DicomSlide(
        self.mock_dwi,
        self.dicom_series_path,
        enable_client_slide_frame_decompression=True,
    )
    self.assertIsNone(slide.label)
    self.assertIsNone(slide.overview)
    self.assertIsNone(slide.thumbnail)

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_get_slide_microscopy_image(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertLen(list(image.levels), 1)
      self.assertLen(list(image.all_levels), 1)
      self.assertEqual(list(image.all_levels), list(image.levels))
      self.assertEqual(
          image.get_image(image.get_level_by_index('1.2.3.4.5'))
          .image_bytes()
          .tobytes(),
          b'abc123abc123',
      )

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_init_slide_microscopy_image_from_metadata(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image_loaded_from_dicom_store = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
    metadata = image_loaded_from_dicom_store.json_metadata()
    image_loaded_from_metadata = dicom_slide.DicomMicroscopeImage(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path),
        json_metadata=metadata,
    )
    self.assertEqual(image_loaded_from_metadata, image_loaded_from_dicom_store)

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_init_slide_microscopy_image_from_series(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      series = dicom_slide.DicomMicroscopeSeries(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertIsNone(series.dicom_slide)
      self.assertIsNotNone(series.dicom_microscope_image)

  def test_init_wsi_image_from_series(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      series = dicom_slide.DicomMicroscopeSeries(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertIsNotNone(series.dicom_slide)
      self.assertIsNone(series.dicom_microscope_image)

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_init_slide_microscopy_image_from_series_metadata(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image_loaded_from_dicom_store = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
    metadata = image_loaded_from_dicom_store.json_metadata()
    series = dicom_slide.DicomMicroscopeSeries(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path),
        json_metadata=metadata,
    )
    self.assertIsNone(series.dicom_slide)
    self.assertEqual(
        series.dicom_microscope_image, image_loaded_from_dicom_store
    )

  def test_init_wsi_slide_microscopy_image_from_series_metadata(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image_loaded_from_dicom_store = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
    metadata = image_loaded_from_dicom_store.json_metadata()
    series = dicom_slide.DicomMicroscopeSeries(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path),
        json_metadata=metadata,
    )
    self.assertIsNone(series.dicom_microscope_image)
    self.assertEqual(series.dicom_slide, image_loaded_from_dicom_store)

  def test_init_dicom_miroscope_series_with_invalid_metadata_raises(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path):
      with self.assertRaises(ez_wsi_errors.InvalidSlideJsonMetadataError):
        dicom_slide.DicomMicroscopeSeries(
            dicom_web_interface.DicomWebInterface(
                credential_factory.CredentialFactory()
            ),
            dicom_path.FromString(series_path),
            json_metadata='{}',
        )

  def test_init_dicom_miroscope_series_with_metadata_missing_imaging_raises(
      self,
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path):
      with self.assertRaisesRegex(
          ez_wsi_errors.InvalidSlideJsonMetadataError,
          'Error decoding JSON metadata does not encode DICOM imaging.',
      ):
        dicom_slide.DicomMicroscopeSeries(
            dicom_web_interface.DicomWebInterface(
                credential_factory.CredentialFactory()
            ),
            dicom_path.FromString(series_path),
            json_metadata='{"sop_class_uid": "123"}',
        )

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_image_from_series_not_concatenated(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertFalse(image.are_instances_concatenated(['1.2.3.4.5']))

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_image_from_series_get_icc_profile_bytes(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    test_instance.ICCProfile = b'abc123abc123'
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertEqual(
          image.get_level_icc_profile_bytes(
              image.get_level_by_index('1.2.3.4.5')
          ),
          b'abc123abc123',
      )

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_image_from_series_get_icc_profile_bytes_none(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      image = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(series_path),
      )
      self.assertEqual(
          image.get_level_icc_profile_bytes(
              image.get_level_by_index('1.2.3.4.5')
          ),
          b'',
      )

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_get_iccprofile_from_resized_level(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3',
        '1.2.3.4',
        '1.2.3.4.5',
        sop_class_uid=sop_class_uid,
    )
    test_instance_1.ICCProfile = b'bad_food'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      image_1 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      level = next(image_1.levels)
      icc_profile = image_1.get_level_icc_profile_bytes(
          level.resize(dicom_slide.ImageDimensions(100, 100))
      )
      self.assertEqual(icc_profile, b'bad_food')

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_images_with_different_paths_not_same(
      self, sop_class_uid
  ):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    test_instance_2 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.5', '1.2.3.4.6', sop_class_uid=sop_class_uid
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      mock_store[dicom_store_path].add_instance(test_instance_2)
      image_1 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      image_2 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_2.StudyInstanceUID}/series/{test_instance_2.SeriesInstanceUID}'
          ),
      )
      self.assertNotEqual(image_1, image_2)

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_images_with_same_path_same(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3', '1.2.3.4', '1.2.3.4.5', sop_class_uid=sop_class_uid
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      image_1 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      image_2 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      self.assertEqual(image_1, image_2)

  @parameterized.named_parameters([
      dict(testcase_name='neg_x', x=-1, y=0, width=3, height=3, expected=False),
      dict(testcase_name='neg_y', x=0, y=-1, width=3, height=3, expected=False),
      dict(
          testcase_name='neg_width',
          x=0,
          y=0,
          width=-3,
          height=3,
          expected=False,
      ),
      dict(
          testcase_name='neg_height',
          x=0,
          y=9,
          width=3,
          height=-3,
          expected=False,
      ),
      dict(
          testcase_name='zero_width',
          x=0,
          y=9,
          width=0,
          height=3,
          expected=False,
      ),
      dict(
          testcase_name='zero_height',
          x=0,
          y=9,
          width=3,
          height=0,
          expected=False,
      ),
      dict(
          testcase_name='to_wide', x=0, y=9, width=25, height=3, expected=False
      ),
      dict(
          testcase_name='to_height',
          x=0,
          y=9,
          width=3,
          height=25,
          expected=False,
      ),
      dict(testcase_name='success', x=0, y=0, width=3, height=3, expected=True),
  ])
  def test_is_patch_fully_in_source_image_dim(
      self, x, y, width, height, expected
  ):
    self.assertEqual(
        dicom_slide.BasePatch(
            x, y, width, height
        ).is_patch_fully_in_source_image_dim(10, 20),
        expected,
    )

  @parameterized.named_parameters([
      dict(testcase_name='2D', ary=np.zeros((2, 2)), expected=1),
      dict(testcase_name='2D_1', ary=np.zeros((2, 2, 1)), expected=1),
      dict(testcase_name='2D_3', ary=np.zeros((2, 2, 3)), expected=3),
  ])
  def test_get_image_bytes_samples_per_pixel(self, ary, expected):
    self.assertEqual(
        dicom_slide.get_image_bytes_samples_per_pixel(ary), expected
    )

  def test_loading_level_into_disabled_cache_is_nop(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_store_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_path),
      )
      ds.preload_level_in_frame_cache(ds.native_level)
      self.assertIsNone(ds.slide_frame_cache)

  def test_loading_level_into_cache(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_store_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_path),
      )
      fc = ds.init_slide_frame_cache()
      ds.preload_level_in_frame_cache(ds.native_level)
      self.assertEqual(fc.cache_stats.number_of_dicom_instances_read, 1)
      self.assertEqual(
          fc.cache_stats.number_of_frames_read_in_dicom_instances, 15
      )

  def test_loading_resized_level_into_cache(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_store_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_path),
      )
      fc = ds.init_slide_frame_cache()
      ds.preload_level_in_frame_cache(
          ds.native_level.resize(dicom_slide.ImageDimensions(100, 100))
      )
      self.assertEqual(fc.cache_stats.number_of_dicom_instances_read, 1)
      self.assertEqual(
          fc.cache_stats.number_of_frames_read_in_dicom_instances, 15
      )

  def test_loading_patch_into_disabled_cache_is_nop(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_store_path}/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_path),
      )
      patch = ds.get_patch(ds.native_level, 5, 5, 100, 100)
      ds.preload_patches_in_frame_cache(patch)
      self.assertIsNone(ds.slide_frame_cache, patch)

  def test_loading_patch_into_level_into_cache(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      fc = ds.init_slide_frame_cache()
      patch = ds.get_patch(ds.native_level, 5, 5, 100, 100)
      ds.preload_patches_in_frame_cache(patch)
      self.assertEqual(fc.cache_stats.number_of_dicom_instances_read, 0)
      self.assertEqual(
          fc.cache_stats.number_of_frames_read_in_dicom_instances, 0
      )
      self.assertEqual(fc.cache_stats.number_of_frame_blocks_read, 1)
      self.assertEqual(fc.cache_stats.number_of_frames_read_in_frame_blocks, 1)
      self.assertEqual(
          fc.cache_stats.number_of_frame_bytes_read_in_frame_blocks, 8418
      )

  def test_loading_patch_list_into_level_into_cache(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2.SOPInstanceUID = '1.2.3.4.555'
    series_path = f'{dicom_store_path}/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      mock_store[dicom_store_path].add_instance(test_instance_2)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_path),
      )
      fc = ds.init_slide_frame_cache()
      patch_list = [ds.get_patch(level, 5, 5, 100, 100) for level in ds.levels]
      self.assertLen(patch_list, 2)
      ds.preload_patches_in_frame_cache(patch_list)
      self.assertEqual(fc.cache_stats.number_of_dicom_instances_read, 0)
      self.assertEqual(
          fc.cache_stats.number_of_frames_read_in_dicom_instances, 0
      )
      self.assertEqual(fc.cache_stats.number_of_frame_blocks_read, 2)
      self.assertEqual(fc.cache_stats.number_of_frames_read_in_frame_blocks, 2)
      self.assertEqual(
          fc.cache_stats.number_of_frame_bytes_read_in_frame_blocks, 16836
      )

  def test_loading_level_from_different_path_than_dicom_slide_raises(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2.StudyInstanceUID = '1.2.3.4.555'
    series_1_path = f'{dicom_store_path}/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
    series_2_path = f'{dicom_store_path}/studies/{test_instance_2.StudyInstanceUID}/series/{test_instance_2.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      mock_store[dicom_store_path].add_instance(test_instance_2)
      ds_1 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_1_path),
      )
      ds_1.init_slide_frame_cache()
      ds_2 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_2_path),
      )
      with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
        ds_1.preload_level_in_frame_cache(ds_2.native_level)

  def test_loading_patch_from_different_path_than_dicom_slide_raises(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2.StudyInstanceUID = '1.2.3.4.555'
    series_1_path = f'{dicom_store_path}/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
    series_2_path = f'{dicom_store_path}/studies/{test_instance_2.StudyInstanceUID}/series/{test_instance_2.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      mock_store[dicom_store_path].add_instance(test_instance_2)
      ds_1 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_1_path),
      )
      ds_1.init_slide_frame_cache()
      ds_2 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_2_path),
      )
      with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
        ds_1.preload_patches_in_frame_cache(
            ds_2.get_patch(ds_2.native_level, 0, 0, 10, 10)
        )

  def test_loading_patches_from_different_path_than_dicom_slide_raises(self):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2 = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    test_instance_2.StudyInstanceUID = '1.2.3.4.555'
    series_1_path = f'{dicom_store_path}/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
    series_2_path = f'{dicom_store_path}/studies/{test_instance_2.StudyInstanceUID}/series/{test_instance_2.SeriesInstanceUID}'
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      mock_store[dicom_store_path].add_instance(test_instance_2)
      ds_1 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_1_path),
      )
      ds_1.init_slide_frame_cache()
      ds_2 = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          dicom_path.FromString(series_2_path),
      )
      with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
        ds_1.preload_patches_in_frame_cache([
            ds_2.get_patch(ds_2.native_level, 0, 0, 10, 10),
            ds_1.get_patch(ds_1.native_level, 0, 0, 10, 10),
        ])

  @parameterized.named_parameters([
      dict(
          testcase_name='accession_number',
          accession_number='ABC',
          expected='ABC:M_0.34520315963921067X:000100x000100+000010+000010',
      ),
      dict(
          testcase_name='no_accession_number',
          accession_number=None,
          expected='M_0.34520315963921067X:000100x000100+000010+000010',
      ),
  ])
  def test_patch_id(self, accession_number, expected):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
          accession_number=accession_number,
      )
      self.assertEqual(
          ds.get_patch(ds.native_level, 10, 10, 100, 100).id, expected
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='accession_number',
          accession_number='ABC',
          expected='ABC:M_0.09862947418263161X:000100x000100+000010+000010',
      ),
      dict(
          testcase_name='no_accession_number',
          accession_number=None,
          expected='M_0.09862947418263161X:000100x000100+000010+000010',
      ),
  ])
  def test_patch_id_resized(self, accession_number, expected):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
          accession_number=accession_number,
      )
      self.assertEqual(
          ds.get_patch(
              ds.native_level.resize(dicom_slide.ImageDimensions(200, 200)),
              10,
              10,
              100,
              100,
          ).id,
          expected,
      )

  def test_level_series_path(self):
    expected = 'https://healthcare.googleapis.com/v1/projects/project_name/locations/us-west1/datasets/dataset_name/dicomStores/dicom_store_name/dicomWeb/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634'
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      self.assertEqual(
          dicom_slide._get_level_series_path(ds.native_level), expected
      )
      self.assertEqual(
          dicom_slide._get_level_series_path(
              ds.native_level.resize((dicom_slide.ImageDimensions(100, 100)))
          ),
          expected,
      )

  def test_dicom_patch_level_is_level_type(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch = ds.get_patch(ds.native_level, 0, 0, 100, 100)
      self.assertIsInstance(patch.level, slide_level_map.Level)

  def test_resized_dicom_patch_level_is_resized_level_type(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch = ds.get_patch(
          ds.native_level.resize(dicom_slide.ImageDimensions(300, 300)),
          0,
          0,
          100,
          100,
      )
      self.assertIsInstance(patch.level, slide_level_map.ResizedLevel)

  def test_dicom_patch_level_frame_numbers(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch = ds.get_patch(ds.native_level, 0, 0, 100, 100)
      self.assertEqual(list(patch.frame_numbers()), [1])

  def test_resized_dicom_patch_level_frame_numbers(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch = ds.get_patch(
          ds.native_level.resize(dicom_slide.ImageDimensions(100, 100)),
          0,
          0,
          100,
          100,
      )
      self.assertEqual(list(patch.frame_numbers()), list(range(1, 16)))

  def test_pryamid_level_patch_cannot_be_created_source_level_not_on_slide(
      self,
  ):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      mk_level = mock.create_autospec(slide_level_map.Level, instance=True)
      type(mk_level).level_index = mock.PropertyMock(return_value=1)
      with self.assertRaises(ez_wsi_errors.LevelNotFoundError):
        dicom_slide._SlidePyramidLevelPatch(ds, mk_level, 0, 0, 10, 10)

  def test_pryamid_level_patch_equal(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 10, 10
      )
      patch_2 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 10, 10
      )
      self.assertEqual(patch_1, patch_2)

  def test_pryamid_level_patch_not_equal_different_coordinates(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 10, 10
      )
      patch_2 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 2, 10, 10
      )
      self.assertNotEqual(patch_1, patch_2)

  def test_pryamid_level_patch_not_equal_different_dim(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 10, 10
      )
      patch_2 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 20, 10
      )
      self.assertNotEqual(patch_1, patch_2)

  def test_pryamid_level_patch_not_equal_different_obj(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = dicom_slide._SlidePyramidLevelPatch(
          ds, ds.native_level, 0, 0, 10, 10
      )
      self.assertNotEqual(patch_1, 'a')

  def test_dicom_patch_equal(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = ds.get_patch(ds.native_level, 0, 0, 10, 10)
      patch_2 = ds.get_patch(ds.native_level, 0, 0, 10, 10)
      self.assertEqual(patch_1, patch_2)

  def test_dicom_patch_not_equal_different_coordinates(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = ds.get_patch(ds.native_level, 0, 0, 10, 10)
      patch_2 = ds.get_patch(ds.native_level, 0, 2, 10, 10)
      self.assertNotEqual(patch_1, patch_2)

  def test_dicom_patch_not_equal_different_dim(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = ds.get_patch(ds.native_level, 0, 0, 10, 10)
      patch_2 = ds.get_patch(ds.native_level, 0, 0, 20, 10)
      self.assertNotEqual(patch_1, patch_2)

  def test_dicom_patch_not_equal_different_obj(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      patch_1 = ds.get_patch(ds.native_level, 0, 0, 10, 10)
      self.assertNotEqual(patch_1, 'a')

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_images_not_equal_other_class(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3',
        '1.2.3.4',
        '1.2.3.4.5',
        sop_class_uid=sop_class_uid,
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      image_1 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      self.assertNotEqual(image_1, 'BAD')

  @parameterized.parameters(slide_level_map.UNTILED_IMAGE_SOP_CLASS_UID)
  def test_slide_microscopy_images_copy(self, sop_class_uid):
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance_1 = dicom_test_utils.create_test_dicom_instance(
        '1.2.3',
        '1.2.3.4',
        '1.2.3.4.5',
        sop_class_uid=sop_class_uid,
    )
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance_1)
      image_1 = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromString(
              f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance_1.StudyInstanceUID}/series/{test_instance_1.SeriesInstanceUID}'
          ),
      )
      image_2 = copy.copy(image_1)
      self.assertEqual(image_1, image_2)
      self.assertIsNot(image_1, image_2)
      self.assertIsNot(image_1._non_tiled_levels, image_2._non_tiled_levels)
      self.assertEqual(
          image_1._non_tiled_levels.to_dict(),
          image_2._non_tiled_levels.to_dict(),
      )

  def test_dicom_slide_copy(self):
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = dicom_path.FromString(
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    )
    dicom_store_path = str(series_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(dicom_store_path) as mock_store:
      mock_store[dicom_store_path].add_instance(test_instance)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          series_path,
      )
      ds_copy = copy.copy(ds)
      self.assertEqual(ds, ds_copy)
      self.assertIsNot(ds, ds_copy)
      self.assertIsNot(ds._level_map, ds_copy._level_map)
      self.assertEqual(
          ds._level_map.to_dict(),
          ds_copy._level_map.to_dict(),
      )


if __name__ == '__main__':
  absltest.main()
