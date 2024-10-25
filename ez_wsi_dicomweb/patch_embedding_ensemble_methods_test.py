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
"""Tests for patch embedding ensemble methods."""

import enum
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import patch_embedding_ensemble_methods
from ez_wsi_dicomweb import patch_embedding_types
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
import pydicom

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


class TestEnum(enum.Enum):
  BAD = 'bad'


class PatchEmbeddingEnsembleMethodsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint()
    dicom_store_path = (
        f'{dicom_test_utils.DEFAULT_DICOMWEB_BASE_URL}/'
        f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb'
    )
    test_instance = pydicom.dcmread(
        dicom_test_utils.test_multi_frame_dicom_instance_path()
    )
    series_path = f'{dicom_test_utils.TEST_STORE_PATH}/dicomWeb/studies/{test_instance.StudyInstanceUID}/series/{test_instance.SeriesInstanceUID}'
    mock_store = self.enter_context(
        dicom_store_mock.MockDicomStores(dicom_store_path)
    )
    mock_store[dicom_store_path].add_instance(test_instance)
    self.mock_store_instance = mock_store[dicom_store_path]
    self.slide = dicom_slide.DicomSlide(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromString(series_path),
        enable_client_slide_frame_decompression=True,
    )
    self.ps = pixel_spacing.PixelSpacing.FromDicomPixelSpacingTag(
        test_instance.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='upper_left',
          location=patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT,
          expected_x=10,
          expected_y=0,
      ),
      dict(
          testcase_name='upper_right',
          location=patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_RIGHT,
          expected_x=286,
          expected_y=0,
      ),
      dict(
          testcase_name='lower_left',
          location=patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.LOWER_LEFT,
          expected_x=10,
          expected_y=376,
      ),
      dict(
          testcase_name='lower_right',
          location=patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.LOWER_RIGHT,
          expected_x=286,
          expected_y=376,
      ),
      dict(
          testcase_name='center',
          location=patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.CENTER,
          expected_x=148,
          expected_y=188,
      ),
  ])
  def test_gen_single_patch_ensemble(self, location, expected_x, expected_y):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(location)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=500,
        height=600,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    results = list(method.generate_ensemble(self.endpoint, test_patch))
    self.assertLen(results, 1)
    patch_embedding_source = results[0]
    self.assertIs(patch_embedding_source.patch.source, test_patch.source)
    self.assertEqual(patch_embedding_source.patch.x, expected_x)
    self.assertEqual(patch_embedding_source.patch.y, expected_y)
    self.assertEqual(
        patch_embedding_source.patch.width, self.endpoint.patch_width()
    )
    self.assertEqual(
        patch_embedding_source.patch.height, self.endpoint.patch_height()
    )

  @parameterized.parameters(['bad', TestEnum.BAD])
  def test_single_patch_ensemble_raises_passed_bad_position(self, location):
    with self.assertRaises(
        ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError
    ):
      patch_embedding_ensemble_methods.SinglePatchEnsemble(location)

  def test_gen_single_patch_ensemble_raises_bad_patch_position(self):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=10,
        y=0,
        width=500,
        height=600,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    method._position = 'bad'
    with self.assertRaises(
        ez_wsi_errors.SinglePatchEmbeddingEnsemblePositionError
    ):
      list(method.generate_ensemble(self.endpoint, test_patch))

  @parameterized.named_parameters([
      dict(
          testcase_name='xc_less_zero',
          x=-1,
          y=0,
          width=1024,
          height=2048,
      ),
      dict(
          testcase_name='yc_less_zero',
          x=0,
          y=-1,
          width=1024,
          height=2048,
      ),
  ])
  def test_gen_single_patch_ensemble_raises_bad_patch_coordinates(
      self, x, y, width, height
  ):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=x,
        y=y,
        width=width,
        height=height,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    method._position = 'bad'
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingDimensionError):
      list(method.generate_ensemble(self.endpoint, test_patch))

  @parameterized.named_parameters([
      dict(
          testcase_name='patch_width_less_endpoint_input_width',
          width=20,
          height=2048,
      ),
      dict(
          testcase_name='patch_height_less_endpoint_input_height',
          width=1024,
          height=20,
      ),
  ])
  def test_gen_single_patch_ensemble_raises_bad_patch_dimensions(
      self, width, height
  ):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=width,
        height=height,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    method._position = 'bad'
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingDimensionError):
      list(method.generate_ensemble(self.endpoint, test_patch))

  def test_single_patch_ensemble_reduce_ensemble_raises_zero_results(self):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    with self.assertRaises(ez_wsi_errors.SinglePatchEmbeddingEnsembleError):
      method.reduce_ensemble(test_patch, [])

  def test_single_patch_ensemble_reduce_ensemble_raises_more_than_one_result(
      self,
  ):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    result = patch_embedding_types.EmbeddingResult(
        test_patch, np.zeros((10, 10))
    )
    with self.assertRaises(ez_wsi_errors.SinglePatchEmbeddingEnsembleError):
      method.reduce_ensemble(test_patch, [result, result])

  def test_single_patch_ensemble_reduce_ensemble_success(self):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    mock_embedding = np.zeros((10, 10))
    result = patch_embedding_types.EmbeddingResult(test_patch, mock_embedding)
    ensemble_result = method.reduce_ensemble(test_patch, [result])
    self.assertIs(ensemble_result.patch, test_patch)
    self.assertIs(ensemble_result.embedding, mock_embedding)

  def test_default_single_patch_ensemble_success(self):
    method = patch_embedding_ensemble_methods.DefaultSinglePatchEnsemble()
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=224,
        height=224,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    self.assertLen(list(method.generate_ensemble(self.endpoint, test_patch)), 1)

  @parameterized.named_parameters([
      dict(
          testcase_name='patch_width_less_than_endpoint_input_width',
          width=223,
          height=224,
      ),
      dict(
          testcase_name='patch_width_greater_than_endpoint_input_width',
          width=225,
          height=224,
      ),
      dict(
          testcase_name='patch_height_less_than_endpoint_input_height',
          width=224,
          height=223,
      ),
      dict(
          testcase_name='patch_height_greater_than_endpoint_input_height',
          width=224,
          height=225,
      ),
  ])
  def test_default_single_patch_ensemble_raises_if_patch_dim_not_match_endpoint(
      self, width, height
  ):
    method = patch_embedding_ensemble_methods.DefaultSinglePatchEnsemble()
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=width,
        height=height,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingDimensionError):
      list(method.generate_ensemble(self.endpoint, test_patch))

  @parameterized.named_parameters([
      dict(
          testcase_name='smallest_step',
          step_x=1,
          step_y=1,
          expected=82944,
      ),
      dict(
          testcase_name='small_step',
          step_x=10,
          step_y=20,
          expected=435,
      ),
      dict(
          testcase_name='medium_step',
          step_x=200,
          step_y=100,
          expected=6,
      ),
      dict(
          testcase_name='large_step_single_patch',
          step_x=500,
          step_y=500,
          expected=1,
      ),
  ])
  def test_gen_mean_patch_embedding_ensemble(self, step_x, step_y, expected):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(
        step_x, step_y
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=512,
        height=512,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    self.assertLen(
        list(method.generate_ensemble(self.endpoint, test_patch)), expected
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='medium_step',
          step_x=200,
          step_y=100,
          expected=[
              (0, 0),
              (200, 0),
              (0, 100),
              (200, 100),
              (0, 200),
              (200, 200),
              (0, 300),
              (200, 300),
          ],
      ),
      dict(
          testcase_name='large_step_single_patch',
          step_x=500,
          step_y=500,
          expected=[(0, 0)],
      ),
  ])
  def test_gen_mean_patch_embedding_ensemble_patch_coordinates(
      self, step_x, step_y, expected
  ):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(
        step_x, step_y
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=512,
        height=512,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    for patch, expected_cooord in zip(
        method.generate_ensemble(self.endpoint, test_patch), expected
    ):
      self.assertIs(patch.ensemble_source_patch, test_patch)
      self.assertEqual(patch.patch.x, expected_cooord[0])
      self.assertEqual(patch.patch.y, expected_cooord[1])
      self.assertEqual(patch.patch.width, self.endpoint.patch_width())
      self.assertEqual(patch.patch.height, self.endpoint.patch_height())

  @parameterized.named_parameters([
      dict(
          testcase_name='zero_x',
          step_x=0,
          step_y=1,
      ),
      dict(
          testcase_name='zero_y',
          step_x=1,
          step_y=0,
      ),
  ])
  def test_gen_mean_patch_embedding_ensemble_raises_zero_step(
      self, step_x, step_y
  ):
    with self.assertRaises(ez_wsi_errors.MeanPatchEmbeddingEnsembleError):
      patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(
          step_x, step_y
      )

  def test_mean_patch_ensemble_reduce_ensemble_raises_zero_results(self):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(10, 10)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    with self.assertRaises(ez_wsi_errors.MeanPatchEmbeddingEnsembleError):
      method.reduce_ensemble(test_patch, [])

  def test_gen_mean_patch_embedding_ensemble_patch_size_validation(self):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(1, 1)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=200,
        height=200,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingDimensionError):
      list(method.generate_ensemble(self.endpoint, test_patch))

  def test_mean_patch_ensemble_reduce_float_ensemble_sucess(self):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(10, 10)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    embedding_result_1 = patch_embedding_types.EmbeddingResult(
        test_patch, np.asarray([1.0, 1.0])
    )
    embedding_result_2 = patch_embedding_types.EmbeddingResult(
        test_patch, np.asarray([2.0, 1.0])
    )
    result = method.reduce_ensemble(
        test_patch, [embedding_result_1, embedding_result_2]
    )
    self.assertIs(result.patch, test_patch)
    self.assertEqual(result.embedding.tolist(), [1.5, 1.0])

  def test_mean_patch_ensemble_reduce_int_ensemble_sucess(self):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(10, 10)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    embedding_result_1 = patch_embedding_types.EmbeddingResult(
        test_patch, np.asarray([1, 1], dtype=np.int32)
    )
    embedding_result_2 = patch_embedding_types.EmbeddingResult(
        test_patch, np.asarray([2, 1], dtype=np.int32)
    )
    result = method.reduce_ensemble(
        test_patch, [embedding_result_1, embedding_result_2]
    )
    self.assertIs(result.patch, test_patch)
    self.assertEqual(result.embedding.tolist(), [1, 1])

  def test_mean_patch_ensemble_reduce_ensemble_raises_if_reducing_error(self):
    method = patch_embedding_ensemble_methods.MeanPatchEmbeddingEnsemble(10, 10)
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    embedding_result_1 = patch_embedding_types.PatchEmbeddingEnsembleResult(
        patch_embedding_types.PatchEmbeddingSource(test_patch, test_patch, '1'),
        None,
        patch_embedding_types.PatchEmbeddingError(
            'error_code', 'error_message'
        ),
    )
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      method.reduce_ensemble(
          test_patch, [embedding_result_1, embedding_result_1]
      )

  def test_single_patch_ensemble_reduce_ensemble_raises_if_reducing_error(self):
    method = patch_embedding_ensemble_methods.SinglePatchEnsemble(
        patch_embedding_ensemble_methods.SinglePatchEnsemblePosition.UPPER_LEFT
    )
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=768,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    embedding_result_1 = patch_embedding_types.PatchEmbeddingEnsembleResult(
        patch_embedding_types.PatchEmbeddingSource(test_patch, test_patch, '1'),
        None,
        patch_embedding_types.PatchEmbeddingError(
            'error_code', 'error_message'
        ),
    )
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      method.reduce_ensemble(test_patch, [embedding_result_1])

  def test_five_patch_mean_ensemble_five_part_sampling(self):
    method = patch_embedding_ensemble_methods.FivePatchMeanEnsemble()
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=1024,
        height=700,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    patches = list(method.generate_ensemble(self.endpoint, test_patch))
    self.assertLen(patches, 5)
    expected_id = patches[0].ensemble_id
    expected_width = self.endpoint.patch_width()
    expected_height = self.endpoint.patch_height()
    expected_pos = [(0, 0), (800, 0), (400, 238), (0, 476), (800, 476)]
    for expected_pos, patch in zip(expected_pos, patches):
      self.assertEqual(patch.ensemble_id, expected_id)
      self.assertEqual(patch.patch.width, expected_width)
      self.assertEqual(patch.patch.height, expected_height)
      self.assertEqual((patch.patch.x, patch.patch.y), expected_pos)

  def test_five_patch_mean_ensemble_single_part_sampling(self):
    method = patch_embedding_ensemble_methods.FivePatchMeanEnsemble()
    test_patch = dicom_slide.DicomPatch(
        source=self.slide,
        x=0,
        y=0,
        width=224,
        height=224,
        source_image_level=self.slide.get_level_by_pixel_spacing(self.ps),
    )
    patches = list(method.generate_ensemble(self.endpoint, test_patch))
    self.assertLen(patches, 1)
    patch = patches[0]
    self.assertEqual(patch.patch.width, self.endpoint.patch_width())
    self.assertEqual(patch.patch.height, self.endpoint.patch_height())
    self.assertEqual((patch.patch.x, patch.patch.y), (0, 0))

  @parameterized.named_parameters([
      dict(
          testcase_name='int_input',
          input_list=[
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
          ],
          np_type=np.int32,
          expected=[4, 5, 6],
      ),
      dict(
          testcase_name='float_input',
          input_list=[
              [1.1, 2, 3],
              [4.1, 5, 6],
              [7.1, 8, 9],
          ],
          np_type=np.float64,
          expected=[4.1, 5.0, 6.0],
      ),
  ])
  def test_mean_patch_embedding(self, input_list, np_type, expected):
    sq_data = [
        patch_embedding_types.EmbeddingResult(
            mock.create_autospec(dicom_slide.DicomPatch, instance=True),
            np.asarray(data, dtype=np_type),
        )
        for data in input_list
    ]
    result = patch_embedding_ensemble_methods.mean_patch_embedding(sq_data)
    self.assertEqual(result.tolist(), expected)

  def test_mean_patch_embedding_raises_if_empty_input(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.MeanPatchEmbeddingEnsembleError,
        'MeanPatchEmbeddingEnsemble requires at least one embedding result.',
    ):
      patch_embedding_ensemble_methods.mean_patch_embedding([])


if __name__ == '__main__':
  absltest.main()
