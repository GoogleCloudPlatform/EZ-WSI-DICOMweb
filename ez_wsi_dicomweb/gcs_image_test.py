# Copyright 2024 Google LLC
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
"""Tests for gcs image."""

import io
import os
import shutil
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import slide_level_map
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import google.api_core.exceptions
import google.cloud.storage
import numpy as np
import PIL.Image

from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


class GcsImageTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            credential_factory,
            'get_default_gcp_project',
            return_value='MOCK_PROJECT',
        )
    )

  def test_gcs_image_init_from_non_npuint8_dtype_raises(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Unsupported image dtype: float64'
    ):
      gcs_image.GcsImage(np.zeros((10, 10)))

  def test_gcs_image_init_from_invalid_shape_raises(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Unsupported image shape: ()'
    ):
      gcs_image.GcsImage(np.zeros(tuple(), dtype=np.uint8))

  def test_gcs_image_init_from_invalid_saples_per_pixel_raises(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Unsupported image samples per pixel: 10'
    ):
      gcs_image.GcsImage(np.zeros((5, 5, 10), dtype=np.uint8))

  def test_gcs_image_init_bytes_empty_raises(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Image bytes is empty.'
    ):
      gcs_image.GcsImage(b'')

  def test_gcs_image_init_bytes_bad_raises(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Error decoding image bytes.'
    ):
      gcs_image.GcsImage(b'badf00d')

  def test_gcs_image_init_rgba_converted_to_rgb(self):
    image = gcs_image.GcsImage(np.zeros((5, 5, 4), dtype=np.uint8))
    self.assertEqual(image.image_bytes().shape, (5, 5, 3))

  @parameterized.parameters(['CMYK', 'LAB', 'RGBA'])
  def test_gcs_image_init_converted_to_rgb(self, mode):
    with io.BytesIO() as img_bytes:
      with PIL.Image.new(mode=mode, size=(1, 1)) as im:
        im.save(img_bytes, format='TIFF')
      image = gcs_image.GcsImage(img_bytes.getvalue())
      self.assertEqual(image.image_bytes().shape, (1, 1, 3))

  def test_gcs_image_init_converted_ycbr_to_rgb(self):
    with io.BytesIO() as img_bytes:
      with PIL.Image.new(mode='YCbCr', size=(1, 1)) as im:
        im.save(img_bytes, format='jpeg')
      image = gcs_image.GcsImage(img_bytes.getvalue())
      self.assertEqual(image.image_bytes().shape, (1, 1, 3))

  def test_gcs_image_init_with_unsupported_mode_raises(self):
    with io.BytesIO() as img_bytes:
      with PIL.Image.new(mode='1', size=(1, 1)) as im:
        im.save(img_bytes, format='TIFF')
      with self.assertRaisesRegex(
          ez_wsi_errors.GcsImageError, 'Unsupported image mode: 1'
      ):
        gcs_image.GcsImage(img_bytes.getvalue())

  def test_source_image_file_bytes(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as img:
      image = gcs_image.GcsImage(img.read())
    self.assertEqual(image.size_bytes_of_source_image, 63117)
    image.clear_source_image_compressed_bytes()
    # saves source bytes after clear
    self.assertEqual(image.size_bytes_of_source_image, 63117)

  def test_source_image_file_bytes_np_array_none(self):
    image = gcs_image.GcsImage(np.zeros((4, 4), dtype=np.uint8))
    self.assertIsNone(image.size_bytes_of_source_image)

  def test_source_image_file_bad_gcs_path(self):
    with self.assertRaisesRegex(
        ez_wsi_errors.GcsImageError, 'Invalid GCS URI: .*'
    ):
      gcs_image.GcsImage(dicom_test_utils.test_jpeg_path())

  def test_gcs_image_init_from_blob(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as _:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage(
            google.cloud.storage.Blob.from_string(
                'gs://test_bucket/test_image.jpg'
            )
        )
        self.assertEqual(image.image_bytes().shape, (156, 454, 3))

  def test_gcs_image_test_uri_does_not_call_auth(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as mock_get_credentials:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
        self.assertEqual(image.uri, 'gs://test_bucket/test_image.jpg')
        mock_get_credentials.assert_not_called()

  def test_gcs_image_test_uri_does_calls_auth(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as mock_get_credentials:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
        self.assertEqual(image.uri, 'gs://test_bucket/test_image.jpg')
        # triggers auth call
        self.assertIsNotNone(image.image_bytes())
        mock_get_credentials.assert_called_once()

  def test_gcs_image_height_initialized_correctly_from_gs_uri(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ):
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
        self.assertEqual(image.height, 156)

  def test_gcs_image_width_initialized_correctly_from_gs_uri(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ):
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage('gs://test_bucket/test_image.jpg')
        self.assertEqual(image.width, 454)

  def test_gcs_image_test_uri_with_bearer_token_does_not_call_auth(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with mock.patch.object(
        credential_factory.CredentialFactory,
        'get_credentials',
        autospec=True,
    ) as mock_get_credentials:
      with gcs_mock.GcsMock({'test_bucket': bucket_path}):
        image = gcs_image.GcsImage(
            'gs://test_bucket/test_image.jpg',
            credential_factory=credential_factory.TokenPassthroughCredentialFactory(
                '1.3.3'
            ),
        )
        self.assertEqual(image.uri, 'gs://test_bucket/test_image.jpg')
        mock_get_credentials.assert_not_called()

  def test_gcs_image_test_width_and_height(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual((image.width, image.height), (454, 156))
      self.assertEqual(image.image_bytes().shape[0:2], (156, 454))

  @parameterized.parameters(
      [(200, 100), (454, 156), (454, 200), (500, 156), (500, 200)]
  )
  def test_gcs_image_test_resample(self, new_width, new_height):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
          image_dimensions=slide_level_map.ImageDimensions(
              new_width, new_height
          ),
      )
      self.assertEqual((image.width, image.height), (new_width, new_height))
      self.assertEqual(image.image_bytes().shape[0:2], (new_height, new_width))

  def test_gcs_get_whole_image_as_patch(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_image_as_patch()
      self.assertTrue(patch.is_patch_fully_in_source_image())
      self.assertEqual((patch.x, patch.y), (0, 0))
      self.assertEqual((patch.width, patch.height), (image.width, image.height))
      self.assertIs(patch.source, image)
      np.testing.assert_array_equal(patch.image_bytes(), image.image_bytes())

  def test_gcs_image_get_patch(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_patch(10, 20, 50, 80)
      self.assertTrue(patch.is_patch_fully_in_source_image())
      self.assertEqual((patch.x, patch.y), (10, 20))
      self.assertEqual((patch.width, patch.height), (50, 80))
      self.assertIs(patch.source, image)
      np.testing.assert_array_equal(
          patch.image_bytes(), image.image_bytes()[20:100, 10:60, :]
      )

  def test_gcs_image_get_patch_round_trip_from_json(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_patch(10, 20, 50, 80)
      patch_metadata = patch.json_metadata()
      gen_patch = gcs_image.GcsPatch.create_from_json(patch_metadata)
      self.assertEqual((gen_patch.x, gen_patch.y), (0, 0))
      self.assertEqual((gen_patch.width, gen_patch.height), (50, 80))
      np.testing.assert_array_equal(
          gen_patch.image_bytes(), patch.image_bytes()
      )
      self.assertTrue(gen_patch.is_patch_fully_in_source_image())
      self.assertNotEqual(gen_patch, patch)

  def test_gcs_image_get_patch_from_patch(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_image_as_patch()
      new_patch = patch.get_patch(10, 20, 50, 80)
      np.testing.assert_array_equal(
          new_patch.image_bytes(), image.image_bytes()[20:100, 10:60, :]
      )

  def test_gcs_image_equal_non_image(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertNotEqual(image, 'abc')

  def test_gcs_patch_equal_non_patch(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_image_as_patch()
      self.assertNotEqual(patch, 'abc')

  def test_gcs_patch_not_in_image(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      patch = image.get_patch(0, 0, 500, 500)
      self.assertFalse(patch.is_patch_fully_in_source_image())

  def test_create_icc_profile_transformation_no_icc_profile(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      transformation = image.create_icc_profile_transformation(
          dicom_slide.get_srgb_icc_profile_bytes()
      )
      self.assertIsNone(transformation)

  @parameterized.named_parameters([
      dict(
          testcase_name='srgb',
          icc_profile_bytes=dicom_slide.get_srgb_icc_profile_bytes(),
      ),
      dict(
          testcase_name='adobergb',
          icc_profile_bytes=dicom_slide.get_adobergb_icc_profile_bytes(),
      ),
      dict(
          testcase_name='rommrgb',
          icc_profile_bytes=dicom_slide.get_rommrgb_icc_profile_bytes(),
      ),
  ])
  def test_create_icc_profile_transformation_has_icc_profile(
      self, icc_profile_bytes
  ):
    bucket_path = self.create_tempdir()
    with PIL.Image.open(dicom_test_utils.test_jpeg_path()) as img:
      img.save(
          os.path.join(bucket_path, 'test_image.jpg'),
          icc_profile=icc_profile_bytes,
      )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      transformation = image.create_icc_profile_transformation(
          dicom_slide.get_srgb_icc_profile_bytes()
      )
      self.assertIsNotNone(transformation)

  @parameterized.named_parameters([
      dict(testcase_name='above_overlap', x=0, y=-1, w=10, h=10),
      dict(testcase_name='left_overlap', x=-1, y=0, w=10, h=10),
      dict(testcase_name='above_left_overlap', x=-1, y=-1, w=10, h=10),
      dict(testcase_name='fully_above', x=0, y=-20, w=10, h=10),
      dict(testcase_name='fully_left', x=-20, y=0, w=10, h=10),
      dict(testcase_name='fully_outside', x=-20, y=-20, w=10, h=10),
      dict(testcase_name='fully_wrap', x=-1, y=-1, w=456, h=158),
      dict(testcase_name='below_overlap', x=0, y=147, w=10, h=10),
      dict(testcase_name='right_overlap', x=455, y=0, w=10, h=10),
      dict(testcase_name='below_right_overlap', x=455, y=147, w=10, h=10),
      dict(testcase_name='fully_below_overlap', x=0, y=156, w=10, h=10),
      dict(testcase_name='fully_right_overlap', x=454, y=0, w=10, h=10),
      dict(testcase_name='fully_below_right_overlap', x=454, y=156, w=10, h=10),
  ])
  def test_get_patch_raises_out_of_image_bounds_error(self, x, y, w, h):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual((image.width, image.height), (454, 156))
      with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
        image.get_patch(x, y, w, h, True)

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
  def test_get_patch_raises_out_of_downsample_image_bounds_error(
      self, x, y, w, h
  ):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
          image_dimensions=gcs_image.ImageDimensions(20, 20),
      )
      self.assertEqual((image.width, image.height), (20, 20))
      with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
        image.get_patch(x, y, w, h, True)

  def test_get_patch_override_raises_image_bounds_error(
      self,
  ):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
          image_dimensions=gcs_image.ImageDimensions(20, 20),
      )
      patch = image.get_patch(0, 0, 10, 10, False)
      with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
        patch.get_patch(0, 0, 50, 50, True)

  def test_get_patch_default_raises_image_bounds_error(
      self,
  ):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
          image_dimensions=gcs_image.ImageDimensions(20, 20),
      )
      patch = image.get_patch(0, 0, 10, 10, True)
      with self.assertRaises(ez_wsi_errors.PatchOutsideOfImageDimensionsError):
        patch.get_patch(0, 0, 50, 50)

  def test_raises_if_initialized_with_bad_image_bytes(self):
    with self.assertRaises(ez_wsi_errors.GcsImageError):
      gcs_image.GcsImage(b'badf00d')

  def test_GcsPatch_raises_if_initalized_json_with_bad_str(self):
    with self.assertRaises(ez_wsi_errors.GcsImageError):
      gcs_image.GcsPatch.create_from_json('%%')

  def test_GcsImage_raises_if_initalized_json_with_bad_str(self):
    with self.assertRaises(ez_wsi_errors.GcsImageError):
      gcs_image.GcsImage.create_from_json('%%')

  def test_get_state_clears_credientals(self):
    bearer_token = '1.3.3'
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            bearer_token
        ),
    )
    self.assertEqual(image.credentials.token, bearer_token)
    state = image.__getstate__()
    self.assertNotIn('_credentials', state)

  def test_set_state_clears_credientals(self):
    bearer_token = '1.3.3'
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            bearer_token
        ),
    )
    self.assertEqual(image.credentials.token, bearer_token)
    state = image.__getstate__()
    image.__setstate__(state)
    self.assertIsNone(image._credentials)
    self.assertEqual(image.credentials.token, bearer_token)

  def test_get_credentials_pass_through(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    cl = image.credentials
    # repeated calls to credentials returns same object
    self.assertIs(cl, image.credentials)

  def test_get_credentials_no_auth(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.NoAuthCredentialsFactory(),
    )
    cl = image.credentials
    # repeated calls to credentials returns same object
    self.assertIs(cl, image.credentials)

  def test_get_credentials_headers_from_no_auth_credential(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.NoAuthCredentialsFactory(),
    )
    cl = image.credentials
    self.assertEqual(image.get_credential_header(cl), {})

  def test_get_credentials_headers_from_token_pass_through_credential(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    cl = image.credentials
    self.assertEqual(
        image.get_credential_header(cl), {'authorization': 'Bearer 1.3.3'}
    )

  @parameterized.parameters([None, gcs_image.ImageDimensions(10, 20)])
  def test_gcs_images_equal(self, image_dimensions):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=image_dimensions,
    )
    image2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=image_dimensions,
    )
    self.assertEqual(image1, image2)

  def test_gcs_images_not_equal_different_dimensions(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 20),
    )
    image2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(20, 10),
    )
    self.assertNotEqual(image1, image2)

  def test_gcs_images_not_equal_different_gcs_path(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_1.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 20),
    )
    image2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_2.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    self.assertNotEqual(image1, image2)

  def test_gcs_images_not_equal_different_dimension(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_1.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 20),
    )
    image2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_1.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(20, 10),
    )
    self.assertNotEqual(image1, image2)

  def test_gcs_images_not_equal_dimension_and_no_dim(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_1.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 20),
    )
    image2 = gcs_image.GcsImage(
        'gs://test_bucket/test_image_1.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    self.assertNotEqual(image1, image2)

  def test_gcs_images_equal_image_bytes(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
        img_bytes = infile.read()
      image2 = gcs_image.GcsImage(img_bytes)
      self.assertEqual(image1, image2)
      self.assertEqual(image2, image1)

  def test_gcs_images_dim_not_equal_image_bytes(self):
    image1 = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 20),
    )
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
        img_bytes = infile.read()
      image2 = gcs_image.GcsImage(img_bytes)
      self.assertNotEqual(image1, image2)
      self.assertNotEqual(image2, image1)

  def test_gcs_images_bytes_equal(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image1 = gcs_image.GcsImage(
        img_bytes,
    )
    image2 = gcs_image.GcsImage(img_bytes)
    self.assertEqual(image1, image2)
    self.assertEqual(image2, image1)

  def test_gcs_images_bytes_not_equal(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image1 = gcs_image.GcsImage(
        np.zeros((4, 4), dtype=np.uint8),
    )
    image2 = gcs_image.GcsImage(img_bytes)
    self.assertNotEqual(image1, image2)
    self.assertNotEqual(image2, image1)

  def test_gcs_images_bytes_not_resized(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image = gcs_image.GcsImage(img_bytes)
    self.assertFalse(image.is_resized)

  def test_gcs_images_bytes_no_change_in_size(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image = gcs_image.GcsImage(
        img_bytes, image_dimensions=gcs_image.ImageDimensions(454, 156)
    )
    self.assertFalse(image.is_resized)

  def test_gcs_images_bytes_resized(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image = gcs_image.GcsImage(
        img_bytes, image_dimensions=gcs_image.ImageDimensions(10, 10)
    )
    self.assertTrue(image.is_resized)

  def test_gcs_image_source_image_not_resized(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    self.assertFalse(image.are_image_bytes_loaded)
    self.assertFalse(image.is_resized)

  def test_gcs_image_source_image_resized(self):
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
        image_dimensions=gcs_image.ImageDimensions(10, 10),
    )
    self.assertFalse(image.are_image_bytes_loaded)
    self.assertTrue(image.is_resized)

  def test_gcs_image_source_image_bytes_per_pixel(self):
    bucket_path = self.create_tempdir()
    shutil.copyfile(
        dicom_test_utils.test_jpeg_path(),
        os.path.join(bucket_path, 'test_image.jpg'),
    )
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual(image.bytes_pre_pixel, 3)

  def test_gcs_image_source_bw_image_bytes_per_pixel(self):
    bucket_path = self.create_tempdir()
    with PIL.Image.fromarray(np.zeros((20, 20), dtype=np.uint8)) as im:
      im.save(os.path.join(bucket_path, 'test_image.jpg'))
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual(image.bytes_pre_pixel, 1)

  def test_gcs_image_source_bw_image_bytes(self):
    bucket_path = self.create_tempdir()
    with PIL.Image.fromarray(np.zeros((20, 20), dtype=np.uint8)) as im:
      im.save(os.path.join(bucket_path, 'test_image.jpg'))
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual(image.image_bytes().shape, (20, 20))

  def test_gcs_image_source_bw_image_patch_bytes(self):
    bucket_path = self.create_tempdir()
    with PIL.Image.fromarray(np.zeros((20, 20), dtype=np.uint8)) as im:
      im.save(os.path.join(bucket_path, 'test_image.jpg'))
    with gcs_mock.GcsMock({'test_bucket': bucket_path}):
      image = gcs_image.GcsImage(
          'gs://test_bucket/test_image.jpg',
          credential_factory=credential_factory.TokenPassthroughCredentialFactory(
              '1.3.3'
          ),
      )
      self.assertEqual(
          image.get_image_as_patch().image_bytes().shape, (20, 20, 1)
      )

  @mock.patch.object(
      google.cloud.storage, 'Client', autospec=True, return_value=None
  )
  @mock.patch.object(
      google.cloud.storage.Blob, 'download_as_bytes', autospec=True
  )
  def test_gcs_image_retry(self, download_as_bytes_mock, unused_mock):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as infile:
      img_bytes = infile.read()
    image = gcs_image.GcsImage(
        'gs://test_bucket/test_image.jpg',
        credential_factory=credential_factory.TokenPassthroughCredentialFactory(
            '1.3.3'
        ),
    )
    download_as_bytes_mock.side_effect = [
        google.api_core.exceptions.TooManyRequests('bad', 'bad', 'bad', 'bad'),
        google.api_core.exceptions.Unauthorized('bad', 'bad', 'bad', 'bad'),
        img_bytes,
    ]
    self.assertEqual(image.image_bytes().shape, (156, 454, 3))
    self.assertEqual(download_as_bytes_mock.call_count, 3)


if __name__ == '__main__':
  absltest.main()
