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
"""Tests for local image."""

from absl.testing import absltest
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import local_image
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np
import PIL.Image


class LocalImageTest(absltest.TestCase):

  def test_local_image_from_file(self):
    img = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    self.assertEqual((img.width, img.height), (454, 156))
    self.assertEqual(img.filename, dicom_test_utils.test_jpeg_path())

  def test_local_image_from_open_file(self):
    with open(dicom_test_utils.test_jpeg_path(), 'rb') as i:
      img = local_image.LocalImage(i)
      self.assertEqual((img.width, img.height), (454, 156))
      self.assertEqual(img.filename, '')

  def test_local_image_from_numpy_array(self):
    with PIL.Image.open(dicom_test_utils.test_jpeg_path()) as i:
      i = np.asarray(i)
    img = local_image.LocalImage(np.asarray(i))
    self.assertEqual((img.width, img.height), (454, 156))
    self.assertEqual(img.filename, '')

  def test_local_image_from_file_can_return_source_image_bytes_metadata(self):
    img = local_image.LocalImage(dicom_test_utils.test_jpeg_path())
    self.assertNotEmpty(img.source_image_bytes_json_metadata())
    self.assertEqual(img.filename, dicom_test_utils.test_jpeg_path())

  def test_resize_local_image_from_file_source_image_bytes_metadata_raises(
      self,
  ):
    img = local_image.LocalImage(
        dicom_test_utils.test_jpeg_path(), gcs_image.ImageDimensions(3, 3)
    )
    self.assertEqual(img.filename, dicom_test_utils.test_jpeg_path())
    with self.assertRaises(ez_wsi_errors.GcsImageError):
      img.source_image_bytes_json_metadata()


if __name__ == '__main__':
  absltest.main()
