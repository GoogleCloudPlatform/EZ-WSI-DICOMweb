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
"""Tests for ez_wsi_dicomweb.dicom_frame_decoder."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb.test_utils import dicom_test_utils
import numpy as np


def _read_test_jpg() -> bytes:
  with open(dicom_test_utils.testdata_path('google.jpg'), 'rb') as infile:
    return infile.read()


def _expected_np_array() -> np.ndarray:
  return np.load(
      dicom_test_utils.testdata_path('google.npy'), allow_pickle=False
  )


class DicomFrameDecoderTest(parameterized.TestCase):

  @parameterized.parameters([
      ('1.2.840.10008.1.2.4.50', True),
      ('1.2.840.10008.1.2.4.90', True),
      ('1.2.840.10008.1.2.4.91', True),
      ('1.2.840.10008.1.2', False),
      ('', False),
      ('1.2.3', False),
  ])
  def test_can_decompress_dicom_transfer_syntax(
      self, transfer_syntax, can_decode
  ):
    self.assertEqual(
        dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
            transfer_syntax
        ),
        can_decode,
    )

  @mock.patch.object(cv2, 'imdecode')
  def test_decode_compressed_frame_bytes_pil(self, mock_imdecode):
    mock_imdecode.return_value = None
    img = _read_test_jpg()
    self.assertTrue(dicom_frame_decoder._PIL_LOADED)
    decoded_img = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(img)
    self.assertTrue(np.array_equal(decoded_img, _expected_np_array()))

  def test_decode_compressed_frame_bytes_cv2(self):
    pil_loaded = dicom_frame_decoder._PIL_LOADED
    img = _read_test_jpg()
    try:
      dicom_frame_decoder._PIL_LOADED = False
      decoded_img = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(img)
      self.assertTrue(np.array_equal(decoded_img, _expected_np_array()))
    finally:
      dicom_frame_decoder._PIL_LOADED = pil_loaded


if __name__ == '__main__':
  absltest.main()
