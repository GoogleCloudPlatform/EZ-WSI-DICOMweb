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


def _read_test_img(path: str) -> bytes:
  with open(dicom_test_utils.testdata_path(path), 'rb') as infile:
    return infile.read()


def _read_test_jpg() -> bytes:
  return _read_test_img('google.jpg')


def _expected_jpeg_np_array() -> np.ndarray:
  return np.load(
      dicom_test_utils.testdata_path('google.npy'), allow_pickle=False
  )


def _rgb_image_almost_equal(
    image_1: np.ndarray, image_2: np.ndarray, threshold: int = 3
) -> bool:
  """Test image RGB bytes values are close."""
  return np.all(np.abs(image_1 - image_2) < threshold)


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
    decoded_img = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        img, dicom_frame_decoder.DicomTransferSyntax.JPEG_BASELINE.value
    )
    self.assertTrue(
        _rgb_image_almost_equal(decoded_img, _expected_jpeg_np_array())
    )

  def test_decode_compressed_frame_bytes_cv2(self):
    img = _read_test_jpg()
    decoded_img = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        img, dicom_frame_decoder.DicomTransferSyntax.JPEG_BASELINE.value
    )
    self.assertTrue(
        _rgb_image_almost_equal(decoded_img, _expected_jpeg_np_array())
    )

  def test_decode_jpeg2k_frame_bytes_similar_to_jpeg_bytes(self):
    jpeg_2000 = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        _read_test_img('dcm_frame_6.j2k'),
        dicom_frame_decoder.DicomTransferSyntax.JPEG_2000.value,
    )
    jpeg = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
        _read_test_img('dcm_frame_6.jpg'),
        dicom_frame_decoder.DicomTransferSyntax.JPEG_BASELINE.value,
    )
    self.assertTrue(_rgb_image_almost_equal(jpeg_2000, jpeg))

  @mock.patch.object(cv2, 'imdecode', autospec=True)
  def test_decode_invalid_jpeg2k_frame_bytes_returns_none(self, im_decode_mock):
    self.assertIsNone(
        dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            b'345723985472345',
            dicom_frame_decoder.DicomTransferSyntax.JPEG_2000.value,
        )
    )
    im_decode_mock.assert_not_called()


if __name__ == '__main__':
  absltest.main()
