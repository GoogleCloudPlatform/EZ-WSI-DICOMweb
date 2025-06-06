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
"""Image encoding, decoding, and downsampling utility."""
import enum
import io
from typing import Optional

import cv2
import imagecodecs
import numpy as np
import PIL.Image


# DICOM Transfer syntax define the encoding of the pixel data in an instance.
# https://www.dicomlibrary.com/dicom/transfer-syntax/
class DicomTransferSyntax(enum.Enum):
  JPEG_BASELINE = '1.2.840.10008.1.2.4.50'
  JPEG_2000_LOSSLESS = '1.2.840.10008.1.2.4.90'
  JPEG_2000 = '1.2.840.10008.1.2.4.91'
  JPEGXL_LOSSLESS = '1.2.840.10008.1.2.4.110'
  JPEGXL_JPEG = '1.2.840.10008.1.2.4.111'
  JPEGXL = '1.2.840.10008.1.2.4.112'


_JPEG2000_TRANSFER_SYNTAXS = {
    DicomTransferSyntax.JPEG_2000_LOSSLESS.value,
    DicomTransferSyntax.JPEG_2000.value,
}

_JPEGXL_TRANSFER_SYNTAXS = {
    DicomTransferSyntax.JPEGXL_LOSSLESS.value,
    DicomTransferSyntax.JPEGXL_JPEG.value,
    DicomTransferSyntax.JPEGXL.value,
}


def can_decompress_dicom_transfer_syntax(transfer_syntax: str) -> bool:
  return transfer_syntax in (
      supported_syntax.value for supported_syntax in DicomTransferSyntax
  )


def _pad_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
  """Pad monochrome frames to 1 channel if channel is ommited."""
  if frame is not None and len(frame.shape) == 2:
    return np.expand_dims(frame, 2)
  return frame


def decode_dicom_compressed_frame_bytes(
    frame: bytes, transfer_syntax: str
) -> Optional[np.ndarray]:
  """Decode compressed frame bytes to RGB image.

  Args:
    frame: Raw image bytes (compressed blob).
    transfer_syntax: DICOM transfer syntax frame pixels are encoded in.

  Returns:
    Decompressed image or None if decompression fails.
  """
  if transfer_syntax == DicomTransferSyntax.JPEGXL_JPEG.value:
    # if JPGXL is encoded JPEG then extract JPG and decode using
    # using JPG pipeline.
    try:
      frame = imagecodecs.jpegxl_decode_jpeg(frame, numthreads=1)
      transfer_syntax = DicomTransferSyntax.JPEG_BASELINE.value
    except ValueError:
      pass
  if transfer_syntax in _JPEGXL_TRANSFER_SYNTAXS:
    return _pad_frame(imagecodecs.jpegxl_decode(frame, numthreads=1))
  if transfer_syntax not in _JPEG2000_TRANSFER_SYNTAXS:
    result = cv2.imdecode(
        np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if result is not None:
      cv2.cvtColor(result, cv2.COLOR_BGR2RGB, dst=result)
      return _pad_frame(result)
  try:
    with io.BytesIO(frame) as frame_bytes:
      with PIL.Image.open(frame_bytes) as p_image:
        return _pad_frame(np.asarray(p_image))
  except PIL.UnidentifiedImageError:
    return None
