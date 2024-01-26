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
import numpy as np
import PIL.Image


# DICOM Transfer syntax define the encoding of the pixel data in an instance.
# https://www.dicomlibrary.com/dicom/transfer-syntax/
class DicomTransferSyntax(enum.Enum):
  JPEG_BASELINE = '1.2.840.10008.1.2.4.50'
  JPEG_2000_LOSSLESS = '1.2.840.10008.1.2.4.90'
  JPEG_2000 = '1.2.840.10008.1.2.4.91'


_JPEG2000_TRANSFER_SYNTAXS = {
    DicomTransferSyntax.JPEG_2000_LOSSLESS.value,
    DicomTransferSyntax.JPEG_2000.value,
}


def can_decompress_dicom_transfer_syntax(transfer_syntax: str) -> bool:
  return transfer_syntax in (
      supported_syntax.value for supported_syntax in DicomTransferSyntax
  )


def decode_dicom_compressed_frame_bytes(
    frame: bytes, transfer_syntax: str
) -> Optional[np.ndarray]:
  """Decode compressed frame bytes to DICOM BGR image.

  Args:
    frame: Raw image bytes (compressed blob).
    transfer_syntax: DICOM transfer syntax frame pixels are encoded in.

  Returns:
    Decompressed image or None if decompression fails.
  """
  if transfer_syntax not in _JPEG2000_TRANSFER_SYNTAXS:
    result = cv2.imdecode(
        np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if result is not None:
      cv2.cvtColor(result, cv2.COLOR_BGR2RGB, dst=result)
      return result
  try:
    return np.asarray(PIL.Image.open(io.BytesIO(frame)))
  except PIL.UnidentifiedImageError:
    return None
