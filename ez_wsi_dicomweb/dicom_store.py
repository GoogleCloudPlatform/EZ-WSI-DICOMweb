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
"""Dicom Web abstraction layer."""
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.dicom_slide import DicomSlide

from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import dicom_web


class DicomStore:
  """Abstraction on top of DicomWeb.

  Designed to be the layer of internal PathDB. Provides the single point of
  access to the hierarchy of slide/image/patch.
  """

  def __init__(
      self,
      dicomstore_path: str,
      enable_client_slide_frame_decompression: bool,
      dicomweb=None,
  ):
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )
    self.dicomstore_path = dicomstore_path
    self.dicomweb = dicomweb
    if dicomweb is None:
      dicom_web_interface.DicomWebInterface(dicom_web.DicomWebClientImpl())

  def get_slide_by_accession_number(self, accession_number: str) -> DicomSlide:
    """Searches a DicomSlide object by accession number.

    Args:
      accession_number: DICOM tag, e.g. AccessionNumber=123.

    Returns:
      A DicomSlide object.

    Raises:
      UnexpectedDicomSlideCountError if the slide count for the series is not 1.
    """
    dicom = self.dicomweb.get_series(
        dicom_path.FromString(self.dicomstore_path),
        {'AccessionNumber': accession_number},
    )
    if len(dicom) != 1:
      raise ez_wsi_errors.UnexpectedDicomSlideCountError(
          f'Expect single slide for {accession_number}, len(dicom)={len(dicom)}'
      )

    return DicomSlide(
        self.dicomweb,
        dicom[0].path,
        enable_client_slide_frame_decompression=self._enable_client_slide_frame_decompression,
        accession_number=accession_number,
    )

  def get_slide(self, study_id_series_id: str) -> DicomSlide:
    """Gets a DicomSlide object.

    Args:
      study_id_series_id: Colon ':' concatenated study Id and series Id.

    Returns:
      A DicomSlide object.

    Raises:
      DicomSlideMissingError if the slide is not constructed correctly.
    """
    store_path = dicom_path.FromString(self.dicomstore_path)
    study_uid, series_uid = study_id_series_id.split(':')
    series_path = dicom_path.FromPath(
        store_path, study_uid=study_uid, series_uid=series_uid
    )
    slide = DicomSlide(
        self.dicomweb,
        series_path,
        enable_client_slide_frame_decompression=self._enable_client_slide_frame_decompression,
    )

    if slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          f'Error constructing DicomSlide for slide {study_id_series_id}.'
      )
    return slide
