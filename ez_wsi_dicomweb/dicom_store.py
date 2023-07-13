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
from typing import Optional

from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import dicomweb_credential_factory
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import pixel_spacing

from hcls_imaging_ml_toolkit import dicom_path


class DicomStore:
  """Abstraction on top of DicomWeb.

  Designed to be the layer of internal PathDB. Provides the single point of
  access to the hierarchy of slide/image/patch.
  """

  def __init__(
      self,
      dicomstore_path: str,
      enable_client_slide_frame_decompression: bool = True,
      credential_factory: Optional[
          dicomweb_credential_factory.AbstractCredentialFactory
      ] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing.PIXEL_SPACING_DIFF_TOLERANCE,
  ):
    """Creates a DicomStore object.

    Args:
      dicomstore_path: The path to the dicom store.
      enable_client_slide_frame_decompression: determines whether frames should
        be decompressed server side or client side. Client side reduces data
        transfer.
      credential_factory: The factory that EZ WSI uses to construct the
        credentials needed to access the DICOM store
      pixel_spacing_diff_tolerance: The tolerance (percentage difference) for
        difference between row and column pixel spacings. This will be used when
        creating DicomSlide objects.

    Returns:
      A DicomStore object.
    """
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )
    self.dicomstore_path = dicomstore_path
    if credential_factory is None:
      credential_factory = dicomweb_credential_factory.CredentialFactory()
    self.dicomweb = dicom_web_interface.DicomWebInterface(credential_factory)
    self._pixel_spacing_diff_tolerance = pixel_spacing_diff_tolerance

  def get_slide_by_accession_number(
      self, accession_number: str
  ) -> dicom_slide.DicomSlide:
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

    return dicom_slide.DicomSlide(
        self.dicomweb,
        dicom[0].path,
        enable_client_slide_frame_decompression=self._enable_client_slide_frame_decompression,
        accession_number=accession_number,
        pixel_spacing_diff_tolerance=self._pixel_spacing_diff_tolerance,
    )

  def get_slide(
      self, study_instance_uid: str, series_instance_uid: str
  ) -> dicom_slide.DicomSlide:
    """Gets a DicomSlide object.

    Args:
      study_instance_uid: DICOM study instance UID.
      series_instance_uid: DICOM study instance UID.

    Returns:
      A DicomSlide object.

    Raises:
      DicomSlideMissingError if the slide is not constructed correctly.
    """
    store_path = dicom_path.FromString(self.dicomstore_path)
    series_path = dicom_path.FromPath(
        store_path, study_uid=study_instance_uid, series_uid=series_instance_uid
    )
    slide = dicom_slide.DicomSlide(
        self.dicomweb,
        series_path,
        enable_client_slide_frame_decompression=self._enable_client_slide_frame_decompression,
        pixel_spacing_diff_tolerance=self._pixel_spacing_diff_tolerance,
    )

    if slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Error constructing DicomSlide for slide StudyInstanceUID: '
          f' {study_instance_uid}; SeriesInstanceUID: {series_instance_uid}.'
      )
    return slide
