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

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import ez_wsi_logging_factory
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb.ml_toolkit import dicom_path


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
          credential_factory_module.AbstractCredentialFactory
      ] = None,
      pixel_spacing_diff_tolerance: float = pixel_spacing.PIXEL_SPACING_DIFF_TOLERANCE,
      logging_factory: Optional[
          ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
      ] = None,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ] = None,
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
      logging_factory: The factory that EZ WSI uses to construct a logging
        interface.
      slide_frame_cache: Slide cache, init to share cache across stores.

    Returns:
      A DicomStore object.
    """
    self._enable_client_slide_frame_decompression = (
        enable_client_slide_frame_decompression
    )
    self.dicomstore_path = dicomstore_path
    if credential_factory is None:
      credential_factory = credential_factory_module.CredentialFactory()
    if logging_factory is None:
      self._logging_factory = ez_wsi_logging_factory.BasePythonLoggerFactory(
          ez_wsi_logging_factory.DEFAULT_EZ_WSI_PYTHON_LOGGER_NAME
      )
    else:
      self._logging_factory = logging_factory
    self.dicomweb = dicom_web_interface.DicomWebInterface(credential_factory)
    self._pixel_spacing_diff_tolerance = pixel_spacing_diff_tolerance
    self._slide_frame_cache = slide_frame_cache

  @property
  def slide_frame_cache(
      self,
  ) -> Optional[local_dicom_slide_cache.InMemoryDicomSlideCache]:
    """Returns DICOM slide frame cache used by slide."""
    return self._slide_frame_cache

  @slide_frame_cache.setter
  def slide_frame_cache(
      self,
      slide_frame_cache: Optional[
          local_dicom_slide_cache.InMemoryDicomSlideCache
      ],
  ) -> None:
    """Sets DICOM slide frame cache used by the store.

    Shared cache's configured using max_cache_frame_memory_lru_cache_size_bytes
    can be used to limit total cache memory utilization across multiple stores.
    It is not recommended to share non-LRU frame caches.

    Args:
      slide_frame_cache: Reference to slide frame cache.
    """
    self._slide_frame_cache = slide_frame_cache

  def init_slide_frame_cache(
      self,
      max_cache_frame_memory_lru_cache_size_bytes: int,
      number_of_frames_to_read: int = local_dicom_slide_cache.DEFAULT_NUMBER_OF_FRAMES_TO_READ_ON_CACHE_MISS,
      max_instance_number_of_frames_to_prefer_whole_instance_download: int = local_dicom_slide_cache.MAX_INSTANCE_NUMBER_OF_FRAMES_TO_PREFER_WHOLE_INSTANCE_DOWNLOAD,
      optimization_hint: local_dicom_slide_cache_types.CacheConfigOptimizationHint = local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM,
  ) -> local_dicom_slide_cache.InMemoryDicomSlideCache:
    """Init a shared DICOM slide frame cache that will back store slides.

    Args:
      max_cache_frame_memory_lru_cache_size_bytes: Maximum size of cache in
        bytes.  Ideally should be in hundreds of megabyts-to-gigabyte size. If
        None, no limit to size.
      number_of_frames_to_read: Number of frames to read on cache miss.
      max_instance_number_of_frames_to_prefer_whole_instance_download: Max
        number of frames to prefer downloading whole instances over retrieving
        frames in batch (Typically faster for small instances e.g. < 10,0000).
        Optimal threshold will depend on average size of instance frame data and
        the size of non frame instance metadata.
      optimization_hint: Optimize cache to minimize data latency or total
        queries to the DICOM store.

    Returns:
      DICOM slide frame cache initialized for the store.
    """
    self._slide_frame_cache = local_dicom_slide_cache.InMemoryDicomSlideCache(
        credential_factory=self.dicomweb.credential_factory,
        max_cache_frame_memory_lru_cache_size_bytes=max_cache_frame_memory_lru_cache_size_bytes,
        number_of_frames_to_read=number_of_frames_to_read,
        max_instance_number_of_frames_to_prefer_whole_instance_download=max_instance_number_of_frames_to_prefer_whole_instance_download,
        optimization_hint=optimization_hint,
        logging_factory=self._logging_factory,
    )
    return self._slide_frame_cache

  def remove_slide_frame_cache(self) -> None:
    """Remove reference to store level DICOM cache."""
    self._slide_frame_cache = None

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
        logging_factory=self._logging_factory,
        slide_frame_cache=self._slide_frame_cache,
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
        logging_factory=self._logging_factory,
        slide_frame_cache=self._slide_frame_cache,
    )

    if slide is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          'Error constructing DicomSlide for slide StudyInstanceUID: '
          f' {study_instance_uid}; SeriesInstanceUID: {series_instance_uid}.'
      )
    return slide
