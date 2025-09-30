# Copyright 2025 Google LLC
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
"""Local DICOM File Interface."""

import io
import math
from typing import Any, Iterator, MutableMapping, Optional, Sequence, Set

import cachetools
from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags
import numpy as np
import pydicom


class _DicomRef:
  """A reference to a DICOM object."""

  def __init__(self, dicom_file_path: str):
    self._path = dicom_file_path
    tags_to_read = [
        tags.SOP_CLASS_UID,
        tags.STUDY_INSTANCE_UID,
        tags.ACCESSION_NUMBER,
    ]
    tags_to_read.extend(dicom_web_interface.DEFAULT_INSTANCE_TAGS)
    tags_to_read = [tag.number for tag in tags_to_read]
    with pydicom.dcmread(self._path, specific_tags=tags_to_read) as dicom_data:
      self._pydicom_metadata = dicom_data

  @property
  def dicom_file_path(self) -> str:
    return self._path

  @property
  def pydicom_metadata(self) -> pydicom.dataset.FileDataset:
    return self._pydicom_metadata

  @property
  def uid_tuple(self) -> tuple[str, str, str]:
    """Returns the UID tuple of the DICOM object."""
    metadata = self.pydicom_metadata
    return (
        metadata.StudyInstanceUID,
        metadata.SeriesInstanceUID,
        metadata.SOPInstanceUID,
    )

  @property
  def json_metadata(self) -> dict[str, Any]:
    """Returns the UID tuple of the DICOM object."""
    metadata = self.pydicom_metadata
    md = metadata.to_json_dict()
    metadata.file_meta.to_json_dict()
    md.update(metadata.file_meta.to_json_dict())
    return md

  @property
  def mock_dicom_store_path(self) -> dicom_path.Path:
    metadata = self.pydicom_metadata
    return dicom_path.FromString(
        f'https://mock_url/studies/{metadata.StudyInstanceUID}/series/{metadata.SeriesInstanceUID}/instances/{metadata.SOPInstanceUID}'
    )

  def get_icc_profile_bytes(self) -> bytes:
    """Returns the ICC profile bytes of the DICOM object."""
    metadata = self.pydicom_metadata
    if 'OpticalPathSequence' in metadata:
      for dataset in metadata.OpticalPathSequence:
        if 'ICCProfile' in dataset:
          return dataset.ICCProfile
    if 'ICCProfile' in metadata:
      return metadata.ICCProfile
    return b''


def _get_pixel_format(dcm: pydicom.dataset.FileDataset) -> np.dtype:
  bytes_per_sample = math.ceil(dcm.BitsAllocated / 8)
  if bytes_per_sample == 1:
    return np.uint8  # pytype: disable=bad-return-type  # numpy-scalars
  else:
    raise ez_wsi_errors.UnsupportedPixelFormatError(
        f'Pixel format not supported. BITS_ALLOCATED = {dcm.BitsAllocated}'
    )


class LocalDicomFileInterface:
  """A Python interface of the DICOMWeb API.

  It wraps around the HealthCare DICOMWeb API to fetch DICOM objects, such as
  studies, series, instances, and dicom_tags associated with those objects.
  """

  def __init__(
      self,
      dicom_file_paths: Sequence[str],
      max_pixel_data_lru_cache_size: int = 1000000000,
  ):
    """Constructor.

    Args:
      dicom_file_paths: Sequence of paths to local DICOM files.
      max_pixel_data_lru_cache_size: Maximum size of pixel data cache.
    """
    # Store mapping of all DICOM instances to their references
    self._dicoms: MutableMapping[tuple[str, str, str], _DicomRef] = {}
    # Long term LRU cache for storing DICOM instance pixel data.
    self._lru_cached_frame_bytes: MutableMapping[
        tuple[str, str, str, int], bytes
    ] = cachetools.LRUCache(
        maxsize=max_pixel_data_lru_cache_size, getsizeof=len
    )
    # Short term cache holds last loaded instance pixel data.
    # Stores whole instance.
    self._single_instance_cache: MutableMapping[
        tuple[str, str, str, int], bytes
    ] = {}
    # Cache key (study_uid, series_uid, instance_uid)
    self._single_instance_instance_cache_key = ('', '', '')
    # Load DICOM file references.
    for file_path in dicom_file_paths:
      dcm_ref = _DicomRef(file_path)
      self._dicoms[dcm_ref.uid_tuple] = dcm_ref

  def add_dicom(self, path: str) -> None:
    dcm_ref = _DicomRef(path)
    clear_pixel_data_cache = dcm_ref.uid_tuple in self._dicoms
    self._dicoms[dcm_ref.uid_tuple] = dcm_ref
    if clear_pixel_data_cache:
      self._lru_cached_frame_bytes.clear()
    if self._single_instance_instance_cache_key == dcm_ref.uid_tuple:
      self._single_instance_cache = {}
      self._single_instance_instance_cache_key = ('', '', '')

  def _get_instances(
      self, study_uid: str, series_uid: str
  ) -> Iterator[_DicomRef]:
    for dcm in self._dicoms.values():
      if (
          dcm.pydicom_metadata.StudyInstanceUID == study_uid or not study_uid
      ) and (
          dcm.pydicom_metadata.SeriesInstanceUID == series_uid or not series_uid
      ):
        yield dcm

  def get_study_uids(self) -> Set[str]:
    """Returns all study instance uids."""
    studies_returned = set()
    for dcm in self._dicoms.values():
      studies_returned.add(dcm.pydicom_metadata.StudyInstanceUID)
    return studies_returned

  def get_series_uids(self, study_uid: str) -> Set[str]:
    """Returns all series instance uids for the given study uid."""
    series_returned = set()
    for dcm in self._get_instances(study_uid, ''):
      series_returned.add(dcm.pydicom_metadata.SeriesInstanceUID)
    return series_returned

  def get_instances(
      self, study_uid: str, series_uid: str
  ) -> Sequence[dicom_web_interface.DicomObject]:
    """Gets all instances under the input parent path.

    Args:
      study_uid: The study uid to fetch instances from.
      series_uid: The series uid to fetch instances from.

    Returns:
      A sequence of the DICOM instance objects.
    """
    instances = []
    for dcm in self._get_instances(study_uid, series_uid):
      # pylint: disable=protected-access
      obj = dicom_web_interface._build_dicom_object(
          dicom_path.Type.INSTANCE,
          dcm.mock_dicom_store_path,
          dcm.json_metadata,
      )
      # pylint: enable=protected-access
      instances.append(obj)
    return instances

  def get_dicom_file_paths(
      self, study_uid: str, series_uid: str
  ) -> Sequence[str]:
    return [
        dcm.dicom_file_path
        for dcm in self._get_instances(study_uid, series_uid)
    ]

  def _cache_frame_bytes(
      self,
      study_uid: str,
      series_uid: str,
      instance_uid: str,
      frame_index: int,
      frame_bytes: bytes,
  ) -> None:
    """Cache instance frame bytes."""
    single_instance_cache_key = (study_uid, series_uid, instance_uid)
    if self._single_instance_instance_cache_key != single_instance_cache_key:
      if self._single_instance_cache:
        # store loaded instance frames in long term lru cache
        self._lru_cached_frame_bytes.update(self._single_instance_cache)
      # clear instance level cache
      self._single_instance_cache = {}
      self._single_instance_instance_cache_key = single_instance_cache_key
    # store frame in instance level cache.
    self._single_instance_cache[
        (study_uid, series_uid, instance_uid, frame_index)
    ] = frame_bytes

  def _get_cached_frame_bytes(
      self, study_uid: str, series_uid: str, instance_uid: str, frame_index: int
  ) -> Optional[bytes]:
    """Return cached instance frame bytes."""
    cache_key = (study_uid, series_uid, instance_uid, frame_index)
    if (
        study_uid,
        series_uid,
        instance_uid,
    ) == self._single_instance_instance_cache_key:
      return self._single_instance_cache.get(cache_key)
    return self._lru_cached_frame_bytes.get(cache_key)

  def get_decoded_frame_image(
      self,
      instance_path: dicom_path.Path,
      frame_index: int,
  ) -> np.ndarray:
    """Returns the decoded frame image for the given instance path and frame index."""
    study_uid, series_uid, instance_uid = (
        instance_path.study_uid,
        instance_path.series_uid,
        instance_path.instance_uid,
    )
    dcm = self._dicoms.get((study_uid, series_uid, instance_uid))
    if dcm is None:
      raise ez_wsi_errors.DicomSlideMissingError(
          f'DICOM instance with Study, Series, and Instance UIDs ({study_uid},'
          f' {series_uid}, {instance_uid}) not in local dicom file interface.'
      )
    frame_index -= 1
    fb = self._get_cached_frame_bytes(
        study_uid, series_uid, instance_uid, frame_index
    )
    if fb is not None:
      dcm = dcm.pydicom_metadata
      if (
          dcm.file_meta.TransferSyntaxUID
          in local_dicom_slide_cache.UNENCAPSULATED_TRANSFER_SYNTAXES
      ):
        return np.frombuffer(fb, _get_pixel_format(dcm)).reshape(
            dcm.Rows, dcm.Columns, dcm.SamplesPerPixel
        )
      else:
        return dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            fb, dcm.file_meta.TransferSyntaxUID
        )
    with io.BytesIO() as buffer:
      with open(dcm.dicom_file_path, 'rb') as infile:
        buffer.write(infile.read())
      buffer.seek(0)
      requested_frame_bytes = None
      instance_frame_accessor = local_dicom_slide_cache.InstanceFrameAccessor(
          buffer
      )
      for count, frame_bytes in enumerate(instance_frame_accessor):
        self._cache_frame_bytes(
            study_uid,
            series_uid,
            instance_uid,
            count,
            frame_bytes,
        )
        if count == frame_index:
          requested_frame_bytes = frame_bytes
      if requested_frame_bytes is None:
        raise ez_wsi_errors.InputFrameNumberOutOfRangeError(
            f'Frame {frame_index+1} not found. DICOM instance has'
            f' {len(instance_frame_accessor)} frames.'
        )
      dcm = dcm.pydicom_metadata
      if (
          dcm.file_meta.TransferSyntaxUID
          in local_dicom_slide_cache.UNENCAPSULATED_TRANSFER_SYNTAXES
      ):
        return np.frombuffer(
            requested_frame_bytes, _get_pixel_format(dcm)
        ).reshape(dcm.Rows, dcm.Columns, dcm.SamplesPerPixel)
      if not dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
          dcm.file_meta.TransferSyntaxUID
      ):
        raise ez_wsi_errors.UnsupportedTransferSyntaxError(
            'Unsupported transfer syntax for local dicom. DICOM transfer'
            f' syntax UID: {dcm.file_meta.TransferSyntaxUID}'
        )
      decoded_frame = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
          requested_frame_bytes, dcm.file_meta.TransferSyntaxUID
      )
      if decoded_frame is None:
        raise ez_wsi_errors.UnsupportedTransferSyntaxError(
            'Error occured decoding frame bytes (size='
            f'{len(requested_frame_bytes)} bytes). DICOM instance encoded in'
            f' Transfer Syntax UID: {dcm.file_meta.TransferSyntaxUID}'
        )
      return decoded_frame

  def get_icc_profile_bytes(self, study_uid: str, series_uid: str) -> bytes:
    for instance in self._get_instances(study_uid, series_uid):
      icc_profile_bytes = instance.get_icc_profile_bytes()
      if icc_profile_bytes is not None:
        return icc_profile_bytes
    return b''
