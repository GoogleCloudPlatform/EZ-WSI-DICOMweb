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
"""DICOMWeb API Interface."""
import dataclasses
import enum
import posixpath
from typing import Any, Collection, Dict, List, MutableMapping, Optional, Sequence
import urllib

from absl import logging
from ez_wsi_dicomweb import dicomweb_credential_factory
from ez_wsi_dicomweb import ez_wsi_errors
from hcls_imaging_ml_toolkit import dicom_json
from hcls_imaging_ml_toolkit import dicom_path
from hcls_imaging_ml_toolkit import dicom_web
from hcls_imaging_ml_toolkit import tags
import google.auth.credentials


# The default Google HealthCare DICOMWeb API entry.
DEFAULT_DICOMWEB_BASE_URL = 'https://healthcare.googleapis.com/v1'
# The default DICOM tags to retrieve for an instance.
_DEFAULT_INSTANCE_TAGS = (
    tags.SOP_INSTANCE_UID,
    tags.INSTANCE_NUMBER,
    tags.ROWS,
    tags.COLUMNS,
    tags.SAMPLES_PER_PIXEL,
    tags.BITS_ALLOCATED,
    tags.HIGH_BIT,
    tags.PIXEL_REPRESENTATION,
    tags.PIXEL_SPACING,
    tags.NUMBER_OF_FRAMES,
    tags.FRAME_INCREMENT_POINTER,
    tags.IMAGE_VOLUME_WIDTH,
    tags.IMAGE_VOLUME_HEIGHT,
    tags.IMAGE_VOLUME_DEPTH,
    tags.TOTAL_PIXEL_MATRIX_COLUMNS,
    tags.TOTAL_PIXEL_MATRIX_ROWS,
    tags.TOTAL_PIXEL_MATRIX_ORIGIN_SEQUENCE,
    tags.SPECIMEN_LABEL_IN_IMAGE,
    tags.EXTENDED_DEPTH_OF_FIELD,
    tags.LABEL_TEXT,
    tags.BARCODE_VALUE,
    tags.CONCATENATION_FRAME_OFFSET_NUMBER,
    tags.IMAGE_TYPE,
    tags.TRANSFER_SYNTAX_UID,
)


# Transfer syntax defining how DICOM frames should be transcoded.
# https://cloud.google.com/healthcare-api/docs/dicom#dicom_frames
class TranscodeDicomFrame(enum.Enum):
  DO_NOT_TRANSCODE = '*'
  UNCOMPRESSED_LITTLE_ENDIAN = '1.2.840.10008.1.2.1'


@dataclasses.dataclass(frozen=True)
class DicomObject:
  """Represents a DICOM object by its path and DICOM tags.

  Attributes:
    path: The DICOM path to the object in a DICOMStore.
    dicom_tags: A list of DICOM tags that are associated with the DICOM object.
  """

  path: dicom_path.Path
  dicom_tags: Dict[str, Any]

  def __post_init__(self):
    """Post __init__.

    Raises:
      DicomPathError if self.path type is not STUDY, SERIES or
      INSTANCE.
    """
    if self.path.type not in (
        dicom_path.Type.STUDY,
        dicom_path.Type.SERIES,
        dicom_path.Type.INSTANCE,
    ):
      raise ez_wsi_errors.DicomPathError(
          f'The DICOM path has an invalid type: {self.path.type}'
      )

  def type(self) -> dicom_path.Type:
    """Returns the type of the DICOM  path."""
    return self.path.type

  def get_value(self, tag: tags.DicomTag) -> Any:
    """Returns the first value for the tag in dicom_tags.

    For many DICOM tags this is going to be the only value in the list. If you
    wish to fetch the entire list of tags instead, use get_list_value().

    Args:
      tag: The tag to lookup values for.

    Returns:
      The first value from dicom_tags corresponding to the tag or None if the
      tag is not present in dicom_tags.
    """
    return dicom_json.GetValue(self.dicom_tags, tag)

  def get_list_value(self, tag: tags.DicomTag) -> Optional[List[Any]]:
    """Returns the value list for the tag from dicom_tags.

    Args:
      tag: The tag to lookup values for.

    Returns:
      The value list corresponding to the tag or None if the tag is not present
      in dicom_tags.
    """
    return dicom_json.GetList(self.dicom_tags, tag)


class DicomWebInterface:
  """A Python interface of the DICOMWeb API.

  It wraps around the HealthCare DICOMWeb API to fetch DICOM objects, such as
  studies, series, instances, and dicom_tags associated with those objects.
  """

  def __init__(
      self,
      credential_factory: dicomweb_credential_factory.AbstractCredentialFactory,
      dicom_web_base_url: str = DEFAULT_DICOMWEB_BASE_URL,
  ):
    """Constructor.

    Args:
      credential_factory: Factory that creates DICOMweb credentials.
      dicom_web_base_url: Optional parameter that is used as the base of the url
        for all REST calls. Defaults to the Google HealthCare DICOMWeb API base.
    """
    self._credentials = None
    self._dicom_web_client = None
    self._dicom_web_base_url = dicom_web_base_url.rstrip('/')
    self._dicomweb_credential_factory = credential_factory

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns class state for pickle serialization."""
    state = self.__dict__.copy()
    del state['_credentials']
    del state['_dicom_web_client']
    return state

  def __setstate__(self, dct: MutableMapping[str, Any]) -> None:
    """Init class state from pickle serialization."""
    self.__dict__ = dct
    self._credentials = None
    self._dicom_web_client = None

  @property
  def dicomweb_credential_factory(
      self,
  ) -> dicomweb_credential_factory.AbstractCredentialFactory:
    return self._dicomweb_credential_factory

  @property
  def dicom_web_base_url(self) -> str:
    return self._dicom_web_base_url

  def credentials(self) -> google.auth.credentials.Credentials:
    """Returns credentials used to acccess DICOM store."""
    if self._credentials is None:
      self._credentials = self._dicomweb_credential_factory.get_credentials()
    dicomweb_credential_factory.refresh_credentials(self._credentials)
    return self._credentials

  @property
  def dicom_web_client(self) -> dicom_web.DicomWebClient:
    """Returns DicomWebClient with refreshed credientals."""
    # Get and refresh, if needed, the DICOM store credentials.
    # The dicom_web_client holds the credentials as reference. Rereshing the
    # credentials stored in the class will also refresh the DicomWebClient
    # credentials.
    credentials = self.credentials()
    if self._dicom_web_client is None:
      self._dicom_web_client = dicom_web.DicomWebClientImpl(credentials)
    return self._dicom_web_client

  def _make_api_url(self, base_path: dicom_path.Path, api_entry) -> str:
    """Constructs a full http url for the input base path and api entry.

    Args:
      base_path: The base path of the resource upon which the api will be
        called. This path should take the form of:
        'projects/{}/locations/{}/datasets/{}/dicomStores/{}'
      api_entry: The api path to attach after the base path.

    Returns:
      The full http url for calling the API.
    """
    if base_path.type == dicom_path.Type.STORE:
      base_path_str = base_path.dicomweb_path_str
    else:
      base_path_str = str(base_path)
    return posixpath.join(self._dicom_web_base_url, base_path_str, api_entry)

  def get_studies(
      self, dicom_store_path: dicom_path.Path
  ) -> Sequence[DicomObject]:
    """Gets all study objects from the input DICOMStore.

    Args:
      dicom_store_path: The path to the DICOMStore to fetch studies from.

    Returns:
      A sequence of the DICOM study objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level
      dicom path.
    """
    if dicom_store_path.type != dicom_path.Type.STORE:
      raise ez_wsi_errors.DicomPathError(
          'A store-level path is required.'
          f'Found: {dicom_store_path.type} path found in {dicom_store_path}'
      )
    api_url = self._make_api_url(dicom_store_path, 'studies')
    json_results = self.dicom_web_client.QidoRs(api_url)
    if not json_results:
      logging.warn('No studies on the requested %s', api_url)
      return []

    return [
        _build_dicom_object(dicom_path.Type.STUDY, dicom_store_path, dicom_tags)
        for dicom_tags in json_results
    ]

  def get_series(
      self,
      parent_path: dicom_path.Path,
      dicom_tags: Optional[Dict[str, Any]] = None,
  ) -> Sequence[DicomObject]:
    """Gets all series objects under the input parent path.

    Args:
      parent_path: The path to a DICOMStore or a study to fetch series from.
      dicom_tags: Tags to search for, if provided.

    Returns:
      A sequence of the series objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level or
      study-level
      DICOM path.
    """
    if parent_path.type not in (dicom_path.Type.STORE, dicom_path.Type.STUDY):
      raise ez_wsi_errors.DicomPathError(
          'A store-level or study-level path is required.'
          f'Found: {parent_path.type} path found in {parent_path}'
      )
    series_query = (
        f'series?{urllib.parse.urlencode(dicom_tags)}'
        if dicom_tags
        else 'series'
    )

    api_url = self._make_api_url(parent_path, series_query)
    json_results = self.dicom_web_client.QidoRs(api_url)
    if not json_results:
      logging.warn('No series on the requested %s', api_url)
      return []

    return [
        _build_dicom_object(dicom_path.Type.SERIES, parent_path, dicom_tags)
        for dicom_tags in json_results
    ]

  def get_instances(
      self, parent_path: dicom_path.Path
  ) -> Sequence[DicomObject]:
    """Gets all instances under the input parent path.

    Args:
      parent_path: The path to a DICOMStore, a study or a series.

    Returns:
      A sequence of the DICOM instance objects.

    Raises:
      DicomPathError: If the input path is not a valid store-level,
      study-level or series-level DICOM path.
    """
    if parent_path.type not in (
        dicom_path.Type.STORE,
        dicom_path.Type.STUDY,
        dicom_path.Type.SERIES,
    ):
      raise ez_wsi_errors.DicomPathError(
          'A store-level, study-level or series-level path is required.'
          f'Found: {parent_path.type} path found in {parent_path}'
      )
    api_url = self._make_api_url(
        parent_path,
        f'instances/?{_get_qido_suffix(_DEFAULT_INSTANCE_TAGS)}',
    )
    json_results = self.dicom_web_client.QidoRs(api_url)
    if not json_results:
      logging.warn('No instances on the requested %s', api_url)
      return []

    return [
        _build_dicom_object(dicom_path.Type.INSTANCE, parent_path, dicom_tags)
        for dicom_tags in json_results
    ]

  def get_frame_image(
      self,
      instance_path: dicom_path.Path,
      frame_index: int,
      transcode_frame: TranscodeDicomFrame,
  ) -> bytes:
    """Retrieves the BGR image pixels of a frame from an instance.

    Args:
      instance_path: The path to a DICOM instance.
      frame_index: The index of the DICOM frame within the requested instance.
      transcode_frame: How DICOM frame should be transcoded.

    Returns:
      The image buffer that contains the pixels of the frame.

    Raises:
      DicomPathError if the input path is not an instance.
    """
    if instance_path.type != dicom_path.Type.INSTANCE:
      raise ez_wsi_errors.DicomPathError(
          'An instance-level path is required. '
          f'Found: {instance_path.type} path found in {instance_path}'
      )
    api_url = self._make_api_url(instance_path, f'frames/{frame_index}')
    return self.dicom_web_client.WadoRs(
        api_url,
        (
            'multipart/related; type=application/octet-stream; '
            f'transfer-syntax={transcode_frame.value}'
        ),
    )


def _build_dicom_object(
    object_type: dicom_path.Type,
    parent_path: dicom_path.Path,
    dicom_tags: Dict[str, Any],
) -> DicomObject:
  """Creates a DICOM object from a DICOM parent path and a set of tags.

  Args:
    object_type: The type of the object to construct.
    parent_path: the DICOM path pointing to the parent of the DICOM object to be
      constructed.
    dicom_tags: DICOM tags associated for the object construction.

  Returns:
    An instance of DicomObject constructed with proper type.
  Raises:
    DicomPathError if the type of the path constructed for the DICOM
    object does not match the input object_type.
  """
  path = dicom_path.FromPath(
      parent_path,
      # the study_uid is None for a store-level path
      study_uid=dicom_json.GetValue(dicom_tags, tags.STUDY_INSTANCE_UID),
      # the series_uid is None for a store-level or tudy-level path
      series_uid=dicom_json.GetValue(dicom_tags, tags.SERIES_INSTANCE_UID),
      instance_uid=dicom_json.GetValue(dicom_tags, tags.SOP_INSTANCE_UID),
  )
  if path.type != object_type:
    raise ez_wsi_errors.DicomPathError(
        f'The parent path and DICOM tags do not yield a path({path.type}) as '
        f'expected: {object_type}'
    )
  return DicomObject(path, dicom_tags)


def _get_qido_suffix(tags_to_fetch: Collection[tags.DicomTag]) -> str:
  """Returns the suffix to be used in a QIDO-RS query.

  Args:
    tags_to_fetch: DICOM tags to be fetched for each instance.
  """
  return urllib.parse.urlencode(
      {'includefield': [tag.number for tag in tags_to_fetch]}, doseq=True
  )
