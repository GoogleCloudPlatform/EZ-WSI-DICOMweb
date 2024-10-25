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
"""Utilities for DICOMweb path manipulation."""
from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Match, Optional
import urllib.parse

import dataclasses_json


class Type(enum.Enum):
  """Type of a resource the path points to."""

  STORE = 'store'
  STUDY = 'study'
  SERIES = 'series'
  INSTANCE = 'instance'


# Used for project ID and location validation
_REGEX_ID_1_TXT = r'[\w-]+'
# Used for dataset ID and dicom store ID validation
_REGEX_ID_2_TXT = r'[\w.-]+'
# Used for DICOM UIDs validation
# '/' is not allowed because the parsing logic in the class uses '/' to
# tokenize the path.
# '@' is not allowed due to security concerns: theoretically it could lead
# to the part before '@' being interpreted as the username, and the part
# after - as the server address, which is a potential vulnerability.
_REGEX_UID_TXT = r'[^/@]+'

_REGEX_BASE_ADDRESS = re.compile(r'https?://.+')
_REGEX_ID_1 = re.compile(_REGEX_ID_1_TXT)
_REGEX_ID_2 = re.compile(_REGEX_ID_2_TXT)
_REGEX_UID = re.compile(_REGEX_UID_TXT)
_REGEX_STORE = re.compile(
    r'projects/(%s)/locations/(%s)/datasets/(%s)/dicomStores/(%s)'
    r'(.*)'
    % (_REGEX_ID_1_TXT, _REGEX_ID_1_TXT, _REGEX_ID_2_TXT, _REGEX_ID_2_TXT)
)
_REGEX_STUDIES = re.compile(r'((.+)/)?studies/(%s)(.*)' % _REGEX_UID_TXT)
_REGEX_SERIES = re.compile(r'series/(%s)(.*)' % _REGEX_UID_TXT)
_REGEX_INSTANCE = re.compile(r'instances/(%s)/?$' % _REGEX_UID_TXT)
_HEALTHCARE_API_URL = 'https://healthcare.googleapis.com'
_DEFAULT_HEALTHCARE_API_VERSION = 'v1'


def DicomPathJoin(*args: str) -> str:
  return '/'.join([arg.strip('/') for arg in args if arg])


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Path:
  """Represents a path to a DICOM Store or a DICOM resource in CHC API.

  Attributes:
    base_address: base address for url
    healthcare_api_version: healthcare api version
    project_id: Project ID.
    location: Location.
    dataset_id: Dataset ID.
    store_id: DICOM Store ID.
    study_prefix: prefix to study UID.
    study_uid: DICOM Study UID.
    series_uid: DICOM Series UID.
    instance_uid: DICOM Instance UID.
  """

  base_address: str
  healthcare_api_version: str
  project_id: str
  location: str
  dataset_id: str
  store_id: str
  study_prefix: str
  study_uid: str
  series_uid: str
  instance_uid: str

  def __post_init__(self) -> None:
    """Validates path configuration.

    Returns:
      None

    Raises:
      ValueError: Invalid configuration.
    """
    if _REGEX_BASE_ADDRESS.fullmatch(self.base_address) is None:
      raise ValueError('Invalid base_address')
    if (
        self.healthcare_api_version
        and _REGEX_ID_1.fullmatch(self.healthcare_api_version) is None
    ):
      raise ValueError('Healthcare API version')
    if self.project_id and _REGEX_ID_1.fullmatch(self.project_id) is None:
      raise ValueError('Invalid project_id')
    if self.location and _REGEX_ID_1.fullmatch(self.location) is None:
      raise ValueError('Invalid location')
    if self.dataset_id and _REGEX_ID_2.fullmatch(self.dataset_id) is None:
      raise ValueError('Invalid dataset_id')
    if self.store_id and _REGEX_ID_2.fullmatch(self.store_id) is None:
      raise ValueError('Invalid store_id')
    id_count_defined = sum([
        1
        for id in [
            self.project_id,
            self.location,
            self.dataset_id,
            self.store_id,
        ]
        if id
    ])
    if (
        self.study_prefix != 'dicomWeb'
        and self.base_address == _HEALTHCARE_API_URL
    ):
      raise ValueError('Invalid study_prefix')
    if id_count_defined != 0 and id_count_defined != 4:
      raise ValueError('Invalid id')
    if self.study_uid and _REGEX_UID.fullmatch(self.study_uid) is None:
      raise ValueError('Invalid study_uid')
    if self.series_uid and _REGEX_UID.fullmatch(self.series_uid) is None:
      raise ValueError('Invalid series_uid')
    if self.instance_uid and _REGEX_UID.fullmatch(self.instance_uid) is None:
      raise ValueError('Invalid instance_uid')
    self._StudyUidMissing(self.study_uid)
    self._SeriesUidMissing(self.series_uid)

  def _StudyUidMissing(self, value: str) -> None:
    if not value:
      if self.series_uid or self.instance_uid:
        raise ValueError(
            'study_uid missing with non-empty series_uid or instance_uid.'
            f' series_uid: {self.series_uid}, instance_uid: {self.instance_uid}'
        )

  def _SeriesUidMissing(self, value: str) -> None:
    if not value:
      if self.instance_uid:
        raise ValueError(
            'series_uid missing with non-empty instance_uid. instance_uid:'
            f' {self.instance_uid}'
        )

  def _BuildGoogleStorePath(self) -> str:
    """Returns component of path identifying google DICOM store."""
    if not self.project_id:
      return ''
    else:
      return DicomPathJoin(
          'projects',
          self.project_id,
          'locations',
          self.location,
          'datasets',
          self.dataset_id,
          'dicomStores',
          self.store_id,
      )

  def _BuildUidPath(self) -> str:
    """Returns UID component of path to imaging in DICOM store."""
    if not self.study_uid:
      return self.study_prefix

    study_path_str = DicomPathJoin(self.study_prefix, 'studies', self.study_uid)
    if not self.series_uid:
      return study_path_str

    series_path_str = DicomPathJoin(study_path_str, 'series', self.series_uid)
    if not self.instance_uid:
      return series_path_str
    return DicomPathJoin(series_path_str, 'instances', self.instance_uid)

  @property
  def complete_url(self) -> str:
    """Returns the complete url of the path."""
    return DicomPathJoin(
        self.base_address,
        self.healthcare_api_version,
        self._BuildGoogleStorePath(),
        self._BuildUidPath(),
    )

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, Path):
      return self.complete_url == other.complete_url
    if isinstance(other, str):
      return self.complete_url == other
    return False

  def __str__(self):
    """Returns the text representation of the path."""
    return self.complete_url

  @property
  def type(self) -> Type:
    """Type of the DICOM resource corresponding to the path."""
    if not self.study_uid:
      return Type.STORE
    elif not self.series_uid:
      return Type.STUDY
    elif not self.instance_uid:
      return Type.SERIES
    return Type.INSTANCE

  def GetStorePath(self) -> Path:
    """Returns the sub-path for the DICOM Store within this path."""
    return Path(
        self.base_address,
        self.healthcare_api_version,
        self.project_id,
        self.location,
        self.dataset_id,
        self.store_id,
        self.study_prefix,
        '',
        '',
        '',
    )

  def GetStudyPath(self) -> Path:
    """Returns the sub-path for the DICOM Study within this path."""
    if self.type == Type.STORE:
      raise ValueError("Can't get a study path from a store path.")
    return Path(
        self.base_address,
        self.healthcare_api_version,
        self.project_id,
        self.location,
        self.dataset_id,
        self.store_id,
        self.study_prefix,
        self.study_uid,
        '',
        '',
    )

  def GetSeriesPath(self) -> Path:
    """Returns the sub-path for the DICOM Series within this path."""
    if self.type in (Type.STORE, Type.STUDY):
      raise ValueError(f"Can't get a series path from a {self.type} path.")
    return Path(
        self.base_address,
        self.healthcare_api_version,
        self.project_id,
        self.location,
        self.dataset_id,
        self.store_id,
        self.study_prefix,
        self.study_uid,
        self.series_uid,
        '',
    )


def _MatchRegex(regex: re.Pattern[str], text_str: str, error_str) -> Match[str]:
  """Matches the regex and returns the match or raises ValueError if failed."""
  match = regex.match(text_str)
  if match is None:
    raise ValueError(error_str)
  return match


def _FromString(path_str: str) -> Path:
  """Parses the string and returns the Path object or raises ValueError if failed."""
  match_err_str = f'Error parsing the path. Path: {path_str}'
  prased_url = urllib.parse.urlparse(path_str)
  if not prased_url.scheme:
    base_address = _HEALTHCARE_API_URL
    healthcare_api_version = _DEFAULT_HEALTHCARE_API_VERSION
  else:
    # check if full url has been provided
    if prased_url.scheme.lower() not in ('http', 'https'):
      raise ValueError(match_err_str)
    if not prased_url.netloc:
      raise ValueError(match_err_str)
    base_address = f'{prased_url.scheme}://{prased_url.netloc}'
    path_str = prased_url.path
    if base_address.lower() != _HEALTHCARE_API_URL:
      healthcare_api_version = ''
    else:
      path_str = path_str.strip('/')
      path_str_parts = path_str.split('/')
      healthcare_api_version = path_str_parts[0]
      if not healthcare_api_version:
        raise ValueError(match_err_str)
      path_str = '/'.join(path_str_parts[1:])
  path_str = path_str.strip('/')
  is_healthcare_api_url = base_address.lower() == _HEALTHCARE_API_URL
  if is_healthcare_api_url:
    store_match = _MatchRegex(_REGEX_STORE, path_str, match_err_str)
    project_id = store_match.group(1)
    location = store_match.group(2)
    dataset_id = store_match.group(3)
    store_id = store_match.group(4)
    store_path_suffix = store_match.group(5)
    study_prefix = 'dicomWeb'
  else:
    project_id = ''
    location = ''
    dataset_id = ''
    store_id = ''
    study_prefix = ''
    store_path_suffix = path_str
  if not store_path_suffix:
    return Path(
        base_address,
        healthcare_api_version,
        project_id,
        location,
        dataset_id,
        store_id,
        study_prefix,
        '',
        '',
        '',
    )
  try:
    studies_match = _MatchRegex(
        _REGEX_STUDIES, store_path_suffix, match_err_str
    )
  except ValueError:
    store_path_suffix = store_path_suffix.strip().strip('/')
    if store_path_suffix:
      study_prefix = store_path_suffix
    if is_healthcare_api_url and study_prefix != 'dicomWeb':
      raise
    return Path(
        base_address,
        healthcare_api_version,
        project_id,
        location,
        dataset_id,
        store_id,
        study_prefix,
        '',
        '',
        '',
    )
  study_prefix = studies_match.group(2)
  if study_prefix is None:
    study_prefix = ''
  if study_prefix:
    study_prefix = study_prefix.strip('/')
  if is_healthcare_api_url and study_prefix != 'dicomWeb':
    raise ValueError(match_err_str)
  study_uid = studies_match.group(3)
  study_path_suffix = studies_match.group(4)

  if not study_path_suffix:
    return Path(
        base_address,
        healthcare_api_version,
        project_id,
        location,
        dataset_id,
        store_id,
        study_prefix,
        study_uid,
        '',
        '',
    )
  study_path_suffix = study_path_suffix.strip('/')
  series_match = _MatchRegex(_REGEX_SERIES, study_path_suffix, match_err_str)
  series_uid = series_match.group(1)
  series_path_suffix = series_match.group(2)
  series_path_suffix = series_path_suffix.strip('/')
  if not series_path_suffix:
    return Path(
        base_address,
        healthcare_api_version,
        project_id,
        location,
        dataset_id,
        store_id,
        study_prefix,
        study_uid,
        series_uid,
        '',
    )

  instance_match = _MatchRegex(
      _REGEX_INSTANCE, series_path_suffix, match_err_str
  )
  instance_uid = instance_match.group(1)

  return Path(
      base_address,
      healthcare_api_version,
      project_id,
      location,
      dataset_id,
      store_id,
      study_prefix,
      study_uid,
      series_uid,
      instance_uid,
  )


def FromString(path_str: str, path_type: Optional[Type] = None) -> Path:
  """Parses the string and returns the Path object or raises ValueError if failed.

  Args:
    path_str: The string containing the path.
    path_type: The expected type of the path or None if no specific type is
      expected.

  Returns:
    The newly constructed Path object.
  Raises:
    ValueError if the path cannot be parsed or the actual path type doesn't
      match the specified expected type.
  """
  path = _FromString(path_str)

  # Validate that the path is of the right type of the type is specified.
  if path_type is not None and path.type != path_type:
    raise ValueError(
        f'Unexpected path type. Expected: {path_type}, actual: {path.type}.'
        f' Path: {path_str}'
    )

  return path


def FromPath(
    base_path: Path,
    store_id: Optional[str] = None,
    study_uid: Optional[str] = None,
    series_uid: Optional[str] = None,
    instance_uid: Optional[str] = None,
) -> Path:
  """Creates a new Path object based on the provided one.

  Replaces the specified path components in the base path to create the new one.

  Args:
    base_path: The base path to use.
    store_id: The store ID to use in the new path or None if the store ID from
      the base path should be used.
    study_uid: The study UID to use in the new path or None if the study UID
      from the base path should be used.
    series_uid: The series UID to use in the new path or None if the series UID
      from the base path should be used.
    instance_uid: The instance UID to use in the new path or None if the
      instance UID from the base path should be used.

  Returns:
    The newly constructed Path object.
  Raises:
    ValueError if the new path is invalid (e.g. if the instance UID is
      specified, but the series UID is None).
  """
  default_study_uid = base_path.study_uid
  default_series_uid = base_path.series_uid
  default_instance_uid = base_path.instance_uid
  if store_id is None:
    store_id = base_path.store_id
  else:
    default_study_uid = ''
    default_series_uid = ''
    default_instance_uid = ''
  if study_uid is None:
    study_uid = default_study_uid
  else:
    default_series_uid = ''
    default_instance_uid = ''
  if series_uid is None:
    series_uid = default_series_uid
  else:
    default_instance_uid = ''
  if instance_uid is None:
    instance_uid = default_instance_uid
  return Path(
      base_path.base_address,
      base_path.healthcare_api_version,
      base_path.project_id,
      base_path.location,
      base_path.dataset_id,
      store_id,
      base_path.study_prefix,
      study_uid,
      series_uid,
      instance_uid,
  )
