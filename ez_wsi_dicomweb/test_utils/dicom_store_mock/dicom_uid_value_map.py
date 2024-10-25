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
"""Data structure to manage data associated with DICOM UID triples.

Usage:
  map = DicomUidValueMap[Type]()

  Methods:
      add_instance(uid, value: Type)
      remove_instances(uid)
      get_instances(uid) -> Iterator[Type]

  uid parameter expected to be Tuple(StudyInstanceUID, SeriesInstanceUID,
    SopInstanceUID)
  StudyInstanceUID, SeriesInstanceUID, or SopInstanceUID can be defined as
  empty, '', for requests to remove_instances and get_instances if the request
  applies to all instances stored under the specified level.
    i.e.,
      ('', '', '') == All instances in store.
      ('1.2.3', '', '') == All instances in store under StudyInstanceUID 1.2.3
      ('1.2.3', '1.2.4', '') == All instances in store under StudyInstanceUID
        1.2.3's series 1.2.4.
"""
import enum
import itertools
from typing import Any, Dict, Generic, Iterator, Mapping, MutableMapping, Tuple, TypeVar, Union

# Required for pre-python 3.12.X; Type definition to allow the data type stored
# DicomUidValueMap to be defined via Pythons generic typing syntax, e.g.,
# foo = DicomUidValueMap[int]()
UidMapType = TypeVar('UidMapType')


class UidValueMapError(Exception):
  pass


class _UIDIndex(enum.Enum):
  STUDY_INSTANCE_UID = 0
  SERIES_INSTANCE_UID = 1
  SOP_INSTANCE_UID = 2


def _increment_uid_index(val: _UIDIndex) -> _UIDIndex:
  if val == _UIDIndex.STUDY_INSTANCE_UID:
    return _UIDIndex.SERIES_INSTANCE_UID
  if val == _UIDIndex.SERIES_INSTANCE_UID:
    return _UIDIndex.SOP_INSTANCE_UID
  raise ValueError(f'Can not increment uid index:{val}')


def _get_uid_index_value(uid: Tuple[str, str, str], index: _UIDIndex) -> str:
  return uid[index.value]


def _raise_if_incorrectly_formatted_for_search_and_remove(
    uid: Tuple[str, str, str]
) -> None:
  """Raises if UID is incorrectly formatted for search and remove."""
  empty_found = False
  for val in uid:
    if not val:
      empty_found = True
    elif empty_found:
      raise UidValueMapError(f'Incorrectly formatted: {uid}.')
  return


class DicomUidValueMap(Generic[UidMapType]):
  """Data structure to manage data associated with DICOM UID triples."""

  def __init__(self):
    self._uid_value_map: Dict[str, Union[DicomUidValueMap, UidMapType]] = {}

  def _get_instances(
      self, uid: Tuple[str, str, str], uid_index: _UIDIndex
  ) -> Iterator[UidMapType]:
    """Returns iterator of the values associated with uid.

       Definition of empty uid value for series or study returns
       all values under empty level.

    Args:
      uid: (DICOM StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
      uid_index: index of uid working on.

    Returns:
      Values associated with UID.
    """
    is_instance_level = uid_index == _UIDIndex.SOP_INSTANCE_UID
    uid_at_level = _get_uid_index_value(uid, uid_index)
    if uid_at_level:
      try:
        child = self._uid_value_map[uid_at_level]
      except KeyError:
        return iter([])
      if is_instance_level:
        return iter([child])
      return child._get_instances(uid, _increment_uid_index(uid_index))  # pylint: disable=protected-access
    if is_instance_level:
      return iter(self._uid_value_map.values())
    return itertools.chain.from_iterable([
        value._get_instances(uid, _increment_uid_index(uid_index))  # pylint: disable=protected-access
        for value in self._uid_value_map.values()
    ])

  def get_instances(self, uid: Tuple[str, str, str]) -> Iterator[UidMapType]:
    """Returns iterator of the values associated with uid.

       Definition of empty uid value for series or study returns
       all values under empty level.

    Args:
      uid: (DICOM StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
        StudyInstanceUID, SeriesInstanceUID, or SopInstanceUID can be defined as
        '', to get_instances when the get_instances applies to all instances
        stored under the specified level. i.e., ('', '', '') == All instances in
        store. ('1.2.3', '', '') == All instances in store under
        StudyInstanceUID 1.2.3 ('1.2.3', '1.2.4', '') == All instances in store
        under StudyInstanceUID 1.2.3's series 1.2.4.

    Returns:
      Iterator of values associated with UID.

    Raises:
      UidValueMapError: incorrectly formatted.
    """
    return self._get_instances(uid, _UIDIndex.STUDY_INSTANCE_UID)

  def add_instance(self, uid: Tuple[str, str, str], value: UidMapType) -> bool:
    """Associates a single value with DICOM study, series, instance UID.

    Args:
      uid: Tuple (DICOM StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
      value: Value to associate with uid.

    Returns:
      True if value added.

    Raises:
      UidValueMapError: Invalid UID.
    """
    uid_value_map = self._uid_value_map
    if any([not val for val in uid]):
      raise UidValueMapError(f'Invalid UID: {uid}')
    for uid_index in (
        _UIDIndex.STUDY_INSTANCE_UID,
        _UIDIndex.SERIES_INSTANCE_UID,
    ):
      study_or_series_uid = _get_uid_index_value(uid, uid_index)
      study_or_series_uid_map = uid_value_map.get(study_or_series_uid)
      if study_or_series_uid_map is None:
        study_or_series_uid_map = DicomUidValueMap()
        uid_value_map[study_or_series_uid] = study_or_series_uid_map
      uid_value_map = study_or_series_uid_map._uid_value_map  # pylint: disable=protected-access
    sop_instance_uid = uid[_UIDIndex.SOP_INSTANCE_UID.value]
    if sop_instance_uid in uid_value_map:
      return False
    uid_value_map[sop_instance_uid] = value
    return True

  def _remove_instance(
      self,
      uid_value_map: MutableMapping[str, Any],
      uid: Tuple[str, str, str],
      uid_index: _UIDIndex,
  ) -> None:
    """Removes values from DICOM study, series, instance UID."""
    uid_index_val = _get_uid_index_value(uid, uid_index)
    if uid_index == _UIDIndex.SOP_INSTANCE_UID or not _get_uid_index_value(
        uid, _increment_uid_index(uid_index)
    ):
      try:
        del uid_value_map[uid_index_val]
      except KeyError as exp:
        raise UidValueMapError(f'UID: {uid} not found in map.') from exp
      return
    series_or_instance_value_map = uid_value_map.get(uid_index_val)
    if series_or_instance_value_map is None:
      raise UidValueMapError(f'UID: {uid} not found in map.')
    child_dict = series_or_instance_value_map._uid_value_map  # pylint: disable=protected-access
    self._remove_instance(child_dict, uid, _increment_uid_index(uid_index))
    if not child_dict:
      del uid_value_map[uid_index_val]

  def remove_instances(self, uid: Tuple[str, str, str]) -> None:
    """Removes values associated with DICOM study, series, instance UID.

    Args:
      uid: (DICOM StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
        StudyInstanceUID, SeriesInstanceUID, or SopInstanceUID can be defined as
        '', to remove_instances when the removal applies to all instances stored
        under the specified level. i.e., ('', '', '') == All instances in store.
        ('1.2.3', '', '') == All instances in store under StudyInstanceUID 1.2.3
        ('1.2.3', '1.2.4', '') == All instances in store under StudyInstanceUID
        1.2.3's series 1.2.4.

    Raises:
      UidValueMapError: UID not found or incorrectly formatted.
    """
    _raise_if_incorrectly_formatted_for_search_and_remove(uid)
    if not uid[0]:
      self._uid_value_map = {}
      return
    self._remove_instance(
        self._uid_value_map, uid, _UIDIndex.STUDY_INSTANCE_UID
    )

  def _to_dict(
      self, uid_value_map: Mapping[str, Any], uid_index: _UIDIndex
  ) -> Mapping[str, Any]:
    if uid_index == _UIDIndex.SOP_INSTANCE_UID:
      return {uid: value for uid, value in uid_value_map.items()}
    return {
        uid: self._to_dict(
            value._uid_value_map, _increment_uid_index(uid_index)  # pylint: disable=protected-access
        )
        for uid, value in uid_value_map.items()
    }

  def to_dict(self) -> Mapping[str, Mapping[str, Mapping[str, UidMapType]]]:
    """Returns a dictionary representation of uid value map for debugging."""
    return self._to_dict(self._uid_value_map, _UIDIndex.STUDY_INSTANCE_UID)
