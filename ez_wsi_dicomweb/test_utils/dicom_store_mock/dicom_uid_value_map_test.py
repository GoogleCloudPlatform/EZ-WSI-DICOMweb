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
"""Tests dicom uid value map."""

from absl.testing import absltest
from absl.testing import parameterized

from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_uid_value_map


class DicomUidValueMapTest(parameterized.TestCase):

  @parameterized.parameters([
      (
          dicom_uid_value_map._UIDIndex.STUDY_INSTANCE_UID,
          dicom_uid_value_map._UIDIndex.SERIES_INSTANCE_UID,
      ),
      (
          dicom_uid_value_map._UIDIndex.SERIES_INSTANCE_UID,
          dicom_uid_value_map._UIDIndex.SOP_INSTANCE_UID,
      ),
  ])
  def test_increment_uid_index(self, index, expected):
    self.assertEqual(dicom_uid_value_map._increment_uid_index(index), expected)

  def test_increment_uid_index_raises_if_cannot_increment(self):
    with self.assertRaises(ValueError):
      dicom_uid_value_map._increment_uid_index(
          dicom_uid_value_map._UIDIndex.SOP_INSTANCE_UID
      )

  @parameterized.parameters([
      (dicom_uid_value_map._UIDIndex.STUDY_INSTANCE_UID, '1'),
      (dicom_uid_value_map._UIDIndex.SERIES_INSTANCE_UID, '2'),
      (dicom_uid_value_map._UIDIndex.SOP_INSTANCE_UID, '3'),
  ])
  def test_get_get_uid_index_value(self, uid_index, expected):
    self.assertEqual(
        dicom_uid_value_map._get_uid_index_value(('1', '2', '3'), uid_index),
        expected,
    )

  def test_get_all_uid_value_from_empty_map(self):
    uid = ('', '', '')
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    self.assertEmpty(list(val_map.get_instances(uid)))
    self.assertEmpty(val_map._uid_value_map)

  @parameterized.parameters(
      [(('1', '2', '3'),), (('1', '2', ''),), (('1', '', ''),)]
  )
  def test_get_undefied_uid_value_from_map_returns_empty(self, uid):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    self.assertEmpty(list(val_map.get_instances(uid)))
    self.assertEmpty(val_map._uid_value_map)

  def test_remove_all_uid_value_from_empty_map(self):
    uid = ('', '', '')
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    val_map.remove_instances(uid)
    self.assertEmpty(val_map._uid_value_map)

  @parameterized.parameters([(('', '2', '3'),), (('1', '', '3'),)])
  def test_remove_bad_formatted_uid_raises(self, uid):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    with self.assertRaises(dicom_uid_value_map.UidValueMapError):
      val_map.remove_instances(uid)
    self.assertEmpty(val_map._uid_value_map)

  @parameterized.parameters(
      [(('1', '2', '3'),), (('1', '2', ''),), (('1', '', ''),)]
  )
  def test_remove_undefined_instances_from_empty_map_raises(self, uid):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    with self.assertRaises(dicom_uid_value_map.UidValueMapError):
      val_map.remove_instances(uid)
    self.assertEmpty(val_map._uid_value_map)

  @parameterized.parameters(
      [(('', '', ''),), (('1', '2', ''),), (('1', '', ''),)]
  )
  def test_add_single_value_to_map_raises_if_uid_not_fuly_defined(self, uid):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    with self.assertRaises(dicom_uid_value_map.UidValueMapError):
      val_map.add_instance(uid, 1)
    self.assertEmpty(val_map._uid_value_map)

  def test_add_instance_returns_true_value_added_false_if_not_added(self):
    uid = ('1', '2', '3')
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    self.assertTrue(val_map.add_instance(uid, 1))
    self.assertFalse(val_map.add_instance(uid, 2))
    self.assertEqual(val_map.to_dict(), {'1': {'2': {'3': 1}}})

  def test_add_instance_same_values_returns_false(self):
    uid = ('1', '2', '3')
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    self.assertTrue(val_map.add_instance(uid, 1))
    self.assertFalse(val_map.add_instance(uid, 1))
    self.assertEqual(val_map.to_dict(), {'1': {'2': {'3': 1}}})

  def test_add_single_value_to_empty_map(self):
    uid = ('1', '2', '3')
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    val_map.add_instance(uid, 5)
    self.assertEqual(val_map.to_dict(), {'1': {'2': {'3': 5}}})

  def test_add_multiple_value_to_same_series_map(self):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    val_map.add_instance(('1', '2', '3'), 5)
    val_map.add_instance(('1', '2', '4'), 6)
    self.assertEqual(val_map.to_dict(), {'1': {'2': {'3': 5, '4': 6}}})

  def test_add_multiple_value_to_different_series_map(self):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    val_map.add_instance(('1', '2', '3'), 5)
    val_map.add_instance(('1', '4', '5'), 6)
    self.assertEqual(val_map.to_dict(), {'1': {'2': {'3': 5}, '4': {'5': 6}}})

  def test_add_multiple_value_to_different_studies_map(self):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    val_map.add_instance(('1', '2', '3'), 5)
    val_map.add_instance(('7', '4', '5'), 6)
    self.assertEqual(
        val_map.to_dict(), {'1': {'2': {'3': 5}}, '7': {'4': {'5': 6}}}
    )

  @parameterized.named_parameters([
      dict(testcase_name='single_value', uid_list=[(('1', '2', '3'), 5)]),
      dict(
          testcase_name='multiple_value_same_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
      ),
      dict(
          testcase_name='multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
      ),
      dict(
          testcase_name='multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
      ),
  ])
  def test_add_and_get_single_values(self, uid_list):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    for uid, value in uid_list:
      val_map.add_instance(uid, value)
    for uid, value in uid_list:
      self.assertEqual(list(val_map.get_instances(uid)), [value])

  @parameterized.named_parameters([
      dict(
          testcase_name='series_query_)single_value',
          uid_list=[(('1', '2', '3'), 5)],
          query=('1', '2', ''),
          expected={5},
      ),
      dict(
          testcase_name='study_query_single_value',
          uid_list=[(('1', '2', '3'), 5)],
          query=('1', '', ''),
          expected={5},
      ),
      dict(
          testcase_name='store_query_single_value',
          uid_list=[(('1', '2', '3'), 5)],
          query=('', '', ''),
          expected={5},
      ),
      dict(
          testcase_name='series_query_multiple_value_same_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          query=('1', '2', ''),
          expected={5, 6},
      ),
      dict(
          testcase_name='study_query_multiple_value_same_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          query=('1', '', ''),
          expected={5, 6},
      ),
      dict(
          testcase_name='store_query_multiple_value_same_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          query=('', '', ''),
          expected={5, 6},
      ),
      dict(
          testcase_name='series_query_1_multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          query=('1', '2', ''),
          expected={5},
      ),
      dict(
          testcase_name='series_query_2_multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          query=('1', '4', ''),
          expected={6},
      ),
      dict(
          testcase_name='study_query_multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          query=('1', '', ''),
          expected={6, 5},
      ),
      dict(
          testcase_name='store_query_multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          query=('', '', ''),
          expected={6, 5},
      ),
      dict(
          testcase_name='series_query_1_multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          query=('1', '2', ''),
          expected={5},
      ),
      dict(
          testcase_name='series_query_2_multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          query=('7', '4', ''),
          expected={6},
      ),
      dict(
          testcase_name='study_query_1_multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          query=('7', '', ''),
          expected={6},
      ),
      dict(
          testcase_name='study_query_2_multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          query=('1', '', ''),
          expected={5},
      ),
      dict(
          testcase_name='store_query_multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          query=('', '', ''),
          expected={5, 6},
      ),
  ])
  def test_get_multiple(self, uid_list, query, expected):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    for uid, value in uid_list:
      val_map.add_instance(uid, value)
    self.assertEqual(set(val_map.get_instances(query)), expected)

  @parameterized.named_parameters([
      dict(testcase_name='single_value', uid_list=[(('1', '2', '3'), 5)]),
      dict(
          testcase_name='multiple_value_same_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
      ),
      dict(
          testcase_name='multiple_value_different_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
      ),
      dict(
          testcase_name='multiple_value_different_studies',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
      ),
  ])
  def test_add_and_remove_all(self, uid_list):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    for uid, value in uid_list:
      val_map.add_instance(uid, value)
    val_map.remove_instances(('', '', ''))
    self.assertEmpty(val_map._uid_value_map)

  @parameterized.named_parameters([
      dict(
          testcase_name='single_value',
          uid_list=[(('1', '2', '3'), 5)],
          remove_uid=('1', '2', '3'),
          expected={},
      ),
      dict(
          testcase_name='multiple_value_same_series_remove_instance_1',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          remove_uid=('1', '2', '3'),
          expected={'1': {'2': {'4': 6}}},
      ),
      dict(
          testcase_name='multiple_value_same_series_remove_instance_2',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          remove_uid=('1', '2', '4'),
          expected={'1': {'2': {'3': 5}}},
      ),
      dict(
          testcase_name='multiple_value_same_series_remove_series',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          remove_uid=('1', '2', ''),
          expected={},
      ),
      dict(
          testcase_name='multiple_value_same_series_remove_studies',
          uid_list=[(('1', '2', '3'), 5), (('1', '2', '4'), 6)],
          remove_uid=('1', '', ''),
          expected={},
      ),
      dict(
          testcase_name='multiple_value_different_series_remove_instance_1',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          remove_uid=('1', '2', '3'),
          expected={'1': {'4': {'5': 6}}},
      ),
      dict(
          testcase_name='multiple_value_different_series_remove_instance_2',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          remove_uid=('1', '4', '5'),
          expected={'1': {'2': {'3': 5}}},
      ),
      dict(
          testcase_name='multiple_value_different_series_remove_series_1',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          remove_uid=('1', '2', ''),
          expected={'1': {'4': {'5': 6}}},
      ),
      dict(
          testcase_name='multiple_value_different_series_remove_series_2',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          remove_uid=('1', '4', ''),
          expected={'1': {'2': {'3': 5}}},
      ),
      dict(
          testcase_name='multiple_value_different_series_remove_study',
          uid_list=[(('1', '2', '3'), 5), (('1', '4', '5'), 6)],
          remove_uid=('1', '', ''),
          expected={},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_instance_1',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('1', '2', '3'),
          expected={'7': {'4': {'5': 6}}},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_instance_2',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('7', '4', '5'),
          expected={'1': {'2': {'3': 5}}},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_series_2',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('7', '4', ''),
          expected={'1': {'2': {'3': 5}}},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_series_1',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('1', '2', ''),
          expected={'7': {'4': {'5': 6}}},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_study_2',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('1', '', ''),
          expected={'7': {'4': {'5': 6}}},
      ),
      dict(
          testcase_name='multiple_value_different_studies_remove_study_1',
          uid_list=[(('1', '2', '3'), 5), (('7', '4', '5'), 6)],
          remove_uid=('7', '', ''),
          expected={'1': {'2': {'3': 5}}},
      ),
  ])
  def test_add_and_remove(self, uid_list, remove_uid, expected):
    val_map = dicom_uid_value_map.DicomUidValueMap[int]()
    for uid, value in uid_list:
      val_map.add_instance(uid, value)
    val_map.remove_instances(remove_uid)
    self.assertEqual(val_map.to_dict(), expected)


if __name__ == '__main__':
  absltest.main()
