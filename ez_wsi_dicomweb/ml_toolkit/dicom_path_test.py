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
"""Tests for dicom path."""

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import test_dicom_path_util as tdpu


class DicomPathTest(parameterized.TestCase):

  def _AssertStoreAttributes(self, path: dicom_path.Path):
    self.assertEqual(path.project_id, tdpu.PROJECT_NAME)
    self.assertEqual(path.location, tdpu.LOCATION)
    self.assertEqual(path.dataset_id, tdpu.DATASET_ID)
    self.assertEqual(path.store_id, tdpu.STORE_ID)

  def testStorePath(self):
    """Store path is parsed correctly and behaves as expected."""
    store_path = dicom_path.FromString(tdpu.STORE_PATH_STR)
    self._AssertStoreAttributes(store_path)
    self.assertEmpty(store_path.study_uid)
    self.assertEmpty(store_path.series_uid)
    self.assertEmpty(store_path.instance_uid)
    self.assertEqual(store_path.type, dicom_path.Type.STORE)
    self.assertEqual(str(store_path), tdpu.STORE_PATH_STR)
    self.assertEqual(str(store_path.GetStorePath()), tdpu.STORE_PATH_STR)

  def testStudyPath(self):
    """Study path is parsed correctly and behaves as expected."""
    study_path = dicom_path.FromString(tdpu.STUDY_PATH_STR)
    self._AssertStoreAttributes(study_path)
    self.assertEqual(study_path.study_uid, tdpu.STUDY_UID)
    self.assertEmpty(study_path.series_uid)
    self.assertEmpty(study_path.instance_uid)
    self.assertEqual(study_path.type, dicom_path.Type.STUDY)
    self.assertEqual(str(study_path), tdpu.STUDY_PATH_STR)
    self.assertEqual(str(study_path.GetStorePath()), tdpu.STORE_PATH_STR)
    self.assertEqual(str(study_path.GetStudyPath()), tdpu.STUDY_PATH_STR)

  def testSeriesPath(self):
    """Series path is parsed correctly and behaves as expected."""
    series_path = dicom_path.FromString(tdpu.SERIES_PATH_STR)
    self._AssertStoreAttributes(series_path)
    self.assertEqual(series_path.study_uid, tdpu.STUDY_UID)
    self.assertEqual(series_path.series_uid, tdpu.SERIES_UID)
    self.assertEmpty(series_path.instance_uid)
    self.assertEqual(series_path.type, dicom_path.Type.SERIES)
    self.assertEqual(str(series_path), tdpu.SERIES_PATH_STR)
    self.assertEqual(str(series_path.GetStorePath()), tdpu.STORE_PATH_STR)
    self.assertEqual(str(series_path.GetStudyPath()), tdpu.STUDY_PATH_STR)
    self.assertEqual(str(series_path.GetSeriesPath()), tdpu.SERIES_PATH_STR)

  def testInstancePath(self):
    """Instance path is parsed correctly and behaves as expected."""
    instance_path = dicom_path.FromString(tdpu.INSTANCE_PATH_STR)
    self._AssertStoreAttributes(instance_path)
    self.assertEqual(instance_path.study_uid, tdpu.STUDY_UID)
    self.assertEqual(instance_path.series_uid, tdpu.SERIES_UID)
    self.assertEqual(instance_path.instance_uid, tdpu.INSTANCE_UID)
    self.assertEqual(instance_path.type, dicom_path.Type.INSTANCE)
    self.assertEqual(str(instance_path), tdpu.INSTANCE_PATH_STR)
    self.assertEqual(str(instance_path.GetStorePath()), tdpu.STORE_PATH_STR)
    self.assertEqual(str(instance_path.GetStudyPath()), tdpu.STUDY_PATH_STR)
    self.assertEqual(str(instance_path.GetSeriesPath()), tdpu.SERIES_PATH_STR)

  # '/' is not allowed because the parsing logic in the class uses '/' to
  # tokenize the path.
  # '@' is not allowed due to a potential security vulnerability.
  @parameterized.parameters('/', '@')
  def testNoForwardSlashOrAt(self, illegal_char):
    """ValueError is raised when an attribute contains '/' or '@'."""
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'project%cid' % illegal_char,
        'l',
        'd',
        's',
        'dicomWeb',
        '',
        '',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'locat%cion' % illegal_char,
        'd',
        's',
        'dicomWeb',
        '',
        '',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'data%cset' % illegal_char,
        's',
        'dicomWeb',
        '',
        '',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        'st%core' % illegal_char,
        'dicomWeb',
        '',
        '',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        's',
        'dicomWeb',
        '1.2%c3' % illegal_char,
        '',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        's',
        'dicomWeb',
        '1.2.3',
        '4.5%c6' % illegal_char,
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        's',
        'dicomWeb',
        '1.2.3',
        '4.5.6',
        '7.8%c9' % illegal_char,
    )

  def testUidMissingError(self):
    """ValueError is raised when an expected UID is missing."""
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        's',
        'dicomWeb',
        '',
        '4.5.6',
        '',
    )
    self.assertRaises(
        ValueError,
        dicom_path.Path,
        'b',
        'v',
        'p',
        'l',
        'd',
        's',
        'dicomWeb',
        'stuid',
        '',
        '7.8.9',
    )

  def testFromStringInvalid(self):
    """ValueError raised when the path string is invalid."""
    self.assertRaises(ValueError, dicom_path.FromString, 'invalid_path')

  def testFromStringTypeError(self):
    """ValueError raised when the expected type doesn't match the actual one."""
    for path_type in dicom_path.Type:
      if path_type != dicom_path.Type.STORE:
        self.assertRaises(
            ValueError, dicom_path.FromString, tdpu.STORE_PATH_STR, path_type
        )
      if path_type != dicom_path.Type.STUDY:
        self.assertRaises(
            ValueError, dicom_path.FromString, tdpu.STUDY_PATH_STR, path_type
        )
      if path_type != dicom_path.Type.SERIES:
        self.assertRaises(
            ValueError, dicom_path.FromString, tdpu.SERIES_PATH_STR, path_type
        )
      if path_type != dicom_path.Type.INSTANCE:
        self.assertRaises(
            ValueError, dicom_path.FromString, tdpu.INSTANCE_PATH_STR, path_type
        )

  @parameterized.named_parameters([
      dict(
          testcase_name='invalid_base_address',
          base_address='www.foo.com',
          version='v1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='invalid_version',
          base_address='https://www.foo.com',
          version=' ',
          project='a',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_project',
          base_address='http://www.foo.com',
          version='v1',
          project='',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_location',
          base_address='http://www.foo.com',
          version='v1',
          project='a',
          location='',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_dataset',
          base_address='http://www.foo.com',
          version='v1',
          project='a',
          location='b',
          dataset='',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_dicom_store',
          base_address='http://www.foo.com',
          version='v1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_two',
          base_address='http://www.foo.com',
          version='v1beta1',
          project='a',
          location='b',
          dataset='',
          dicom_store='',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='missing_three',
          base_address='http://www.foo.com',
          version='v1beta1',
          project='',
          location='b',
          dataset='',
          dicom_store='',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='bad_study_uid_prefix',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='',
      ),
      dict(
          testcase_name='invalid_project',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='@',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='dicomWeb',
      ),
      dict(
          testcase_name='invalid_location',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='a',
          location='@',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='dicomWeb',
      ),
      dict(
          testcase_name='invalid_dataset',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='a',
          location='b',
          dataset='@',
          dicom_store='d',
          study_uid_prefix='dicomWeb',
      ),
      dict(
          testcase_name='invalid_dicom_store',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='@',
          study_uid_prefix='dicomWeb',
      ),
  ])
  def testInvalidGoogleDicomStorePath(
      self,
      base_address,
      version,
      project,
      location,
      dataset,
      dicom_store,
      study_uid_prefix,
  ):
    with self.assertRaises(ValueError):
      dicom_path.Path(
          base_address,
          version,
          project,
          location,
          dataset,
          dicom_store,
          study_uid_prefix,
          '',
          '',
          '',
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='invalid_study_uid',
          study_uid='/',
          series_uid='',
          instance_uid='',
      ),
      dict(
          testcase_name='invalid_series_uid',
          study_uid='1',
          series_uid='/',
          instance_uid='',
      ),
      dict(
          testcase_name='invalid_instance_uid',
          study_uid='1',
          series_uid='1.2',
          instance_uid='/',
      ),
      dict(
          testcase_name='missing_study_uid_1',
          study_uid='',
          series_uid='1.2',
          instance_uid='',
      ),
      dict(
          testcase_name='missing_study_uid_2',
          study_uid='',
          series_uid='1.2',
          instance_uid='1.2.3',
      ),
      dict(
          testcase_name='missing_study_and_series_uid',
          study_uid='',
          series_uid='',
          instance_uid='1.2.3',
      ),
      dict(
          testcase_name='missing_series_uid',
          study_uid='1',
          series_uid='',
          instance_uid='1.2.3',
      ),
  ])
  def testInvalidUID(self, study_uid, series_uid, instance_uid):
    with self.assertRaises(ValueError):
      dicom_path.Path(
          dicom_path._HEALTHCARE_API_URL,
          'v1beta1',
          'a',
          'b',
          'c',
          'd',
          'dicomWeb',
          study_uid,
          series_uid,
          instance_uid,
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='generic_store',
          base_address='http://www.foo.com',
          version='',
          project='',
          location='',
          dataset='',
          dicom_store='',
          study_uid_prefix='',
          expected='http://www.foo.com',
      ),
      dict(
          testcase_name='health_care_api_v1_store',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='dicomWeb',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/a/locations/b/datasets/c/dicomStores/d/dicomWeb',
      ),
      dict(
          testcase_name='health_care_api_v1beta1_store',
          base_address=dicom_path._HEALTHCARE_API_URL,
          version='v1beta1',
          project='a',
          location='b',
          dataset='c',
          dicom_store='d',
          study_uid_prefix='dicomWeb',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/a/locations/b/datasets/c/dicomStores/d/dicomWeb',
      ),
  ])
  def testValidGoogleDicomStorePath(
      self,
      base_address,
      version,
      project,
      location,
      dataset,
      dicom_store,
      study_uid_prefix,
      expected,
  ):
    self.assertEqual(
        str(
            dicom_path.Path(
                base_address,
                version,
                project,
                location,
                dataset,
                dicom_store,
                study_uid_prefix,
                '',
                '',
                '',
            )
        ),
        expected,
    )

  def testGetStudyPathRaisesFromStorePath(self):
    with self.assertRaises(ValueError):
      dicom_path.Path(
          dicom_path._HEALTHCARE_API_URL,
          'v1beta1',
          'a',
          'b',
          'c',
          'd',
          'dicomWeb',
          '',
          '',
          '',
      ).GetStudyPath()

  @parameterized.named_parameters([
      dict(
          testcase_name='store_path',
          study_uid='',
      ),
      dict(
          testcase_name='study_path',
          study_uid='1',
      ),
  ])
  def testGetSeriesPathRaisesFromStoreOrStudyPath(self, study_uid):
    with self.assertRaises(ValueError):
      dicom_path.Path(
          dicom_path._HEALTHCARE_API_URL,
          'v1beta1',
          'a',
          'b',
          'c',
          'd',
          'dicomWeb',
          study_uid,
          '',
          '',
      ).GetSeriesPath()

  @parameterized.named_parameters([
      dict(
          testcase_name='invalid_base_url',
          path='ftp://foo.bar',
      ),
      dict(
          testcase_name='missing_netloc',
          path='http://',
      ),
      dict(
          testcase_name='missing_healthcare_api_version',
          path=f'{dicom_path._HEALTHCARE_API_URL}',
      ),
      dict(
          testcase_name='missing_healthcare_api_empty_version',
          path=f'{dicom_path._HEALTHCARE_API_URL}//',
      ),
      dict(
          testcase_name='bad_healthcare_api_study_prefix',
          path=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/foo',
      ),
      dict(
          testcase_name='bad_healthcare_api_study_prefix_with_study',
          path=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/foo/studies/1',
      ),
  ])
  def testFromStringPathRaises(self, path):
    with self.assertRaises(ValueError):
      dicom_path.FromString(path)

  @parameterized.named_parameters([
      dict(
          testcase_name='generic_http',
          path='http://foo.bar',
          expected='http://foo.bar',
      ),
      dict(
          testcase_name='generic_https',
          path='https://foo.bar',
          expected='https://foo.bar',
      ),
      dict(
          testcase_name='generic_https_study_prefix',
          path='https://foo.bar/abc/efg/',
          expected='https://foo.bar/abc/efg',
      ),
      dict(
          testcase_name='generic_https_study_prefix_with_study',
          path='https://foo.bar/abc/efg/studies/1',
          expected='https://foo.bar/abc/efg/studies/1',
      ),
      dict(
          testcase_name='generic_https_study_no_prefix_with_study',
          path='https://foo.bar/studies/1',
          expected='https://foo.bar/studies/1',
      ),
      dict(
          testcase_name='generic_https_study_no_prefix_with_study_series',
          path='https://foo.bar/studies/1/series/2/',
          expected='https://foo.bar/studies/1/series/2',
      ),
      dict(
          testcase_name='generic_https_study_no_prefix_with_study_series_inst',
          path='https://foo.bar/studies/1/series/2/instances/3/',
          expected='https://foo.bar/studies/1/series/2/instances/3',
      ),
      dict(
          testcase_name='legacy_path',
          path='projects/p/locations/l/datasets/d/dicomStores/ds',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb',
      ),
      dict(
          testcase_name='full_hcapi_path',
          path=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb',
      ),
      dict(
          testcase_name='full_hcapi_path_studies',
          path=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1.2.3',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1.2.3',
      ),
  ])
  def testFromStringPath(self, path, expected):
    self.assertEqual(str(dicom_path.FromString(path)), expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='no_params',
          store=None,
          study=None,
          series=None,
          instance=None,
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3',
      ),
      dict(
          testcase_name='store_only',
          store='a',
          study=None,
          series=None,
          instance=None,
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/a/dicomWeb',
      ),
      dict(
          testcase_name='store_and_study',
          store='a',
          study='b',
          series=None,
          instance=None,
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/a/dicomWeb/studies/b',
      ),
      dict(
          testcase_name='store_and_study_series',
          store='a',
          study='b',
          series='c',
          instance=None,
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/a/dicomWeb/studies/b/series/c',
      ),
      dict(
          testcase_name='store_and_study_series_instance',
          store='a',
          study='b',
          series='c',
          instance='d',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/a/dicomWeb/studies/b/series/c/instances/d',
      ),
      dict(
          testcase_name='instances',
          store=None,
          study=None,
          series=None,
          instance='d',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/d',
      ),
      dict(
          testcase_name='series_instances',
          store=None,
          study=None,
          series='c',
          instance='d',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/c/instances/d',
      ),
      dict(
          testcase_name='study_series_instances',
          store=None,
          study='b',
          series='c',
          instance='d',
          expected=f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/b/series/c/instances/d',
      ),
  ])
  def testFromPath(self, store, study, series, instance, expected):
    path = dicom_path.FromString(
        f'{dicom_path._HEALTHCARE_API_URL}/v1beta1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3'
    )
    self.assertEqual(
        dicom_path.FromPath(path, store, study, series, instance).complete_url,
        expected,
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='dicom_path',
          test=dicom_path.FromString(
              '/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3'
          ),
      ),
      dict(
          testcase_name='string',
          test=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3',
      ),
  ])
  def testPathEqualTrue(self, test):
    path = dicom_path.FromString(
        f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3'
    )
    self.assertEqual(path, test)

  @parameterized.named_parameters([
      dict(
          testcase_name='dicom_store_path',
          test=dicom_path.FromString(
              '/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb'
          ),
      ),
      dict(
          testcase_name='dicom_study_path',
          test=dicom_path.FromString(
              '/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1'
          ),
      ),
      dict(
          testcase_name='dicom_series_path',
          test=dicom_path.FromString(
              '/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2'
          ),
      ),
      dict(
          testcase_name='string',
          test=f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/4',
      ),
      dict(testcase_name='number', test=1),
  ])
  def testPathEqualFalse(self, test):
    path = dicom_path.FromString(
        f'{dicom_path._HEALTHCARE_API_URL}/v1/projects/p/locations/l/datasets/d/dicomStores/ds/dicomWeb/studies/1/series/2/instances/3'
    )
    self.assertNotEqual(path, test)

  @parameterized.parameters([
      'https://foo.com/studies/1.2.3/series/4.5.6/instances/7.8.9',
      'https://foo.com//studies/1.2.3/series/4.5.6/instances/7.8.9',
  ])
  def testPathNoStudyUidPrefix(self, path):
    path = dicom_path.FromString(path)
    self.assertEqual(path.study_prefix, '')
    self.assertEqual(path.study_uid, '1.2.3')
    self.assertEqual(path.series_uid, '4.5.6')
    self.assertEqual(path.instance_uid, '7.8.9')
    self.assertEqual(
        path.complete_url,
        'https://foo.com/studies/1.2.3/series/4.5.6/instances/7.8.9',
    )


if __name__ == '__main__':
  absltest.main()
