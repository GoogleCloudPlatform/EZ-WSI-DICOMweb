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
"""Shared testing utilities."""
import collections
import dataclasses
import inspect
import os
from typing import List, Mapping, Optional, Tuple

from absl.testing import absltest


@dataclasses.dataclass(frozen=True)
class TestFileState:
  """Temp file test state."""

  path: str
  size: int

  @property
  def filename(self) -> str:
    _, filename = os.path.split(self.path)
    return filename

  @property
  def dirname(self) -> str:
    dirname, _ = os.path.split(self.path)
    return dirname


@dataclasses.dataclass(frozen=True)
class TestFiles:
  root: str
  files: Mapping[str, TestFileState]

  def list_mocked_file_paths(self) -> List[str]:
    return list(self.files.keys())


def create_mock_test_files(
    dir_path: str, mock_file_descriptions: List[Tuple[str, int]]
) -> TestFiles:
  """Create files within temporary directory for testing."""
  files = collections.OrderedDict()
  for file_path, file_size in mock_file_descriptions:
    fpath = os.path.join(dir_path, file_path)
    dirname, _ = os.path.split(fpath)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(fpath, 'wt') as outfile:
      outfile.write('*' * file_size)
    files[file_path] = TestFileState(fpath, file_size)
  return TestFiles(dir_path, files)


def test_mock_methods_and_actual_methods_have_same_signatures(
    test_case: absltest.TestCase,
    actual_class: object,
    mocked_class: object,
    ignored_attributes: Optional[List[str]] = None,
):
  """Validate mock's methods implements actual methods signature correctly.

  Args:
    test_case: TestCase instance.
    actual_class: Instance of class being mocked.
    mocked_class: Instance of mock implementation.  Mock autospec validation
      validates that callers call mocks with the correct parameters. This test
      validates that the implementation of the mock also matches.  For mocked
      implementations to function in all cases mocked methods are required to
      have method prototypes(signatures) that match those in the actual method.
      This is required to enable mock to be called correctly, with parameter by
      order, by keyword, and with parameter defaults. This test validates that
      all public methods and the __init__ function on BucketMock meet these
      requirements. If this test fails the failing method prototype/ signature
      should be compared with the actual signature of the method on
      google.cloud.storage.Bucket.
    ignored_attributes: Attributes to exclude from testing.
  """
  if ignored_attributes is None:
    ignored_attributes = []
  # Create actual instances of both objects.
  for mocked_attribute in dir(mocked_class):
    if mocked_attribute != '__init__' and (
        (mocked_attribute.startswith('__') and mocked_attribute.endswith('__'))
        or (
            mocked_attribute.startswith('_')
            and not mocked_attribute.endswith('__')
        )
    ):
      # ignore default python methods and private methods.
      continue
    mock_class_attribute = mocked_class.__getattribute__(mocked_attribute)
    try:
      actual_class_attribute = actual_class.__getattribute__(mocked_attribute)
    except AttributeError:
      continue
    try:
      mock_sig = inspect.signature(mock_class_attribute)
    except TypeError:
      # Skip properties.
      continue
    actual_sig = inspect.signature(actual_class_attribute)
    mock_sig_params = mock_sig.parameters
    actual_sig_params = actual_sig.parameters
    # Test methods have parameters with the same names in the same order.

    if mocked_attribute in ignored_attributes:
      continue
    test_case.assertEqual(
        list(mock_sig_params),
        list(actual_sig_params),
        f'Testing {mocked_attribute}',
    )
    # Test attributes associated with parameters match.
    for param_name, mock_parameter in mock_sig_params.items():
      actual_parameter = actual_sig_params[param_name]
      # Do not test retry parameter. Not handled by mock and init in actual
      # classes by default to instance of google.api_core.retry.Retry
      if param_name == 'retry':
        continue
      if actual_parameter.default != inspect.Signature.empty:
        test_case.assertEqual(
            mock_parameter.default,
            actual_parameter.default,
            f'Testing {mocked_attribute}({param_name})',
        )
      if actual_parameter.annotation != inspect.Signature.empty:
        test_case.assertEqual(
            mock_parameter.annotation,
            actual_parameter.annotation,
            f'Testing {mocked_attribute}({param_name})',
        )
      if actual_parameter.kind != inspect.Signature.empty:
        test_case.assertEqual(
            mock_parameter.kind,
            actual_parameter.kind,
            f'Testing {mocked_attribute}({param_name})',
        )


def create_single_file_mock(
    temp_dir_path: str,
    bucket_name: str,
    blob_name: str,
    blob_size_bytes: int = 10,
) -> Mapping[str, str]:
  return {
      bucket_name: create_mock_test_files(
          temp_dir_path, [(blob_name, blob_size_bytes)]
      ).root
  }
