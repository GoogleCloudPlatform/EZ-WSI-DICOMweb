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
"""Tests for ez wsi errors."""
import inspect
import sys

from absl.testing import absltest
from absl.testing import parameterized

from ez_wsi_dicomweb import ez_wsi_errors


class EZWsiErrorsTest(parameterized.TestCase):

  def test_all_errors_use_base_class(self):
    for _, cls in inspect.getmembers(
        sys.modules[ez_wsi_errors.__name__], inspect.isclass
    ):
      self.assertTrue(issubclass(cls, ez_wsi_errors.EZWsiError))


if __name__ == '__main__':
  absltest.main()
