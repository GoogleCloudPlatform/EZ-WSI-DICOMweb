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
"""Tests for patch embedding types."""
from unittest import mock

from absl.testing import absltest
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import patch_embedding_types
import numpy as np


class PathEmbeddingTypeTest(absltest.TestCase):

  def test_patch_embedding_ensemble_result_raises_if_incorrect_init(self):
    source = mock.create_autospec(
        patch_embedding_types.PatchEmbeddingSource, instance=True
    )
    with self.assertRaises(ValueError):
      patch_embedding_types.PatchEmbeddingEnsembleResult(source, None, None)
    with self.assertRaises(ValueError):
      patch_embedding_types.PatchEmbeddingEnsembleResult(
          source,
          np.zeros((1,), dtype=np.uint8),
          patch_embedding_types.PatchEmbeddingError('a', 'a'),
      )

  def test_raises_if_error_and_accessing_embedding(self):
    source = mock.create_autospec(
        patch_embedding_types.PatchEmbeddingSource, instance=True
    )
    result = patch_embedding_types.PatchEmbeddingEnsembleResult(
        source, None, patch_embedding_types.PatchEmbeddingError('a', 'a')
    )
    with self.assertRaises(ez_wsi_errors.PatchEmbeddingEndpointError):
      _ = result.embedding


if __name__ == '__main__':
  absltest.main()
