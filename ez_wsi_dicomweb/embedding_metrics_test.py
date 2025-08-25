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
"""Tests for slide cache logger base python implementation."""
from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import embedding_metrics


class EmbeddingMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='same_direction',
          e1=[4, 4],
          e2=[1, 1],
          expected=1.0,
      ),
      dict(
          testcase_name='opposite_direction',
          e1=[1, 1],
          e2=[-1, -1],
          expected=-1.0,
      ),
      dict(
          testcase_name='perpendicular',
          e1=[1, 1],
          e2=[1, -1],
          expected=0.0,
      ),
  ])
  def test_cosine_similarity(self, e1, e2, expected):
    self.assertEqual(
        round(embedding_metrics.cosine_similarity(e1, e2), 5), expected
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='same_direction',
          e1=[4, 4],
          e2=[4, 4],
          expected=0.0,
      ),
      dict(
          testcase_name='opposite_direction',
          e1=[1, 1],
          e2=[-1, 1],
          expected=2.0,
      ),
  ])
  def test_euclidian_distance(self, e1, e2, expected):
    self.assertEqual(embedding_metrics.euclidian_distance(e1, e2), expected)


if __name__ == '__main__':
  absltest.main()
