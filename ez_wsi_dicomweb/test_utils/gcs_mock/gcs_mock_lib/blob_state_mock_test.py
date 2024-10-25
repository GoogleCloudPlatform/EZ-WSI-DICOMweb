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
"""Blob State Mock Tests."""
import copy
import dataclasses
import datetime

from absl.testing import absltest

from ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib import blob_state_mock


class BlobMockStateTest(absltest.TestCase):

  def test_gcs_blob_mock_state_constructor(self):
    size_in_bytes = 5
    md5_hash = 'abcdef'
    metadata = {'foo': 'bar'}
    generation = 6
    metageneration = 8
    etag = 'MockEtag'
    content_type = 'text'
    time_created = datetime.datetime.now()
    time_deleted = datetime.datetime.now()
    updated = datetime.datetime.now()
    component_count = 1
    blob_state = blob_state_mock.BlobStateMock(
        size_in_bytes,
        md5_hash,
        metadata,
        generation,
        metageneration,
        etag,
        content_type,
        time_created,
        time_deleted,
        updated,
        component_count,
    )
    self.assertEqual(
        dataclasses.asdict(blob_state),
        {
            'size_in_bytes': size_in_bytes,
            'md5_hash': md5_hash,
            'metadata': metadata,
            'generation': generation,
            'metageneration': metageneration,
            'etag': etag,
            'content_type': content_type,
            'time_created': time_created,
            'time_deleted': time_deleted,
            'updated': updated,
            'component_count': component_count,
        },
    )

  def test_gcs_blob_mock_state_copy(self):
    size_in_bytes = 5
    md5_hash = 'abcdef'
    metadata = {'foo': 'bar'}
    generation = 6
    metageneration = 8
    etag = 'MockEtag'
    content_type = 'text'
    time_created = datetime.datetime.now()
    time_deleted = datetime.datetime.now()
    updated = datetime.datetime.now()
    component_count = 1
    blob_state = blob_state_mock.BlobStateMock(
        size_in_bytes,
        md5_hash,
        metadata,
        generation,
        metageneration,
        etag,
        content_type,
        time_created,
        time_deleted,
        updated,
        component_count,
    )
    blob_state = copy.copy(blob_state)
    self.assertEqual(
        dataclasses.asdict(blob_state),
        {
            'size_in_bytes': size_in_bytes,
            'md5_hash': md5_hash,
            'metadata': metadata,
            'generation': generation,
            'metageneration': metageneration,
            'etag': etag,
            'content_type': content_type,
            'time_created': time_created,
            'time_deleted': time_deleted,
            'updated': updated,
            'component_count': component_count,
        },
    )


if __name__ == '__main__':
  absltest.main()
