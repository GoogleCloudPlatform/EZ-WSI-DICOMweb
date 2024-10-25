# Copyright 2024 Google LLC
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
"""Test actual endpoint."""

import enum
import logging
import sys
import time
from typing import Iterator, Mapping, Sequence, Union

from absl import app
from absl import flags
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import patch_embedding
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import pixel_spacing
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np


DICOM_SERIES_PATH_FLAG = flags.DEFINE_string(
    'dicom_series_path',
    None,
    'DICOMweb path to DICOM series',
)

DICOM_IMAGING_MAGNIFICATION_FLAG = flags.DEFINE_string(
    'magnification',
    '20X',
    'Approximate magnification of DICOM imaging to generate embeddings from.',
)


class SupportedEndpoints(enum.Enum):
  V1 = 'V1'
  V2 = 'V2'


PATHOLOGY_ENDPOINT_FLAG = flags.DEFINE_enum_class(
    'endpoint', SupportedEndpoints.V2, SupportedEndpoints, 'Endpoint to test.'
)

VERTEX_ENDPOINT_GCP_PROJECT_FLAG = flags.DEFINE_string(
    'vertex_endpoint_gcp_project',
    '',
    'GCP project hosting Vertex endpoint if "" uses endpoint default.',
)

VERTEX_ENDPOINT_LOCATION_FLAG = flags.DEFINE_string(
    'vertex_endpoint_location',
    '',
    'Location/region hosting vertex endpoint if "" uses endpoint default.',
)

VERTEX_ENDPOINT_ID_FLAG = flags.DEFINE_string(
    'endpoint_id',
    '',
    'Vertex endpoint id "" uses endpoint default.',
)


ICC_PROFILE_FLAG = flags.DEFINE_enum_class(
    'icc_profile',
    patch_embedding_endpoints.IccProfileNormalization.NONE,
    patch_embedding_endpoints.IccProfileNormalization,
    'ICC Profile normalization to perform on imaging with source icc profile.',
)

TEST_RUNS_PER_EMBEDDING_REQUEST_FLAG = flags.DEFINE_integer(
    'number_of_runs_per_embedding',
    10,
    'Number of times to benchmark each embedding request.',
    lower_bound=1,
)

EMBEDDING_REQUEST_COUNTS_FLAG = flags.DEFINE_multi_integer(
    'request_counts',
    (1),
    'Number of embeddings to request in test run',
    lower_bound=1,
)


def _endpoint_flag_configuration() -> Mapping[str, str]:
  """Returns configuration parameters of parameters or endpoints."""
  params = {}
  if VERTEX_ENDPOINT_GCP_PROJECT_FLAG.value:
    params['project_id'] = VERTEX_ENDPOINT_GCP_PROJECT_FLAG.value
  if VERTEX_ENDPOINT_LOCATION_FLAG.value:
    params['endpoint_location'] = VERTEX_ENDPOINT_LOCATION_FLAG.value
  if VERTEX_ENDPOINT_ID_FLAG.value:
    params['endpoint_id'] = VERTEX_ENDPOINT_ID_FLAG.value
  return params


def _create_patch_iterator(
    endpoint: patch_embedding_endpoints.AbstractPatchEmbeddingEndpoint,
    ds: dicom_slide.DicomSlide,
    level: Union[dicom_slide.Level, dicom_slide.ResizedLevel],
    number_of_embeddings: int,
) -> Iterator[dicom_slide.DicomPatch]:
  """Yields non-overlapping patches."""
  px = 0
  py = 0
  width = endpoint.patch_width()
  height = endpoint.patch_height()
  for _ in range(number_of_embeddings):
    yield ds.get_patch(
        level,
        px,
        py,
        width,
        height,
    )
    px += width
    if px + width >= level.width:
      py += height
      px = 0


def main(unused_argv: Sequence[str]) -> None:
  # Log to standard out.
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logging.getLogger().addHandler(handler)
  logging.getLogger().setLevel(logging.INFO)

  if PATHOLOGY_ENDPOINT_FLAG.value == SupportedEndpoints.V1:
    logging.info('Using V1 endpoint')
    endpoint = patch_embedding_endpoints.V1PatchEmbeddingEndpoint(
        **_endpoint_flag_configuration()
    )
  else:
    logging.info('Using V2 endpoint')
    endpoint = patch_embedding_endpoints.V2PatchEmbeddingEndpoint(
        icc_profile_normalization=ICC_PROFILE_FLAG.value,
        **_endpoint_flag_configuration()
    )

  # Connect to DICOM Slide; retrieves slide metadata; pixel data not retrieved.
  ds = dicom_slide.DicomSlide(
      path=dicom_path.FromString(DICOM_SERIES_PATH_FLAG.value),
      dwi=dicom_web_interface.DicomWebInterface(
          credential_factory.DefaultCredentialFactory()
      ),
  )

  # Identify DICOM pyramid level to generate embeddings from
  level = ds.get_level_by_pixel_spacing(
      pixel_spacing.PixelSpacing.FromMagnificationString(
          DICOM_IMAGING_MAGNIFICATION_FLAG.value
      ),
      maximum_downsample=8.0,  # maximum of 8x downsampling.
  )

  # Iterates over range of embedding request clounds.
  for number_of_embeddings in EMBEDDING_REQUEST_COUNTS_FLAG.value:
    run_time = []
    count = TEST_RUNS_PER_EMBEDDING_REQUEST_FLAG.value
    # Measure average time to get embedding request for across 10 tests.
    # Perform one request, to warm up embedding model before averging time.
    for i in range(count + 1):
      patches = _create_patch_iterator(
          endpoint, ds, level, number_of_embeddings
      )
      start_time = time.time()
      # Request embeddings for patches; transform to iterator to list to
      # pull all values.
      list(patch_embedding.generate_patch_embeddings(endpoint, patches))
      # Request returns iterator, iterate over all embedddings
      if i == 0:
        # skip timing on first request to give vertex time to warm up.
        continue
      elapsed_time = time.time() - start_time
      # Log time to fulfill embedding request.
      logging.info('%f sec', elapsed_time)
      run_time.append(elapsed_time)
    # Log average time for to fulfill embedding request.
    logging.info(
        '%d: Average time(%d runs): %f+-%f sec',
        number_of_embeddings,
        count,
        np.mean(run_time),
        np.std(run_time),
    )


if __name__ == '__main__':
  app.run(main)
