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
"""Mock V1 Embedding Endpoint."""

import http
import io
import json
from typing import Optional

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import requests
import requests_mock


_VERSION = 'V1EmbeddingEndpointMock'
_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys


class V1DicomEmbeddingEndpointMock:
  """Mocks V1 Pathology Embedding Enpoint."""

  def __init__(
      self, mock_request: requests_mock.Mocker, mock_endpoint_url: str
  ):
    self._mock_endpoint_url = mock_endpoint_url
    mock_request.add_matcher(self._handle_request)

  def _handle_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Handles a request for the mock embedding endpoint.

    Args:
      request: The request to handle.

    Returns:
      None if request not handled otherwise mock V1 embedding response.
      Mock embedding is mean channel value per patch.
    """
    if not request.url.startswith(self._mock_endpoint_url):
      return None
    message = request.json()
    embedding_results = []
    for embedding_request in message[_EndpointJsonKeys.INSTANCES]:
      bearer_token = embedding_request[_EndpointJsonKeys.BEARER_TOKEN]
      dicom_web_store_url = embedding_request[
          _EndpointJsonKeys.DICOM_WEB_STORE_URL
      ]
      dicom_study_uid = embedding_request[_EndpointJsonKeys.DICOM_STUDY_UID]
      dicom_series_uid = embedding_request[_EndpointJsonKeys.DICOM_SERIES_UID]
      dicom_instance_uids = embedding_request[_EndpointJsonKeys.INSTANCE_UIDS]
      metadata = embedding_request[_EndpointJsonKeys.EZ_WSI_STATE]
      dcf = credential_factory_module.TokenPassthroughCredentialFactory(
          bearer_token
      )
      dwi = dicom_web_interface.DicomWebInterface(dcf)
      path = dicom_path.FromPath(
          dicom_path.FromString(dicom_web_store_url),
          study_uid=dicom_study_uid,
          series_uid=dicom_series_uid,
          instance_uid='',
      )
      slide = dicom_slide.DicomSlide(
          dwi=dwi,
          path=path,
          enable_client_slide_frame_decompression=True,
          json_metadata=metadata,
      )
      ps = slide.get_instance_pixel_spacing(dicom_instance_uids[0])
      slide_embeddings = {}
      slide_embeddings[_EndpointJsonKeys.DICOM_STUDY_UID] = dicom_study_uid
      slide_embeddings[_EndpointJsonKeys.DICOM_SERIES_UID] = dicom_series_uid
      slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS] = []
      for patch_coordinates in embedding_request[
          _EndpointJsonKeys.PATCH_COORDINATES
      ]:
        x = patch_coordinates[_EndpointJsonKeys.X_ORIGIN]
        y = patch_coordinates[_EndpointJsonKeys.Y_ORIGIN]
        width = patch_coordinates[_EndpointJsonKeys.WIDTH]
        height = patch_coordinates[_EndpointJsonKeys.HEIGHT]
        slide_patch = slide.get_patch(ps, x, y, width, height)
        image_bytes = slide_patch.image_bytes()

        # Mock the embedding as the per channel patch average.
        image_bytes = np.mean(image_bytes, axis=(0, 1))
        patch_embedding = {}
        patch_embedding[_EndpointJsonKeys.EMBEDDINGS] = image_bytes.tolist()
        patch_embedding[_EndpointJsonKeys.PATCH_COORDINATE] = patch_coordinates
        slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS].append(
            patch_embedding
        )
      if slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS]:
        embedding_results.append(slide_embeddings)
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    ez_wsi_error = None
    msg = json.dumps({
        _EndpointJsonKeys.PREDICTIONS: [
            embedding_results,
            ez_wsi_error,
            _VERSION,
        ]
    }).encode('utf-8')
    resp.raw = io.BytesIO(msg)
    return resp


class V1GcsEmbeddingEndpointMock:
  """Mocks V1 Pathology Embedding Enpoint."""

  def __init__(
      self, mock_request: requests_mock.Mocker, mock_endpoint_url: str
  ):
    self._mock_endpoint_url = mock_endpoint_url
    mock_request.add_matcher(self._handle_request)

  def _handle_request(
      self, request: requests.Request
  ) -> Optional[requests.Response]:
    """Handles a request for the mock embedding endpoint.

    Args:
      request: The request to handle.

    Returns:
      None if request not handled otherwise mock V1 embedding response.
      Mock embedding is mean channel value per patch.
    """
    if not request.url.startswith(self._mock_endpoint_url):
      return None
    message = request.json()
    embedding_results = []
    for embedding_request in message[_EndpointJsonKeys.INSTANCES]:
      credential_factory = (
          credential_factory_module.TokenPassthroughCredentialFactory(
              embedding_request[_EndpointJsonKeys.BEARER_TOKEN]
          )
      )
      gcs_image_url = embedding_request[_EndpointJsonKeys.GCS_IMAGE_URL]
      metadata = embedding_request[_EndpointJsonKeys.EZ_WSI_STATE]
      if metadata:
        patch_metadata = metadata.get(_EndpointJsonKeys.PATCHES, [])
        image_metadata = metadata.get(_EndpointJsonKeys.IMAGE, None)
        if image_metadata is not None:
          image = gcs_image.GcsImage.create_from_json(image_metadata)
          # Optimization remove image compressed bytes.
          # Bytes only needed to be retained to call embedding api.
          image.clear_source_image_compressed_bytes()
        else:
          image = None
      else:
        patch_metadata = []
        image = None
      slide_embeddings = {}
      slide_embeddings[_EndpointJsonKeys.GCS_IMAGE_URL] = gcs_image_url
      slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS] = []
      for index, patch_coordinates in enumerate(
          embedding_request[_EndpointJsonKeys.PATCH_COORDINATES]
      ):
        x = patch_coordinates[_EndpointJsonKeys.X_ORIGIN]
        y = patch_coordinates[_EndpointJsonKeys.Y_ORIGIN]
        width = patch_coordinates[_EndpointJsonKeys.WIDTH]
        height = patch_coordinates[_EndpointJsonKeys.HEIGHT]
        if patch_metadata:
          patch = gcs_image.GcsPatch.create_from_json(patch_metadata[index])
          # Optimization remove image compressed bytes.
          # Bytes only needed to be retained to call embedding api.
          patch.source.clear_source_image_compressed_bytes()
          if patch.x != 0 or patch.y != 0:
            raise ValueError('InvalidPatch Position')
        else:
          if image is None:
            image = gcs_image.GcsImage(
                gcs_image_url,
                credential_factory=credential_factory,
            )
          patch = image.get_patch(x, y, width, height)
        if patch.width != width or patch.height != height:
          raise ValueError('PatchDimInvalid')
        image_bytes = patch.image_bytes()

        # Mock the embedding as the per channel patch average.
        image_bytes = np.mean(image_bytes, axis=(0, 1))
        patch_embedding = {}
        patch_embedding[_EndpointJsonKeys.EMBEDDINGS] = image_bytes.tolist()
        patch_embedding[_EndpointJsonKeys.PATCH_COORDINATE] = patch_coordinates
        slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS].append(
            patch_embedding
        )
      if slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS]:
        embedding_results.append(slide_embeddings)
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    ez_wsi_error = None
    msg = json.dumps({
        _EndpointJsonKeys.PREDICTIONS: [
            embedding_results,
            ez_wsi_error,
            _VERSION,
        ]
    }).encode('utf-8')
    resp.raw = io.BytesIO(msg)
    return resp
