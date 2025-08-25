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
"""Mock Embedding Endpoint."""

import base64
import dataclasses
import functools
import http
import io
import json
from typing import Any, List, MutableMapping, Optional

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb import pixel_spacing as pixel_spacing_module
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import PIL.Image
import requests
import requests_mock

_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys


class V2DicomEmbeddingEndpointMock:
  """Mocks V2 Pathology Embedding Enpoint."""

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
      bearer_token = embedding_request.get(_EndpointJsonKeys.BEARER_TOKEN, '')
      dcm_path = embedding_request[_EndpointJsonKeys.DICOM_PATH]
      path = dicom_path.FromString(dcm_path[_EndpointJsonKeys.SERIES_PATH])
      dicom_instance_uids = dcm_path[_EndpointJsonKeys.INSTANCE_UIDS]
      try:
        metadata = embedding_request[_EndpointJsonKeys.EXTENSIONS][
            _EndpointJsonKeys.EZ_WSI_STATE
        ]
      except KeyError:
        metadata = ''
      dcf = credential_factory_module.TokenPassthroughCredentialFactory(
          bearer_token
      )
      dwi = dicom_web_interface.DicomWebInterface(dcf)
      slide = dicom_slide.DicomSlide(
          dwi=dwi,
          path=path,
          enable_client_slide_frame_decompression=True,
          json_metadata=metadata,
      )
      ps = slide.get_instance_pixel_spacing(dicom_instance_uids[0])
      slide_embeddings = {}
      slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS] = []
      for patch_coordinates in embedding_request[
          _EndpointJsonKeys.PATCH_COORDINATES
      ]:
        x = patch_coordinates[_EndpointJsonKeys.X_ORIGIN]
        y = patch_coordinates[_EndpointJsonKeys.Y_ORIGIN]
        width = patch_coordinates[_EndpointJsonKeys.WIDTH]
        height = patch_coordinates[_EndpointJsonKeys.HEIGHT]
        level = slide.get_level_by_pixel_spacing(ps)
        slide_patch = slide.get_patch(level, x, y, width, height)
        image_bytes = slide_patch.image_bytes()

        # Mock the embedding as the per channel patch average.
        image_bytes = np.mean(image_bytes, axis=(0, 1))
        patch_embedding = {}
        patch_embedding[_EndpointJsonKeys.EMBEDDING_VECTOR] = (
            image_bytes.tolist()
        )
        patch_embedding[_EndpointJsonKeys.PATCH_COORDINATE] = patch_coordinates
        slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS].append(
            patch_embedding
        )
      if slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS]:
        embedding_results.append({_EndpointJsonKeys.RESULT: slide_embeddings})
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    msg = json.dumps({_EndpointJsonKeys.PREDICTIONS: embedding_results}).encode(
        'utf-8'
    )
    resp.raw = io.BytesIO(msg)
    return resp


class V2GcsEmbeddingEndpointMock:
  """Mocks V2 Pathology Embedding Enpoint."""

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
              embedding_request.get(_EndpointJsonKeys.BEARER_TOKEN, '')
          )
      )
      gcs_image_url = embedding_request[_EndpointJsonKeys.IMAGE_FILE_URI]
      try:
        metadata = embedding_request[_EndpointJsonKeys.EXTENSIONS][
            _EndpointJsonKeys.EZ_WSI_STATE
        ]
      except KeyError:
        metadata = ''
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
      slide_embeddings[_EndpointJsonKeys.IMAGE_FILE_URI] = gcs_image_url
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
        patch_embedding[_EndpointJsonKeys.EMBEDDING_VECTOR] = (
            image_bytes.tolist()
        )
        patch_embedding[_EndpointJsonKeys.PATCH_COORDINATE] = patch_coordinates
        slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS].append(
            patch_embedding
        )
      if slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS]:
        embedding_results.append({_EndpointJsonKeys.RESULT: slide_embeddings})
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    msg = json.dumps({_EndpointJsonKeys.PREDICTIONS: embedding_results}).encode(
        'utf-8'
    )
    resp.raw = io.BytesIO(msg)
    return resp


@dataclasses.dataclass(frozen=True)
class _Patch:
  x: int
  y: int
  width: int
  height: int


def _get_image_bytes(
    image_bytes: np.ndarray,
    patch_index: int,
    patch: Optional[_Patch],
) -> np.ndarray:
  """Returns the image bytes for a patch."""
  del patch_index
  if patch is None:
    height, width = image_bytes.shape[0:2]
    patch = _Patch(0, 0, width, height)
  return image_bytes[
      patch.y : patch.y + patch.height, patch.x : patch.x + patch.width
  ]


def _get_dicom_patch_image_bytes(
    slide: dicom_slide.DicomSlide,
    ps: Optional[pixel_spacing_module.PixelSpacing],
    patch_index: int,
    patch: Optional[_Patch],
) -> np.ndarray:
  """Returns the image bytes for a dicom patch."""
  del patch_index
  level = slide.get_level_by_pixel_spacing(ps)
  if patch is None:
    patch = _Patch(0, 0, level.width, level.height)
  return slide.get_patch(
      level, patch.x, patch.y, patch.width, patch.height
  ).image_bytes()


def _get_gcs_patch_image_bytes(
    gcs_image_url: str,
    credential_factory: credential_factory_module.CredentialFactory,
    image: Optional[gcs_image.GcsImage],
    patch_metadata: Optional[List[str]],
    patch_index: int,
    patch: Optional[_Patch],
) -> np.ndarray:
  """Returns the image bytes for a gcs patch."""
  if patch_metadata is not None and patch_metadata:
    patch = gcs_image.GcsPatch.create_from_json(patch_metadata[patch_index])
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
    if patch is None:
      patch = _Patch(0, 0, image.width, image.height)
    patch = image.get_patch(patch.x, patch.y, patch.width, patch.height)
  return patch.image_bytes()


def _add_mock_temp_bias_input_type(
    instance: MutableMapping[str, Any], input_type: str
) -> MutableMapping[str, Any]:
  instance[_EndpointJsonKeys.MODEL_TEMPERATURE] = 1.2
  instance[_EndpointJsonKeys.MODEL_BIAS] = 1.0
  instance['input_type'] = input_type
  return instance


class MedSigLipEmbeddingEndpointMock(
    patch_embedding_endpoints.MedSigLipEmbeddingEndpoint
):
  """Mocks MedSigLipEmbeddingEndpoint."""

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
      text_request = embedding_request.get(_EndpointJsonKeys.MEDSIGLIP_TEXT)
      image_request = embedding_request.get(_EndpointJsonKeys.MEDSIGLIP_IMAGE)
      if image_request is not None:
        bearer_token = image_request.get(
            _EndpointJsonKeys.MEDSIGLIP_ACCESS_CREDENTIAL
        )
        dcf = credential_factory_module.TokenPassthroughCredentialFactory(
            bearer_token
        )
        gcs_uri = image_request.get(_EndpointJsonKeys.MEDSIGLIP_GCS_URI)
        input_bytes = image_request.get(_EndpointJsonKeys.MEDSIGLIP_INPUT_BYTES)
        dcm_path = image_request.get(_EndpointJsonKeys.MEDSIGLIP_DICOMWEB_URI)

        if dcm_path is not None:
          path = dicom_path.FromString(dcm_path)
          try:
            metadata = image_request[_EndpointJsonKeys.EXTENSIONS][
                _EndpointJsonKeys.EZ_WSI_STATE
            ]
          except KeyError:
            metadata = ''
          dwi = dicom_web_interface.DicomWebInterface(dcf)
          slide = dicom_slide.DicomSlide(
              dwi=dwi,
              path=path.GetSeriesPath(),
              enable_client_slide_frame_decompression=True,
              json_metadata=metadata,
          )
          ps = slide.get_instance_pixel_spacing(path.instance_uid)
          get_patch_image_bytes = functools.partial(
              _get_dicom_patch_image_bytes, slide, ps
          )
        elif gcs_uri is not None:
          try:
            metadata = image_request[_EndpointJsonKeys.EXTENSIONS][
                _EndpointJsonKeys.EZ_WSI_STATE
            ]
          except KeyError:
            metadata = ''
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
          get_patch_image_bytes = functools.partial(
              _get_gcs_patch_image_bytes,
              gcs_uri,
              dcf,
              image,
              patch_metadata,
          )
        elif input_bytes is not None:
          image_bytes = base64.b64decode(input_bytes)
          with io.BytesIO(image_bytes) as frame_bytes:
            with PIL.Image.open(frame_bytes) as p_image:
              image_bytes = np.asarray(p_image)
          get_patch_image_bytes = functools.partial(
              _get_image_bytes,
              image_bytes,
          )
        else:
          raise ValueError('Invalid image input')
      elif text_request is not None:
        txt_bytes = np.frombuffer(text_request.encode('utf-8'), dtype=np.uint8)
        mean_text_embedding = np.round(np.mean(txt_bytes), 3)
        text_embedding = {}
        txt_length = txt_bytes.shape[0]
        text_embedding[_EndpointJsonKeys.EMBEDDING] = [
            mean_text_embedding,
            txt_length,
            1.0,
        ]
        embedding_results.append(
            _add_mock_temp_bias_input_type(text_embedding, 'text')
        )
        continue
      else:
        raise ValueError('Invalid input')

      slide_embeddings = {}
      slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS] = []
      patch_coordinates = image_request.get(
          _EndpointJsonKeys.MEDSIGLIP_PATCH_COORDINATES
      )
      if not patch_coordinates:
        image_bytes = get_patch_image_bytes(0, None)

        # Mock the embedding as the per channel patch average.
        image_bytes = np.round(np.mean(image_bytes, axis=(0, 1)), 3)
        patch_embedding = {}
        patch_embedding[_EndpointJsonKeys.EMBEDDING] = image_bytes.tolist()
        embedding_results.append(
            _add_mock_temp_bias_input_type(patch_embedding, 'image')
        )
        continue
      for index, patch_coordinate in enumerate(patch_coordinates):
        x = patch_coordinate[_EndpointJsonKeys.X_ORIGIN]
        y = patch_coordinate[_EndpointJsonKeys.Y_ORIGIN]
        width = patch_coordinate[_EndpointJsonKeys.WIDTH]
        height = patch_coordinate[_EndpointJsonKeys.HEIGHT]
        image_bytes = get_patch_image_bytes(index, _Patch(x, y, width, height))

        # Mock the embedding as the per channel patch average.
        image_bytes = np.round(np.mean(image_bytes, axis=(0, 1)), 3)
        patch_embedding = {}
        patch_embedding[_EndpointJsonKeys.EMBEDDING] = image_bytes.tolist()
        patch_embedding[_EndpointJsonKeys.PATCH_COORDINATE] = patch_coordinate
        slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS].append(
            patch_embedding
        )
      if slide_embeddings[_EndpointJsonKeys.PATCH_EMBEDDINGS]:
        embedding_results.append(
            _add_mock_temp_bias_input_type(slide_embeddings, 'image')
        )
    resp = requests.Response()
    resp.status_code = http.HTTPStatus.OK
    msg = json.dumps({_EndpointJsonKeys.PREDICTIONS: embedding_results}).encode(
        'utf-8'
    )
    resp.raw = io.BytesIO(msg)
    return resp
