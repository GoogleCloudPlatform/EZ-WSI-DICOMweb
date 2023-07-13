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
"""DICOMweb Abstract Credential Factory and Default implementation."""
import abc
import copy
import json
import os
from typing import Any, List, Mapping, Optional, Union

import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests  # Required by: google.auth.transport.requests.Request()

_SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/cloud-healthcare',
]


class AbstractCredentialFactory(metaclass=abc.ABCMeta):
  """Generates the credentials used to access DICOM store.

  Implementations of the abstract credential factory should be compatible with
  pickle serialization. The purpose of the credential factory is to enable
  EZ-WSI to construct the credentials needed to access the DICOM store
  following pickle deserialization. As one example, this enables EZ-WSI DICOM
  Store and DICOM Slide classes to be initalized once and passed through a
  cloud dataflow pipeline.
  """

  @abc.abstractmethod
  def get_credentials(self) -> google.auth.credentials.Credentials:
    """Return credentials to use to access DICOM Store."""


def refresh_credentials(
    auth_credentials: google.auth.credentials.Credentials,
) -> google.auth.credentials.Credentials:
  """Refreshs credentials."""
  if not auth_credentials.valid:
    auth_credentials.refresh(google.auth.transport.requests.Request())
  return auth_credentials


class CredentialFactory(AbstractCredentialFactory):
  """Factory for default or service account credential creation."""

  def __init__(
      self,
      json_param: Optional[
          Union[Mapping[str, Any], str, bytes, os.PathLike[Any]]
      ] = None,
      scopes: Optional[List[str]] = None,
  ) -> None:
    """Credential Factory Constructor.

    Args:
      json_param: Optional parameter that defines location of JSON file, or
        loaded JSON that contains service account credentials which should be
        used for auth.  If undefined, then the default credentials of the
        running environment are used.
      scopes: Credential scopes if undefined defaults to:
        ['https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/cloud-healthcare',]
    """
    if not json_param:
      self._json = {}
    elif (
        isinstance(json_param, str)
        or isinstance(json_param, bytes)
        or isinstance(json_param, os.PathLike)
    ):
      # Read JSON from file."""
      with open(json_param, 'rt') as infile:
        self._json = json.load(infile)
    else:
      # Use in memory JSON loaded in memory as python Dict.
      self._json = copy.copy(json_param)
    self._scopes = _SCOPES if scopes is None else copy.copy(scopes)

  def get_credentials(self) -> google.auth.credentials.Credentials:
    """Returns credentials to use to accessing DICOM store."""
    if self._json:
      return refresh_credentials(
          service_account.Credentials.from_service_account_info(
              self._json, scopes=self._scopes
          )
      )
    return refresh_credentials(google.auth.default(scopes=self._scopes)[0])
