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
from typing import Optional

from ez_wsi_dicomweb import credential_factory as credential_factory_module
import google.auth

# Deprecated, use ez_wsi_dicomweb.credential_factory instead.

AbstractCredentialFactory = credential_factory_module.AbstractCredentialFactory
CredentialFactory = credential_factory_module.CredentialFactory


def refresh_credentials(
    auth_credentials: google.auth.credentials.Credentials,
    credential_factory: Optional[AbstractCredentialFactory] = None,
) -> google.auth.credentials.Credentials:
  """Refreshs credentials."""
  return credential_factory_module.refresh_credentials(
      auth_credentials, credential_factory
  )
