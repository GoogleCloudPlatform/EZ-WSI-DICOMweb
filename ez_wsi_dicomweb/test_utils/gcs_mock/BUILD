load("@ez_wsi_pip_deps//:requirements.bzl", "requirement")

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

licenses(["notice"])

package(
    default_applicable_licenses = ["//ez_wsi_dicomweb:license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "gcs_mock",
    srcs = ["gcs_mock.py"],
    deps = [
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:blob_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:bucket_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:client_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:gcs_mock_types",
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:gcs_state_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib:transfer_manager_mock",
    ],
)

py_test(
    name = "gcs_mock_test",
    srcs = ["gcs_mock_test.py"],
    deps = [
        ":gcs_mock",
    ],
)
