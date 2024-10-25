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
    default_visibility = ["//ez_wsi_dicomweb/test_utils/gcs_mock:__subpackages__"],
)

py_library(
    name = "client_mock",
    srcs = ["client_mock.py"],
    deps = [
        ":gcs_mock_types",
        ":gcs_state_mock",
    ],
)

py_library(
    name = "bucket_mock",
    srcs = ["bucket_mock.py"],
    deps = [
        ":client_mock",
        ":gcs_mock_types",
        ":gcs_mock_utils",
    ],
)

py_library(
    name = "blob_mock",
    srcs = ["blob_mock.py"],
    deps = [
        ":blob_state_mock",
        ":client_mock",
        ":gcs_mock_constants",
        ":gcs_mock_types",
        ":gcs_mock_utils",
    ],
)

py_library(
    name = "gcs_state_mock",
    srcs = ["gcs_state_mock.py"],
    deps = [
        ":blob_state_mock",
        ":gcs_mock_constants",
        ":gcs_mock_types",
        ":gcs_mock_utils",
    ],
)

py_library(
    name = "gcs_mock_constants",
    srcs = ["gcs_mock_constants.py"],
)

py_library(
    name = "gcs_mock_types",
    srcs = ["gcs_mock_types.py"],
)

py_library(
    name = "blob_state_mock",
    srcs = ["blob_state_mock.py"],
)

py_test(
    name = "blob_mock_test",
    srcs = ["blob_mock_test.py"],
    deps = [
        ":blob_mock",
        ":blob_state_mock",
        ":bucket_mock",
        ":gcs_mock_constants",
        ":gcs_mock_types",
        ":gcs_test_utils",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "bucket_mock_test",
    srcs = ["bucket_mock_test.py"],
    deps = [
        ":bucket_mock",
        ":gcs_mock_types",
        ":gcs_test_utils",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "client_mock_test",
    srcs = ["client_mock_test.py"],
    deps = [
        ":client_mock",
        ":gcs_mock_types",
        ":gcs_state_mock",
        ":gcs_test_utils",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "gcs_state_mock_test",
    srcs = ["gcs_state_mock_test.py"],
    deps = [
        ":gcs_mock_types",
        ":gcs_state_mock",
        ":gcs_test_utils",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "blob_state_mock_test",
    srcs = ["blob_state_mock_test.py"],
    deps = [":blob_state_mock"],
)

py_library(
    name = "gcs_test_utils",
    srcs = ["gcs_test_utils.py"],
)

py_library(
    name = "gcs_mock_utils",
    srcs = ["gcs_mock_utils.py"],
    deps = [
        ":gcs_mock_constants",
        ":gcs_mock_types",
    ],
)

py_test(
    name = "gcs_mock_utils_test",
    srcs = ["gcs_mock_utils_test.py"],
    deps = [
        ":gcs_mock_constants",
        ":gcs_mock_utils",
    ],
)
