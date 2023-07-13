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

package(
    default_applicable_licenses = ["//ez_wsi_dicomweb:license"],
    default_visibility = ["//ez_wsi_dicomweb:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "dicom_store_mock",
    srcs = [
        "dicom_store_mock.py",
    ],
    srcs_version = "PY3",
    deps = [":dicom_store_mock_types"],
)

py_library(
    name = "dicom_store_mock_types",
    srcs = [
        "dicom_store_mock_types.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "dicom_store_mock_test",
    srcs = [
        "dicom_store_mock_test.py",
    ],
    data = [
        "//ez_wsi_dicomweb:testdata_files",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dicom_store_mock",
        ":dicom_store_mock_types",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
    ],
)
