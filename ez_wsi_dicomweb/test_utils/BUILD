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

package(
    default_applicable_licenses = ["//ez_wsi_dicomweb:license"],
    default_visibility = ["//ez_wsi_dicomweb:__subpackages__"],
)

py_library(
    name = "dicom_test_utils",
    srcs = ["dicom_test_utils.py"],
    data = ["//ez_wsi_dicomweb:testdata_files"],
    deps = [
        "//ez_wsi_dicomweb:dicom_web_interface",
        "//ez_wsi_dicomweb:dicomweb_credential_factory",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        requirement("pydicom"),
    ],
)

py_library(
    name = "embedding_endpoint_mock",
    srcs = ["embedding_endpoint_mock.py"],
    deps = [
        "//ez_wsi_dicomweb:credential_factory",
        "//ez_wsi_dicomweb:dicom_slide",
        "//ez_wsi_dicomweb:dicom_web_interface",
        "//ez_wsi_dicomweb:gcs_image",
        "//ez_wsi_dicomweb:patch_embedding_endpoints",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        requirement("numpy"),
        requirement("requests_mock"),
    ],
)

py_binary(
    name = "e2e_embedding_test",
    srcs = ["e2e_embedding_test.py"],
    deps = [
        "//ez_wsi_dicomweb:credential_factory",
        "//ez_wsi_dicomweb:dicom_slide",
        "//ez_wsi_dicomweb:dicom_web_interface",
        "//ez_wsi_dicomweb:patch_embedding",
        "//ez_wsi_dicomweb:patch_embedding_endpoints",
        "//ez_wsi_dicomweb:pixel_spacing",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        requirement("numpy"),
    ],
)
