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
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "dicom_json",
    srcs = ["dicom_json.py"],
    deps = [
        ":tags",
    ],
)

py_library(
    name = "dicom_path",
    srcs = ["dicom_path.py"],
    deps = [requirement("dataclasses_json")],
)

py_library(
    name = "tags",
    srcs = ["tags.py"],
)

py_library(
    name = "test_dicom_path_util",
    srcs = ["test_dicom_path_util.py"],
    deps = [
        ":dicom_path",
    ],
)

py_library(
    name = "tag_values",
    srcs = ["tag_values.py"],
)

py_library(
    name = "dicom_builder",
    srcs = ["dicom_builder.py"],
    deps = [
        ":dicom_json",
        ":tag_values",
        ":tags",
        requirement("numpy"),
    ],
)

py_library(
    name = "test_dwc_util",
    srcs = ["test_dwc_util.py"],
    deps = [
        ":dicom_builder",
        ":dicom_json",
        ":tag_values",
        ":tags",
    ],
)

py_test(
    name = "dicom_json_test",
    srcs = ["dicom_json_test.py"],
    deps = [
        ":dicom_json",
        ":tags",
    ],
)

py_test(
    name = "dicom_path_test",
    srcs = ["dicom_path_test.py"],
    deps = [
        ":dicom_path",
        ":test_dicom_path_util",
    ],
)

py_test(
    name = "dicom_builder_test",
    srcs = ["dicom_builder_test.py"],
    deps = [
        ":dicom_builder",
        ":dicom_json",
        ":tag_values",
        ":tags",
        ":test_dwc_util",
        requirement("numpy"),
    ],
)
