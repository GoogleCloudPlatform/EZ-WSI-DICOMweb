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
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "ez_wsi_dicomweb",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "ez_wsi_dicomweb",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":abstract_slide_frame_cache",
        ":dicom_frame_decoder",
        ":dicom_slide",
        ":dicom_store",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":magnification",
        ":patch_generator",
        ":slide_level_map",
    ],
)

py_library(
    name = "dicom_web_interface",
    srcs = ["dicom_web_interface.py"],
    srcs_version = "PY3",
    deps = [":ez_wsi_errors"],
)

py_library(
    name = "slide_level_map",
    srcs = ["slide_level_map.py"],
    srcs_version = "PY3",
    deps = [
        ":dicom_web_interface",
        ":ez_wsi_errors",
    ],
)

py_library(
    name = "dicom_slide",
    srcs = ["dicom_slide.py"],
    srcs_version = "PY3",
    deps = [
        ":abstract_slide_frame_cache",
        ":dicom_frame_decoder",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":magnification",
        ":slide_level_map",
    ],
)

py_library(
    name = "magnification",
    srcs = ["magnification.py"],
    srcs_version = "PY3",
)

py_library(
    name = "patch_generator",
    srcs = ["patch_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":dicom_slide",
        ":ez_wsi_errors",
        ":magnification",
    ],
)

filegroup(
    name = "testdata_files",
    srcs = glob(["testdata/**"]),
)

py_test(
    name = "dicom_web_interface_test",
    srcs = ["dicom_web_interface_test.py"],
    data = [":testdata_files"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dicom_test_utils",
        ":dicom_web_interface",
        ":ez_wsi_errors",
    ],
)

py_test(
    name = "slide_level_map_test",
    srcs = ["slide_level_map_test.py"],
    data = [":testdata_files"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dicom_test_utils",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":slide_level_map",
    ],
)

py_test(
    name = "dicom_slide_test",
    srcs = ["dicom_slide_test.py"],
    data = [":testdata_files"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":abstract_slide_frame_cache",
        ":dicom_slide",
        ":dicom_test_utils",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":magnification",
    ],
)

py_test(
    name = "magnification_test",
    srcs = ["magnification_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":magnification"],
)

py_test(
    name = "patch_generator_test",
    srcs = ["patch_generator_test.py"],
    data = [":testdata_files"],
    deps = [
        ":dicom_slide",
        ":dicom_test_utils",
        ":ez_wsi_errors",
        ":magnification",
        ":patch_generator",
    ],
)

py_library(
    name = "dicom_frame_decoder",
    srcs = ["dicom_frame_decoder.py"],
    srcs_version = "PY3",
)

py_library(
    name = "ez_wsi_errors",
    srcs = ["ez_wsi_errors.py"],
)

py_test(
    name = "ez_wsi_errors_test",
    srcs = ["ez_wsi_errors_test.py"],
    deps = [":ez_wsi_errors"],
)

py_test(
    name = "dicom_store_test",
    srcs = ["dicom_store_test.py"],
    data = [":testdata_files"],
    deps = [
        ":dicom_store",
        ":dicom_test_utils",
    ],
)

py_test(
    name = "dicom_frame_decoder_test",
    srcs = ["dicom_frame_decoder_test.py"],
    data = [":testdata_files"],
    deps = [
        ":dicom_frame_decoder",
        ":dicom_test_utils",
    ],
)

py_library(
    name = "abstract_slide_frame_cache",
    srcs = ["abstract_slide_frame_cache.py"],
)

py_library(
    name = "dicom_store",
    srcs = ["dicom_store.py"],
    deps = [
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
    ],
)

py_library(
    name = "dicom_test_utils",
    srcs = ["dicom_test_utils.py"],
    deps = [":dicom_web_interface"],
)
