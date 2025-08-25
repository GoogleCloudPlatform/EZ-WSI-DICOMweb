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

exports_files([
    "LICENSE",
    "requirements.txt",
])

py_library(
    name = "ez_wsi_dicomweb",
    srcs = ["__init__.py"],
    deps = [
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
    deps = [
        ":credential_factory",
        ":error_retry_util",
        ":ez_wsi_errors",
        requirement("dataclasses_json"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_json",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        requirement("requests-toolbelt"),
        requirement("retrying"),
    ],
)

py_library(
    name = "slide_level_map",
    srcs = ["slide_level_map.py"],
    deps = [
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":pixel_spacing",
        requirement("cachetools"),
        requirement("dataclasses_json"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_json",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        requirement("numpy"),
        requirement("pydicom"),
    ],
)

py_library(
    name = "dicom_slide",
    srcs = ["dicom_slide.py"],
    data = ["//third_party:icc_profiles"],
    deps = [
        ":credential_factory",
        ":dicom_frame_decoder",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":ez_wsi_logging_factory",
        ":local_dicom_slide_cache",
        ":local_dicom_slide_cache_types",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("pillow"),
        requirement("opencv-python-headless"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        requirement("numpy"),
        requirement("pydicom"),
    ],
)

py_library(
    name = "gcs_image",
    srcs = ["gcs_image.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":error_retry_util",
        ":ez_wsi_errors",
        ":slide_level_map",
        requirement("pillow"),
        requirement("opencv-python-headless"),
        requirement("numpy"),
        requirement("retrying"),
    ],
)

py_library(
    name = "local_image",
    srcs = ["local_image.py"],
    deps = [
        ":credential_factory",
        ":gcs_image",
        requirement("numpy"),
    ],
)

py_library(
    name = "magnification",
    srcs = ["magnification.py"],
)

py_library(
    name = "patch_generator",
    srcs = ["patch_generator.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("pillow"),
        requirement("opencv-python-headless"),
        requirement("numpy"),
    ],
)

filegroup(
    name = "testdata_files",
    srcs = glob(["testdata/**"]),
)

py_test(
    name = "dicom_web_interface_test",
    srcs = ["dicom_web_interface_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_json",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("pydicom"),
        requirement("requests_mock"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_test(
    name = "slide_level_map_test",
    srcs = ["slide_level_map_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":pixel_spacing",
        ":slide_level_map",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("pydicom"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_test(
    name = "dicom_slide_test",
    srcs = ["dicom_slide_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":ez_wsi_logging_factory",
        ":local_dicom_slide_cache",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("pillow"),
        requirement("opencv-python-headless"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
        requirement("pydicom"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_test(
    name = "gcs_image_test",
    srcs = ["gcs_image_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":ez_wsi_errors",
        ":gcs_image",
        ":slide_level_map",
        requirement("pillow"),
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "local_image_test",
    srcs = ["local_image_test.py"],
    deps = [
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        requirement("pillow"),
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
    ],
)

py_test(
    name = "magnification_test",
    srcs = ["magnification_test.py"],
    deps = [":magnification"],
)

py_test(
    name = "patch_generator_test",
    srcs = ["patch_generator_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        ":patch_generator",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("pillow"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
        requirement("pydicom"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_library(
    name = "dicom_frame_decoder",
    srcs = ["dicom_frame_decoder.py"],
    deps = [
        requirement("pillow"),
        requirement("opencv-python-headless"),
        requirement("imagecodecs"),
        requirement("numpy"),
    ],
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
    deps = [
        ":credential_factory",
        ":dicom_store",
        ":ez_wsi_logging_factory",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_test(
    name = "dicom_frame_decoder_test",
    srcs = ["dicom_frame_decoder_test.py"],
    deps = [
        ":dicom_frame_decoder",
        requirement("opencv-python-headless"),
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
    ],
)

py_library(
    name = "dicom_store",
    srcs = ["dicom_store.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":ez_wsi_logging_factory",
        ":local_dicom_slide_cache",
        ":local_dicom_slide_cache_types",
        ":pixel_spacing",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
    ],
)

py_library(
    name = "credential_factory",
    srcs = ["credential_factory.py"],
    deps = [requirement("cachetools")],
)

py_library(
    name = "dicomweb_credential_factory",
    srcs = ["dicomweb_credential_factory.py"],
    deps = [":credential_factory"],
)

py_library(
    name = "pixel_spacing",
    srcs = ["pixel_spacing.py"],
    deps = [
        ":ez_wsi_errors",
        requirement("dataclasses_json"),
    ],
)

py_library(
    name = "pixel_spacing_converter",
    srcs = ["pixel_spacing_converter.py"],
    deps = [
        ":ez_wsi_errors",
        ":magnification",
        ":pixel_spacing",
    ],
)

py_test(
    name = "pixel_spacing_test",
    srcs = ["pixel_spacing_test.py"],
    deps = [
        ":ez_wsi_errors",
        ":pixel_spacing",
    ],
)

py_test(
    name = "pixel_spacing_converter_test",
    srcs = ["pixel_spacing_converter_test.py"],
    deps = [
        ":ez_wsi_errors",
        ":magnification",
        ":pixel_spacing_converter",
    ],
)

py_library(
    name = "local_dicom_slide_cache",
    srcs = ["local_dicom_slide_cache.py"],
    deps = [
        ":credential_factory",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":ez_wsi_logging_factory",
        ":local_dicom_slide_cache_types",
        ":slide_level_map",
        requirement("cachetools"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        requirement("psutil"),
        requirement("pydicom"),
    ],
)

py_test(
    name = "local_dicom_slide_cache_test",
    srcs = ["local_dicom_slide_cache_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_logging_factory",
        ":local_dicom_slide_cache",
        ":local_dicom_slide_cache_types",
        ":slide_level_map",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("pydicom"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_library(
    name = "ez_wsi_logging_factory",
    srcs = ["ez_wsi_logging_factory.py"],
)

py_test(
    name = "ez_wsi_logging_factory_test",
    srcs = ["ez_wsi_logging_factory_test.py"],
    deps = [":ez_wsi_logging_factory"],
)

py_library(
    name = "local_dicom_slide_cache_types",
    srcs = ["local_dicom_slide_cache_types.py"],
    deps = [
        ":slide_level_map",
    ],
)

py_library(
    name = "patch_embedding",
    srcs = ["patch_embedding.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":error_retry_util",
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        ":patch_embedding_endpoints",
        ":patch_embedding_ensemble_methods",
        ":patch_embedding_types",
        ":patch_generator",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("numpy"),
        requirement("retrying"),
    ],
)

py_library(
    name = "patch_embedding_types",
    srcs = ["patch_embedding_types.py"],
    deps = [
        ":dicom_slide",
        ":ez_wsi_errors",
        ":gcs_image",
        ":slide_level_map",
        requirement("numpy"),
    ],
)

py_library(
    name = "error_retry_util",
    srcs = ["error_retry_util.py"],
    deps = [
        ":ez_wsi_errors",
    ],
)

py_library(
    name = "embedding_metrics",
    srcs = ["embedding_metrics.py"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "embedding_metrics_test",
    srcs = ["embedding_metrics_test.py"],
    deps = [":embedding_metrics"],
)

py_library(
    name = "patch_embedding_endpoints",
    srcs = ["patch_embedding_endpoints.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":embedding_metrics",
        ":error_retry_util",
        ":ez_wsi_errors",
        ":gcs_image",
        ":patch_embedding_types",
        ":slide_level_map",
        requirement("pillow"),
        requirement("cachetools"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/ml_toolkit:tags",
        requirement("numpy"),
        requirement("retrying"),
    ],
)

py_library(
    name = "patch_embedding_ensemble_methods",
    srcs = ["patch_embedding_ensemble_methods.py"],
    deps = [
        ":ez_wsi_errors",
        ":patch_embedding_endpoints",
        ":patch_embedding_types",
        requirement("numpy"),
    ],
)

py_test(
    name = "patch_embedding_ensemble_methods_test",
    srcs = ["patch_embedding_ensemble_methods_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":patch_embedding_endpoints",
        ":patch_embedding_ensemble_methods",
        ":patch_embedding_types",
        ":pixel_spacing",
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
        requirement("pydicom"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
    ],
)

py_test(
    name = "patch_embedding_endpoints_test",
    srcs = ["patch_embedding_endpoints_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        ":patch_embedding",
        ":patch_embedding_endpoints",
        ":patch_embedding_types",
        ":pixel_spacing",
        ":slide_level_map",
        requirement("pillow"),
        requirement("cachetools"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        requirement("numpy"),
        requirement("pydicom"),
        requirement("requests_mock"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "patch_embedding_test",
    srcs = ["patch_embedding_test.py"],
    deps = [
        ":credential_factory",
        ":dicom_slide",
        ":dicom_web_interface",
        ":ez_wsi_errors",
        ":gcs_image",
        ":local_image",
        ":patch_embedding",
        ":patch_embedding_endpoints",
        ":patch_embedding_ensemble_methods",
        ":patch_embedding_types",
        ":pixel_spacing",
        requirement("pillow"),
        requirement("opencv-python-headless"),
        "//ez_wsi_dicomweb/ml_toolkit:dicom_path",
        "//ez_wsi_dicomweb/test_utils:dicom_test_utils",
        "//ez_wsi_dicomweb/test_utils:embedding_endpoint_mock",
        requirement("numpy"),
        requirement("pydicom"),
        requirement("requests_mock"),
        "//ez_wsi_dicomweb/test_utils/dicom_store_mock",
        "//ez_wsi_dicomweb/test_utils/gcs_mock",
    ],
)

py_test(
    name = "credential_factory_test",
    srcs = ["credential_factory_test.py"],
    deps = [
        ":credential_factory",
        ":dicomweb_credential_factory",
        requirement("cachetools"),
    ],
)

py_test(
    name = "patch_embedding_types_test",
    srcs = ["patch_embedding_types_test.py"],
    deps = [
        ":ez_wsi_errors",
        ":patch_embedding_types",
        requirement("numpy"),
    ],
)
