load("@ez_wsi_pip_deps//:requirements.bzl", "requirement")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:private"],
)

license(
    name = "license",
    package_name = "third_party",
)

licenses(["by_exception_only"])

exports_files(["LICENSE"])

filegroup(
    name = "icc_profiles",
    srcs = glob(["**/*.icc"]),
    visibility = ["//ez_wsi_dicomweb:__subpackages__"],
)
