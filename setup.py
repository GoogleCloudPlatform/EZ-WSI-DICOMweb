# !/usr/bin/python
#
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
"""Install script for ez-wsi-dicomweb."""

import setuptools

setuptools.setup(
    name='ez_wsi_dicomweb',
    version='3.1.0',
    url='https://github.com/GoogleCloudPlatform/ez-wsi-dicomweb',
    author='Google LLC.',
    author_email='no-reply@google.com',
    license='Apache 2.0',
    description=(
        'A library that provides the ability to extract an image patch from a'
        ' pathology DICOM whole slide image.'
    ),
    install_requires=[
        'absl-py',
        'cachetools',
        'google-auth',
        'google_auth_httplib2',
        'httplib2',
        'numpy',
        'opencv-python-headless',
        'pillow',
        'psutil',
        'pydicom',
        'requests',
        'requests_mock',
        'requests_toolbelt',
        'hcls-imaging-ml-toolkit-ez-wsi',
    ],
    packages=setuptools.find_packages(),
)
