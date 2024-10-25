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
    version='6.0.7',
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
        'dataclasses-json',
        'google-auth',
        'google_auth_httplib2',
        'google_cloud_storage',
        'numpy',
        'opencv-python-headless',
        'pillow',
        'psutil',
        'pydicom',
        'requests',
        'requests_mock',
        'requests_toolbelt',
        'retrying',
    ],
    package_dir={
        'ez_wsi_dicomweb': 'ez_wsi_dicomweb',
        'ez_wsi_dicomweb.ml_toolkit': 'ez_wsi_dicomweb/ml_toolkit',
        'ez_wsi_dicomweb.test_utils': 'ez_wsi_dicomweb/test_utils',
        'ez_wsi_dicomweb.test_utils.dicom_store_mock': (
            'ez_wsi_dicomweb/test_utils/dicom_store_mock'
        ),
        'ez_wsi_dicomweb.test_utils.dicom_store_mock.testdata': (
            'ez_wsi_dicomweb/test_utils/dicom_store_mock/testdata'
        ),
        'ez_wsi_dicomweb.test_utils.gcs_mock.': (
            'ez_wsi_dicomweb/test_utils/gcs_mock'
        ),
        'ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib': (
            'ez_wsi_dicomweb/test_utils/gcs_mock/gcs_mock_lib'
        ),
        'third_party': 'third_party',
    },
    package_data={
        'ez_wsi_dicomweb': ['*.md'],
        'third_party': ['*.icc', 'LICENSE'],
        'ez_wsi_dicomweb.test_utils.dicom_store_mock.testdata': ['*.dcm'],
    },
    packages=setuptools.find_packages(
        include=[
            'ez_wsi_dicomweb',
            'ez_wsi_dicomweb.ml_toolkit',
            'ez_wsi_dicomweb.test_utils',
            'ez_wsi_dicomweb.test_utils.dicom_store_mock',
            'ez_wsi_dicomweb.test_utils.dicom_store_mock.testdata',
            'ez_wsi_dicomweb.test_utils.gcs_mock',
            'ez_wsi_dicomweb.test_utils.gcs_mock.gcs_mock_lib',
            'third_party',
        ]
    ),
    python_requires='>=3.10',
)
