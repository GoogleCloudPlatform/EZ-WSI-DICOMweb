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
"""Tests for slide cache logger base python implementation."""
import logging
from typing import Any, Mapping, Optional
from unittest import mock

from absl.testing import absltest
from ez_wsi_dicomweb import ez_wsi_logging_factory


def _create_logger(
    name: Optional[str] = None, signature: Optional[Mapping[str, Any]] = None
) -> ez_wsi_logging_factory._BasePythonLogger:
  return ez_wsi_logging_factory.BasePythonLoggerFactory(
      name
  ).create_logger(signature)


class LocalDicomSlideCacheLoggerTest(absltest.TestCase):

  @mock.patch.object(logging.Logger, 'debug', autospec=True)
  def test_get_logger_debug(self, mock_logger):
    _create_logger(name='named_logger').debug('test')
    mock_logger.assert_called_once_with(
        logging.getLogger('named_logger'), 'test'
    )

  @mock.patch.object(logging.Logger, 'warning', autospec=True)
  def test_get_logger_warning(self, mock_logger):
    _create_logger().warning('test')
    mock_logger.assert_called_once_with(logging.getLogger(''), 'test')

  @mock.patch.object(logging.Logger, 'info', autospec=True)
  def test_get_logger_info(self, mock_logger):
    _create_logger().info('test')
    mock_logger.assert_called_once_with(logging.getLogger(''), 'test')

  @mock.patch.object(logging.Logger, 'error', autospec=True)
  def test_get_logger_error(self, mock_logger):
    _create_logger().error('test')
    mock_logger.assert_called_once_with(logging.getLogger(), 'test')

  @mock.patch.object(logging.Logger, 'critical', autospec=True)
  def test_get_logger_critical(self, mock_logger):
    _create_logger().critical('test')
    mock_logger.assert_called_once_with(logging.getLogger(), 'test')

  @mock.patch.object(logging.Logger, 'debug', autospec=True)
  def test_get_logger_debug_complex_structure(self, mock_logger):
    _create_logger(None, {'create': 'create_val'}).debug(
        'test', {'a': 1}, ValueError('Bad_Value'), None, {}
    )
    mock_logger.assert_called_once_with(
        logging.getLogger(),
        'test; a: 1; create: create_val; EXCEPTION: Bad_Value',
    )


if __name__ == '__main__':
  absltest.main()
