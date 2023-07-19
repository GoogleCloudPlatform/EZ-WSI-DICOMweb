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
"""Abstract and reference python logging interfaces for local DICOM slide cache."""
from __future__ import annotations

import abc
import collections
import copy
import logging
from typing import Any, Mapping, MutableMapping, Optional, Union


OptionalStructureElements = Union[Exception, Mapping[str, Any], None]
DEFAULT_EZ_WSI_PYTHON_LOGGER_NAME = 'ez-wsi-DICOMweb'


class AbstractLoggingInterface(metaclass=abc.ABCMeta):
  """Logging interface for local DICOM slide cache."""

  @abc.abstractmethod
  def debug(self, msg: str, *args: OptionalStructureElements) -> None:
    """Logs debug message.

    Args:
      msg: Message to log.
      *args: Optional arguments to log as structured logs or additional msg.

    Returns:
      None
    """

  @abc.abstractmethod
  def info(self, msg: str, *args: OptionalStructureElements) -> None:
    """Logs info message.

    Args:
      msg: Message to log.
      *args: Optional arguments to log as structured logs or additional msg.

    Returns:
      None
    """

  @abc.abstractmethod
  def warning(self, msg: str, *args: OptionalStructureElements) -> None:
    """Logs warning message.

    Args:
      msg: Message to log.
      *args: Optional arguments to log as structured logs or additional msg.

    Returns:
      None
    """

  @abc.abstractmethod
  def error(self, msg: str, *args: OptionalStructureElements) -> None:
    """Logs error message.

    Args:
      msg: Message to log.
      *args: Optional arguments to log as structured logs or additional msg.

    Returns:
      None
    """

  @abc.abstractmethod
  def critical(self, msg: str, *args: OptionalStructureElements) -> None:
    """Logs critical message.

    Args:
      msg: Message to log.
      *args: Optional arguments to log as structured logs or additional msg.

    Returns:
      None
    """


class AbstractLoggingInterfaceFactory(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def create_logger(
      self, signature: Optional[Mapping[str, Any]] = None
  ) -> AbstractLoggingInterface:
    """Creates an instance of the logger.

    Args:
      signature: Optional signature element to include as structure or message
        elements in all logs.
    """


def _merge_dict_sorted_by_key(
    merge_dict: MutableMapping[str, Any], element: Mapping[str, Any]
) -> None:
  for key in sorted(element):
    merge_dict[key] = element[key]


class _BasePythonLogger(AbstractLoggingInterface):
  """Reference implementation for python logging interface."""

  def __init__(
      self,
      pylogger: logging.Logger,
      signature: Optional[Mapping[str, Any]] = None,
  ):
    self._logger = pylogger
    self._signature = {} if signature is None else copy.copy(signature)

  def _build_msg(self, msg: str, *args: OptionalStructureElements) -> str:
    """Builds log msg from msg and structure elements."""
    optional_args = collections.OrderedDict()
    found_exception = None
    for element in args:
      if element is None or not element:
        continue
      if isinstance(element, Mapping):
        _merge_dict_sorted_by_key(optional_args, element)
      elif isinstance(element, Exception):
        found_exception = element
    if self._signature:
      _merge_dict_sorted_by_key(optional_args, self._signature)
    if found_exception is not None:
      optional_args['EXCEPTION'] = found_exception
    structure = '; '.join(
        [f'{key}: {value}' for key, value in optional_args.items()]
    )
    if structure:
      return f'{msg}; {structure}'
    return msg

  def debug(self, msg: str, *args: OptionalStructureElements) -> None:
    self._logger.debug(self._build_msg(msg, *args))

  def info(self, msg: str, *args: OptionalStructureElements) -> None:
    self._logger.info(self._build_msg(msg, *args))

  def warning(self, msg: str, *args: OptionalStructureElements) -> None:
    self._logger.warning(self._build_msg(msg, *args))

  def error(self, msg: str, *args: OptionalStructureElements) -> None:
    self._logger.error(self._build_msg(msg, *args))

  def critical(self, msg: str, *args: OptionalStructureElements) -> None:
    self._logger.critical(self._build_msg(msg, *args))


class BasePythonLoggerFactory(AbstractLoggingInterfaceFactory):
  """Factory class to contruct Python logger for DICOM Slide Cache."""

  def __init__(
      self,
      name: Optional[str] = None,
  ):
    self._name = name

  def create_logger(
      self, signature: Optional[Mapping[str, Any]] = None
  ) -> _BasePythonLogger:
    return _BasePythonLogger(logging.getLogger(self._name), signature)
