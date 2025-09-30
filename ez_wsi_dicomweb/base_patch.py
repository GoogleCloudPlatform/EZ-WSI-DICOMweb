# Copyright 2025 Google LLC
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
"""Base class for patch implementations."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class PatchBounds:
  """A bounding rectangle of a patch, in pixel units."""

  x_origin: int  # The upper leftmost x coordinate of the patch intersection.
  y_origin: int  # The upper leftmost y coordinate of the patch intersection.
  width: int
  height: int


class BasePatch:
  """A rectangular patch/tile/view of an Image at a specific pixel spacing."""

  def __init__(
      self,
      x: int,
      y: int,
      width: int,
      height: int,
  ):
    self._x = x
    self._y = y
    self._width = width
    self._height = height

  @property
  def x(self) -> int:
    return self._x

  @property
  def y(self) -> int:
    return self._y

  @property
  def width(self) -> int:
    return self._width

  @property
  def height(self) -> int:
    return self._height

  @property
  def patch_bounds(self) -> PatchBounds:
    return PatchBounds(
        x_origin=self.x, y_origin=self.y, width=self.width, height=self.height
    )

  def is_patch_fully_in_source_image_dim(self, width: int, height: int) -> bool:
    if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
      return False
    return self.x + self.width <= width and self.y + self.height <= height
