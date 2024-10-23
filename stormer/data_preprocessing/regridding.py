# Adopted from WeatherBench 2 at https://github.com/google-research/weatherbench2/blob/main/weatherbench2/regridding.py
"""Routines for horizontal regridding.

This module supports three types of regridding:
- Nearest neighbor: suitable for interpolating non-continuous fields (e.g.,
  categrorical land-surface type).
- Bilinear interpolation: most suitable for regridding to finer grids.
- Linear conservative regridding: most suitable for regridding to coarser grids.

Only rectalinear grids (one dimensional lat/lon coordinates) are supported, but
irregular spacing is OK.

Conservative regridding schemes are adapted from:
https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import xarray

Array = Union[np.ndarray, jax.Array]


@dataclasses.dataclass(frozen=True)
class Grid:
  """Representation of a rectalinear grid."""

  lon: np.ndarray
  lat: np.ndarray

  @classmethod
  def from_degrees(cls, lon: np.ndarray, lat: np.ndarray) -> Grid:
    return cls(np.deg2rad(lon), np.deg2rad(lat))

  @property
  def shape(self) -> tuple[int, int]:
    return (len(self.lon), len(self.lat))

  def _to_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(self.lon.tolist()), tuple(self.lat.tolist())

  def __eq__(self, other):  # needed for hashability
    return isinstance(other, Grid) and self._to_tuple() == other._to_tuple()

  def __hash__(self):
    return hash(self._to_tuple())


@dataclasses.dataclass(frozen=True)
class Regridder:
  """Base class for regridding."""

  source: Grid
  target: Grid

  def regrid_array(self, field: Array) -> jax.Array:
    """Regrid an array with dimensions (..., lon, lat) from source to target."""
    raise NotImplementedError

  def regrid_dataset(self, dataset: xarray.Dataset) -> xarray.Dataset:
    """Regrid an xarray.Dataset from source to target."""
    if not (dataset['latitude'].diff('latitude') > 0).all():
      # ensure latitude is increasing
      dataset = dataset.isel(latitude=slice(None, None, -1))  # reverse
    assert (dataset['latitude'].diff('latitude') > 0).all()
    dataset = xarray.apply_ufunc(
        self.regrid_array,
        dataset,
        dask='parallelized',
        input_core_dims=[['longitude', 'latitude']],
        output_core_dims=[['longitude', 'latitude']],
        exclude_dims={'longitude', 'latitude'},
        dask_gufunc_kwargs={
          'output_sizes': {
            'longitude': self.target.lon.shape[0],
            'latitude': self.target.lat.shape[0],
          }
        },
        output_dtypes=[np.float32],
    )
    return dataset


def _assert_increasing(x: np.ndarray) -> None:
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _latitude_cell_bounds(x: Array) -> jax.Array:
  pi_over_2 = jnp.array([np.pi / 2], dtype=x.dtype)
  return jnp.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _latitude_overlap(
    source_points: Array,
    target_points: Array,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  source_bounds = _latitude_cell_bounds(source_points)
  target_bounds = _latitude_cell_bounds(target_points)
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  # normalized cell area: integral from lower to upper of cos(latitude)
  return (upper > lower) * (jnp.sin(upper) - jnp.sin(lower))


def _conservative_latitude_weights(
    source_points: Array, target_points: Array
) -> jax.Array:
  """Create a weight matrix for conservative regridding along latitude.

  Args:
    source_points: 1D latitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D latitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (target, source). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _latitude_overlap(source_points, target_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


def _align_phase_with(x, target, period):
  """Align the phase of a periodic number to match another.

  The returned number is equivalent to the original (modulo the period) with
  the smallest distance from the target, among the values
  `{x - period, x, x + period}`.

  Args:
    x: number to adjust.
    target: number with phase to match.
    period: periodicity.

  Returns:
    x possibly shifted up or down by `period`.
  """
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  x_plus = _align_phase_with(jnp.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  x_minus = _align_phase_with(jnp.roll(x, +1), x, period)
  return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
  # valid as long as no intervals are larger than period/2
  y0 = _align_phase_with(y0, x0, period)
  y1 = _align_phase_with(y1, x0, period)
  upper = jnp.minimum(x1, y1)
  lower = jnp.maximum(x0, y0)
  return jnp.maximum(upper - lower, 0)


def _longitude_overlap(
    first_points: Array,
    second_points: Array,
    period: float = 2 * np.pi,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  first_points = first_points % period
  first_upper = _periodic_upper_bounds(first_points, period)
  first_lower = _periodic_lower_bounds(first_points, period)

  second_points = second_points % period
  second_upper = _periodic_upper_bounds(second_points, period)
  second_lower = _periodic_lower_bounds(second_points, period)

  return jnp.vectorize(functools.partial(_periodic_overlap, period=period))(
      first_lower[:, jnp.newaxis],
      first_upper[:, jnp.newaxis],
      second_lower[jnp.newaxis, :],
      second_upper[jnp.newaxis, :],
  )


def _conservative_longitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> jax.Array:
  """Create a weight matrix for conservative regridding along longitude.

  Args:
    source_points: 1D longitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D longitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _longitude_overlap(target_points, source_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


class ConservativeRegridder(Regridder):
  """Regrid with linear conservative regridding."""

  @functools.partial(jax.jit, static_argnums=0)
  def _mean(self, field: Array) -> jax.Array:
    """Computes cell-averages of field on the target grid."""
    lon_weights = _conservative_longitude_weights(
        self.source.lon, self.target.lon
    )
    lat_weights = _conservative_latitude_weights(
        self.source.lat, self.target.lat
    )
    return jnp.einsum(
        'ab,cd,...bd->...ac',
        lon_weights,
        lat_weights,
        field,
        precision='highest',
    )

  @functools.partial(jax.jit, static_argnums=0)
  def _nanmean(self, field: Array) -> jax.Array:
    """Compute cell-averages skipping NaNs like np.nanmean."""
    nulls = jnp.isnan(field)
    total = self._mean(jnp.where(nulls, 0, field))
    count = self._mean(jnp.logical_not(nulls))
    return total / count  # intentionally NaN if count == 0

  regrid_array = _nanmean
