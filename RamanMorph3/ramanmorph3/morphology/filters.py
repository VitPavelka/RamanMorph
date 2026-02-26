# ramanmorph3/morphology/filters.py
from __future__ import annotations

from typing import Literal, Callable, Tuple
from functools import lru_cache
import numpy as np

from ramanmorph3.logging_utils import get_logger
logger = get_logger(__name__)

BoundaryMode = Literal["reflect", "mirror", "nearest", "wrap", "constant"]


@lru_cache(maxsize=1)
def _require_scipy() -> Tuple[Callable, Callable]:
	"""
	Import SciPy ndimage filters once (cached).

	:raises RuntimeError: If SciPy is not installed.
	:return:
	"""
	try:
		from scipy.ndimage import maximum_filter1d, minimum_filter1d  # type: ignore
	except Exception as e:
		raise RuntimeError(
			"SciPy is required for morphology filters.\n"
			"Install it e.g.:\n"
			"  pip install scipy\n"
		) from e
	
	return maximum_filter1d, minimum_filter1d


def _validate_half_window(half_window: int) -> int:
	"""Validate `half_window` and return as int."""
	hw = int(half_window)
	if hw < 0:
		raise ValueError(f"half_window must be >= 0, got {half_window}")
	return hw


def _size_from_half_window(half_window: int) -> int:
	"""Convert `half_window` to an odd full window size: size = 1 + 2*half_window."""
	hw = _validate_half_window(half_window)
	return 1 + 2 * hw


def dilate_1d(
		y: np.ndarray,
		half_window: int,
		*,
		axis: int = -1,
		mode: BoundaryMode = "reflect",
		cval: float = 0.0,
) -> np.ndarray:
	"""
	1D morphological dilation (max filter) with symmetric structuring element.

	:param y: Input array. Works for 1D spectra or N-D stacks; filtering is applied along 'axis'.
	:param half_window: Half-width of the symmetric window. Full size = 1 + 2*half_window.
	:param axis: Axis along which to apply the filter (default: last axis).
	:param mode: Boundary handling passed to SciPy.
	:param cval: Used only when mode="constant".
	:return: Dilated array of the same shape as `y`.
	"""
	maximum_filter1d, _ = _require_scipy()
	size = _size_from_half_window(half_window)
	return maximum_filter1d(y, size=size, axis=axis, mode=mode, cval=cval)


def erode_1d(
		y: np.ndarray,
		half_window: int,
		*,
		axis: int = -1,
		mode: BoundaryMode = "reflect",
		cval: float = 0.0,
) -> np.ndarray:
	"""
	1D morphological erosion (min filter) with symmetric structuring element.

	:param y: Input array. Works for 1D spectra or N-D stacks; filtering is applied along 'axis'.
	:param half_window: Half-width of the symmetric window. Full size = 1 + 2*half_window.
	:param axis: Axis along which to apply the filter (default: last axis).
	:param mode: Boundary handling passed to SciPy.
	:param cval: Used only when mode="constant".
	:return: Eroded array of the same shape as `y`.
	"""
	_, minimum_filter1d = _require_scipy()
	size = _size_from_half_window(half_window)
	return minimum_filter1d(y, size=size, axis=axis, mode=mode, cval=cval)


def contact_mask(
		y: np.ndarray,
		y_filtered: np.ndarray,
		*,
		atol: float = 0.0,
		rtol: float = 0.0,
) -> np.ndarray:
	"""
	Compute a boolean mask of "contact points" between the original signal and its
	morphological envelope.

	Here the contact points are defined as indices where y == y_filtered (within tolerance),
	i.e., the sample itself reaches the max/min of its neighborhood.

	:param y: Original signal(s).
	:param y_filtered: Output of `dilate_1d` or `erode_1d`, same shape as `y`.
	:param atol: Tolerance for float comparisons.
	:param rtol: Tolerance for float comparisons.
	:return: Boolean mask with the same shape as `y`. For a single spectrum you can do `np.flatnonzero(mask)`.
	"""
	if y.shape != y_filtered.shape:
		raise ValueError(f"y and a_filtered must have same shape, got {y.shape} vs {y_filtered.shape}.")
	return np.isclose(y, y_filtered, atol=atol, rtol=rtol)
