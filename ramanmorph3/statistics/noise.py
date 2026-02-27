# ramanmorph3/statistics/noise.py
from __future__ import annotations

from typing import List, Sequence, Union, Literal

import numpy as np

from ramanmorph3.logging_utils import get_logger
logger = get_logger(__name__)


def _mask_from_peaks(
		n: int,
		peaks: Sequence,
		*,
		region: Literal["peak", "base"] = "peak",  # "peak" → [left, right], "base" → [left_base, right_base]
) -> np.ndarray:
	"""
	Build a boolean mask of length n marking indices belonging to peak regions.
	"""
	if region not in {"base", "peak"}:
		raise ValueError("region must be 'base' or 'peak'.")

	mask = np.zeros(n, dtype=bool)
	for p in peaks:
		if region == "base":
			a, b = int(p.left_base), int(p.right_base)
		else:
			a, b = int(p.left), int(p.right)

		a = max(0, min(n - 1, a))
		b = max(0, min(n - 1, b))
		if b < a:
			a, b = b, a
		mask[a:b + 1] = True
	return mask


def noise_std_1d(
		y: np.ndarray,
		baseline: np.ndarray,
		peaks: Sequence,
		*,
		region: Literal["peak", "base"] = "peak",
		ddof: int = 1,
		use_residual: Literal["baseline", "raw"] = "baseline",
) -> float:
	"""
	Compute noise standard deviation from regions NOT covered by peaks.

	Note
	----
	Legacy attribute settings are `region`="peak" and use_residual=`baseline`.

	:param y: Raw spectrum (1D).
	:param baseline: Baseline (1D). Used if use_residual="baseline"
	:param peaks: list[Peak]
	:param region: Which peak region to exclude: 'base' excludes [left_base, right_base],
				   'peak' excludes [left,right].
	:param ddof: Passed to np.std.
	:param use_residual:
			  - "baseline": compute std(y - baseline) outside peaks
			  - "raw": compute std(y) outside peaks
	:return: float
	"""
	y = np.asarray(y, dtype=float)
	baseline = np.asarray(baseline, dtype=float)
	if y.ndim != 1 or baseline.ndim != 1 or y.size != baseline.size:
		raise ValueError("noise_std_1d expects y and baseline as 1D arrays of equal length.")

	n = y.size
	mask = _mask_from_peaks(n, peaks, region=region)
	idx = ~mask

	if not np.any(idx):
		return float("nan")

	if use_residual == "baseline":
		vals = (y - baseline)[idx]
	elif use_residual == "raw":
		vals = y[idx]
	else:
		raise ValueError("use_residual must be 'baseline' or 'raw'.")

	vals = vals[np.isfinite(vals)]
	if vals.size < 2:
		return float("nan")

	return float(np.std(vals, ddof=ddof))


def noise_std_nd(
		y: np.ndarray,
		baseline: np.ndarray,
		peaks_nd: Union[np.ndarray, List],
		*,
		axis: int = -1,
		region: Literal["peak", "base"] = "peak",
		ddof: int = 1,
		use_residual: Literal["baseline", "raw"] = "raw",
) -> np.ndarray:
	"""
	ND wrapper: compute noise std per spectrum.

	:param y: (..., n) with spectral axis `axis`.
	:param baseline: (..., n) with spectral axis `axis`.
	:param peaks_nd: object array of shape (...) where each cell is list[Peak]
	:param axis:
	:param region:
	:param ddof:
	:param use_residual:
	:return: np.ndarray of shape (...) with noise std per spectrum.
	"""
	y = np.asarray(y)
	baseline = np.asarray(baseline)
	peaks_arr = np.asarray(peaks_nd, dtype=object)

	y_m = np.moveaxis(y, axis, -1)
	b_m = np.moveaxis(baseline, axis, -1)

	if y_m.shape != b_m.shape:
		raise ValueError("y and baseline must have the same shape.")

	if peaks_arr.ndim == 0:
		return np.asarray(
			noise_std_1d(
				y=y_m.reshape(-1),
				baseline=b_m.reshape(-1),
				peaks=peaks_arr.item(),
				region=region,
				ddof=ddof,
				use_residual=use_residual,
			), dtype=float
		)

	if peaks_arr.shape != y_m.shape[:-1]:
		raise ValueError("peaks_nd shape must match y/baseline spatial shape (all dims except spectral axis).")

	out = np.empty(peaks_arr.shape, dtype=float)
	for idx in np.ndindex(peaks_arr.shape):
		out[idx] = noise_std_1d(
			y=y_m[idx],
			baseline=b_m[idx],
			peaks=peaks_arr[idx],
			region=region,
			ddof=ddof,
			use_residual=use_residual,
		)

	return out
