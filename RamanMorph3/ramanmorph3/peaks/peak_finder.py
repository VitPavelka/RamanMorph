# ramanmorph3/peaks/peak_finder.py
"""
Peak detection and line (peakline/baseline) derivation based on morphological envelopes.

Workflow
--------
1) Peaks are defined first from dilation/erosion (tips + tails + bases).
2) Then peakline and baseline are derived from peak boundaries.
3) Constrained linear smoothing is applied to ensure lines never exceed the spectrum.
4) Supports ND stacks (time series, maps, volumes) along an arbitrary spectral axis.
5) Produce Peak objects with boundaries + metrics (except AUC-angle / ROC, which are left for later).

Expected inputs
---------------
- x: (n,) spectral axis
- y: (..., n) signal(s)
- y_dilated: same shape as y (upper envelope)
- y_eroded_peak: same shape as y (narrow erosion / "tail" contacts)
- y_eroded_base: same shape as y (wide erosion / "base" contacts)

If you already have an envelopes helper, you can compute these upstream
and call find_peaks_* function.

Notes
-----
- This module uses "contact points" defined by y == envelope (with tolerance).
- Multi-peaks splitting is done by inserting a valley anchor between adjacent tips sharing the same
tail interval (mimic the legacy logic from version 1.0).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ramanmorph3.logging_utils import get_logger
from ramanmorph3.morphology.interpolation import auto_atol, contact_indices, derive_peakline_and_baseline_1d, correct_peak_boundaries_1d
from .identification import Peak, characterize_peaks_1d
from .metrics import compute_peak_metrics_1d

logger = get_logger(__name__)


# --- Public API ---
def find_peaks_1d(
		x: np.ndarray,
		y: np.ndarray,
		y_dilated: np.ndarray,
		y_eroded_peak: np.ndarray,
		y_eroded_base: np.ndarray,
		*,
		refine_lines: bool = True,
		iteration_max: int = 20,
		atol: Optional[float] = None,
		correct_boundaries: bool = True,
		compute_metrics: bool = True,
) -> Tuple[List[Peak], np.ndarray, np.ndarray]:
	"""
	Full 1D pipeline:
	  1) detect candidates (tips/tails/bases) from contact with envelopes
	  2) characterize peaks from candidates (incl. multi-peak splitting)
	  3) derive peakline + baseline from peak boundaries
	  4) optional boundary correction after refinement
	  5) optional metric computation

	:param x:
	:param y:
	:param y_dilated:
	:param y_eroded_peak:
	:param y_eroded_base:
	:param refine_lines:
	:param iteration_max:
	:param atol:
	:param correct_boundaries:
	:param compute_metrics:
	:return: (peaks, peakline, baseline)
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	y_dilated = np.asarray(y_dilated)
	y_eroded_peak = np.asarray(y_eroded_peak)
	y_eroded_base = np.asarray(y_eroded_base)

	if x.ndim != 1 or y.ndim != 1:
		raise ValueError(f"find_peaks_1d expects 1D x and 1D y: "
		                 f"Got {x.ndim}D x and {y.ndim}D y")
	if not (x.size == y.size == y_dilated.size == y_eroded_peak.size == y_eroded_base.size):
		raise ValueError("All 1D inputs must have the same length.")

	cmp_atol = auto_atol(y, atol)

	tips = contact_indices(y, y_dilated, atol=cmp_atol)
	tails = contact_indices(y, y_eroded_peak, atol=cmp_atol)
	bases = contact_indices(y, y_eroded_base, atol=cmp_atol)

	use_wide = not np.array_equal(y_eroded_peak, y_eroded_base)

	from time import time
	start = time()
	peaks = characterize_peaks_1d(
		y=y,
		y_eroded_peak=y_eroded_peak,
		tips=tips,
		tails=tails,
		bases=bases,
		use_wide_bases=use_wide,
	)
	pe = time()
	# print(f"\npeaks time {pe - start}")

	peakline, baseline = derive_peakline_and_baseline_1d(
		x=x,
		y=y,
		y_eroded_peak=y_eroded_peak,
		y_eroded_base=y_eroded_base,
		peaks=peaks,
		refine=refine_lines,
		iteration_max=iteration_max,
		atol=atol
	)
	li = time()
	# print(f"lines time {li - pe}")

	if correct_boundaries and peaks:
		correct_peak_boundaries_1d(peaks, y, peakline, baseline, atol=atol)
	cb = time()
	# print(f"bound {cb - li}")

	if compute_metrics and peaks:
		compute_peak_metrics_1d(peaks, x, y, peakline, baseline)

	cm = time()
	# print(f"metrics {cm - cb}")

	return peaks, peakline, baseline


# --- Public API ND ---
def find_peaks_nd(
		x: np.ndarray,
		y: np.ndarray,
		y_dilated: np.ndarray,
		y_eroded_peak: np.ndarray,
		y_eroded_base: np.ndarray,
		*,
		axis: int = -1,
		refine_lines: bool = True,
		iteration_max: int = 20,
		atol: Optional[float] = None,
		correct_boundaries: bool = True,
		compute_metrics: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	ND wrapper over find_peaks_1d.

	:param x:
	:param y:
	:param y_dilated:
	:param y_eroded_peak:
	:param y_eroded_base:
	:param axis:
	:param refine_lines:
	:param iteration_max:
	:param atol:
	:param correct_boundaries:
	:param compute_metrics:
	:return:
		- peaks_grid: np.ndarray(dtype=object) of shape y.shape without spectral axis,
		  each element is List[Peak]
		- peakline: np.ndarray same shape as y
		- baseline: np.ndarray same shape as y
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	y_dilated = np.asarray(y_dilated)
	y_eroded_peak = np.asarray(y_eroded_peak)
	y_eroded_base = np.asarray(y_eroded_base)

	if x.ndim != 1:
		raise ValueError(f"x must be 1D; Got {x.ndim}D x")
	if y.shape != y_dilated.shape or y.shape != y_eroded_peak.shape or y.shape != y_eroded_base.shape:
		raise ValueError("y, y_dilated, y_eroded_peak, and y_eroded_base must have identical shapes.")
	if y.shape[axis] != x.size:
		raise ValueError("Spectral axis length of y must match len(x).")

	# Move spectral axis to last
	y_m = np.moveaxis(y, axis, -1)
	yd_m = np.moveaxis(y_dilated, axis, -1)
	yep_m = np.moveaxis(y_eroded_peak, axis, -1)
	yeb_m = np.moveaxis(y_eroded_base, axis, -1)

	spatial_shape = y_m.shape[:-1]
	n_spec = y_m.shape[-1]

	# Flatten spatial dims
	Y = y_m.reshape(-1, n_spec)
	YD = yd_m.reshape(-1, n_spec)
	YEP = yep_m.reshape(-1, n_spec)
	YEB = yeb_m.reshape(-1, n_spec)

	peakline_out = np.empty_like(Y, dtype=float)
	baseline_out = np.empty_like(Y, dtype=float)
	peaks_grid = np.empty((Y.shape[0],), dtype=object)

	for i in tqdm(range(Y.shape[0])):
		peaks_i, pl_i, bl_i = find_peaks_1d(
			x=x,
			y=Y[i],
			y_dilated=YD[i],
			y_eroded_peak=YEP[i],
			y_eroded_base=YEB[i],
			refine_lines=refine_lines,
			iteration_max=iteration_max,
			atol=atol,
			correct_boundaries=correct_boundaries,
			compute_metrics=compute_metrics,
		)
		peaks_grid[i] = peaks_i
		peakline_out[i] = pl_i
		baseline_out[i] = bl_i

	# Reshape back
	peaks_grid = peaks_grid.reshape(spatial_shape)
	peakline_out = peakline_out.reshape((*spatial_shape, n_spec))
	baseline_out = baseline_out.reshape((*spatial_shape, n_spec))

	# Move spectral axis back
	peakline_out = np.moveaxis(peakline_out, -1, axis)
	baseline_out = np.moveaxis(baseline_out, -1, axis)

	if isinstance(peaks_grid, np.ndarray) and peaks_grid.ndim == 0:
		peaks_grid = peaks_grid.item()

	return peaks_grid, peakline_out, baseline_out
