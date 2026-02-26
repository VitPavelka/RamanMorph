# ramanmorph3/peaks/metrics.py
from __future__ import annotations

from typing import Sequence

import numpy as np
from .identification import Peak


# --- Helpers ---
def _trapz(x: np.ndarray, y: np.ndarray) -> float:
	"""Tiny wrapper to avoid dtype surprises."""
	return float(np.trapezoid(y.astype(float, copy=False), x.astype(float, copy=False)))


def _interp_crossing_x(x0: float, y0: float, x1: float, y1: float, y_target: float) -> float:
	"""Linear interpolation for x where y crosses y_target between (x0,y0) and (x1,y1)."""
	if y1 == y0:
		return float(x0)
	t = (y_target - y0) / (y1 - y0)
	return float(x0 + t * (x1 - x0))


def compute_peak_metrics_1d(
		peaks: Sequence[Peak],
		x: np.ndarray,
		y: np.ndarray,
		peakline: np.ndarray,
		baseline: np.ndarray
) -> None:
	"""
	Fill Peak metrics in-place.

	Areas are integrated between [left, right] (NOT using left_base/right_base for integration limits):
		- area: above peakline
		- area_base: above baseline (but within left...right)
		- area_abs: above 0 within left...right

	:param peaks:
	:param x:
	:param y:
	:param peakline:
	:param baseline:
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	peakline = np.asarray(peakline)
	baseline = np.asarray(baseline)

	for p in peaks:
		left, right, apex = int(p.left), int(p.right), int(p.apex)
		if right <= left or apex <= left or apex >= right:
			continue

		p.height_abs = float(y[apex])
		p.height = float(y[apex] - peakline[apex])
		p.height_base = float(y[apex] - baseline[apex])

		# Areas over [left, right]
		xx = x[left:right + 1]
		yy = y[left:right + 1]

		p.area_abs = abs(_trapz(xx, yy))
		p.area = abs(_trapz(xx, (yy - peakline[left:right + 1])))
		p.area_base = abs(_trapz(xx, (yy - baseline[left:right + 1])))

		# FWHM above peakline
		half = float(peakline[apex] + p.height / 2.0)

		# Find left crossing
		left_cross = None
		for i in range(apex, left, -1):
			if (y[i] >= half > y[i - 1]) or (y[i] <= half < y[i - 1]):
				left_cross = _interp_crossing_x(float(x[i - 1]), float(y[i - 1]),
				                                float(x[i]), float(y[i]), half)
				break

		# Find right crossing
		right_cross = None
		for i in range(apex, right):
			if (y[i] >= half > y[i + 1]) or (y[i] <= half < y[i + 1]):
				right_cross = _interp_crossing_x(float(x[i]), float(y[i]),
				                                 float(x[i + 1]), float(y[i + 1]), half)
				break

		if left_cross is not None and right_cross is not None:
			p.fwhm = abs(float(right_cross - left_cross))
		else:
			p.fwhm = float("nan")
