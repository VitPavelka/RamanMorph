# ramanmorph3/statistics/cumulative.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ramanmorph3.logging_utils import get_logger
logger = get_logger(__name__)


SUPPORTED_METRICS = {
	"height", "height_base", "height_abs",
	"area", "area_base", "area_abs"
}


def peak_metric_value(peak, metric: str) -> float:
	"""
	Read a numeric metric from a Peak-like object.

	:param peak: Peak instance (must have attribute `metric`).
	:param metric: One of SUPPORTED_METRICS ("height", "height_base", "area", "area_base", "area_abs").
	:return: The metric value.
	"""
	if metric not in SUPPORTED_METRICS:
		raise ValueError(f"Unsupported metric '{metric}'. Supported: {sorted(SUPPORTED_METRICS)}")

	v = float(getattr(peak, metric))
	return abs(v)


@dataclass(frozen=True)
class CumulativeCurve:
	"""
	Rank-ordered cumulative contribution curve.

	x: fraction of included peaks in [0,1], length n
	y: cumulative sum in [0,1], length n+1 (starts at 0, ends at 1 if total>0)
	auc: area under y(x) in [0,1]
	order: indices of peaks sorted by metric desc (len n)
	portions: per-peak normalized contributions (sorted order, len n)
	"""
	x: np.ndarray
	y: np.ndarray
	auc: float
	order: np.ndarray
	portions: np.ndarray


def cumulative_curve_from_peaks(
		peaks: Sequence,
		*,
		metric: str = "area"
) -> CumulativeCurve:
	"""
	Build normalized cumulative curve from peaks sorted by a chosen metric.

	Notes
	-----
	  - If `peaks` is empty or total metric is 0, returns a degenerate curve with auc=0.
	  - Uses trapezoidal integration for AUC.

	:param peaks:
	:param metric:
	:return: CumulativeCurve instance.
	"""
	n = len(peaks)
	if n == 0:
		x = np.array([0.0], dtype=float)
		y = np.array([0.0], dtype=float)
		return CumulativeCurve(
			x=x, y=y,
			auc=0.0,
			order=np.array([], dtype=int),
			portions=np.array([], dtype=float)
		)

	values = np.array([peak_metric_value(p, metric) for p in peaks], dtype=float)
	order = np.argsort(values)[::-1]  # desc
	values_sorted = values[order]

	total = float(np.nansum(values_sorted))
	if not np.isfinite(total) or total <= 0.0:
		x = np.linspace(0.0, 1.0, n + 1, dtype=float)
		y = np.zeros(n + 1, dtype=float)
		return CumulativeCurve(
			x=x, y=y,
			auc=0.0,
			order=order.astype(int),
			portions=np.zeros(n, dtype=float)
		)

	portions = values_sorted / total
	# legacy peak_order: 0...1 with step 1/(n-1)
	if n == 1:
		x = np.array([0.0], dtype=float)
	else:
		x = (np.arange(n, dtype=float) / float(n - 1))

	# peak_normal_areas: cumulative normalized sum (no initial 0-point)
	y = np.cumsum(portions)  # length n, starts at first peak portion

	# legacy AUC: mean of the cumulative curve
	auc = float(np.mean(y))  # legacy AUC

	return CumulativeCurve(
		x=x, y=y,
		auc=auc,
		order=order.astype(int),
		portions=portions.astype(float)
	)


# def finite_difference_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
# 	"""
# 	Numerical derivative dy/dx using numpy.gradient.
#
# 	:param x:
# 	:param y:
# 	:return:
# 	"""
# 	x = np.asarray(x, dtype=float)
# 	y = np.asarray(y, dtype=float)
# 	if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
# 		raise ValueError("finite_difference_derivative expects 1D arrays of equal length.")
#
# 	if x.size < 2:
# 		return np.zeros_like(y)
#
# 	return np.gradient(y, x)


# def tangent_angles_deg(derivative: np.ndarray) -> np.ndarray:
# 	"""
# 	Convert slope dy/dx into tangent angle in degrees: arctan(slope).
#
# 	:param derivative:
# 	:return:
# 	"""
# 	derivative = np.asarray(derivative, dtype=float)
# 	return np.degrees(np.arctan(derivative))
