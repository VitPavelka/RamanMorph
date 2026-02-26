# ramanmorph3/statistics/scoring.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate

from ramanmorph3.logging_utils import get_logger
from .cumulative import cumulative_curve_from_peaks

logger = get_logger(__name__)


def _set_peak_fields(peak, *, auc_angle: Optional[float], roc: Optional[float]):
	"""
	Update Peak.

	:param peak:
	:param auc_angle:
	:param roc:
	:return:
	"""
	if auc_angle is not None:
		setattr(peak, "auc_angle", auc_angle)
	if roc is not None:
		setattr(peak, "roc", roc)
	return peak


# def roc_from_slope(
# 		slope: float,
# 		*,
# 		ok_tangent_point: float,
# 		limit_tangent_point: float,
# ) -> float:
# 	"""
# 	Map slope (dy/dx) to a [0,1] "probability-like" ROC score.
#
# 	- slope <= ok_tangent_point     → 0.0
# 	- slope >= limit_tangent_point  → 1.0
# 	- linear ramp in-between
#
# 	:param slope:
# 	:param ok_tangent_point:
# 	:param limit_tangent_point:
# 	:return:
# 	"""
# 	if not np.isfinite(slope):
# 		return 0.0
#
# 	if limit_tangent_point <= ok_tangent_point:
# 		raise ValueError(
# 			f"limit_tangent_point ({limit_tangent_point}) must be "
# 			f"> ok_tangent_point ({ok_tangent_point}). "
# 		)
#
# 	if slope <= ok_tangent_point:
# 		return 0.0
# 	if slope >= limit_tangent_point:
# 		return 1.0
# 	return float((slope - ok_tangent_point) / limit_tangent_point - ok_tangent_point)


def annotate_peaks_with_auc_and_roc(
		peak: Sequence,
		*,
		metric: str = "area",
		use_abs: bool = True,
		ok_tangent_point: float = 30,
		limit_tangent_point: float = 45,
) -> Tuple[List, Dict[str, Any]]:
	"""


	auc_angle:

	roc:


	:param peak:
	:param metric:
	:param use_abs:
	:param ok_tangent_point:
	:param limit_tangent_point:
	:return:
	"""
	peaks = list(peak)
	curve = cumulative_curve_from_peaks(peaks, metric=metric)

	n = len(peaks)
	if n < 2:
		der = np.array([0.0], dtype=float)
	else:
		k = min(3, n - 1)
		tck = interpolate.splrep(curve.x, curve.y, s=0, k=k)
		der = np.asarray(interpolate.splev(curve.x, interpolate.splder(tck)), dtype=float)

	# tangent angle is degrees
	angles_deg = np.degrees(np.arctan(der))

	updated = list(peaks)
	classes = {"ok": [], "maybe": [], "no": []}

	# curve.order maps sorted rank → original peak index
	for rank_pos, peak_idx in enumerate(curve.order):
		slope = float(der[rank_pos])
		angle = float(angles_deg[rank_pos])
		cum_y = float(curve.y[rank_pos])

		# classification: keep your current thresholds (slopes)
		if np.isfinite(slope) and slope >= limit_tangent_point:
			classes["ok"].append(int(peak_idx))
		elif np.isfinite(slope) and slope >= ok_tangent_point:
			classes["maybe"].append(int(peak_idx))
		else:
			classes["no"].append(int(peak_idx))

		updated[int(peak_idx)] = _set_peak_fields(
			updated[int(peak_idx)],
			auc_angle=cum_y,
			roc=angle
		)

	stats = {
		"auc": float(curve.auc),
		"peak_order": curve.x,
		"peak_normal_areas": curve.y,
		"derivative": der,
		"tangent_angle_deg": angles_deg,
		"order": curve.order,
		"classes_by_index": classes,
		"metric": metric,
		"use_abs": use_abs,
		"ok_tangent_point": ok_tangent_point,
		"limit_tangent_point": limit_tangent_point,
	}
	return updated, stats


def annotate_peaks_nd(
		peaks_nd: Union[np.ndarray, List],
		*,
		metric: str = "area",
		use_abs: bool = True,
		ok_tangent_point: float = 30,
		limit_tangent_point: float = 45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	ND wrapper: peaks_nd is an object array where each cell is list[Peak].

	Returns
	-------
	peaks_out : np.ndarray(dtype=object), same shape as peaks_nd (each cell list[Peak] updated)
	auc_map   : np.ndarray(float), same shape as peaks_nd (per-spectrum AUC)
	n_peaks   : np.ndarray(int), same shape as peaks_nd (counts)

	:param peaks_nd:
	:param metric:
	:param use_abs:
	:param ok_tangent_point:
	:param limit_tangent_point:
	"""
	peaks_arr = np.asarray(peaks_nd, dtype=object)

	peaks_out = np.empty_like(peaks_arr, dtype=object)
	auc_map = np.empty(peaks_arr.shape, dtype=float)
	n_map = np.empty(peaks_arr.shape, dtype=int)

	if peaks_arr.ndim == 0:
		updated, stats = annotate_peaks_with_auc_and_roc(
			peaks_arr.item(),
			metric=metric,
			use_abs=use_abs,
			ok_tangent_point=ok_tangent_point,
			limit_tangent_point=limit_tangent_point,
		)
		peaks_out[...] = updated
		auc_map[...] = stats["auc"]
		n_map[...] = len(updated)
		return peaks_out, auc_map, n_map

	for idx in np.ndindex(peaks_arr.shape):
		plist = peaks_arr[idx]
		updated, stats = annotate_peaks_with_auc_and_roc(
			plist,
			metric=metric,
			use_abs=use_abs,
			ok_tangent_point=ok_tangent_point,
			limit_tangent_point=limit_tangent_point,
		)
		peaks_out[idx] = updated
		auc_map[idx] = stats["auc"]
		n_map[idx] = len(updated)

	return peaks_out, auc_map, n_map
