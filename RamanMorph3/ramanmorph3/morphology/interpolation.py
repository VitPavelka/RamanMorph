# ramanmorph3/morphology/interpolation.py
from __future__ import annotations

from typing import Optional, Tuple, List, Sequence, Literal

import numpy as np

from ramanmorph3.logging_utils import get_logger
from ramanmorph3.peaks.identification import Peak

logger = get_logger(__name__)


# --- Universal helpers ---
def auto_atol(y: np.ndarray, atol: Optional[float]) -> float:
	"""Scale-aware absolute tolerance for float comparisons."""
	if atol is not None:
		return float(atol)
	y = np.asarray(y)
	if y.size == 0:
		return 0.0
	dtype = y.dtype if np.issubdtype(y.dtype, np.floating) else np.float64
	eps = np.finfo(dtype).eps
	finite = np.isfinite(y)
	scale = float(np.nanmax(np.abs(y[finite]))) if np.any(finite) else 1.0
	return float(eps * 100.0 * max(1.0, scale))


def contact_indices(y: np.ndarray, env: np.ndarray, *, atol: float) -> np.ndarray:
	"""Indices where y touches env (within atol)."""
	return np.flatnonzero(np.isclose(y, env, atol=atol, rtol=0.0)).astype(int)


# --- Constraint smoothing Helpers ---
def _segments_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Return inclusive [start, end] indices of True-runs in a boolean mask.
	"""
	m = np.asarray(mask, dtype=bool)
	n = m.size
	if n == 0:
		return np.array([], dtype=int), np.array([], dtype=int)

	d = np.diff(m.astype(np.int8))
	starts = np.flatnonzero(np.r_[m[0], d == 1])
	ends = np.flatnonzero(np.r_[d == -1, m[-1]])

	return starts, ends


def _piecewise_linear_through_points(
		x: np.ndarray,
		y: np.ndarray,
		anchors: np.ndarray,
		line: Optional[np.ndarray] = None,
) -> np.ndarray:
	"""
	Piecewise-linear curve going through (x[a], y[a]) for each anchor 'a'.
	Returned array has length len(y). Anchors must be sorted unique.
	"""
	# x = np.asarray(x)
	# y = np.asarray(y)
	# anchors = np.asarray(anchors, dtype=int).ravel()
	# if anchors.size < 2:
	# 	# nothing to interpolate; fallback to y or line
	# 	return y.astype(float, copy=True) if line is None else np.asarray(line, dtype=float, copy=True)
	#
	# anchors = np.unique(anchors)
	# anchors.sort()
	#
	# # np.interp requires strictly increasing xp; Raman x should be monotone,
	# # but keep a conservative fallback if not
	# xp = x[anchors]
	# if np.any(np.diff(xp) <= 0):
	# 	# fallback to the original slow loop (keep your old implementation here)
	# 	out = np.empty_like(y, dtype=float) if line is None else np.asarray(line, dtype=float, copy=True)
	# 	for a0, a1 in zip(anchors[:-1], anchors[1:]):
	# 		if a1 <= a0:
	# 			continue
	# 		x0, y0 = float(x[a0]), float(y[a0])
	# 		x1, y1 = float(x[a1]), float(y[a1])
	# 		den = (x1 - x0)
	# 		if den == 0.0:
	# 			out[a0:a1] = y0
	# 		else:
	# 			slope = (y1 - y0) / den
	# 			xs = x[a0:a1]
	# 			out[a0:a1] = y0 + slope * (xs - x0)
	# 		if line is not None:
	# 			out[a1] = float(y[a1])
	# 	out[-1] = float(y[-1])
	# 	return out
	#
	# if line is None:
	# 	out = np.interp(x, xp, y[anchors]).astype(float, copy=False)
	# 	out[anchors] = y[anchors]  # exact anchor values
	# 	return out
	#
	# out = np.asarray(line, dtype=float, copy=True)
	# a0, a1 = int(anchors[0]), int(anchors[-1])
	# out[a0:a1 + 1] = np.interp(x[a0:a1 + 1], xp, y[anchors]).astype(float, copy=False)
	# out[anchors] = y[anchors]
	# return out

	if line is None:
		n = y.size
		out = np.empty(n, dtype=float)
	else:
		out = np.asarray(line, copy=True)

	for a0, a1 in zip(anchors[:-1], anchors[1:]):
		if a1 <= a0:
			continue
		x0, y0 = float(x[a0]), float(y[a0])
		x1, y1 = float(x[a1]), float(y[a1])
		den = (x1 - x0)
		if den == 0.0:
			out[a0:a1] = y0
		else:
			slope = (y1 - y0) / den
			xs = x[a0:a1]  # excludes a1
			out[a0:a1] = y0 + slope * (xs - x0)
		if line is not None:
			out[a1] = float(y[a1])

	out[-1] = float(y[-1])
	return out


# --- Constrained smoothing  (legacy "improved_linear_interpolation" logic) ---
def constrained_linear_smoothing(
		x: np.ndarray,
		y_nominal: np.ndarray,
		y_line: np.ndarray,
		*,
		anchors: np.ndarray,
		iteration_max: int = 20,
		atol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Iteratively replace segments where `y_line` > `y_nominal` by inserting a new anchor at the
	maximum violation, then re-interpolating a global polyline through y_nominal at anchors.

	Workflow
		- detect "above spectrum" segments of the current line
		- within each segment pick the index of minimum `(y_nominal - y_line)` (most negative)
		  i.e. maximum `(y_line - y_nominal)`
		- interpolate through those anchors using `y_nominal` values

	Guarantee (within atol): result <= `y_nominal` + `atol`

	:param x:
	:param y_nominal:
	:param y_line:
	:param anchors:
	:param iteration_max:
	:param atol:
	:return:
	"""
	x = np.asarray(x)
	y_nominal = np.asarray(y_nominal)
	y_line = np.asarray(y_line)

	if x.ndim != 1 or y_nominal.ndim != 1 or y_line.ndim != 1:
		raise ValueError("constrained_linear_smoothing requires exactly 1D arrays x, y_nominal, y_line.")
	if not (x.size == y_nominal.size == y_line.size):
		raise ValueError("x, y_nominal, y_line must have the same length.")

	cmp_atol = auto_atol(y_nominal, atol)
	n = x.size

	anchors = np.asarray(anchors, dtype=int).ravel()
	anchors = np.unique(np.concatenate([anchors, np.array([0, n - 1], dtype=int)]))
	anchors.sort()

	cur = y_line.astype(float, copy=True)

	for it in range(iteration_max):
		above = cur > (y_nominal + cmp_atol)
		if not np.any(above):
			return cur, anchors

		starts, ends = _segments_from_mask(above)

		new_anchors: List[int] = []
		for s, e in zip(starts, ends):
			seg = cur[s:e + 1] - y_nominal[s:e + 1]
			if seg.size == 0 or not np.any(np.isfinite(seg)):
				continue

			j = int(s) + int(np.nanargmax(seg))
			new_anchors.append(j)

		if not new_anchors:
			return np.minimum(cur, y_nominal), anchors

		anchors_new = np.unique(np.concatenate([anchors, np.asarray(new_anchors, dtype=int)]))
		if anchors_new.size == anchors.size:
			return np.minimum(cur, y_nominal), anchors

		anchors = anchors_new
		anchors.sort()

		cur = _piecewise_linear_through_points(x, y_nominal, anchors, line=None)

	return np.minimum(cur, y_nominal), anchors


# --- Derive peakline / baseline Helpers ---
def _collect_edges(peaks: Sequence[Peak], left_attr: str, right_attr: str) -> np.ndarray:
	edges: List[int] = []
	for p in peaks:
		edges.append(int(getattr(p, left_attr)))
		edges.append(int(getattr(p, right_attr)))
	edges = np.unique(np.asarray(edges, dtype=int))
	edges.sort()
	return np.asarray(edges, dtype=int)


def _derive_line_from_edges(
		x: np.ndarray,
		y: np.ndarray,
		y_fallback: np.ndarray,
		edges: np.ndarray,
		*,
		refine: bool,
		iteration_max: int,
		atol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Build a piecewise linear line that goes through y at 'edges' and uses y_fallback outside.
	Then optionally apply constrained smoothing to ensure it stays under y.
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	y_fallback = np.asarray(y_fallback)

	if edges.size == 0:
		line = y_fallback.astype(float, copy=True)
		return constrained_linear_smoothing(
			x, y, line, anchors=edges, iteration_max=iteration_max, atol=atol
		) if refine else (line, edges)

	edges = np.unique(edges.astype(int))
	edges.sort()

	line = y_fallback.astype(float, copy=True)

	# Fill between consecutive edges by linear interpolation through spectrum values
	line = _piecewise_linear_through_points(x, y, edges, line=line)
	return constrained_linear_smoothing(
		x, y, line, anchors=edges, iteration_max=iteration_max, atol=atol
	) if refine else (line, edges)


def derive_peakline_and_baseline_1d(
		x: np.ndarray,
		y: np.ndarray,
		y_eroded_peak: np.ndarray,
		y_eroded_base: np.ndarray,
		peaks: Sequence[Peak],
		*,
		refine: bool = True,
		iteration_max: int = 20,
		atol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Derive peakline (narrow) and baseline (wide) from already characterized peaks.

	:param x:
	:param y:
	:param y_eroded_peak:
	:param y_eroded_base:
	:param peaks:
	:param refine:
	:param iteration_max:
	:param atol:
	:return:
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	y_eroded_peak = np.asarray(y_eroded_peak)
	y_eroded_base = np.asarray(y_eroded_base)

	if y.ndim != 1:
		raise ValueError("derive_peakline_and_baseline_1d expects 1D array.")

	edges_peak = _collect_edges(peaks, "left", "right")
	edges_base = _collect_edges(peaks, "left_base", "right_base")

	peakline, edges_peak = _derive_line_from_edges(
		x, y, y_eroded_peak, edges_peak,
		refine=refine, iteration_max=iteration_max, atol=atol
	)

	baseline, edges_base = _derive_line_from_edges(
		x, y, y_eroded_base, edges_base,
		refine=refine, iteration_max=iteration_max, atol=atol
	)

	cmp_atol = auto_atol(peakline, atol)
	if np.any(baseline > peakline + cmp_atol):
		baseline, edges_base = constrained_linear_smoothing(
			x=x,
			y_nominal=peakline,
			y_line=baseline,
			anchors=edges_base,
			iteration_max=iteration_max,
			atol=atol
		)

	# Diagnostic only for now: baseline should stay below peakline
	if np.any(baseline > peakline + auto_atol(peakline, atol)):
		logger.warning("baseline exceeds peakline at some points; check it")

	return peakline, baseline


# --- Boundary correction ---
def _shift_boundaries(
		peak: Peak,
		y: np.ndarray,
		line: np.ndarray,
		cmp_atol: float,
		choice: Literal["peakline", "baseline"],
) -> None:
	""""""
	if choice == "peakline":
		left, right = (int(peak.left), int(peak.right))
	else:
		left, right = (int(peak.left_base), int(peak.right_base))

	for i in range(left, peak.apex + 1):
		if np.isclose(y[i], line[i], atol=cmp_atol, rtol=0.0):
			left = i
			break

	for i in range(peak.apex, right + 1):
		if np.isclose(y[i], line[i], atol=cmp_atol, rtol=0.0):
			right = i
			break

	peak.left, peak.right = left, right


def correct_peak_boundaries_1d(
		peaks: Sequence[Peak],
		y: np.ndarray,
		peakline: np.ndarray,
		baseline: np.ndarray,
		*,
		atol: Optional[float] = None,
) -> None:
	"""
	In-place correction of peak boundaries based on intersections after line refinement.
	This is a conservative isclose-based variant of the legacy `tail_correction` method.

	:param peaks:
	:param y:
	:param peakline:
	:param baseline:
	:param atol:
	:return:
	"""
	y = np.asarray(y)
	peakline = np.asarray(peakline)
	baseline = np.asarray(baseline)
	cmp_atol = auto_atol(y, atol)

	contacts_bl = np.flatnonzero(np.isclose(y, baseline, atol=cmp_atol, rtol=0.0))
	contacts_pl = np.flatnonzero(np.isclose(y, peakline, atol=cmp_atol, rtol=0.0))

	def first_contact(contacts: np.ndarray, start: int, end: int) -> Optional[int]:
		j = np.searchsorted(contacts, start)
		if j < contacts.size and contacts[j] <= end:
			return int(contacts[j])
		return None

	for p in peaks:
		# baseline boundaries
		lb = first_contact(contacts_bl, p.left_base, p.apex)
		if lb is not None:
			p.left_base = lb
		rb = first_contact(contacts_bl, p.apex, p.right_base)
		if rb is not None:
			p.right_base = rb

		# peakline boundaries
		l = first_contact(contacts_pl, p.left, p.apex)
		if l is not None:
			p.left = l
		r = first_contact(contacts_pl, p.apex, p.right)
		if r is not None:
			p.right = r

	# for peak in peaks:
	# 	# Baseline boundaries: search for first contact with baseline on each side
	# 	# left_base in [left_base, apex]
	# 	_shift_boundaries(peak=peak, y=y, line=baseline, cmp_atol=cmp_atol, choice="baseline")
	# 	_shift_boundaries(peak=peak, y=y, line=peakline, cmp_atol=cmp_atol, choice="peakline")
