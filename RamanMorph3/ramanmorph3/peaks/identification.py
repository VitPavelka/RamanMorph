# ramanmorph3/peaks/identification.py
from __future__ import annotations

from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np

from ramanmorph3.logging_utils import get_logger

get_logger = get_logger(__name__)


# --- Peak data model ---
@dataclass(slots=True)
class Peak:
	"""
	Peak boundaries and basic metrics.

	All indices are integer indices into the spectral axis (0...n-1).

	left_base  ... where spectrum touches BASELINE
	left       ... where spectrum touches PEAKLINE
	apex       ... where spectrum touches DILATION (tip candidate)
	right      ... where spectrum touches PEAKLINE
	right_base ... where spectrum touches BASELINE

	Metrics are computed by `compute_peak_metrics(...)`.
	`auc_angle` and `roc` are placeholders for later statistical evaluation.
	"""
	left_base: int
	left: int
	apex: int
	right: int
	right_base: int

	# Metrics filled later
	height: float = float("nan")
	height_base: float = float("nan")
	height_abs: float = float("nan")

	area: float = float("nan")
	area_base: float = float("nan")
	area_abs: float = float("nan")

	fwhm: float = float("nan")

	auc_angle: Optional[float] = None
	roc: Optional[float] = None


# --- Helpers ---
def _ensure_edges(idxs: np.ndarray, n: int) -> np.ndarray:
	"""Ensure 0 and n-1 are present, return sorted unique int array."""
	if n <= 0:
		return np.array([], dtype=int)
	need = np.array([0, n - 1], dtype=int)
	out = np.unique(np.concatenate([idxs.astype(int, copy=False), need]))
	out.sort()
	return out


def _detect_boundaries(boundaries: np.ndarray, tip: int) -> Tuple[Optional[int], Optional[int]]:
	""""""
	pos = int(np.searchsorted(boundaries, tip))

	if pos <= 0 or pos >= boundaries.size:
		# Tip on/near edge → skip (or clamp). Keeping skip is safer.
		return None, None

	left_pos = int(boundaries[pos - 1])
	right_pos = int(boundaries[pos])

	if not (left_pos < tip < right_pos):
		return None, None

	return left_pos, right_pos


def _split_neighboring_tips_by_valley(
		peaks: list[Peak],
		y: np.ndarray,
		y_eroded_peak: np.ndarray,
) -> List[Peak]:
	"""
	Multi-peak splitting: if adjacent peaks share the same tail interval (same left boundary),
	insert a boundary at the valley between their apices.

	This matches the legacy constraint: pseudo-candidate creation.
	"""
	for i in range(1, len(peaks)):
		p0 = peaks[i - 1]
		p1 = peaks[i]

		# If they share the same left tail, they are inside the same tail interval.
		if p1.left == p0.left:
			a0 = p0.apex
			a1 = p1.apex
			if a1 <= a0 + 1:
				# Adjacent tips → still try a 1-points valley between them if possible
				continue

			seg = (y[a0:a1] - y_eroded_peak[a0:a1])
			if seg.size <= 0:
				continue
			valley = int(np.argmin(seg)) + int(a0)

			# Update narrow boundaries
			p0.right = valley
			p1.left = valley

	return peaks


# --- Peak characterization from candidates ---
def characterize_peaks_1d(
		y: np.ndarray,
		y_eroded_peak: np.ndarray,
		tips: np.ndarray,
		tails: np.ndarray,
		bases: np.ndarray,
		*,
		use_wide_bases: bool,
) -> List[Peak]:
	"""
	Build Peak objects (boundaries + apex) from candidate indices.

	This follows the legacy idea:
	  - For each tip, find surrounding tails → (left, right)
	  - Optionally find also surrounding bases → (left_base, right_base) -- if hw1 != hw2
	  - If multiple tips share the same tail interval, insert a valley boundary between them
	    at argmin(y - y_eroded_peak) between adjacent tips.

	:param y:
	:param y_eroded_peak:
	:param tips:
	:param tails:
	:param bases:
	:param use_wide_bases:
	:return:
	"""
	y = np.asarray(y)
	y_eroded_peak = np.asarray(y_eroded_peak)

	n = y.size
	if n == 0:
		return []

	tips = np.asarray(tips, dtype=int)
	tails = np.asarray(tails, dtype=int)
	bases = np.asarray(bases, dtype=int)

	tails = _ensure_edges(tails, n)
	bases = _ensure_edges(bases, n)

	tips = np.unique(tips)
	tips.sort()

	if tips.size == 0:
		return []

	# Build initial peaks
	peaks: List[Peak] = []

	for t in tips:
		# Surrounding tails
		left, right = _detect_boundaries(tails, t)
		if left is None or right is None:
			continue

		# Surrounding base-tails (if hw1 != hw2)
		if use_wide_bases:
			left_base, right_base = _detect_boundaries(bases, t)
			if left_base is None or right_base is None:
				continue
		else:
			left_base, right_base = left, right

		peaks.append(Peak(
			left_base=left_base,
			left=left,
			apex=int(t),
			right=right,
			right_base=right_base,
		))

	# Sort by apex (important for multipeak splitting)
	peaks.sort(key=lambda p: p.apex)
	peaks = _split_neighboring_tips_by_valley(peaks, y, y_eroded_peak)

	return peaks
