# ramanmorph3/morphology/envelopes.py
"""
Morphological envelopes for 1D spectra.

Computes:
- top envelope via dilation (max filter)
- two bottom envelopes via erosion (min filter)
	- smaller (peakline anchors / "tails")
	- larger (peakline anchors)

Also provides contact masks (touch points).
Derives interpolated peakline/baseline using linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramanmorph3.logging_utils import get_logger
from .filters import BoundaryMode, dilate_1d, erode_1d, contact_mask
from .interpolation import linear_interpolation_nd

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Envelopes:
	"""Container for morphological envelopes and contact masks."""

	# Raw envelopes
	y_dilated: np.ndarray
	y_eroded_small: np.ndarray
	y_eroded_big: np.ndarray

	# Contact masks (True where y touches the envelope)
	tip_mask: np.ndarray
	tail_mask: np.ndarray
	base_mask: np.ndarray

	# Derived (interpolated) lines
	peakline: np.ndarray
	baseline: np.ndarray


def compute_envelopes(
		x: np.ndarray,
		y: np.ndarray,
		*,
		hw_signal: int,
		hw_peakline: int,
		hw_baseline: int,
		axis: int = -1,
		interpolation_atol: float = 0.0,
		mode: BoundaryMode = "reflect",
		cval: float = 0.0,
) -> Envelopes:
	"""
	Compute morphological envelopes + contact masks.

	:param y: Input array (1D or ND).
	:param hw_signal: Half-window for dilation (signal/top envelope).
	:param hw_peakline: Half-window for erosion (local minima; "tail" candidates).
	:param hw_baseline: Half-window for erosion (broad minima; baseline candidates).
	:param axis: Axis along which to apply 1D morphology.
	:return: Envelopes plus boolean masks (tip/tail/base).
	"""
	y = np.asarray(y)

	if hw_signal < 0 or hw_peakline < 0 or hw_baseline < 0:
		raise ValueError("Half-windows must be positive.")

	y_dil = dilate_1d(y, half_window=hw_signal, axis=axis, mode=mode, cval=cval)
	y_ero_s = erode_1d(y, half_window=hw_peakline, axis=axis, mode=mode, cval=cval)
	y_ero_b = erode_1d(y, half_window=hw_baseline, axis=axis, mode=mode, cval=cval)

	peakline = linear_interpolation_nd(x, y, y_ero_s)
	baseline = linear_interpolation_nd(x, peakline, linear_interpolation_nd(x, y, y_ero_b))

	# Contact masks: "touching the envelope"
	tip = contact_mask(y, y_dil)

	cmp_atol = interpolation_atol
	if cmp_atol <= 0.0:
		finite = np.isfinite(y)
		scale = float(np.nanmax(np.abs(y[finite]))) if np.any(finite) else 1.0
		cmp_atol = float(np.finfo(float).eps * 100.0 * max(1.0, scale))

	tail_mask = contact_mask(y=y, y_filtered=peakline, atol=cmp_atol, rtol=0.0)
	base_mask = contact_mask(y=y, y_filtered=baseline, atol=cmp_atol, rtol=0.0)

	logger.debug(
		"Envelopes computed (hw_signal=%d, hw_peakline=%d, hw_baseline=%d).",
		hw_signal, hw_peakline, hw_baseline
	)

	return Envelopes(
		y_dilated=y_dil,
		y_eroded_small=y_ero_s,
		y_eroded_big=y_ero_b,
		tip_mask=tip,
		tail_mask=tail_mask,
		base_mask=base_mask,
		peakline=peakline,
		baseline=baseline,
	)
