# ramanmorph3/__init__.py
from __future__ import annotations

__version__ = "3.0.0"

from .io import SpectralData, load_any, load_npz, save_npz, load_wdf
from .morphology.envelopes import compute_envelopes
from .peaks import find_peaks_1d, find_peaks_nd
from .statistics import (
	CumulativeCurve,
	cumulative_curve_from_peaks,
	annotate_peaks_with_auc_and_roc,
	annotate_peaks_nd,
	noise_std_1d,
	noise_std_nd,
)

__all__ = [
	"SpectralData",
	"load_any",
	"load_npz",
	"save_npz",
	"load_wdf",
	"compute_envelopes",
	"find_peaks_1d",
	"find_peaks_nd",
	"CumulativeCurve",
	"cumulative_curve_from_peaks",
	"annotate_peaks_with_auc_and_roc",
	"annotate_peaks_nd",
	"noise_std_1d",
	"noise_std_nd",
	"__version__",
]
