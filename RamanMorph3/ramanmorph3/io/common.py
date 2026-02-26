# ramanmorph3/io/common.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ramanmorph3.logging_utils import get_logger
logger = get_logger(__name__)


@dataclass
class SpectralData:
	"""
	Container for spectral data.

	x: 1D spectral axis
	y: (..., n_points)
	meta: JSON-friendly metadata
	aux: additional arrays aligned with spectra grid (e.g. time, spatial coords, whitelight image bytes)
	"""
	x: np.ndarray
	y: np.ndarray
	meta: Dict[str, Any] = field(default_factory=dict)
	aux: Dict[str, np.ndarray] = field(default_factory=dict)
	config: Dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.x = np.array(self.x)
		self.y = np.array(self.y)
		self.aux = {k: np.asarray(v) for k, v in dict(self.aux).items()} if self.aux else {}

		if self.x.ndim != 1:
			raise ValueError(f"`x` must be 1D, got shape {self.x.shape}")
		if self.y.ndim < 1:
			raise ValueError(f"`y` must be at least 1D, got shape {self.y.shape}")
		if self.y.shape[-1] != self.x.shape[0]:
			raise ValueError(
				"Last axis of `y` must match length of `x`: "
				f"y.shape[-1]={self.y.shape[-1]} vs len(x)={self.x.shape[0]}"
			)

	@property
	def n_points(self) -> int:
		"""Number of spectral points."""
		return int(self.x.shape[0])

	@property
	def spectra_shape(self) -> Tuple[int, ...]:
		"""Shape of the spectra grid (all axes except the spectral axis)."""
		return tuple(self.y.shape[:-1])

	def as_2d(self) -> Tuple[np.ndarray, Tuple[int, ...]]:
		"""
		Flatten all non-spectral axes so spectra become (n_spectra, n_points).

		:return:
			y2d: Reshaped vies/copy of y with shape (n_spectra, n_points).
			original_shape: The original non-spectral shape, to allow reshaping back.
		"""
		orig = self.spectra_shape
		y2d = self.y.reshape((-1, self.n_points))
		return y2d, orig

	def with_y(self, y_new: np.ndarray, *, meta_update: Optional[Dict[str, Any]] = None) -> SpectralData:
		"""
		Return a new SpectralData with the same x and updated y.

		:param y_new: New spectra array with last axis matching x.
		:param meta_update: Metadata updates merged into a copy of existing meta.
		:return: New SpectralData object.
		"""
		meta = dict(self.meta)
		if meta_update:
			meta.update(meta_update)
		return SpectralData(x=self.x, y=y_new, meta=meta, aux=dict(self.aux))


def _json_dumps_safe(obj: Any) -> str:
	"""Serialize to JSON with a safe fallback for non-serializable objects."""
	return json.dumps(obj, ensure_ascii=False, default=str)


def guess_format(path: Any[str, Path]) -> str:
	"""Guess input format from suffix."""
	p = Path(path)
	suf = p.suffix.lower()
	if suf == ".wdf":
		return "wdf"
	if suf == ".npz":
		return "npz"
	raise ValueError(f"Unsupported input format: {p.name} (suffix: {suf})")


def load_any(path: Any[str, Path]) -> SpectralData:
	"""
	Load spectral data from a supported file.

	Supported:
		- .wdf (via renishaw-wdf)
		- .npz (RamanMorph3 container)
	"""
	fmt = guess_format(path)
	if fmt == "wdf":
		from .wdf import load_wdf
		return load_wdf(path)
	if fmt == "npz":
		from .npz import load_npz
		return load_npz(path)

	raise RuntimeError("Unreachable")
