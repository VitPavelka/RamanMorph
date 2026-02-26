# ramanmorph3/io/npz.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

from ramanmorph3.logging_utils import get_logger
from .common import SpectralData, _json_dumps_safe

logger = get_logger(__name__)


# --- Peak packing (no pickle) ---
PEAK_DTYPE = np.dtype([
	("left_base", "<i4"),
	("left", "<i4"),
	("apex", "<i4"),
	("right", "<i4"),
	("right_base", "<i4"),
	("height", "<f8"),
	("height_base", "<f8"),
	("height_abs", "<f8"),
	("area", "<f8"),
	("area_base", "<f8"),
	("area_abs", "<f8"),
	("fwhm", "<f8"),
	("auc_angle", "<f8"),
	("roc", "<f8"),
])


def _pack_peaks_any(peaks_any: Any) -> Dict[str, np.ndarray]:
	"""
	Pack peaks into (table, offsets, shape) arrays so we can store them in NPZ
	with allow_pickle=False.

	peaks_any can be:
	  - list[Peak] for single spectrum
	  - object ndarray of shape spectra_shape, each cell list[Peak]

	:param peaks_any:
	:return:
	"""
	peaks_arr = np.asarray(peaks_any, dtype=object)

	if peaks_arr.ndim == 0:
		shape = ()
		cells = [peaks_arr.item()]
	else:
		shape = peaks_arr.shape
		cells = [peaks_arr[idx] for idx in np.ndindex(shape)]

	# prefix-sum offsets
	offsets = np.zeros(len(cells) + 1, dtype=np.int64)
	rows: List[tuple] = []

	for i, plist in enumerate(cells):
		plist = list(plist) if plist is not None else []
		offsets[i + 1] = offsets[i] + len(plist)

		for p in plist:
			rows.append((
				int(p.left_base), int(p.left), int(p.apex), int(p.right), int(p.right_base),
				float(p.height), float(p.height_base), float(p.height_abs),
				float(p.area), float(p.area_base), float(p.area_abs),
				float(p.fwhm),
				float(getattr(p, "auc_angle", np.nan)),
				float(getattr(p, "roc", np.nan)),
			))

	table = np.array(rows, dtype=PEAK_DTYPE) if rows else np.zeros((0,), dtype=PEAK_DTYPE)

	return {
		"peaks_table": table,
		"peaks_offsets": offsets,
		"peaks_shape": np.asarray(shape, dtype=np.int64),
	}


def _unpack_peaks(table: np.ndarray, offsets: np.ndarray, shape: tuple[int, ...]) -> Any:
	"""
	Reconstruct peaks as list[Peak] (single) or object ndarray of lists (ND).
	:param table:
	:param offsets:
	:param shape:
	:return:
	"""
	from ramanmorph3.peaks.peak_finder import Peak

	n_cells = int(offsets.size - 1)
	lists: List[list] = []

	for i in range(n_cells):
		a = int(offsets[i])
		b = int(offsets[i + 1])
		seg = table[a:b]
		plist = [
			Peak(
				left_base=int(r["left_base"]),
				left=int(r["left"]),
				apex=int(r["apex"]),
				right=int(r["right"]),
				right_base=int(r["right_base"]),
				height=float(r["height"]),
				height_base=float(r["height_base"]),
				height_abs=float(r["height_abs"]),
				area=float(r["area"]),
				area_base=float(r["area_base"]),
				area_abs=float(r["area_abs"]),
				fwhm=float(r["fwhm"]),
				auc_angle=float(r["auc_angle"]),
				roc=float(r["roc"]),
			)
			for r in seg
		]
		lists.append(plist)

	if shape == ():
		return lists[0]

	out = np.empty(shape, dtype=object)
	for idx, plist in zip(np.ndindex(shape), lists):
		out[idx] = plist
	return out


def save_npz(
		path: Any[str, Path],
		*,
		data: SpectralData,
		peakline: Optional[np.ndarray] = None,
		baseline: Optional[np.ndarray] = None,
		peaks: Optional[np.ndarray] = None,
		spectrum_metrics: Optional[np.ndarray] = None,
		parameters: Optional[Dict[str, Any]] = None,
		meta_extra: Optional[Dict[str, Any]] = None,
		stats: Optional[Dict[str, Any]] = None,
		arrays_extra: Optional[Dict[str, Any]] = None,
		schema_version: int = 1,
		compress: bool = True,
) -> Path:
	"""
	Save RamanMorph3 result to a .npz container.

	:param path: Output file path.
	:param data: Input spectral data (x, y_raw, meta).
	:param peakline: Array with same shape as data.y (optional).
	:param baseline: Array with same shape as data.y (optional).
	:param peaks: Structured array (per-peak-table) or any ndarray (optional).
	:param spectrum_metrics: Structured array (per-spectrum metrics) or any ndarray (optional).
	:param parameters: Parameters used (saved as JSON string).
	:param meta_extra: Extra metadata merged into data.meta (saved as JSON string).
	:param stats:
	:param arrays_extra:
	:param schema_version:
	:param compress: If True uses np.savez_compressed, else np.savez.
	:return: Path to saved file.
	"""
	out = Path(path)
	out.parent.mkdir(parents=True, exist_ok=True)

	meta = dict(data.meta)
	if meta_extra:
		meta.update(meta_extra)

	payload: Dict[str, Any] = {
		"x": np.array(data.x),
		"y_raw": np.array(data.y),
		"config_json": _json_dumps_safe(parameters or {}),
		"meta_json": _json_dumps_safe(meta),
		"schema_version": np.array(schema_version, dtype=np.int64),
		"stats_json": _json_dumps_safe(stats or {}),
	}

	if peakline is not None:
		payload["peakline"] = np.asarray(peakline)
	if baseline is not None:
		payload["baseline"] = np.asarray(baseline)
	if peaks is not None:
		# If peaks is an object array / list-of-Peak, pack it to numeric arrays (no pickle).
		peaks_arr = np.asarray(peaks, dtype=object)
		packed = _pack_peaks_any(peaks_arr)
		payload.update(packed)
	if spectrum_metrics is not None:
		payload["spectrum_metrics"] = np.asarray(spectrum_metrics)
	if arrays_extra:
		for k, v in arrays_extra.items():
			if k in payload:
				raise KeyError(f"arrays_extra key collides with reserved name: {k}")
			payload[k] = np.asarray(v)

	aux_keys = list(getattr(data, "aux", {}).keys())
	payload["aux_keys_json"] = json.dumps(aux_keys, ensure_ascii=False)

	for k in aux_keys:
		storage_key = f"aux__{k}"
		if storage_key in payload:
			raise KeyError(f"aux key collides with reserved name: {storage_key}")
		payload[storage_key] = np.asarray(data.aux[k])

	saver = np.savez_compressed if compress else np.savez
	saver(out, **payload)
	logger.info("Saved NPZ: %s", out)
	return out


def load_npz(path: Any[str, Path]) -> SpectralData:
	"""
	Load a RamanMorph3 .npz container.

	Notes
	-----
	This returns only the SpectralData (x, y_raw, meta).
	Other array can be accessed via `np.load` if needed.

	:param path: Path to .npz file.
	"""
	p = Path(path)
	with np.load(p, allow_pickle=False) as z:
		x = z["x"]
		y = z["y_raw"]

		meta: Dict[str, Any] = {}
		aux: Dict[str, Any] = {}
		if "aux_keys_json" in z.files:
			try:
				raw = z["aux_keys_json"]
				aux_keys = json.loads(raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else str(raw))
				for k in aux_keys:
					storage_key = f"aux__{k}"
					if storage_key in z.files:
						aux[k] = z[storage_key]
			except Exception as e:
				logger.warning("Failed to parse aux_keys_json(%s): %s", p, e)

		if "meta_json" in z.files:
			try:
				raw = z["meta_json"]
				meta = json.loads(raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else str(raw))
			except Exception as e:
				logger.warning("Failed to parse meta_json (%s): %s", p, e)

		if "config_json" in z.files:
			try:
				raw = z["config_json"]
				config = json.loads(raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else str(raw))
			except Exception as e:
				logger.warning("Failed to parse config_json (%s): %s", p, e)

	meta.setdefault("source_file", str(p))
	meta.setdefault("source_format", "npz")
	return SpectralData(x=x, y=y, meta=meta, aux=aux, config=config)


def load_npz_result(path: Any[str, Path]) -> Dict[str, Any]:
	"""
	Load NPZ and reconstruct a full RamanMorph3 result.

	Returns dict with:
		- data: SpectralData(x, y_raw, meta, aux, config)
		- peakline / baseline (if present)
		- peaks (list[Peak] or object ndarray of list[Peak]) if packed peaks present
		- parameters (dict) and stats (dict)
		- any extra arrays stored in arrays_extra
	:param path:
	:return:
	"""
	p = Path(path)
	with np.load(p, allow_pickle=False) as z:
		# --- base data ---
		x = z["x"]
		y_raw = z["y_raw"]

		meta = {}
		if "meta_json" in z.files:
			raw = z["meta_json"]
			meta = json.loads(raw.item() if raw.ndim == 0 else str(raw))
		meta.setdefault("source_file", str(p))
		meta.setdefault("source_format", "npz")

		parameters = {}
		if "config_json" in z.files:
			raw = z["config_json"]
			parameters = json.loads(raw.item() if raw.ndim == 0 else str(raw))

		stats = {}
		if "stats_json" in z.files:
			raw = z["stats_json"]
			stats = json.loads(raw.item() if raw.ndim == 0 else str(raw))

		# --- aux ---
		aux: Dict[str, Any] = {}
		if "aux_keys_json" in z.files:
			aux_keys = json.loads(z["aux_keys_json"].item() if z["aux_keys_json"].ndim == 0 else str(z["aux_keys_json"]))
			for k in aux_keys:
				key = f"aux__{k}"
				if key in z.files:
					aux[k] = z[key]

		data = SpectralData(x=x, y=y_raw, meta=meta, aux=aux)

		out: Dict[str, Any] = {
			"data": data,
			"parameters": parameters,
			"stats": stats,
		}

		# optional computed arrays
		if "peakline" in z.files:
			out["peakline"] = z["peakline"]
		if "baseline" in z.files:
			out["baseline"] = z["baseline"]
		if "spectrum_metrics" in z.files:
			out["spectrum_metrics"] = z["spectrum_metrics"]

		# packed peaks → reconstructed peaks
		if all(k in z.files for k in ("peaks_table", "peaks_offsets", "peaks_shape")):
			shape = tuple(int(v) for v in z["peaks_shape"].tolist())
			out["peaks"] = _unpack_peaks(z["peaks_table"], z["peaks_offsets"], shape)

		# include any other arrays (arrays_extra) transparently
		reserved = {
			"x", "y_raw", "config_json", "meta_json", "stats_json", "schema_version",
			"peakline", "baseline", "spectrum_metrics",
			"aux_keys_json",
			# aux__*
			"peaks_table", "peaks_offsets", "peaks_shape",
		}
		for k in z.files:
			if k in reserved or k.startswith("aux__"):
				continue
			out[k] = z[k]
		return out


def load_npz_full(path: Any[str, Path]) -> Dict[str, Any]:
	"""
	Load full NPZ payload into a dict.

	Use this if you need peakline/baseline/peaks/etc.
	"""
	p = Path(path)
	with np.load(p, allow_pickle=False) as z:
		return {k: z[k] for k in z.files}
