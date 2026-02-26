# ramanmorph3/io/wdf.py
from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import timezone

from ramanmorph3.logging_utils import get_logger
from .common import SpectralData

logger = get_logger(__name__)


def _import_wdf() -> Any:
	"""
	Import `renishaw-wdf` (module name: `wdf`).

	The PyPI package is `renishaw-wdf` but it installs/imports as `wdf`.
	"""
	try:
		import wdf  # type: ignore
	except Exception as e:
		raise RuntimeError(
			"WDF support is not available. Install the official Renishaw package:\n"
			"  pip install renishaw-wdf\n"
			"or if you use extras:\n"
			"  pip install 'RamanMorph3_0[wdf]'\n"
		) from e
	return wdf


def _try_get_map_shape(wdf_obj: Any) -> Optional[Tuple[int, ...]]:
	"""
	Attempt to infer map/volume dimensions from renishaw-wdf object.
	Returns shape excluding spectral axis (e.g. (ny, nx) or (nz, ny, nx)).
	"""
	try:
		ma = getattr(wdf_obj, "map_area", None)
		if ma is None:
			return None
		cnt = getattr(ma, "count", None)
		if cnt is None:
			return None

		# count often has x,y,(z)
		dims = []
		for name in ("z", "y", "x"):  # keep internal order (z,y,x)
			if hasattr(cnt, name):
				v = int(getattr(cnt, name))
				if v > 0:
					dims.append(v)

		return tuple(dims) if dims else None
	except Exception:
		return None


def _safe_prop_value(v: Any, *, max_str_len: int = 20000) -> Any:
	"""
	Convert section property value into something JSON-friendly.
	"""
	# unwrap "property" objects
	try:
		if hasattr(v, "value"):
			v = v.value
	except Exception:
		pass

	# recursively unwrap nested property bags
	if hasattr(v, "keys") or hasattr(v, "__getitem__"):
		# but avoid treating strings/bytes as mappings
		if not isinstance(v, (str, bytes, bytearray)):
			# best effort: only if it yields something reasonable
			try:
				d = _props_to_dict(v)
				if d and d != {"_raw": str(v)}:
					return d
			except Exception:
				pass

	if isinstance(v, (int, float, bool)) or v is None:
		return v

	if isinstance(v, (bytes, bytearray)):
		return f"<bytes len={len(v)}>"

	# datetime, enums, Path-like, etc.
	s = str(v)
	if len(s) > max_str_len:
		return s[:max_str_len] + " ... <TRUNCATED>"
	return s


def _props_to_dict(props: Any) -> Dict[str, Any]:
	"""
	Convert renishaw-wdf property containers (dict-like or Pset-like) to plain dict.
	"""
	if props is None:
		return {}

	# Get keys
	keys = None
	if hasattr(props, "keys"):
		try:
			keys = list(props.keys())
		except Exception:
			keys = None

	if keys is None:
		# Some containers are iterable over keys
		try:
			keys = list(props)
		except Exception:
			return {"_raw": _safe_prop_value(props)}

	out: Dict[str, Any] = {}
	for k in keys:
		try:
			v = props[k]
		except Exception:
			continue
		out[str(k)] = _safe_prop_value(v)
	return out


def _collect_section_properties(data: Any, wdf: Any) -> Dict[str, Any]:
	"""
	Best-effort: iterate over WdfBlockId constants and read the FIRST instance (-1).
	"""
	sections: Dict[str, Any] = {}
	WdfBlockId = getattr(wdf, "WdfBlockId", None)
	if WdfBlockId is None:
		return sections

	for name in dir(WdfBlockId):
		if name.startswith("_"):
			continue
		block = getattr(WdfBlockId, name)
		# filter out non-constants (defensive)
		if not isinstance(block, (int,)):
			continue

		try:
			props = data.get_section_properties(block, -1)  # first available instance
		except Exception:
			continue

		out = _props_to_dict(props)
		if out:
			sections[name] = out

	return sections


def _extract_whitelight(data: Any, wdf: Any) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
	"""
	Extract WHITELIGHT JPEG bytes as uint8 array and parse a few EXIF custom tags if Pillow is installed.
	"""
	WdfBlockId = getattr(wdf, "WdfBlockId", None)
	if WdfBlockId is None or not hasattr(WdfBlockId, "WHITELIGHT"):
		return None, {}

	try:
		stream = data.get_section_stream(WdfBlockId.WHITELIGHT, -1)
	except Exception:
		return None, {}

	jpg_bytes = None
	exif_info: Dict[str, Any] = {}

	with stream:
		raw = stream.read(-1)
		if raw:
			jpg_bytes = np.frombuffer(raw, dtype=np.uint8)

	# EXIF parse (optional)
	try:
		import PIL.Image  # type: ignore
		from io import BytesIO

		if jpg_bytes is not None:
			img = PIL.Image.open(BytesIO(jpg_bytes.tobytes()))
			exif = img.getexif()

			# renishaw-wdf docs show these custom tags
			# 0xfea0 position, 0xfea1 fov, 0xfea2 objective (if present)
			for tag, key in ((0xFEA0, "position"), (0xFEA1, "fov"), (0xFEA2, "objective")):
				if tag in exif:
					exif_info[key] = exif[tag]
	except Exception:
		pass

	return jpg_bytes, exif_info


def _collect_origins(data: Any, *, n_spec: int) -> Dict[str, np.ndarray]:
	"""Read ALL origins into aux as 'origin_<dtype>' arrays."""
	aux: Dict[str, np.ndarray] = {}
	origins = getattr(data, "origins", None)
	if not origins:
		return aux

	for dtype in origins:
		name = str(dtype)  # e.g. 'WdfDataType.Spatial_X'
		short = name.split(".")[-1].lower()
		key = f"origin_{short}"

		try:
			vals = list(origins[dtype])
		except Exception:
			continue

		if len(vals) != n_spec:
			# still keep, but don't assume alignment
			try:
				aux[key] = np.asarray(vals)
			except Exception:
				aux[key] = np.asarray([str(v) for v in vals], dtype=object)
			continue

		# datetime handling
		if vals and hasattr(vals[0], "year") and hasattr(vals[0], "month"):
			cleaned = []
			for dt in vals:
				tz = getattr(dt, "tzinfo", None)
				if tz is not None:
					dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
				cleaned.append(dt)
			aux[key] = np.array([np.datetime64(dt) for dt in cleaned], dtype="datetime64[ms]")
			continue

		# numeric / fallback
		try:
			aux[key] = np.asarray(vals, dtype=float)
		except Exception:
			aux[key] = np.asarray([str(v) for v in vals], dtype=object)

	return aux


def load_wdf(path: Any[str, Path]) -> SpectralData:
	"""
	Load a Renishaw .wdf file using the official `renishaw-wdf` library.

	:param path: Path to .wdf
	:return: SpectralData
		x: 1D spectral axis
		y: spectral array with shape (..., n_points) if a map shape can be inferred,
		   otherwise (n_spectra, n_points).
		meta: best-effort metadata
	"""
	wdf = _import_wdf()
	p = Path(path)

	meta: Dict[str, Any] = {
		"source_file": str(p),
		"source_format": "wdf",
	}

	with wdf.Wdf(p) as data:
		# x-axis
		x = np.asarray(list(data.xlist()), dtype=float)

		# number of spectra: Wdf supports indexing and is iterable
		n_spec = len(data)
		n_points = x.shape[0]

		y2d = np.empty((n_spec, n_points), dtype=float)
		for i in range(n_spec):
			y2d[i, :] = np.asarray(data[i], dtype=float)

		# sections/properties
		meta["wdf_sections"] = _collect_section_properties(data, wdf)

		# whitelight image (optional)
		wl_jpg, wl_exif = _extract_whitelight(data, wdf)
		if wl_exif:
			meta["whitelight_exif"] = wl_exif

		# origins → aux (time, spatial coords, ...)
		aux = _collect_origins(data, n_spec=n_spec)

		# shape
		map_shape_zyx = _try_get_map_shape(data)
		if map_shape_zyx is not None and int(np.prod(map_shape_zyx)) == n_spec:
			y = y2d.reshape((*map_shape_zyx, n_points))

			# If it's a 2D map stored as z=1, squeeze it to (y, x, n_points)
			if len(map_shape_zyx) == 3 and map_shape_zyx[0] == 1:
				y = y.reshape((map_shape_zyx[1], map_shape_zyx[2], n_points))
				meta["spectra_shape"] = (map_shape_zyx[1], map_shape_zyx[2])
				meta["grid_dim_order"] = "(y, x)"
				# reshape origins accordingly
				for k, v in list(aux.items()):
					if v.shape == (n_spec,):
						aux[k] = v.reshape((map_shape_zyx[1], map_shape_zyx[2]))
			else:
				meta["spectra_shape"] = map_shape_zyx
				meta["grid_dim_order"] = "(z, y, x)"
				for k, v in list(aux.items()):
					if v.shape == (n_spec,):
						aux[k] = v.reshape(map_shape_zyx)

		else:
			# non-map: series or single
			if n_spec == 1:
				y = y2d[0, :]  # 1D spectrum
				meta["spectra_shape"] = ()
			else:
				y = y2d
				meta["spectra_shape"] = (n_spec,)

		# attach whitelight jpeg as aux array (so it can be saved into npz)
		if wl_jpg is not None:
			aux["whitelight_jpeg"] = wl_jpg

		# quick timestamps
		if "origin_time" in aux and aux["origin_time"].size >= 1:
			try:
				meta["timestamp_start"] = str(aux["origin_time"].reshape(-1)[0])
				meta["timestamp_end"] = str(aux["origin_time"].reshape(-1)[-1])
			except Exception:
				pass

	return SpectralData(x=x, y=y, meta=meta, aux=aux)
