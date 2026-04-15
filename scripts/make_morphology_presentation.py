#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from ramanmorph3.io import load_any
from ramanmorph3.morphology import dilate_1d, erode_1d
from ramanmorph3.morphology.interpolation import auto_atol, contact_indices, derive_peakline_and_baseline_1d
from ramanmorph3.peaks.identification import characterize_peaks_1d


@dataclass
class SpectrumRun:
	x: np.ndarray
	y: np.ndarray
	y_dil: np.ndarray
	y_ero_peak: np.ndarray
	y_ero_base: np.ndarray
	tips: np.ndarray
	tails: np.ndarray
	bases: np.ndarray
	peaks: list
	peakline: np.ndarray
	baseline: np.ndarray


@dataclass
class FrameInfo:
	phase: str
	index: int
	step_no: int
	total_steps: int


def _choose_1d_spectrum(y: np.ndarray, index: int) -> np.ndarray:
	if y.ndim == 1:
		return y
	flat = y.reshape(-1, y.shape[-1])
	if not (0 <= index < flat.shape[0]):
		raise ValueError(f"spectrum-index out of range: {index}, available 0..{flat.shape[0]-1}")
	return flat[index]


def run_pipeline(x: np.ndarray, y: np.ndarray, hw_peak: int, hw_base: int) -> SpectrumRun:
	y_dil = dilate_1d(y=y, half_window=hw_peak)
	y_ero_peak = erode_1d(y=y, half_window=hw_peak)
	y_ero_base = erode_1d(y=y, half_window=hw_base)

	cmp_atol = auto_atol(y, atol=None)
	tips = contact_indices(y, y_dil, atol=cmp_atol)
	tails = contact_indices(y, y_ero_peak, atol=cmp_atol)
	bases = contact_indices(y, y_ero_base, atol=cmp_atol)

	peaks = characterize_peaks_1d(
		y=y,
		y_eroded_peak=y_ero_peak,
		tips=tips,
		tails=tails,
		bases=bases,
		use_wide_bases=not np.array_equal(y_ero_peak, y_ero_base),
	)

	peakline, baseline = derive_peakline_and_baseline_1d(
		x=x,
		y=y,
		y_eroded_peak=y_ero_peak,
		y_eroded_base=y_ero_base,
		peaks=peaks,
		refine=True,
	)

	return SpectrumRun(
		x=x,
		y=y,
		y_dil=y_dil,
		y_ero_peak=y_ero_peak,
		y_ero_base=y_ero_base,
		tips=tips,
		tails=tails,
		bases=bases,
		peaks=peaks,
		peakline=peakline,
		baseline=baseline,
	)


def save_overview(run: SpectrumRun, out_dir: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(run.x, run.y, color="black", lw=1.3, label="Raw spectrum")
	ax.plot(run.x, run.y_dil, color="tab:orange", lw=1.0, alpha=0.9, label="Dilation")
	ax.plot(run.x, run.y_ero_peak, color="tab:blue", lw=1.0, alpha=0.9, label="Erosion (peak)")
	ax.plot(run.x, run.y_ero_base, color="tab:purple", lw=1.0, alpha=0.8, label="Erosion (base)")
	ax.set_title("Overview: raw spectrum + morphology envelopes")
	ax.set_xlabel("Raman shift")
	ax.set_ylabel("Intensity")
	ax.legend(loc="best")
	ax.grid(alpha=0.2)
	fig.tight_layout()
	fig.savefig(out_dir / "01_overview.png", dpi=180)
	plt.close(fig)


def save_candidates(run: SpectrumRun, out_dir: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(run.x, run.y, color="black", lw=1.2, label="Raw")
	ax.plot(run.x[run.tips], run.y[run.tips], "o", ms=5, color="tab:orange", label="Tip candidates")
	ax.plot(run.x[run.tails], run.y[run.tails], "o", ms=4, color="tab:blue", label="Tail candidates")
	ax.plot(run.x[run.bases], run.y[run.bases], "o", ms=4, color="tab:green", label="Base candidates")
	ax.set_title("Candidates from morphology contacts")
	ax.set_xlabel("Raman shift")
	ax.set_ylabel("Intensity")
	ax.grid(alpha=0.2)
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig(out_dir / "02_candidates.png", dpi=180)
	plt.close(fig)


def save_filtered_peaks(run: SpectrumRun, out_dir: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(run.x, run.y, color="black", lw=1.2, label="Raw")

	for i, p in enumerate(run.peaks):
		label = "Accepted peaks (apex)" if i == 0 else None
		ax.plot(run.x[p.apex], run.y[p.apex], "o", ms=6, color="crimson", label=label)
		ax.axvline(run.x[p.left], color="tab:blue", alpha=0.35, lw=1)
		ax.axvline(run.x[p.right], color="tab:blue", alpha=0.35, lw=1)
		ax.axvline(run.x[p.left_base], color="tab:green", alpha=0.22, lw=1)
		ax.axvline(run.x[p.right_base], color="tab:green", alpha=0.22, lw=1)

	ax.set_title("Filtered peaks + boundaries")
	ax.set_xlabel("Raman shift")
	ax.set_ylabel("Intensity")
	ax.grid(alpha=0.2)
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig(out_dir / "03_filtered_peaks.png", dpi=180)
	plt.close(fig)


def save_final_lines(run: SpectrumRun, out_dir: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(run.x, run.y, color="black", lw=1.2, label="Raw")
	ax.plot(run.x, run.peakline, color="tab:blue", lw=1.6, label="Peakline")
	ax.plot(run.x, run.baseline, color="tab:green", lw=1.8, label="Baseline")

	if run.peaks:
		ax.plot(
			run.x[[p.apex for p in run.peaks]],
			run.y[[p.apex for p in run.peaks]],
			"o",
			color="crimson",
			ms=5,
			label="Final peaks",
		)

	ax.set_title("Final lines: peakline + baseline")
	ax.set_xlabel("Raman shift")
	ax.set_ylabel("Intensity")
	ax.grid(alpha=0.2)
	ax.legend(loc="best")
	fig.tight_layout()
	fig.savefig(out_dir / "04_final_lines.png", dpi=180)
	plt.close(fig)


def _sample_indices(n: int, step: int, max_frames: int) -> list[int]:
	idx = list(range(0, n, max(1, step)))
	if idx[-1] != n - 1:
		idx.append(n - 1)
	if len(idx) > max_frames:
		sel = np.linspace(0, len(idx) - 1, max_frames).astype(int)
		idx = [idx[i] for i in sel]
	return idx


def _build_storyboard(run: SpectrumRun, step: int, max_frames: int) -> list[FrameInfo]:
	idx = _sample_indices(run.y.size, step=step, max_frames=max_frames)
	frames: list[FrameInfo] = []

	for k, i in enumerate(idx):
		frames.append(FrameInfo(phase="dilation", index=i, step_no=k + 1, total_steps=len(idx)))
	for k, i in enumerate(idx):
		frames.append(FrameInfo(phase="erosion", index=i, step_no=k + 1, total_steps=len(idx)))

	# 20 frames to show candidate pruning
	for k in range(20):
		frames.append(FrameInfo(phase="filter", index=k, step_no=k + 1, total_steps=20))

	for k, i in enumerate(idx):
		frames.append(FrameInfo(phase="interpolation", index=i, step_no=k + 1, total_steps=len(idx)))

	return frames


def save_pipeline_video(run: SpectrumRun, out_dir: Path, step: int, max_frames: int, fps: int) -> Path:
	storyboard = _build_storyboard(run, step=step, max_frames=max_frames)

	fig, ax = plt.subplots(figsize=(13, 6))
	y_margin = (float(np.nanmax(run.y)) - float(np.nanmin(run.y))) * 0.08
	ax.set_xlim(run.x[0], run.x[-1])
	ax.set_ylim(float(np.nanmin(run.y)) - y_margin, float(np.nanmax(run.y)) + y_margin)

	ax.plot(run.x, run.y, color="black", lw=1.2, label="Raw spectrum")
	dil_built = np.full(run.y.shape, np.nan, dtype=float)
	ero_built = np.full(run.y.shape, np.nan, dtype=float)
	pl_built = np.full(run.y.shape, np.nan, dtype=float)
	bl_built = np.full(run.y.shape, np.nan, dtype=float)

	dil_line, = ax.plot(run.x, dil_built, color="tab:orange", lw=2.0, label="Dilation (built)")
	ero_line, = ax.plot(run.x, ero_built, color="tab:blue", lw=2.0, label="Erosion (built)")
	peakline_line, = ax.plot(run.x, pl_built, color="tab:cyan", lw=1.6, label="Peakline (interp)")
	baseline_line, = ax.plot(run.x, bl_built, color="tab:green", lw=1.8, label="Baseline (interp)")

	tips_seen: list[int] = []
	tails_seen: list[int] = []
	bases_seen: list[int] = []
	apex_arr = np.array([p.apex for p in run.peaks], dtype=int) if run.peaks else np.array([], dtype=int)

	tips_scatter = ax.scatter([], [], s=34, c="tab:orange", label="Tip contacts")
	tails_scatter = ax.scatter([], [], s=30, c="tab:blue", label="Tail contacts")
	bases_scatter = ax.scatter([], [], s=30, c="tab:green", label="Base contacts")
	apex_scatter = ax.scatter([], [], s=44, c="crimson", label="Accepted peaks")

	cursor = ax.axvline(run.x[0], color="gray", lw=1.0, ls="--", alpha=0.65)
	status = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=10)

	ax.set_xlabel("Raman shift")
	ax.set_ylabel("Intensity")
	ax.set_title("RamanMorph3 pipeline animation")
	ax.grid(alpha=0.2)
	ax.legend(loc="upper right", fontsize=8)

	tips_set = set(int(i) for i in run.tips.tolist())
	tails_set = set(int(i) for i in run.tails.tolist())
	bases_set = set(int(i) for i in run.bases.tolist())
	prev_dil = -1
	prev_ero = -1
	prev_interp = -1
	frames: list[Image.Image] = []

	for fr in tqdm(storyboard, desc="Rendering pipeline frames"):
		i = fr.index

		if fr.phase == "dilation":
			a = prev_dil + 1
			b = i
			dil_built[a:b + 1] = run.y_dil[a:b + 1]
			for j in range(a, b + 1):
				if j in tips_set and j not in tips_seen:
					tips_seen.append(j)
			prev_dil = i
			status.set_text(f"Phase: dilation {fr.step_no}/{fr.total_steps}\nTouch -> tip candidate")
		elif fr.phase == "erosion":
			a = prev_ero + 1
			b = i
			ero_built[a:b + 1] = run.y_ero_peak[a:b + 1]
			for j in range(a, b + 1):
				if j in tails_set and j not in tails_seen:
					tails_seen.append(j)
				if j in bases_set and j not in bases_seen:
					bases_seen.append(j)
			prev_ero = i
			status.set_text(f"Phase: erosion {fr.step_no}/{fr.total_steps}\nTouch -> tail/base candidate")
		elif fr.phase == "filter":
			status.set_text(f"Phase: candidate filtering {fr.step_no}/{fr.total_steps}\nNon-peaks removed")
		elif fr.phase == "interpolation":
			a = prev_interp + 1
			b = i
			pl_built[a:b + 1] = run.peakline[a:b + 1]
			bl_built[a:b + 1] = run.baseline[a:b + 1]
			prev_interp = i
			status.set_text(f"Phase: interpolation {fr.step_no}/{fr.total_steps}\nBuilding peakline/baseline")

		dil_line.set_ydata(dil_built)
		ero_line.set_ydata(ero_built)
		peakline_line.set_ydata(pl_built)
		baseline_line.set_ydata(bl_built)
		cursor.set_xdata([run.x[i], run.x[i]])

		if tips_seen:
			tips_xy = np.column_stack((run.x[tips_seen], run.y[tips_seen]))
		else:
			tips_xy = np.empty((0, 2))
		if tails_seen:
			tails_xy = np.column_stack((run.x[tails_seen], run.y[tails_seen]))
		else:
			tails_xy = np.empty((0, 2))
		if bases_seen:
			bases_xy = np.column_stack((run.x[bases_seen], run.y[bases_seen]))
		else:
			bases_xy = np.empty((0, 2))

		tips_scatter.set_offsets(tips_xy)
		tails_scatter.set_offsets(tails_xy)
		bases_scatter.set_offsets(bases_xy)

		if fr.phase == "filter" and apex_arr.size > 0:
			n_keep = max(1, int(np.ceil((fr.step_no / fr.total_steps) * apex_arr.size)))
			curr = apex_arr[:n_keep]
			apex_scatter.set_offsets(np.column_stack((run.x[curr], run.y[curr])))
		elif apex_arr.size > 0 and fr.phase in {"interpolation"}:
			apex_scatter.set_offsets(np.column_stack((run.x[apex_arr], run.y[apex_arr])))

		fig.canvas.draw()
		rgba = np.asarray(fig.canvas.buffer_rgba())
		frames.append(Image.fromarray(rgba.copy(), mode="RGBA"))

	gif_path = out_dir / "05_pipeline_animation.gif"
	duration_ms = int(1000 / max(1, fps))
	frames[0].save(
		gif_path,
		save_all=True,
		append_images=frames[1:],
		duration=duration_ms,
		loop=0,
		optimize=False,
	)
	plt.close(fig)
	return gif_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Create presentation visuals and one full pipeline animation.")
	parser.add_argument("input", type=Path, help="Input .wdf or .npz path")
	parser.add_argument("--out-dir", type=Path, default=Path("presentation_outputs"), help="Output directory")
	parser.add_argument("--spectrum-index", type=int, default=0, help="For ND data: flattened spectrum index")
	parser.add_argument("--hw-peak", type=int, default=10, help="Half-window for peak erosion/dilation")
	parser.add_argument("--hw-base", type=int, default=15, help="Half-window for baseline erosion")
	parser.add_argument("--step", type=int, default=3, help="Frame sampling step")
	parser.add_argument("--max-frames", type=int, default=160, help="Maximum frames per long phase")
	parser.add_argument("--fps", type=int, default=15, help="Animation FPS")
	args = parser.parse_args()

	args.out_dir.mkdir(parents=True, exist_ok=True)

	data = load_any(args.input)
	x = np.asarray(data.x)
	y = _choose_1d_spectrum(np.asarray(data.y), args.spectrum_index)

	print("Computing morphology pipeline...")
	run = run_pipeline(x=x, y=y, hw_peak=args.hw_peak, hw_base=args.hw_base)

	print("Saving static plots...")
	save_overview(run, args.out_dir)
	save_candidates(run, args.out_dir)
	save_filtered_peaks(run, args.out_dir)
	save_final_lines(run, args.out_dir)

	print("Saving combined animation (single GIF)...")
	gif_path = save_pipeline_video(
		run,
		args.out_dir,
		step=args.step,
		max_frames=args.max_frames,
		fps=args.fps,
	)

	print("Done.")
	print(f"Output dir: {args.out_dir.resolve()}")
	print(f"Peaks found: {len(run.peaks)}")
	print(f"Animation: {gif_path.name}")
	print("\nGenerated key files:")
	for name in [
		"01_overview.png",
		"02_candidates.png",
		"03_filtered_peaks.png",
		"04_final_lines.png",
		"05_pipeline_animation.gif",
	]:
		p = args.out_dir / name
		print(f" - {name}: {'OK' if p.exists() else 'missing'}")


if __name__ == "__main__":
	main()
