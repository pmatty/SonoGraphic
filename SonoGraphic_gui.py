#!/usr/bin/env python3
"""
SonoGraphic: bar-aligned sonograms for a fixed-media staff.

Desktop GUI conversion of SonoGraphic_workbook.ipynb.
Dependencies:
    numpy scipy matplotlib pillow soundfile mido pdfplumber pymupdf PySide6
Install with:
    python -m pip install numpy scipy matplotlib pillow soundfile mido pdfplumber pymupdf PySide6
Run with:
    python sonographic_gui.py
"""

from __future__ import annotations

import bisect
import csv
import io
import sys
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.interpolate import interp1d

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Analysis functions (adapted from the notebook)
# ---------------------------------------------------------------------------

def bars_from_midi(path: str) -> list[float]:
    import mido

    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    tempos = {0: 500000}
    sigs = {0: (4, 4)}
    end_tick = 0
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                tempos[t] = msg.tempo
            elif msg.type == "time_signature":
                sigs[t] = (msg.numerator, msg.denominator)
        end_tick = max(end_tick, t)

    tl = sorted(tempos.items())
    tl_ticks = [t for t, _ in tl]
    cum = [0.0]
    for (t0, tp0), (t1, _) in zip(tl, tl[1:]):
        cum.append(cum[-1] + (t1 - t0) * tp0 / 1e6 / tpb)

    def seconds(tick: int) -> float:
        i = bisect.bisect_right(tl_ticks, tick) - 1
        return cum[i] + (tick - tl[i][0]) * tl[i][1] / 1e6 / tpb

    sl = sorted(sigs.items())
    sl_ticks = [t for t, _ in sl]
    boundaries = [0]
    t = 0
    while end_tick - t > tpb // 4:
        num, den = sl[bisect.bisect_right(sl_ticks, t) - 1][1]
        t += num * 4 * tpb // den
        boundaries.append(t)
    return [seconds(b) for b in boundaries]


def bars_from_csv(path: str, duration: float) -> list[float]:
    starts, end = [], None
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            key = row[0].strip().lower()
            try:
                value = float(row[1])
            except ValueError:
                continue
            if key == "end":
                end = value
            else:
                starts.append((int(float(key)), value))
    starts.sort()
    bounds = [v for _, v in starts]
    bounds.append(end if end is not None else duration)
    return bounds


def bars_constant(tempo: float, meter: str, n_bars: int) -> list[float]:
    num, den = (int(x) for x in meter.split("/"))
    bar = num * (4 / den) * 60.0 / tempo
    return [i * bar for i in range(n_bars + 1)]


def load_audio(path: str, channels: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    idx = [int(c.strip()) - 1 for c in channels.split(",") if c.strip()]
    if not idx or min(idx) < 0 or max(idx) >= data.shape[1]:
        raise ValueError(
            f"Audio has {data.shape[1]} channel(s); requested channel(s) {channels!r} are unavailable."
        )
    return data[:, idx].mean(axis=1), sr


def log_sonogram(x, sr, n_fft, hop, fmin, fmax, rows, floor_db, tilt, gamma):
    """Return v [rows, frames] in 0..1, low frequency first."""
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float32), (pad, pad + n_fft))
    n_frames = len(x) // hop + 1
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    grid = np.geomspace(fmin, min(fmax, sr / 2), rows).astype(np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    db = np.empty((rows, n_frames), np.float32)
    block = 1024
    for a in range(0, n_frames, block):
        b = min(a + block, n_frames)
        idx = (np.arange(a, b) * hop)[:, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(xp[idx] * win, axis=1)).astype(np.float32)
        d = 20 * np.log10(mag + 1e-9)
        db[:, a:b] = interp1d(
            f, d, axis=1, kind="linear", assume_sorted=True
        )(grid).T
    db -= db.max()
    if tilt:
        db += (tilt * np.log2(grid / 1000.0))[:, None].astype(np.float32)
        db -= db.max()
    v = np.clip(1.0 + db / floor_db, 0.0, 1.0) ** gamma
    return v.astype(np.float32), hop / sr


def render_bar(v, dt, t0, t1, px_per_s, style, cmap, width=None):
    width = width or max(1, round((t1 - t0) * px_per_s))
    tt = t0 + (np.arange(width) + 0.5) / width * (t1 - t0)
    pos = np.clip(tt / dt, 0, v.shape[1] - 1.001)
    i0 = pos.astype(int)
    w = (pos - i0).astype(np.float32)
    img = v[:, i0] * (1 - w) + v[:, i0 + 1] * w
    img = img[::-1]
    if cmap:
        from matplotlib import colormaps
        rgb = (colormaps[cmap](img)[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")
    if style == "transparent":
        rgba = np.zeros(img.shape + (4,), np.uint8)
        rgba[..., 3] = (img * 255).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")
    grey = (255 * (1 - img)).astype(np.uint8)
    return Image.fromarray(np.dstack([grey] * 3), "RGB")


def systems_from_pdf(path: str, label: str = "Tape"):
    import pdfplumber

    systems = []
    with pdfplumber.open(path) as pdf:
        for pi, page in enumerate(pdf.pages):
            hl = [l for l in page.lines
                  if abs(l["top"] - l["bottom"]) < 0.5 and (l["x1"] - l["x0"]) > 250]
            vl = [l for l in page.lines
                  if abs(l["x0"] - l["x1"]) < 0.5 and 4 < (l["bottom"] - l["top"]) < 12]
            for w in page.extract_words():
                if w["text"] != label or w["x0"] > 140:
                    continue
                cy = (w["top"] + w["bottom"]) / 2
                cands = [l for l in hl if abs(l["top"] - cy) < 6]
                if not cands:
                    continue
                line = min(cands, key=lambda l: abs(l["top"] - cy))
                y = line["top"]
                xs = sorted(l["x0"] for l in vl
                            if l["top"] - 0.5 <= y <= l["bottom"] + 0.5
                            and line["x0"] < l["x0"] <= line["x1"] + 1)
                merged = []
                for x in xs:
                    if not merged or x - merged[-1] > 4:
                        merged.append(x)
                end = merged[-1] if merged and line["x1"] - merged[-1] < 15 else line["x1"]
                inner = [x for x in merged if x < end - 4]
                systems.append({
                    "page": pi, "y": y, "x0": line["x0"], "x1": end,
                    "edges": [line["x0"]] + inner + [end]
                })
    return systems


def build_pdf_systems(v, dt, bounds, systems, first_bar, args, out):
    ppp = args.dpi / 72.0
    n_bars = len(bounds) - 1
    k = 0
    rows_out, previews = [], []
    for s, sysm in enumerate(systems):
        edges = list(sysm["edges"])
        n = len(edges) - 1
        mode = "RGBA" if args.style == "transparent" else "RGB"
        bg = (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)
        total_w = max(1, round((edges[-1] - edges[0]) * ppp))
        canvas = Image.new(mode, (total_w, round(args.height_mm / 25.4 * args.dpi)), bg)
        for j in range(n):
            if k + j >= n_bars:
                break
            left = edges[j] + (args.inset_pt if j == 0 else 0)
            w_px = max(1, round((edges[j + 1] - left) * ppp))
            img = render_bar(
                v, dt, bounds[k + j], bounds[k + j + 1], 0,
                args.style, args.cmap, width=w_px
            )
            canvas.paste(img.convert(mode), (round((left - edges[0]) * ppp), 0))
            if args.barlines:
                for dx in range(args.barlines):
                    xline = min(round((edges[j] - edges[0]) * ppp) + dx, total_w - 1)
                    for yy in range(canvas.height):
                        canvas.putpixel(
                            (xline, yy),
                            (0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0)
                        )
        a = first_bar + k
        b = first_bar + min(k + n, n_bars) - 1
        name = f"system_{s + 1:02d}_bars_{a:03d}-{b:03d}.png"
        target = out / name
        canvas.save(target, dpi=(args.dpi, args.dpi))
        rows_out.append([
            s + 1, sysm["page"] + 1, f"{sysm['y']:.1f}",
            f"{edges[0]:.1f}", f"{edges[-1]:.1f}",
            f"{(edges[-1] - edges[0]) / 72 * 25.4:.1f}", a, b, name
        ])
        previews.append((sysm, target))
        k += n
    with open(out / "systems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "system", "pdf_page", "staff_y_pt", "x_left_pt", "x_right_pt",
            "width_mm", "first_bar", "last_bar", "file"
        ])
        w.writerows(rows_out)
    return previews


def stamp_pdf(src_pdf, dst_pdf, previews, height_mm):
    import fitz

    doc = fitz.open(src_pdf)
    h = height_mm / 25.4 * 72
    for sysm, img in previews:
        page = doc[sysm["page"]]
        rect = fitz.Rect(
            sysm["x0"], sysm["y"] - h / 2,
            sysm["x1"], sysm["y"] + h / 2
        )
        page.insert_image(rect, filename=str(img), overlay=True)
    doc.save(dst_pdf)


def bar_range(path):
    a, b = path.stem.split("_bars_")[1].split("-")
    return int(a), int(b)


# ---------------------------------------------------------------------------
# Worker: performs the notebook's stages and sends completed outputs to GUI
# ---------------------------------------------------------------------------

class AnalysisWorker(QObject):
    stage = Signal(str, str, object)  # title, kind, payload
    log = Signal(str)
    progress = Signal(int)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def emit_log(self, message):
        self.log.emit(message)

    def run(self):
        try:
            c = self.cfg
            out = Path(c["out"]).expanduser()
            out.mkdir(parents=True, exist_ok=True)

            self.emit_log("Reading audio metadata…")
            audio_info = sf.info(c["audio"])
            audio_duration = audio_info.frames / audio_info.samplerate - c["offset"]
            if c["tempo_mode"] == "MIDI":
                bounds = bars_from_midi(c["midi"])
                tempo_source = "MIDI tempo map exported from Sibelius"
            elif c["tempo_mode"] == "CSV":
                bounds = bars_from_csv(c["csv"], audio_duration)
                tempo_source = "supplied tempo map"
            else:
                bounds = bars_constant(c["tempo"], c["meter"], c["n_bars"])
                tempo_source = "constant tempo"

            if len(bounds) < 2 or any(b <= a for a, b in zip(bounds, bounds[1:])):
                raise ValueError("The tempo map must contain at least two increasing boundaries.")

            n_bars = len(bounds) - 1
            durations = np.diff(bounds)
            self.emit_log(
                f"Audio: {audio_duration:.2f} s | tempo map: {n_bars} bars, "
                f"ends at {bounds[-1]:.2f} s"
            )
            if abs(bounds[-1] - audio_duration) > 0.5:
                self.emit_log(
                    f"WARNING: tempo map and audio differ by "
                    f"{bounds[-1] - audio_duration:+.2f} s. Check export or OFFSET."
                )
            if c["expected_duration"] and abs(bounds[-1] - c["expected_duration"]) > 0.5:
                self.emit_log(
                    f"WARNING: expected {c['expected_duration']:.0f} s, "
                    f"tempo map gives {bounds[-1]:.2f} s."
                )

            # 1. Bar duration plot
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 2.6))
            ax.bar(
                range(c["first_bar"], c["first_bar"] + n_bars),
                durations, color="#1F3A5F", width=0.9
            )
            ax.set_xlabel("bar number")
            ax.set_ylabel("bar duration (s)")
            ax.set_title(
                "Bar durations from the tempo map "
                "(steps mark tempo or metre changes)", fontsize=10
            )
            fig.tight_layout()
            duration_path = out / "bar_durations.png"
            fig.savefig(duration_path, dpi=160)
            plt.close(fig)
            self.stage.emit("01 · Tempo map and bar durations", "image", str(duration_path))
            self.progress.emit(15)

            # 2. Load and analyse audio
            self.emit_log("Loading selected audio channels and computing the sonogram…")
            x, sr = load_audio(c["audio"], c["channels"])
            if c["offset"] > 0:
                x = x[int(round(c["offset"] * sr)):]
            elif c["offset"] < 0:
                x = np.pad(x, (int(round(-c["offset"] * sr)), 0))

            rows = round(c["height_mm"] / 25.4 * c["dpi"])
            px_per_s = c["mm_per_second"] / 25.4 * c["dpi"]
            v, dt = log_sonogram(
                x, sr, c["n_fft"], c["hop"], c["fmin"], c["fmax"],
                rows, c["floor_db"], c["tilt"], c["gamma"]
            )
            self.emit_log(
                f"Sonogram: {v.shape[0]} frequency rows × {v.shape[1]} time frames"
            )
            self.progress.emit(40)

            # 3. Overview image
            self.emit_log("Creating whole-piece overview…")
            fig, axes = plt.subplots(4, 1, figsize=(11.7, 8.3), squeeze=False)
            total = bounds[-1]
            chunk = total / 4
            marks = c["marks"]
            for r, ax in enumerate(axes[:, 0]):
                a, b = r * chunk, (r + 1) * chunk
                ia = int(a / dt)
                ib = min(int(b / dt) + 1, v.shape[1])
                ax.imshow(
                    1 - v[::-1, ia:ib], aspect="auto", cmap="gray",
                    vmin=0, vmax=1,
                    extent=[ia * dt, ib * dt, np.log10(c["fmin"]), np.log10(c["fmax"])]
                )
                ax.set_xlim(a, b)
                ticks = [
                    (bounds[i], c["first_bar"] + i)
                    for i in range(len(bounds) - 1)
                    if a - 1e-6 <= bounds[i] < b
                    and (c["first_bar"] + i) % c["bar_number_every"] == 0
                ]
                ax.set_xticks([t for t, _ in ticks])
                ax.set_xticklabels([str(n) for _, n in ticks], fontsize=7)
                freqs = [f for f in (100, 1000, 10000) if c["fmin"] <= f <= c["fmax"]]
                ax.set_yticks([np.log10(f) for f in freqs])
                ax.set_yticklabels(
                    [f"{f // 1000}k" if f >= 1000 else str(f) for f in freqs],
                    fontsize=7
                )
                for i in range(len(bounds) - 1):
                    if a <= bounds[i] < b:
                        ax.axvline(bounds[i], color="#1F3A5F", lw=0.3, alpha=0.5)
                for label, bar in marks.items():
                    i = bar - c["first_bar"]
                    if 0 <= i < len(bounds) - 1 and a <= bounds[i] < b:
                        ax.axvline(bounds[i], color="#1F3A5F", lw=1.2)
                        ax.text(
                            bounds[i], np.log10(c["fmax"]), label,
                            color="#1F3A5F", fontsize=9,
                            fontweight="bold", va="bottom", ha="left"
                        )
            axes[-1, 0].set_xlabel("bar number", fontsize=8)
            fig.tight_layout()
            overview_path = out / "overview.png"
            fig.savefig(overview_path, dpi=200)
            plt.close(fig)
            self.stage.emit("02 · Whole-piece sonogram overview", "image", str(overview_path))
            self.progress.emit(55)

            # 4. One image per bar and manifest
            self.emit_log("Rendering one sonogram image per bar…")
            images, manifest = [], []
            for i in range(n_bars):
                t0, t1 = bounds[i], bounds[i + 1]
                img = render_bar(
                    v, dt, t0, t1, px_per_s, c["style"], c["cmap"]
                )
                number = c["first_bar"] + i
                name = f"bar_{number:03d}.png"
                img.save(out / name, dpi=(c["dpi"], c["dpi"]))
                images.append(img)
                manifest.append([
                    number, f"{t0:.3f}", f"{t1:.3f}", f"{t1 - t0:.3f}",
                    name, f"{img.width / c['dpi'] * 25.4:.1f}"
                ])
                if i % max(1, n_bars // 10) == 0:
                    self.progress.emit(55 + int(15 * (i + 1) / n_bars))
            with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["bar", "start_s", "end_s", "duration_s", "file", "width_mm"])
                w.writerows(manifest)

            # Make a compact preview of first four bars
            strip = images[:4]
            if strip:
                sheet = Image.new(
                    "RGB",
                    (sum(i.width for i in strip) + 20 * len(strip), strip[0].height),
                    "white"
                )
                x0 = 0
                for im in strip:
                    rgba = im.convert("RGBA")
                    sheet.paste(rgba, (x0, 0), rgba)
                    x0 += im.width + 20
                sheet.thumbnail((1600, 500))
                bar_preview = out / "bar_images_preview.png"
                sheet.save(bar_preview)
                self.stage.emit(
                    f"03 · Individual bar images ({n_bars} bars)", "image",
                    str(bar_preview)
                )
            self.emit_log(f"Wrote {n_bars} bar images and manifest.csv to {out}")
            self.progress.emit(72)

            # 5. Score-aligned system images (optional)
            previews, systems = [], []
            if c["pdf"]:
                self.emit_log("Reading Tape staff barlines from score PDF…")
                systems = systems_from_pdf(c["pdf"])
                pdf_bars = sum(len(s["edges"]) - 1 for s in systems)
                self.emit_log(
                    f"Score PDF: {len(systems)} systems, {pdf_bars} bars on the Tape staff"
                )
                if pdf_bars != n_bars:
                    self.emit_log(
                        f"WARNING: the PDF has {pdf_bars} bars and the tempo map has {n_bars}."
                    )
                cfg = SimpleNamespace(
                    dpi=c["dpi"], style=c["style"], height_mm=c["height_mm"],
                    cmap=c["cmap"], inset_pt=c["inset_pt"], barlines=c["barlines"]
                )
                previews = build_pdf_systems(
                    v, dt, bounds, systems, c["first_bar"], cfg, out
                )
                self.emit_log(f"Wrote {len(previews)} system images and systems.csv")
                if previews:
                    preview_paths = [p for _, p in previews]
                    # Contact sheet, preserving readable thumbnails
                    thumbs = []
                    for p in preview_paths:
                        im = Image.open(p).convert("RGBA")
                        im.thumbnail((1200, 180))
                        thumbs.append((p.name, im.copy()))
                    sheet_w = max(im.width for _, im in thumbs)
                    sheet_h = sum(im.height + 28 for _, im in thumbs)
                    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(sheet)
                    y = 0
                    for name, im in thumbs:
                        draw.text((4, y), name, fill="black")
                        y += 20
                        sheet.paste(im, (0, y), im)
                        y += im.height + 8
                    system_preview = out / "system_images_preview.png"
                    sheet.save(system_preview)
                    self.stage.emit(
                        f"04 · Score-aligned system images ({len(previews)} systems)",
                        "image", str(system_preview)
                    )
                self.progress.emit(84)

                # 6. Stamp strips on PDF and render relevant page
                stamped = out / "score_with_strips.pdf"
                stamp_pdf(c["pdf"], str(stamped), previews, c["height_mm"])
                self.emit_log(f"Wrote {stamped}")
                import fitz
                doc = fitz.open(str(stamped))
                selected = None
                for sysm, p in previews:
                    a, b = bar_range(p)
                    if a <= c["check_bar"] <= b:
                        selected = sysm["page"]
                        break
                if selected is None and previews:
                    selected = previews[0][0]["page"]
                if selected is not None:
                    pix = doc[selected].get_pixmap(dpi=110, alpha=False)
                    qimg = QImage(
                        pix.samples, pix.width, pix.height, pix.stride,
                        QImage.Format.Format_RGB888
                    ).copy()
                    self.stage.emit(
                        f"05 · Score check (PDF page {selected + 1})",
                        "qimage", qimg
                    )
                doc.close()
            else:
                self.emit_log("No score PDF selected; skipping system alignment and stamped PDF.")

            # 7. Reproducibility / method record
            method = (
                f"Sonograms were computed from the mean of channels {c['channels']} "
                f"of the {sr} Hz tape file,\n"
                f"using a Hann window of {c['n_fft']} samples "
                f"({c['n_fft'] / sr * 1000:.0f} ms) and a step of {c['hop']} samples "
                f"({c['hop'] / sr * 1000:.1f} ms).\n"
                f"Magnitudes are shown in decibels relative to the peak of the whole file, "
                f"on a range of {c['floor_db']} dB,\n"
                f"with a logarithmic frequency axis from {c['fmin']} Hz to "
                f"{c['fmax'] / 1000:g} kHz. Dark areas are loud.\n"
                f"Bar boundaries come from the {tempo_source} "
                f"({n_bars} bars, {bounds[-1]:.1f} s). "
                f"Images are cropped to each bar and set at {c['dpi']} dpi."
            )
            (out / "method.txt").write_text(method, encoding="utf-8")
            self.stage.emit("06 · Method record", "text", method)
            self.progress.emit(100)
            self.emit_log("Analysis complete.")
            self.finished.emit()

        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# GUI widgets
# ---------------------------------------------------------------------------

class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.toggle = QPushButton("▾  " + title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setStyleSheet(
            "QPushButton { text-align: left; font-weight: 600; padding: 9px; "
            "border: 1px solid #c8c8c8; border-radius: 5px; background: #f3f3f3; }"
        )
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.content.setVisible)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)


class FilePicker(QWidget):
    # Class-level variable to store the shared initial directory across all pickers
    last_directory = str(Path.home())

    def __init__(self, label, filters="", save=False, directory=False):
        super().__init__()
        self.filters = filters
        self.save = save
        self.directory = directory
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Optional" if "optional" in label.lower() else "Choose a file…")
        self.button = QPushButton("Browse")
        self.button.setStyleSheet(
            "QPushButton { background-color: #3c3f41; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #4c5052; }"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        self.button.clicked.connect(self.browse)

    def browse(self):
        # Priority: 1. Text already in the box -> 2. Last folder picked anywhere -> 3. Home directory
        current_text = self.edit.text().strip()
        if current_text:
            start_dir = current_text if Path(current_text).is_dir() else str(Path(current_text).parent)
        else:
            start_dir = FilePicker.last_directory

        if self.directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", start_dir)
        elif self.save:
            value, _ = QFileDialog.getSaveFileName(self, "Choose file", start_dir, self.filters)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose file", start_dir, self.filters)

        if value:
            self.edit.setText(value)
            # Store folder path globally for future file picks
            p = Path(value)
            FilePicker.last_directory = str(p if p.is_dir() else p.parent)

    def text(self):
        return self.edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SonoGraphic")
        self.resize(1500, 950)
        self.thread = None
        self.worker = None
        self._build_ui()
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background-color: #2b2b2b; 
                color: #ffffff; 
            }
            QLabel { 
                color: #ffffff; 
            }
            QGroupBox { 
                font-weight: 600; 
                border: 1px solid #555555;
                border-radius: 6px; 
                margin-top: 10px; 
                padding: 8px; 
                color: #ffffff; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 4px; 
                color: #ffffff; 
            }
            QPushButton { 
                padding: 7px 10px; 
                background-color: #3c3f41; 
                color: #ffffff; 
                border: 1px solid #555555; 
                border-radius: 4px; 
            }
            QPushButton:hover { 
                background-color: #4c5052; 
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                padding: 5px; 
                border: 1px solid #555555; 
                border-radius: 4px;
                background-color: #ffffff; 
                color: #000000;
            }
        """)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        header = QHBoxLayout()
        title = QLabel("SonoGraphic")
        title.setStyleSheet("font-size: 23px; font-weight: 700; color: #1f3a5f;")
        subtitle = QLabel("Bar-aligned sonograms for a fixed-media staff")
        subtitle.setStyleSheet("color: #666;")
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(subtitle)
        header.addStretch()
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # LEFT: inputs and settings
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        files = QGroupBox("Input files")
        ff = QFormLayout(files)
        self.audio = FilePicker("Audio", "Audio (*.wav *.aiff *.aif *.flac);;All files (*)")
        self.midi = FilePicker("MIDI", "MIDI (*.mid *.midi);;All files (*)")
        self.csv = FilePicker("CSV", "CSV (*.csv);;All files (*)")
        self.pdf = FilePicker("Score PDF", "PDF (*.pdf);;All files (*)")
        ff.addRow("Audio file", self.audio)
        ff.addRow("MIDI file", self.midi)
        ff.addRow("Tempo CSV", self.csv)
        ff.addRow("Score PDF", self.pdf)
        left_layout.addWidget(files)

        tempo = QGroupBox("Tempo map")
        tf = QFormLayout(tempo)
        self.tempo_mode = QComboBox()
        self.tempo_mode.addItems(["MIDI", "CSV", "Constant"])
        self.tempo = QDoubleSpinBox(); self.tempo.setRange(1, 500); self.tempo.setValue(54)
        self.meter = QLineEdit("4/4")
        self.n_bars = QSpinBox(); self.n_bars.setRange(1, 10000); self.n_bars.setValue(130)
        tf.addRow("Source", self.tempo_mode)
        tf.addRow("Constant tempo (qpm)", self.tempo)
        tf.addRow("Metre", self.meter)
        tf.addRow("Number of bars", self.n_bars)
        left_layout.addWidget(tempo)

        score = QGroupBox("Score and timing")
        sfm = QFormLayout(score)
        self.first_bar = QSpinBox(); self.first_bar.setRange(-10000, 10000); self.first_bar.setValue(1)
        self.offset = QDoubleSpinBox(); self.offset.setRange(-3600, 3600); self.offset.setDecimals(3); self.offset.setValue(0)
        self.expected = QDoubleSpinBox(); self.expected.setRange(0, 100000); self.expected.setDecimals(1); self.expected.setValue(400)
        self.expected_check = QCheckBox("Check expected duration"); self.expected_check.setChecked(True)
        self.check_bar = QSpinBox(); self.check_bar.setRange(-10000, 10000); self.check_bar.setValue(28)
        self.marks = QLineEdit("A:7, B:15, C:24, D:50, E:55, F:65, G:86, H:98, I:110")
        sfm.addRow("First bar number", self.first_bar)
        sfm.addRow("Audio offset (s)", self.offset)
        sfm.addRow(self.expected_check, self.expected)
        sfm.addRow("Score check bar", self.check_bar)
        sfm.addRow("Rehearsal marks", self.marks)
        left_layout.addWidget(score)

        son = QGroupBox("Sonogram settings")
        sform = QFormLayout(son)
        self.channels = QLineEdit("1,2")
        self.n_fft = QSpinBox(); self.n_fft.setRange(128, 65536); self.n_fft.setSingleStep(256); self.n_fft.setValue(4096)
        self.hop = QSpinBox(); self.hop.setRange(16, 16384); self.hop.setValue(256)
        self.fmin = QDoubleSpinBox(); self.fmin.setRange(1, 20000); self.fmin.setValue(40)
        self.fmax = QDoubleSpinBox(); self.fmax.setRange(10, 96000); self.fmax.setValue(16000)
        self.floor = QDoubleSpinBox(); self.floor.setRange(1, 160); self.floor.setValue(70)
        self.tilt = QDoubleSpinBox(); self.tilt.setRange(-60, 60); self.tilt.setDecimals(2); self.tilt.setValue(0)
        self.gamma = QDoubleSpinBox(); self.gamma.setRange(0.05, 5); self.gamma.setDecimals(2); self.gamma.setValue(1)
        sform.addRow("Channels to average", self.channels)
        sform.addRow("FFT window", self.n_fft)
        sform.addRow("Hop size", self.hop)
        sform.addRow("Minimum frequency (Hz)", self.fmin)
        sform.addRow("Maximum frequency (Hz)", self.fmax)
        sform.addRow("Dynamic range (dB)", self.floor)
        sform.addRow("Spectral tilt (dB/oct)", self.tilt)
        sform.addRow("Gamma", self.gamma)
        left_layout.addWidget(son)

        image = QGroupBox("Image and engraving")
        imf = QFormLayout(image)
        self.dpi = QSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(600)
        self.height = QDoubleSpinBox(); self.height.setRange(1, 100); self.height.setValue(14)
        self.mmps = QDoubleSpinBox(); self.mmps.setRange(0.1, 1000); self.mmps.setValue(12)
        self.style = QComboBox(); self.style.addItems(["ink", "transparent"])
        self.cmap = QLineEdit("")
        self.inset = QDoubleSpinBox(); self.inset.setRange(0, 100); self.inset.setValue(15)
        self.barlines = QSpinBox(); self.barlines.setRange(0, 20); self.barlines.setValue(0)
        self.every = QSpinBox(); self.every.setRange(1, 100); self.every.setValue(5)
        imf.addRow("DPI", self.dpi)
        imf.addRow("Image height (mm)", self.height)
        imf.addRow("Bar width (mm/s)", self.mmps)
        imf.addRow("Style", self.style)
        imf.addRow("Matplotlib colour map (optional)", self.cmap)
        imf.addRow("System first-bar inset (pt)", self.inset)
        imf.addRow("System barline width (px)", self.barlines)
        imf.addRow("Overview bar-number interval", self.every)
        left_layout.addWidget(image)

        output = QGroupBox("Output")
        of = QFormLayout(output)
        self.out = FilePicker("Output folder", "", directory=True)
        self.out.edit.setText(str(Path.cwd() / "strips"))
        of.addRow("Save results to", self.out)
        left_layout.addWidget(output)

        self.run_button = QPushButton("▶  Run analysis")
        self.run_button.setStyleSheet(
            "QPushButton { background: #1f3a5f; color: white; font-weight: 700; "
            "padding: 12px; border-radius: 5px; }"
            "QPushButton:disabled { background: #999; }"
        )
        self.run_button.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(390)
        left_scroll.setMaximumWidth(520)
        splitter.addWidget(left_scroll)

        # RIGHT: outputs and log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_header = QHBoxLayout()
        out_title = QLabel("Analysis outputs")
        out_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        right_header.addWidget(out_title)
        right_header.addStretch()
        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.clicked.connect(self.open_output_folder)
        right_header.addWidget(self.open_folder_button)
        right_layout.addLayout(right_header)

        self.output_scroll = QScrollArea()
        self.output_scroll.setWidgetResizable(True)
        self.output_container = QWidget()
        self.output_layout = QVBoxLayout(self.output_container)
        self.output_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.output_layout.setSpacing(10)
        self.output_scroll.setWidget(self.output_container)
        right_layout.addWidget(self.output_scroll, 1)

        log_box = QGroupBox("Process log")
        log_layout = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        log_layout.addWidget(self.log)
        right_layout.addWidget(log_box)
        splitter.addWidget(right)
        splitter.setSizes([440, 1000])

        self.tempo_mode.currentTextChanged.connect(self._update_tempo_fields)
        self._update_tempo_fields()

    def _update_tempo_fields(self):
        mode = self.tempo_mode.currentText()
        self.midi.setEnabled(mode == "MIDI")
        self.csv.setEnabled(mode == "CSV")
        self.tempo.setEnabled(mode == "Constant")
        self.meter.setEnabled(mode == "Constant")
        self.n_bars.setEnabled(mode == "Constant")

    def _parse_marks(self):
        result = {}
        raw = self.marks.text().strip()
        if not raw:
            return result
        for item in raw.split(","):
            if ":" not in item:
                raise ValueError(f"Invalid rehearsal mark {item!r}. Use e.g. A:7, B:15.")
            label, bar = item.split(":", 1)
            result[label.strip()] = int(bar.strip())
        return result

    def _config(self):
        mode = self.tempo_mode.currentText()
        audio = self.audio.text()
        if not audio:
            raise ValueError("Please select an audio file.")
        if mode == "MIDI" and not self.midi.text():
            raise ValueError("Please select a MIDI tempo map, or change the tempo map source.")
        if mode == "CSV" and not self.csv.text():
            raise ValueError("Please select a tempo CSV, or change the tempo map source.")
        if self.fmin.value() >= self.fmax.value():
            raise ValueError("Minimum frequency must be lower than maximum frequency.")
        if self.n_fft.value() <= 0 or self.hop.value() <= 0:
            raise ValueError("FFT window and hop size must be positive.")
        return {
            "audio": audio, "midi": self.midi.text(), "csv": self.csv.text(),
            "pdf": self.pdf.text(), "out": self.out.text() or "strips",
            "tempo_mode": mode, "tempo": self.tempo.value(),
            "meter": self.meter.text().strip(), "n_bars": self.n_bars.value(),
            "first_bar": self.first_bar.value(), "offset": self.offset.value(),
            "expected_duration": self.expected.value() if self.expected_check.isChecked() else None,
            "marks": self._parse_marks(), "check_bar": self.check_bar.value(),
            "channels": self.channels.text().strip(),
            "n_fft": self.n_fft.value(), "hop": self.hop.value(),
            "fmin": self.fmin.value(), "fmax": self.fmax.value(),
            "floor_db": self.floor.value(), "tilt": self.tilt.value(),
            "gamma": self.gamma.value(), "dpi": self.dpi.value(),
            "height_mm": self.height.value(), "mm_per_second": self.mmps.value(),
            "style": self.style.currentText(), "cmap": self.cmap.text().strip() or None,
            "inset_pt": self.inset.value(), "barlines": self.barlines.value(),
            "bar_number_every": self.every.value(),
        }

    def start_analysis(self):
        try:
            cfg = self._config()
        except Exception as exc:
            QMessageBox.warning(self, "Check settings", str(exc))
            return

        self.clear_outputs()
        self.log.clear()
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")

        self.thread = QThread()
        self.worker = AnalysisWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.stage.connect(self.add_stage)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def clear_outputs(self):
        while self.output_layout.count():
            item = self.output_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_stage(self, title, kind, payload):
        section = CollapsibleSection(title)
        if kind == "image":
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(160)
            pix = QPixmap(payload)
            label.setPixmap(pix.scaled(
                1100, 650, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            label.setToolTip(payload)
            section.addWidget(label)
            path_label = QLabel(str(payload))
            path_label.setStyleSheet("color: #777; font-size: 10px;")
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            section.addWidget(path_label)
            save_button = QPushButton("Open image")
            save_button.clicked.connect(lambda checked=False, p=payload: self._open_path(p))
            section.addWidget(save_button)
        elif kind == "qimage":
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(QPixmap.fromImage(payload).scaled(
                1100, 700, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            section.addWidget(label)
        elif kind == "text":
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(payload)
            text.setMinimumHeight(130)
            section.addWidget(text)
            copy_button = QPushButton("Copy method text")
            copy_button.clicked.connect(lambda checked=False, t=payload: QApplication.clipboard().setText(t))
            section.addWidget(copy_button)

        self.output_layout.addWidget(section)
        self.output_scroll.verticalScrollBar().setValue(
            self.output_scroll.verticalScrollBar().maximum()
        )

    def append_log(self, message):
        self.log.append(message)

    def analysis_finished(self):
        self.run_button.setEnabled(True)
        self.run_button.setText("▶  Run analysis")
        self.append_log("All outputs have been written to: " + (self.out.text() or "strips"))
        QMessageBox.information(self, "SonoGraphic", "Analysis complete.")

    def analysis_failed(self, message):
        self.run_button.setEnabled(True)
        self.run_button.setText("▶  Run analysis")
        self.append_log(message)
        QMessageBox.critical(
            self, "Analysis failed",
            "The analysis stopped because of an error. The full traceback is in the Process log."
        )

    def _open_path(self, path):
        import subprocess
        import os
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])

    def open_output_folder(self):
        path = self.out.text().strip()
        if path and Path(path).exists():
            self._open_path(path)
        else:
            QMessageBox.information(self, "Output folder", "The output folder has not been created yet.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SonoGraphic")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
