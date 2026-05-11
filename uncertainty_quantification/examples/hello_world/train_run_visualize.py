"""Synthetic hello world for uq-mlip: train, predict, and plot.

This example avoids MACE/UMA downloads so every developer can verify the core
UQ workflow quickly. Backend-specific extraction smoke tests are handled by
scripts/run_hello_world.sh.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from uq_mlip.data import EmbeddingData, save_embeddings
from uq_mlip.model import UQModel


def make_bundle(n_frames: int, atoms_per_frame: int, shifted: bool = False) -> EmbeddingData:
    rng = np.random.default_rng(7 if not shifted else 11)
    features = []
    energies = []
    elements = []
    num_atoms = []

    for frame in range(n_frames):
        drift = frame / max(n_frames - 1, 1)
        if shifted and 0.55 <= drift <= 0.82:
            drift += 0.75

        for atom_idx in range(atoms_per_frame):
            element = 8 if atom_idx == 0 else 1
            local = atom_idx / max(atoms_per_frame - 1, 1)
            feature = np.array(
                [
                    drift,
                    local,
                    np.sin(2 * np.pi * drift),
                    1.0 if element == 8 else 0.25,
                ]
            )
            noise = 0.015 * rng.normal()
            energy = (
                0.7 * feature[0]
                + 0.2 * feature[1]
                + 0.4 * feature[2]
                + 0.3 * feature[3]
                + noise
            )
            features.append(feature)
            energies.append(energy)
            elements.append(element)
        num_atoms.append(atoms_per_frame)

    return EmbeddingData(
        node_feats=np.asarray(features, dtype=float),
        node_energies=np.asarray(energies, dtype=float),
        node_type=np.asarray(elements, dtype=int),
        num_atoms=np.asarray(num_atoms, dtype=int),
    )


def write_prediction_csv(bundle: EmbeddingData, predictions: dict[str, np.ndarray], path: Path) -> Path:
    sample_idx = np.concatenate(
        [np.full(n_atoms, frame) for frame, n_atoms in enumerate(bundle.num_atoms)]
    )
    atom_idx = np.concatenate([np.arange(n_atoms) for n_atoms in bundle.num_atoms])
    df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "atom_idx": atom_idx,
            "element": bundle.node_type,
            "uq_lower": predictions["lower"],
            "uq_upper": predictions["upper"],
            "uq": predictions["uncertainty"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return path


def plot_profile(csv_path: Path, output: Path) -> Path:
    df = pd.read_csv(csv_path)
    profile = (
        df.groupby("sample_idx")
        .agg(mean_uq=("uq", "mean"), max_uq=("uq", "max"))
        .reset_index()
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 940, 520
    left, right, top, bottom = 78, 34, 64, 74
    xs = profile["sample_idx"].to_numpy(dtype=float)
    mean = profile["mean_uq"].to_numpy(dtype=float)
    max_uq = profile["max_uq"].to_numpy(dtype=float)
    x_span = max(xs.max() - xs.min(), 1.0)
    y_max = max(float(max_uq.max()) * 1.15, 1e-6)

    def xmap(value):
        return left + (value - xs.min()) / x_span * (width - left - right)

    def ymap(value):
        return height - bottom - value / y_max * (height - top - bottom)

    mean_points = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, mean))
    max_points = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, max_uq))
    area_points = (
        " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, max_uq))
        + " "
        + " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs[::-1], mean[::-1]))
    )
    shifted_x = xmap(16)
    shifted_width = xmap(24) - shifted_x
    grid = "\n".join(
        f'<line x1="{left}" y1="{ymap(y):.1f}" x2="{width-right}" y2="{ymap(y):.1f}" stroke="#e6eef2" />'
        for y in [y_max * i / 4 for i in range(5)]
    )
    dots = "\n".join(
        f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="4" fill="#d73027" />'
        for x, y in zip(xs, max_uq)
    )
    title = "uq-mlip hello world: uncertainty profile"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfcfd" />
  <text x="{left}" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#24343d">{escape(title)}</text>
  {grid}
  <rect x="{shifted_x:.1f}" y="{top}" width="{shifted_width:.1f}" height="{height-top-bottom}" fill="#fdae61" opacity="0.18" />
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#6b7c85" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#6b7c85" />
  <polygon points="{area_points}" fill="#41b6c4" opacity="0.26" />
  <polyline points="{max_points}" fill="none" stroke="#d73027" stroke-width="1.5" opacity="0.65" />
  <polyline points="{mean_points}" fill="none" stroke="#225ea8" stroke-width="3.2" />
  {dots}
  <text x="{width / 2:.1f}" y="{height - 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#334">configuration index</text>
  <text x="24" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="15" fill="#334">predicted per-atom UQ</text>
  <rect x="{width - 294}" y="72" width="250" height="84" rx="8" fill="#ffffff" stroke="#d9e2e7" />
  <line x1="{width - 274}" y1="96" x2="{width - 238}" y2="96" stroke="#225ea8" stroke-width="3.2" />
  <text x="{width - 228}" y="101" font-family="Arial, sans-serif" font-size="13" fill="#334">mean atom UQ</text>
  <circle cx="{width - 256}" cy="120" r="4.5" fill="#d73027" />
  <text x="{width - 228}" y="125" font-family="Arial, sans-serif" font-size="13" fill="#334">max atom UQ</text>
  <rect x="{width - 276}" y="137" width="38" height="12" fill="#fdae61" opacity="0.25" />
  <text x="{width - 228}" y="148" font-family="Arial, sans-serif" font-size="13" fill="#334">shifted region</text>
</svg>
"""
    output.write_text(svg)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="examples/hello_world/outputs")
    parser.add_argument("--estimators", default=80, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)

    train_bundle = make_bundle(n_frames=32, atoms_per_frame=6, shifted=False)
    run_bundle = make_bundle(n_frames=32, atoms_per_frame=6, shifted=True)

    train_npz = save_embeddings(train_bundle, outdir / "embeddings" / "embedding_info_train.npz")
    run_npz = save_embeddings(run_bundle, outdir / "embeddings" / "embedding_info_run.npz")

    model = UQModel(outdir / "uq-model", n_estimators=args.estimators)
    model.fit(train_bundle)
    predictions = model.predict_embeddings(run_bundle)

    csv_path = write_prediction_csv(run_bundle, predictions, outdir / "results" / "UQ_synthetic_run.csv.gz")
    plot_path = plot_profile(csv_path, outdir / "uq_profile.svg")

    print(f"training embeddings: {train_npz}")
    print(f"run embeddings: {run_npz}")
    print(f"prediction csv: {csv_path}")
    print(f"visualization: {plot_path}")


if __name__ == "__main__":
    main()
