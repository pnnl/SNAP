"""Plot a per-frame UQ profile from a uq-mlip prediction CSV as SVG."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="UQ CSV produced by uq-mlip predict.")
    parser.add_argument("--output", required=True, help="Path to write the PNG plot.")
    parser.add_argument("--title", default="uq-mlip uncertainty profile")
    return parser


def main() -> Path:
    args = build_parser().parse_args()
    df = pd.read_csv(args.csv)
    profile = (
        df.groupby("sample_idx")
        .agg(mean_uq=("uq", "mean"), max_uq=("uq", "max"), atoms=("atom_idx", "count"))
        .reset_index()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 500
    left, right, top, bottom = 74, 28, 62, 70
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
    dots = "\n".join(
        f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="4.5" fill="#d73027" />'
        for x, y in zip(xs, max_uq)
    )
    grid = "\n".join(
        f'<line x1="{left}" y1="{ymap(y):.1f}" x2="{width-right}" y2="{ymap(y):.1f}" stroke="#e6eef2" />'
        for y in [y_max * i / 4 for i in range(5)]
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfcfd" />
  <text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#24343d">{escape(args.title)}</text>
  {grid}
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#6b7c85" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#6b7c85" />
  <polygon points="{area_points}" fill="#41b6c4" opacity="0.24" />
  <polyline points="{max_points}" fill="none" stroke="#d73027" stroke-width="1.6" opacity="0.7" />
  <polyline points="{mean_points}" fill="none" stroke="#225ea8" stroke-width="3.2" />
  {dots}
  <text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#334">configuration index</text>
  <text x="24" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="15" fill="#334">uncertainty</text>
  <rect x="{width - 250}" y="70" width="205" height="62" rx="8" fill="#ffffff" stroke="#d9e2e7" />
  <line x1="{width - 232}" y1="94" x2="{width - 196}" y2="94" stroke="#225ea8" stroke-width="3.2" />
  <text x="{width - 188}" y="99" font-family="Arial, sans-serif" font-size="13" fill="#334">mean atom UQ</text>
  <circle cx="{width - 214}" cy="116" r="4.5" fill="#d73027" />
  <text x="{width - 188}" y="121" font-family="Arial, sans-serif" font-size="13" fill="#334">max atom UQ</text>
</svg>
"""
    output.write_text(svg)
    return output


if __name__ == "__main__":
    print(main())
