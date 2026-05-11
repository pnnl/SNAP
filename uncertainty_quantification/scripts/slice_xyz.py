#!/usr/bin/env python3
"""Write a small complete-frame slice from an XYZ/extxyz trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_frame(handle, frame_number: int) -> tuple[list[str], int] | None:
    """Read one XYZ frame, preserving text exactly."""

    atom_count_line = handle.readline()
    if atom_count_line == "":
        return None

    try:
        atom_count = int(atom_count_line.strip())
    except ValueError as exc:
        raise ValueError(
            f"Frame {frame_number} starts with an invalid atom count: "
            f"{atom_count_line.rstrip()!r}"
        ) from exc

    comment_line = handle.readline()
    if comment_line == "":
        raise ValueError(f"Frame {frame_number} is missing its comment line.")

    atom_lines = []
    for atom_index in range(atom_count):
        atom_line = handle.readline()
        if atom_line == "":
            raise ValueError(
                f"Frame {frame_number} ended after {atom_index} atom lines; "
                f"expected {atom_count}."
            )
        atom_lines.append(atom_line)

    return [atom_count_line, comment_line, *atom_lines], atom_count


def slice_xyz(source: Path, output: Path, frames: int, start: int = 0) -> tuple[int, int]:
    """Copy complete XYZ frames from source to output."""

    if frames < 1:
        raise ValueError("--frames must be at least 1.")
    if start < 0:
        raise ValueError("--start must be non-negative.")

    output.parent.mkdir(parents=True, exist_ok=True)
    written_frames = 0
    written_atoms = 0

    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        frame_number = 0
        while written_frames < frames:
            frame = read_frame(src, frame_number)
            if frame is None:
                break

            lines, atom_count = frame
            if frame_number >= start:
                dst.writelines(lines)
                written_frames += 1
                written_atoms += atom_count

            frame_number += 1

    if written_frames < frames:
        raise ValueError(
            f"Requested {frames} frame(s) from start {start}, but only wrote "
            f"{written_frames}."
        )

    return written_frames, written_atoms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames, atoms = slice_xyz(args.source, args.output, args.frames, args.start)
    print(f"Wrote {frames} frame(s), {atoms} atom(s): {args.output}")


if __name__ == "__main__":
    main()
