"""Command line interface for uq-mlip."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from uq_mlip.backends import get_extractor
from uq_mlip.data import load_embeddings
from uq_mlip.model import UQModel


def _add_alpha_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upper-alpha", default=0.95, type=float)
    parser.add_argument("--lower-alpha", default=0.05, type=float)


def _model_kwargs(args: argparse.Namespace) -> dict:
    return {
        "lower_alpha": args.lower_alpha,
        "upper_alpha": args.upper_alpha,
        "n_estimators": getattr(args, "estimators", 100),
        "device": getattr(args, "device", "cpu"),
    }


def extract_command(args: argparse.Namespace) -> Path:
    kwargs = {
        "checkpoint": args.checkpoint,
        "device": args.device,
    }
    # Fall back to each backend's own default model when unspecified.
    if args.model_size is not None:
        kwargs["model"] = args.model_size
    if args.backend == "uma":
        kwargs["head"] = args.head
        kwargs["batch_size"] = args.batch_size
    elif args.backend == "chgnet":
        kwargs["batch_size"] = args.batch_size
        kwargs["on_isolated_atoms"] = args.on_isolated_atoms
    extractor = get_extractor(args.backend, **kwargs)
    output = extractor.extract_file(args.sample, args.savedir, index=args.index)
    print(output)
    return output


def train_command(args: argparse.Namespace) -> Path:
    model = UQModel.train_from_file(
        args.embeddings,
        args.savedir,
        **_model_kwargs(args),
    )
    print(model.model_path)
    return model.model_path


def predict_command(args: argparse.Namespace) -> Path:
    embeddings = load_embeddings(args.embeddings)
    model = UQModel.from_dir(args.model_dir, **_model_kwargs(args))
    predictions = model.predict_embeddings(embeddings)

    sample_idx = np.concatenate(
        [np.full(n_atoms, index) for index, n_atoms in enumerate(embeddings.num_atoms)]
    )
    atom_idx = np.concatenate([np.arange(n_atoms) for n_atoms in embeddings.num_atoms])
    df = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "atom_idx": atom_idx,
            "element": embeddings.node_type,
            "uq_lower": predictions["lower"],
            "uq_upper": predictions["upper"],
            "uq": predictions["uncertainty"],
        }
    )

    output_dir = Path(args.savedir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"UQ_{Path(args.embeddings).stem}.csv.gz"
    df.to_csv(output, index=False, compression="gzip")
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uq-mlip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract per-atom embeddings.")
    extract.add_argument("--backend", required=True, choices=["mace", "uma", "chgnet"])
    extract.add_argument("--sample", required=True)
    extract.add_argument("--savedir", required=True)
    extract.add_argument(
        "--model-size",
        default=None,
        help="Backend model identifier. Defaults per backend: mace=medium-0b, "
        "uma=uma-m-1p1, chgnet=0.3.0.",
    )
    extract.add_argument("--checkpoint", default=None)
    extract.add_argument("--index", default=":")
    extract.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    extract.add_argument("--batch-size", default=4, type=int)
    extract.add_argument(
        "--head",
        default="omat",
        choices=["oc20", "omat", "omol", "odac", "omc"],
    )
    extract.add_argument(
        "--on-isolated-atoms",
        default="warn",
        choices=["ignore", "warn", "error"],
        help="CHGNet only: how graph conversion handles atoms with no neighbor "
        "inside the cutoff.",
    )
    extract.set_defaults(func=extract_command)

    train = subparsers.add_parser("train", help="Train a UQ model.")
    train.add_argument("--embeddings", required=True)
    train.add_argument("--savedir", required=True)
    train.add_argument("--estimators", default=100, type=int)
    train.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    _add_alpha_args(train)
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict", help="Predict per-atom UQ from embeddings.")
    predict.add_argument("--embeddings", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--savedir", required=True)
    predict.add_argument("--estimators", default=100, type=int)
    predict.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    _add_alpha_args(predict)
    predict.set_defaults(func=predict_command)

    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
