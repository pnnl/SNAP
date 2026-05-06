from uq_mlip.cli import build_parser


def test_extract_parser_accepts_default_backends():
    parser = build_parser()

    mace_args = parser.parse_args(
        [
            "extract",
            "--backend",
            "mace",
            "--sample",
            "validation.xyz",
            "--savedir",
            "embeddings",
        ]
    )
    uma_args = parser.parse_args(
        [
            "extract",
            "--backend",
            "uma",
            "--sample",
            "validation.xyz",
            "--savedir",
            "embeddings",
        ]
    )

    assert mace_args.backend == "mace"
    assert uma_args.backend == "uma"
