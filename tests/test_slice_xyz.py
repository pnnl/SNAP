from pathlib import Path
import importlib.util


def load_slice_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "slice_xyz.py"
    spec = importlib.util.spec_from_file_location("slice_xyz", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_slice_xyz_writes_complete_frames(tmp_path):
    module = load_slice_module()
    source = tmp_path / "source.xyz"
    output = tmp_path / "slice.xyz"
    source.write_text(
        "\n".join(
            [
                "2",
                'Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3',
                "H 0 0 0",
                "O 0 0 1",
                "1",
                'Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3',
                "Si 0 0 0",
                "",
            ]
        )
    )

    frames, atoms = module.slice_xyz(source, output, frames=1, start=1)

    assert frames == 1
    assert atoms == 1
    assert output.read_text() == (
        '1\nLattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3\n'
        "Si 0 0 0\n"
    )
