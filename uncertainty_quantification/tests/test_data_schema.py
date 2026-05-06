import numpy as np
import pytest

from uq_mlip.data import EmbeddingData, load_embeddings, save_embeddings


def test_embedding_round_trip(tmp_path):
    bundle = EmbeddingData(
        node_feats=np.array([[0.0, 1.0], [1.0, 2.0]]),
        node_energies=np.array([0.2, 0.8]),
        node_type=np.array([8, 1]),
        num_atoms=np.array([2]),
    )

    path = save_embeddings(bundle, tmp_path / "embedding_info_test.npz")
    loaded = load_embeddings(path)

    np.testing.assert_allclose(loaded.node_feats, bundle.node_feats)
    np.testing.assert_allclose(loaded.node_energies, bundle.node_energies)
    np.testing.assert_array_equal(loaded.node_type, bundle.node_type)
    np.testing.assert_array_equal(loaded.num_atoms, bundle.num_atoms)


def test_embedding_schema_rejects_mismatched_atom_count():
    bundle = EmbeddingData(
        node_feats=np.array([[0.0, 1.0], [1.0, 2.0]]),
        node_energies=np.array([0.2, 0.8]),
        node_type=np.array([8, 1]),
        num_atoms=np.array([3]),
    )

    with pytest.raises(ValueError, match="sum\\(num_atoms\\)"):
        bundle.validate()
