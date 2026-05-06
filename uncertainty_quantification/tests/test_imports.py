def test_public_imports():
    import uq_mlip

    assert uq_mlip.UQCalculator is not None
    assert uq_mlip.UQModel is not None
    assert uq_mlip.with_uq is not None
