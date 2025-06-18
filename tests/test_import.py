def test_import():
    import maynard

    assert hasattr(maynard, "__version__")
