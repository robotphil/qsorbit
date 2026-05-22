"""Smoke test: verify the package imports and basic invariants hold."""

import qsorbit


def test_package_imports():
    """The qsorbit package should be importable."""
    assert qsorbit is not None


def test_version_is_set():
    """The package should expose a __version__ string."""
    assert hasattr(qsorbit, "__version__")
    assert isinstance(qsorbit.__version__, str)
    assert len(qsorbit.__version__) > 0
