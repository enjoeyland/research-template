import os

from src.cache_env import CACHE_VARS, default_cache_root, set_cache_defaults


def _clear(monkeypatch):
    for var in list(CACHE_VARS) + ["CACHE_DIR"]:
        monkeypatch.delenv(var, raising=False)


def test_fills_unset_variables_under_cache_dir(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    assert set_cache_defaults() == str(tmp_path)
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")


def test_never_overrides_what_is_already_set(monkeypatch, tmp_path) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    set_cache_defaults()
    assert os.environ["HF_HOME"] == "/somewhere/else"
    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")


def test_no_scratch_no_change(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("USER", "no-such-user-xyz")
    assert default_cache_root() is None
    assert set_cache_defaults() is None
    assert "HF_HOME" not in os.environ
