import pytest

from src.utils import require_data_path


def test_require_data_path_ok(tmp_path) -> None:
    assert require_data_path(tmp_path) == tmp_path


def test_require_data_path_missing_names_the_fix(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.*ln -sfn"):
        require_data_path(tmp_path / "nope")


def test_require_data_path_dangling_symlink(tmp_path) -> None:
    link = tmp_path / "ds"
    link.symlink_to(tmp_path / "gone")
    with pytest.raises(FileNotFoundError, match="dangling symlink"):
        require_data_path(link)
    (tmp_path / "gone").mkdir()
    assert require_data_path(link) == link
