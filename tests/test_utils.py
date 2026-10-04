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


def test_link_checkpoints_dir_puts_the_symlink_in_the_experiment_folder(tmp_path) -> None:
    from omegaconf import OmegaConf

    from src.utils import link_checkpoints_dir

    exp = tmp_path / "logs" / "runs" / "exp1"
    long_term = tmp_path / "lustre" / "runs" / "exp1" / "checkpoints"
    cfg = OmegaConf.create({"paths": {"exp_dir": str(exp), "ckpt_dir": str(long_term)}})
    exp.mkdir(parents=True)
    link_checkpoints_dir(cfg)
    link_checkpoints_dir(cfg)  # idempotent
    assert (exp / "checkpoints").is_symlink() and (exp / "checkpoints").resolve() == long_term.resolve()
    assert long_term.is_dir()  # created, so the link never dangles

    # default layout: ckpt_dir already is <exp_dir>/checkpoints -> a real directory, no link
    exp2 = tmp_path / "logs" / "runs" / "exp2"
    exp2.mkdir(parents=True)
    cfg2 = OmegaConf.create({"paths": {"exp_dir": str(exp2), "ckpt_dir": str(exp2 / "checkpoints")}})
    link_checkpoints_dir(cfg2)
    assert not (exp2 / "checkpoints").is_symlink()
