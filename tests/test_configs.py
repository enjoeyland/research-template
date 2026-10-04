import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_wandb_project_comes_from_env_then_project_name_then_repo_folder(monkeypatch) -> None:
    """WANDB_PROJECT > PROJECT_NAME (.env) > the repo folder name."""
    from pathlib import Path

    import rootutils
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    import src.utils  # noqa: F401  (registers the ${basename:...} resolver)

    root = rootutils.find_root(indicator=".project-root")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    cases = (
        ({}, Path(root).name),
        ({"PROJECT_NAME": "my-project"}, "my-project"),
        ({"PROJECT_NAME": "my-project", "WANDB_PROJECT": "other"}, "other"),
    )
    for env, expected in cases:
        for var in ("WANDB_PROJECT", "PROJECT_NAME"):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(config_name="train.yaml", overrides=["logger=wandb"])
            assert cfg.logger.wandb.project == expected, env
    GlobalHydra.instance().clear()


def test_every_experiment_config_composes_for_train_eval_and_analyze() -> None:
    """Experiment configs are shared by train.py / eval.py / analyze.py: a group an experiment overrides (e.g.
    /callbacks) must exist in all three defaults lists, or `experiment/train=<x>` fails with 'Could not override'."""
    import rootutils
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    root = rootutils.find_root(indicator=".project-root")
    exps = sorted(
        str(p.relative_to(root / "configs" / "experiment" / "train").with_suffix(""))
        for p in (root / "configs" / "experiment" / "train").rglob("*.yaml")
    )
    assert exps, "no experiment configs found"
    for config_name, extra in (("train.yaml", []), ("eval.yaml", ["ckpt_path=."]), ("analyze.yaml", [])):
        for exp in exps:
            GlobalHydra.instance().clear()
            with initialize(version_base="1.3", config_path="../configs"):
                cfg = compose(config_name=config_name, overrides=[f"experiment/train={exp}"] + extra)
                assert cfg.experiment_name == exp.rsplit("/", 1)[-1], (config_name, exp)
    GlobalHydra.instance().clear()


def test_resumable_callbacks_build_for_any_max_epochs(monkeypatch) -> None:
    """default_resumable has three ModelCheckpoints; Lightning rejects equal state_keys, which used to happen when
    max_epochs equalled the resume interval (10)."""
    import rootutils
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from lightning import Trainer

    from src.utils import instantiate_callbacks

    root = rootutils.find_root(indicator=".project-root")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    for max_epochs in (10, 11, 50):
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(config_name="train.yaml", overrides=["callbacks=default_resumable", f"trainer.max_epochs={max_epochs}"])
        callbacks = instantiate_callbacks(cfg.callbacks)
        Trainer(callbacks=callbacks, max_epochs=max_epochs, accelerator="cpu", logger=False)  # validates the callbacks
    GlobalHydra.instance().clear()
