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


def test_wandb_project_defaults_to_repo_folder_and_env_overrides(monkeypatch) -> None:
    """Default wandb project = the repo folder name (so a copied template names itself); WANDB_PROJECT wins."""
    from pathlib import Path

    import rootutils
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    import src.utils  # noqa: F401  (registers the ${basename:...} resolver)

    root = rootutils.find_root(indicator=".project-root")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    for env, expected in ((None, Path(root).name), ("my-project", "my-project")):
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        if env:
            monkeypatch.setenv("WANDB_PROJECT", env)
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(config_name="train.yaml", overrides=["logger=wandb"])
            assert cfg.logger.wandb.project == expected
    GlobalHydra.instance().clear()
