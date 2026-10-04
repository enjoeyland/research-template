"""Synthetic classification datamodule (no download) -- a placeholder to replace with real data."""

from typing import Optional, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split


class ToyDataModule(LightningDataModule):
    """Gaussian blobs: ``n_classes`` clusters in ``n_features`` dimensions.

    ``fold`` selects the train/val/test split (the sweep axis of scripts/sbatch/template.sh); the data
    itself is fixed.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_features: int = 16,
        n_classes: int = 3,
        train_val_test_split: Sequence[float] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        fold: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[TensorDataset] = None
        self.data_val: Optional[TensorDataset] = None
        self.data_test: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return
        hp = self.hparams
        g = torch.Generator().manual_seed(0)  # fixed data, independent of the training seed
        y = torch.randint(0, hp.n_classes, (hp.n_samples,), generator=g)
        centers = 3.0 * torch.randn(hp.n_classes, hp.n_features, generator=g)
        x = centers[y] + torch.randn(hp.n_samples, hp.n_features, generator=g)
        dataset = TensorDataset(x, y)
        split_g = torch.Generator().manual_seed(hp.fold)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, list(hp.train_val_test_split), generator=split_g
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)
