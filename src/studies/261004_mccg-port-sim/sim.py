"""Simulation: does the MCCG SCM-cascade loss / metric structure fit the new src/losses + src/metrics design?

A miniature of ``TabImgSCMCascadeModule`` (which is a 1824-line module whose 580-line ``forward`` computes the loss
inside nested closures) rebuilt on the template's ``ModelOutput`` / ``CompositeLoss`` / ``TaskMetrics``. It keeps the
features that made MCCG's loss code hard to structure:

  * 8 weighted loss terms, some training-only (margin loss) or zero-weight,
  * a ground-truth pass and a self-predicted pass blended with weight ``w`` (``PassBlend``),
  * terms computed by the model itself (KL / orthogonality regularizers) and diagnostics that must NOT enter the loss,
  * a second metric stream (``gt_preds``) that only exists in training,
  * ``continuous_loss`` chosen by a string flag in MCCG -> chosen by ``_target_`` here,
  * per-column concept accuracy that MCCG reads through a forward hook (``_capture_cat_logits``).

The real MCCG functions (``pairwise_margin_loss``, ``continuous_column_loss``, ``soft_cross_entropy``,
``reconcbm_concept_bins``) are loaded from the MCCG checkout and wrapped as terms / metrics, so the numbers are the
real ones. Checks print PASS/FAIL; the report goes to ``logs/studies/261004_mccg-port-sim/report.json``.

Run through srun/sbatch (CLAUDE.md §1):  python src/studies/261004_mccg-port-sim/sim.py
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.losses import CompositeLoss, LossTerm  # noqa: E402
from src.metrics import MetricGroup, MetricHandler  # noqa: E402
from src.utils.callbacks import MetricTrends  # noqa: E402
from src.utils.model_output import ModelOutput, get_field  # noqa: E402

MCCG = Path("/home/khmin1104/workspace/Medical-CausalInference/medical_concept_causal_graph/src")
OUT = ROOT / "logs" / "studies" / "261004_mccg-port-sim"


def _load(rel: str, name: str):
    """Load one MCCG module by file path (they only depend on torch), without importing MCCG's ``src`` package."""
    spec = importlib.util.spec_from_file_location(name, MCCG / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pm = _load("models/components/pairwise_margin.py", "mccg_pairwise_margin")
cgt = _load("models/components/concept_ground_truth.py", "mccg_concept_ground_truth")
cacc = _load("metrics/concept_accuracy.py", "mccg_concept_accuracy")

CAT_SIZES = [3, 4, 5, 2]  # columns 0-2 categorical; column 3 is continuous with k=2 anchors (MCCG §25/§28)
CONT = (3,)
J, P, D, NCLS, HID = len(CAT_SIZES), 4, 6, 3, 32
PRIORS = [0.5, 0.3, 0.2]


# ----------------------------------------------------------------------------------------------- loss terms (new)
class CatRecon(LossTerm):
    """Masked per-column reconstruction loss of ``_loss_cat`` (MCCG), as a term. The continuous-column loss is
    selected by the subclass (``_target_``), not by a string flag + ``assert``."""

    requires = ("cat_logits", "input_mask")
    mode = "soft_ce"

    def __init__(self, continuous_columns=CONT) -> None:
        super().__init__()
        self.continuous = set(int(c) for c in continuous_columns)

    def forward(self, outputs, batch):
        cat_logits, mask = get_field(outputs, "cat_logits"), get_field(outputs, "input_mask")
        cols, has = [], []
        for j, lg in enumerate(cat_logits):
            m = mask[:, j]
            safe = torch.where(m.unsqueeze(-1), lg, torch.zeros_like(lg))  # MCCG's NaN-safe masking
            if j in self.continuous:
                v = batch["cat_value"][:, j]
                if self.mode == "soft_ce":
                    per = cgt.soft_cross_entropy(safe, torch.stack([1 - v, v], dim=-1))
                else:
                    per = cgt.continuous_column_loss(safe, v, self.mode)
            else:
                per = cgt.soft_cross_entropy(safe, F.one_hot(batch["cat"][:, j], lg.shape[-1]).to(lg.dtype))
            mf = m.float()
            denom = mf.sum()
            cols.append((per * mf).sum() / denom.clamp(min=1e-6))
            has.append((denom > 0).float())
        cols, has = torch.stack(cols), torch.stack(has)
        return (cols * has).sum() / has.sum().clamp(min=1.0)


class CatReconSoftCE(CatRecon):
    mode = "soft_ce"


class CatReconMSE(CatRecon):
    mode = "mse"


class CatReconGNLL(CatRecon):
    mode = "gaussian_nll"


class TaskMargin(LossTerm):
    """``_loss_task``: pairwise margin loss in training, plain CE at eval (MCCG applies the margin only when
    ``self.training``). The class priors are THIS term's state (config), not something the model passes."""

    requires = ("logits", "target")

    def __init__(self, class_priors, tau_pos: float = 1.0, tau_neg: float = 1.0, adaptive: float = 0.0) -> None:
        super().__init__()
        pos, neg = pm.build_margin_terms(torch.tensor(list(class_priors)), tau_pos, tau_neg, adaptive)
        self.register_buffer("pos", pos)
        self.register_buffer("neg", neg)

    def forward(self, outputs, batch):
        logits, target = get_field(outputs, "logits"), get_field(outputs, "target")
        if self.training:
            return pm.pairwise_margin_loss(logits, target, self.pos, self.neg)
        return F.cross_entropy(logits, target)


class MaskedPatchMSE(LossTerm):
    requires = ("recon_image", "patch_mask")

    def forward(self, outputs, batch):
        recon, mask = get_field(outputs, "recon_image"), get_field(outputs, "patch_mask").float()
        per_patch = ((recon - batch["image"]) ** 2).mean(-1)
        return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------------------------- metrics (new)
class ConceptAccuracy(MeanMetric, MetricHandler):
    """Per-step mean concept accuracy from ``outputs["cat_logits"]`` -- MCCG gets these through a forward hook
    (``_capture_cat_logits``) because its forward never returns them."""

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        logits = get_field(outputs, "cat_logits")
        if logits is None:
            return None
        pred, true = cacc.reconcbm_concept_bins(logits, batch["cat"], batch["cat_value"], set(CONT))
        self.update((pred == true).float().mean())
        return self

    def on_epoch_end(self, split, *args, **kwargs):
        return self.compute() if self.update_called else None


class ConceptAccGroup(MetricGroup):
    def init_metrics(self) -> None:
        for split_dict in (self.metrics_train, self.metrics_valid, self.metrics_test):
            split_dict["concept/acc_mean"] = ConceptAccuracy()


# ------------------------------------------------------------------------------------------------- model / data
class SimData(Dataset):
    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.cat = torch.stack([torch.randint(0, k, (n,), generator=g) for k in CAT_SIZES], dim=1)
        self.cat[:, 3] = 0
        self.cat_value = torch.zeros(n, J)
        self.cat_value[:, 3] = torch.rand(n, generator=g)
        self.y = (self.cat[:, 0] + self.cat[:, 1]) % NCLS
        self.image = torch.randn(n, P, D, generator=g) + self.y.view(-1, 1, 1).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return {"image": self.image[i], "cat": self.cat[i], "cat_value": self.cat_value[i], "y": self.y[i]}


class SimDM(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(SimData(256, 0), batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(SimData(64, 1), batch_size=32)

    def test_dataloader(self):
        return DataLoader(SimData(64, 2), batch_size=32)


class MiniCascade(LightningModule):
    """Forward only: returns ONE ModelOutput per step. No loss closures, no hooks, no logged-name filtering."""

    def __init__(self, loss: CompositeLoss, metrics) -> None:
        super().__init__()
        self.loss_fn, self.metrics = loss, metrics
        self.enc = nn.Sequential(nn.Linear(P * D + J, HID), nn.ReLU())
        self.rec = nn.Linear(HID, P * D)
        self.heads = nn.ModuleList([nn.Linear(HID, k) for k in CAT_SIZES])
        self.gt_proj = nn.Linear(sum(CAT_SIZES), HID)
        self.predict_head = nn.Linear(sum(CAT_SIZES), NCLS)

    def _pass(self, fused, parts, cond, y, input_mask):
        h = fused if cond is None else fused + self.gt_proj(cond)
        cat_logits = [head(h) for head in self.heads]
        logits = self.predict_head(parts(cat_logits))
        return ModelOutput(
            logits=logits, target=y,
            extras={
                "cat_logits": cat_logits, "input_mask": input_mask,
                "exogenous_kl": h.pow(2).mean(), "orthogonality": h[:, :4].mean().pow(2),
            },
        )

    def forward(self, batch):
        img, cat, v, y = batch["image"], batch["cat"], batch["cat_value"], batch["y"]
        B = img.shape[0]
        patch_mask = torch.rand(B, P, device=img.device) < 0.5  # sampled ONCE here, read by the loss
        input_mask = torch.rand(B, J, device=img.device) < 0.5
        tab = torch.where(input_mask, torch.zeros_like(v), cat.float() / 5 + v)
        fused = self.enc(torch.cat([(img * (~patch_mask).unsqueeze(-1)).flatten(1), tab], dim=1))
        recon = self.rec(fused).view(B, P, D)

        def self_parts(lgs):
            return torch.cat([torch.softmax(lg, -1) for lg in lgs], dim=-1)

        gt_dist = [F.one_hot(cat[:, j], k).float() for j, k in enumerate(CAT_SIZES)]
        gt_dist[3] = torch.stack([1 - v[:, 3], v[:, 3]], dim=-1)
        gt_cond = torch.cat(gt_dist, dim=-1)

        def gt_parts(_):
            return gt_cond

        out_self = self._pass(fused, self_parts, None, y, input_mask)
        extras = dict(out_self.extras)
        extras.update(recon_image=recon, patch_mask=patch_mask)
        # diagnostics (MCCG: loss_cat_entropy_floor, gate_temperature): measurements, never in the loss
        extras["entropy_floor"] = (-torch.xlogy(gt_dist[3], gt_dist[3]).sum(-1)).mean().detach()
        extras["gate_temperature"] = torch.tensor(0.5, device=img.device)
        if self.training:
            out_gt = self._pass(fused, gt_parts, gt_cond, y, input_mask)
            extras["passes"] = {"gt": out_gt, "self": out_self}
            extras["gt_preds"] = out_gt.logits.argmax(-1)  # only exists in training
        return ModelOutput(logits=out_self.logits, target=y, extras=extras)

    def _step(self, split, batch, batch_idx):
        outputs = self(batch)
        loss_dict = self.loss_fn(outputs, batch)
        key = "val" if split == "valid" else split
        logged = {f"{key}/{k}": v for k, v in loss_dict.items()}
        logged.update(self.metrics.on_step(split, outputs, batch, batch_idx))
        self.log_dict(logged, on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._step("valid", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._step("test", batch, batch_idx)

    def _end(self, split):
        values = self.metrics.on_epoch_end(split)
        if values:
            self.log_dict(values)

    def on_train_epoch_end(self):
        self._end("train")

    def on_validation_epoch_end(self):
        self._end("valid")

    def on_test_epoch_end(self):
        self._end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


# ----------------------------------------------------------------------------------------------------- configs
W = {"gt": 0.5, "self": 0.5}


def loss_cfg(cat_target: str = "CatReconSoftCE", kl_weight: float = 0.01) -> Dict[str, Any]:
    me = __name__
    blend = lambda term: {"_target_": "src.losses.PassBlend", "weights": dict(W), "term": term}  # noqa: E731
    return {
        "_target_": "src.losses.CompositeLoss",
        "terms": {
            "image": {"weight": 1.0, "term": {"_target_": f"{me}.MaskedPatchMSE"}},
            "cat": {"weight": 1.0, "term": blend({"_target_": f"{me}.{cat_target}", "continuous_columns": list(CONT)})},
            "task": {"weight": 1.0, "term": blend({"_target_": f"{me}.TaskMargin", "class_priors": PRIORS,
                                                   "tau_pos": 1.0, "tau_neg": 1.0, "adaptive": 0.0})},
            "exogenous_kl": {"weight": kl_weight, "term": blend({"_target_": "src.losses.FieldTerm", "key": "exogenous_kl"})},
            "orthogonality": {"weight": 0.0, "term": blend({"_target_": "src.losses.FieldTerm", "key": "orthogonality"})},
        },
    }


def metrics_cfg() -> Dict[str, Any]:
    return {
        "_target_": "src.metrics.TaskMetrics", "monitor_metric": "val/macro_f1", "monitor_mode": "max",
        "metric_groups": [
            {"_target_": "src.metrics.ClassificationMetricGroup", "num_classes": NCLS},
            # the second stream is just another group with other keys (MCCG: a subclass + on_step override)
            {"_target_": "src.metrics.ClassificationMetricGroup", "num_classes": NCLS,
             "preds_key": "gt_preds", "name_prefix": "gt/"},
            {"_target_": "src.metrics.FieldMeanGroup", "keys": ["entropy_floor", "gate_temperature"]},
            {"_target_": f"{__name__}.ConceptAccGroup"},
        ],
    }


def build(cat_target="CatReconSoftCE", **kw) -> MiniCascade:
    return MiniCascade(
        hydra.utils.instantiate(OmegaConf.create(loss_cfg(cat_target, **kw))),
        hydra.utils.instantiate(OmegaConf.create(metrics_cfg())),
    )


# ------------------------------------------------------------------------------------------- MCCG-style reference
def ref_cat(view: ModelOutput, batch, mode: str) -> torch.Tensor:
    """Independent re-implementation of MCCG's ``_loss_cat`` reduction (categorical via F.cross_entropy, which
    MCCG documents as identical to soft-CE with a one-hot target)."""
    cols, has = [], []
    for j, lg in enumerate(view.get("cat_logits")):
        m = view.get("input_mask")[:, j]
        safe = torch.where(m.unsqueeze(-1), lg, torch.zeros_like(lg))
        if j in CONT:
            v = batch["cat_value"][:, j]
            per = (cgt.soft_cross_entropy(safe, torch.stack([1 - v, v], -1)) if mode == "soft_ce"
                   else cgt.continuous_column_loss(safe, v, mode))
        else:
            per = F.cross_entropy(safe, batch["cat"][:, j], reduction="none")
        mf = m.float()
        cols.append((per * mf).sum() / mf.sum().clamp(min=1e-6))
        has.append((mf.sum() > 0).float())
    cols, has = torch.stack(cols), torch.stack(has)
    return (cols * has).sum() / has.sum().clamp(min=1.0)


def ref_task(view: ModelOutput, batch, training: bool) -> torch.Tensor:
    if training:
        pos, neg = pm.build_margin_terms(torch.tensor(PRIORS), 1.0, 1.0, 0.0)
        return pm.pairwise_margin_loss(view.logits, batch["y"], pos, neg)
    return F.cross_entropy(view.logits, batch["y"])


def ref_total(outputs: ModelOutput, batch, training: bool, mode="soft_ce", kl_w=0.01) -> torch.Tensor:
    """MCCG ``forward``'s assembly: ``lambda * (w*gt + (1-w)*self)`` per term, zero-weight terms dropped."""
    w = 0.5
    img = MaskedPatchMSE()(outputs, batch)
    if training:
        g, s = outputs.get("passes")["gt"], outputs.get("passes")["self"]
        cat = w * ref_cat(g, batch, mode) + (1 - w) * ref_cat(s, batch, mode)
        task = w * ref_task(g, batch, True) + (1 - w) * ref_task(s, batch, True)
        kl = w * g.get("exogenous_kl") + (1 - w) * s.get("exogenous_kl")
    else:
        cat, task, kl = ref_cat(outputs, batch, mode), ref_task(outputs, batch, False), outputs.get("exogenous_kl")
    return img + cat + task + kl_w * kl


# --------------------------------------------------------------------------------------------------------- checks
RESULTS: List[Dict[str, Any]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append({"check": name, "ok": True, "detail": detail})
        print(f"PASS  {name}: {detail}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append({"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def a_batch():
    return next(iter(SimDM().train_dataloader()))


def check_fit():
    model = build()
    logger = CSVLogger(str(OUT), name="fit")
    trainer = Trainer(max_epochs=4, accelerator="cpu", logger=logger, enable_checkpointing=False,
                      enable_progress_bar=False, callbacks=[MetricTrends("val/macro_f1", "max")],
                      enable_model_summary=False)
    trainer.fit(model, datamodule=SimDM())
    trainer.test(model, datamodule=SimDM())
    df = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
    cols = set(df.columns)
    need = {"train/loss", "train/loss_image", "train/loss_cat", "train/loss_task", "train/loss_exogenous_kl",
            "val/loss", "val/macro_f1", "train/gt/acc", "val/diag/entropy_floor", "val/concept/acc_mean",
            "val/macro_f1_best", "val/overfit_gap"}
    assert need <= cols, f"missing logged keys: {sorted(need - cols)}"
    assert "loss_orthogonality" not in " ".join(cols), "zero-weight term must not be computed/logged"
    assert not any(c.startswith("val/gt/") or c.startswith("test/gt/") for c in cols), "gt stream must not exist at eval"
    assert not any("entropy" in c and "loss" in c for c in cols), "a diagnostic leaked into the loss"
    first, last = df["train/loss"].dropna().iloc[0], df["train/loss"].dropna().iloc[-1]
    assert torch.isfinite(torch.tensor(last)) and last < first, f"train/loss did not decrease: {first:.3f} -> {last:.3f}"
    return f"{len(cols)} logged keys; train/loss {first:.3f} -> {last:.3f}"


def check_equivalence():
    model = build()
    batch = a_batch()
    torch.manual_seed(0)
    model.train()
    out = model(batch)
    new = float(model.loss_fn(out, batch)["loss"].detach())
    ref = float(ref_total(out, batch, True))
    assert abs(new - ref) < 1e-5, f"train: new={new} ref={ref}"
    model.eval()
    with torch.no_grad():
        out_e = model(batch)
        new_e = float(model.loss_fn(out_e, batch)["loss"])
        ref_e = float(ref_total(out_e, batch, False))
    assert abs(new_e - ref_e) < 1e-5, f"eval: new={new_e} ref={ref_e}"
    return f"train {new:.6f} == {ref:.6f}, eval {new_e:.6f} == {ref_e:.6f}"


def check_swap_continuous_loss():
    batch, vals = a_batch(), {}
    torch.manual_seed(1)
    base = build()
    base.train()
    out = base(batch)
    for target, mode in [("CatReconSoftCE", "soft_ce"), ("CatReconMSE", "mse"), ("CatReconGNLL", "gaussian_nll")]:
        model = build(target)  # only the loss config's _target_ differs; the model code is the same class
        total = float(model.loss_fn(out, batch)["loss_cat"])
        ref = 0.5 * float(ref_cat(out.get("passes")["gt"], batch, mode)) + 0.5 * float(ref_cat(out.get("passes")["self"], batch, mode))
        assert abs(total - ref) < 1e-5, f"{mode}: {total} vs {ref}"
        vals[mode] = round(total, 4)
    assert len(set(vals.values())) == 3, f"the three losses should differ: {vals}"
    return f"loss_cat per mode {vals} (each == MCCG function)"


def check_requires_error():
    model = build()
    model.train()
    out = model(a_batch())
    del out.extras["passes"]["self"].extras["exogenous_kl"]
    try:
        model.loss_fn(out, a_batch())
    except (KeyError, TypeError) as e:
        msg = str(e)
    else:
        raise AssertionError("expected an error for the missing field")
    assert "exogenous_kl" in msg and "'self'" in msg, f"error must name the pass and the field: {msg}"
    return f"missing per-pass field -> {msg[:110]!r}"


def check_zero_weight_needs_no_field():
    model = build()
    model.eval()
    out = model(a_batch())
    out.extras.pop("orthogonality", None)  # weight 0 -> not required
    model.loss_fn(out, a_batch())
    return "orthogonality weight 0: field not required"


def check_grads():
    model = build()
    model.train()
    batch = a_batch()
    model.loss_fn(model(batch), batch)["loss"].backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"no gradient for {dead}"
    return "all parameters receive gradient through PassBlend/CompositeLoss"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("fit: terms/streams/diagnostics logged, gt stream train-only", check_fit),
        ("numeric equivalence with MCCG-style assembly (train + eval)", check_equivalence),
        ("swap continuous_loss by _target_ only", check_swap_continuous_loss),
        ("missing model field -> error names it", check_requires_error),
        ("zero-weight term needs no field", check_zero_weight_needs_no_field),
        ("gradients reach every parameter", check_grads),
    ]:
        check(name, fn)
    (OUT / "report.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed -> {OUT / 'report.json'}")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
