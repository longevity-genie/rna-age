"""RM30: the model (TabICLv2, initialised from its released weights), its training on expression episodes, and a
sklearn-like regressor around a trained model.

- :class:`ExprICL`: two ``tabicl`` ``TabICL`` modules sharing one body (column embedder's set transformer, row
  interactor, in-context transformer): ``reg`` (``max_classes=0``, 999-quantile head, pinball loss) for age, masked
  gene and perturbation, ``cls`` (``max_classes=10``, cross-entropy) for health. Only the target encoders and the
  decoder are the classification head's own.
- :func:`from_pretrained`: the released TabICLv2 regressor checkpoint (``jingang/TabICL``,
  ``tabicl-regressor-v2-20260212.ckpt``, found or downloaded with ``hf_hub_download`` under ``$HF_HOME``, exactly
  as ``TabICLRegressor._load_model``) into ``reg`` and so into the shared body; the classification head's own
  tensors from the classifier checkpoint when it can be had (its body differs: LayerNorms with biases, so that head
  starts close to, not at, a trained state).
- Forward = TabICL's training path (column embedding → row interaction → in-context learning, no inference manager);
  with the released weights it reproduces ``TabICLRegressor``'s predictions on the same preprocessed table
  (``tests/test_expr_icl.py``).
- :func:`train`: mixed-task steps with task weights; AdamW, warm-up + cosine, gradient clipping, bf16 autocast on
  CUDA; checkpoints under ``data/interim/checkpoints/<run>/``, per-step metrics and the periodic quick evaluation
  on held-out episodes in ``<run dir>/train_metrics.parquet``.

``tabicl`` is not a project dependency: run with ``uv run --with tabicl``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from rna_age.train.expr_icl import CACHE, TASKS, Corpus, Episode, EpisodeSampler, standardise

log = logging.getLogger(__name__)

REPO_ID = "jingang/TabICL"
REGRESSOR_CKPT = "tabicl-regressor-v2-20260212.ckpt"
CLASSIFIER_CKPT = "tabicl-classifier-v2-20260212.ckpt"
MAX_CLASSES = 10
CLS_OWN = ("col_embedder.y_encoder.", "icl_predictor.y_encoder.", "icl_predictor.decoder.")


def checkpoint_path(filename: str = REGRESSOR_CKPT, allow_download: bool = True) -> Path:
    """The released checkpoint from the Hugging Face cache (``$HF_HOME``), downloaded if absent (TabICLRegressor's
    own loader does the same)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=REPO_ID, filename=filename, local_files_only=True))
    except Exception:
        if not allow_download:
            raise
        log.info("downloading %s from %s", filename, REPO_ID)
        return Path(hf_hub_download(repo_id=REPO_ID, filename=filename))


class ExprICL(nn.Module):
    def __init__(self, config: dict, max_classes: int = MAX_CLASSES, recompute: bool = False):
        super().__init__()
        from tabicl._model.tabicl import TabICL

        config = {k: v for k, v in config.items() if k != "max_classes"}
        if recompute:  # TabICL's own gradient checkpointing (activation recomputation): memory for compute
            config["recompute"] = True
        self.config = config
        self.max_classes = max_classes
        self.reg = TabICL(max_classes=0, **config)
        self.cls = TabICL(max_classes=max_classes, **config)
        r, c = self.reg, self.cls  # one body: the classification model reuses the regression model's modules
        c.col_embedder.in_linear = r.col_embedder.in_linear
        c.col_embedder.tf_col = r.col_embedder.tf_col
        c.row_interactor = r.row_interactor
        c.icl_predictor.tf_icl = r.icl_predictor.tf_icl
        if hasattr(r.icl_predictor, "ln"):
            c.icl_predictor.ln = r.icl_predictor.ln

    def forward(self, X: torch.Tensor, y_ctx: torch.Tensor, classification: bool = False) -> torch.Tensor:
        """X: (B, n_ctx + n_qry, H), y_ctx: (B, n_ctx) → (B, n_qry, 999 quantiles | max_classes logits)."""
        net = self.cls if classification else self.reg
        emb = net.col_embedder._train_forward(X, y_ctx, d=None, embed_with_test=False)
        rep = net.row_interactor._train_forward(emb, d=None)
        return net.icl_predictor._icl_predictions(rep, y_ctx)[:, y_ctx.shape[1] :]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def from_pretrained(
    regressor: Path | None = None, classifier: Path | None | bool = None, recompute: bool = False
) -> ExprICL:
    """TabICLv2's released weights. ``classifier=False`` skips the classification head's own tensors (they stay at
    their initialisation); ``recompute`` turns on TabICL's gradient checkpointing (same weights, less memory)."""
    ck = torch.load(regressor or checkpoint_path(), map_location="cpu", weights_only=True)
    model = ExprICL(ck["config"], recompute=recompute)
    model.reg.load_state_dict(ck["state_dict"])  # strict: every tensor of the released regressor
    if classifier is not False:
        try:
            cs = torch.load(classifier or checkpoint_path(CLASSIFIER_CKPT), map_location="cpu", weights_only=True)
            own = model.cls.state_dict()
            take = {k: v for k, v in cs["state_dict"].items() if k.startswith(CLS_OWN) and k in own
                    and own[k].shape == v.shape}  # fmt: skip
            model.cls.load_state_dict(take, strict=False)
            log.info("classification head: %d tensors from the released classifier", len(take))
        except Exception as e:  # no network or no file: the head starts at its initialisation
            log.warning("classifier checkpoint not loaded (%s); classification head from initialisation", e)
    return model


# ---------------------------------------------------------------- batches, loss, prediction


def to_tensors(eps: list[Episode], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = np.stack([np.concatenate([e.X_ctx, e.X_qry]) for e in eps])
    yc = np.stack([e.y_ctx for e in eps])
    yq = np.stack([e.y_qry for e in eps])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)  # noqa: E731
    return t(X), t(yc), t(yq)


def episode_loss(model: ExprICL, eps: list[Episode], device: str) -> torch.Tensor:
    X, yc, yq = to_tensors(eps, device)
    cls = eps[0].classification
    out = model(X, yc, classification=cls)
    if cls:
        return nn.functional.cross_entropy(out.flatten(end_dim=-2).float(), yq.long().flatten())
    q = out.shape[-1]
    alphas = torch.linspace(0.0, 1.0, q + 2, device=out.device, dtype=torch.float32)[1:-1].view(1, 1, -1)
    err = yq.unsqueeze(-1) - out.float()
    return torch.maximum(alphas * err, (alphas - 1) * err).mean()


@torch.no_grad()
def predict_episode(model: ExprICL, e: Episode, device: str) -> np.ndarray:
    """Query predictions in the target's own units (regression: quantile mean; classification: P(class 1))."""
    X, yc, _ = to_tensors([e], device)
    out = model(X, yc, classification=e.classification)[0].float()
    if e.classification:
        return torch.softmax(out[:, :2], -1)[:, 1].cpu().numpy()
    return out.mean(-1).cpu().numpy() * e.y_sd + e.y_mean


def episode_metrics(e: Episode, pred: np.ndarray) -> dict[str, float]:
    if e.classification:
        from sklearn.metrics import roc_auc_score

        y = e.y_qry
        auc = float(roc_auc_score(y, pred)) if 0 < y.sum() < len(y) else float("nan")
        return {"auc": auc, "accuracy": float(((pred > 0.5) == (y > 0.5)).mean())}
    y = e.y_qry * e.y_sd + e.y_mean
    mae = float(np.abs(pred - y).mean())
    mae0 = float(np.abs(e.y_mean - y).mean())  # the context mean
    with np.errstate(all="ignore"):
        r = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else float("nan")
    return {"mae": mae, "mae_ctx_mean": mae0, "skill": 1 - mae / mae0 if mae0 > 0 else float("nan"), "r": r}


def quick_eval(model: ExprICL, eps: list[Episode], device: str) -> list[dict]:
    model.eval()
    rows = []
    for t in TASKS:
        te = [e for e in eps if e.task == t]
        if not te:
            continue
        ms = [episode_metrics(e, predict_episode(model, e, device)) for e in te]
        for k in ms[0]:
            v = np.array([m[k] for m in ms], dtype=np.float64)
            value = float(v[~np.isnan(v)].mean()) if (~np.isnan(v)).any() else float("nan")
            rows.append({"task": t, "metric": k, "value": value, "episodes": len(te)})
    model.train()
    return rows


# ---------------------------------------------------------------- sklearn-like wrapper


class ExprICLRegressor:
    """fit(X, y) keeps the context; predict(X) runs the model in context. Columns: at most ``max_features``, the
    most variable on the context; standardised on the context; ``n_estimators`` column orders averaged (the first
    is the given order)."""

    def __init__(self, model: ExprICL, device: str = "cpu", max_features: int = 500, n_estimators: int = 4,
                 chunk: int = 512, seed: int = 0, amp: bool = True):  # fmt: skip
        self.model, self.device, self.max_features = model, device, max_features
        self.n_estimators, self.chunk, self.seed, self.amp = n_estimators, chunk, seed, amp

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in ("model", "device", "max_features", "n_estimators", "chunk", "seed", "amp")}

    def fit(self, X: np.ndarray, y: np.ndarray) -> ExprICLRegressor:
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-d with one row per target")
        self.cols_ = np.sort(np.argsort(-X.var(0))[: self.max_features])
        self.X_, self.y_ = X, y
        self.n_features_in_ = X.shape[1]
        return self

    def _autocast(self) -> dict:
        """Half precision on a GPU at inference (bf16 where the card has it natively, else fp16), fp32 elsewhere."""
        from rna_age.train.runs import has_native_bf16

        cuda = str(self.device).startswith("cuda")
        dtype = torch.bfloat16 if cuda and has_native_bf16() else torch.float16
        return {"device_type": "cuda" if cuda else "cpu", "dtype": dtype, "enabled": bool(self.amp and cuda)}

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_"):
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, fit had {self.n_features_in_}")
        Xc, Xq = standardise(self.X_[:, self.cols_], X[:, self.cols_])
        mu, sd = float(self.y_.mean()), float(self.y_.std()) + 1e-6
        yc = torch.from_numpy(((self.y_ - mu) / sd).astype(np.float32))[None].to(self.device)
        rng = np.random.default_rng(self.seed)
        self.model.eval()
        out = np.zeros(len(X))
        for e in range(self.n_estimators):
            perm = np.arange(Xc.shape[1]) if e == 0 else rng.permutation(Xc.shape[1])
            for a in range(0, len(X), self.chunk):
                T = np.concatenate([Xc[:, perm], Xq[a : a + self.chunk][:, perm]])[None]
                with torch.autocast(**self._autocast()):
                    q = self.model(torch.from_numpy(T).to(self.device), yc)[0].float().mean(-1)
                out[a : a + self.chunk] += q.cpu().numpy()
        return out / self.n_estimators * sd + mu


# ---------------------------------------------------------------- training


@dataclass
class TrainConfig:
    steps: int = 20000
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    micro_batch: int = 4  # episodes of one task and shape per forward
    accum: int = 4  # forwards per optimiser step (each picks its task by the weights)
    task_weights: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TASKS})
    n_rows: tuple[int, int] = (64, 512)
    n_cols: tuple[int, int] = (100, 500)
    eval_every: int = 500
    eval_episodes: int = 32
    ckpt_every: int = 2000
    log_every: int = 50
    seed: int = 0
    device: str = "auto"
    amp: bool = True  # bf16 where the card has it, else fp16 with a loss scaler
    recompute: bool = False  # gradient checkpointing inside TabICL (for 11 GB cards)


def _device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def save_checkpoint(path: Path, model: ExprICL, opt, step: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "max_classes": model.max_classes,
                "step": step, "optimizer": opt.state_dict(), "train_config": asdict(cfg)}, path)  # fmt: skip


def load_trained(run_id: str, step: str = "last") -> ExprICL:
    """A trained model of run ``run_id`` (its ``checkpoint_dir``/``<step>.pt``)."""
    from rna_age.train.runs import RUNS

    conf = json.loads((RUNS / run_id / "config.json").read_text())
    ck = torch.load(Path(conf["checkpoint_dir"]) / f"{step}.pt", map_location="cpu", weights_only=True)
    model = ExprICL(ck["config"], ck["max_classes"])
    model.load_state_dict(ck["state_dict"])
    return model


def train(
    cfg: TrainConfig,
    corpus: Corpus | None = None,
    model: ExprICL | None = None,
    name: str = "expricl",
    cache: Path = CACHE,
    resume: str | None = None,
) -> Path:
    """Train on mixed-task episodes; returns the run directory. ``resume``: a run id whose ``last.pt`` (weights,
    optimiser, step) this run continues from, in a new run directory, up to ``cfg.steps`` in total; the cosine
    schedule is read at the global step, so a longer ``cfg.steps`` than the first run's is a warm restart."""
    from rna_age.train.expr_icl import load_corpus
    from rna_age.train.runs import new_run

    t0 = time.time()
    torch.manual_seed(cfg.seed)
    dev = _device(cfg.device)
    corpus = corpus or load_corpus(cache)
    tr = EpisodeSampler(corpus, "train", n_rows=cfg.n_rows, n_cols=cfg.n_cols, seed=cfg.seed)
    ev = EpisodeSampler(corpus, "eval", n_rows=cfg.n_rows, n_cols=cfg.n_cols, seed=cfg.seed + 1)
    evset = ev.fixed_set(cfg.eval_episodes) if cfg.eval_episodes else []
    start, ck = 0, None
    if resume:
        from rna_age.train.runs import RUNS

        conf = json.loads((RUNS / resume / "config.json").read_text())
        ck = torch.load(Path(conf["checkpoint_dir"]) / "last.pt", map_location="cpu", weights_only=True)
        model = ExprICL(ck["config"], ck["max_classes"], recompute=cfg.recompute)
        model.load_state_dict(ck["state_dict"])
        start = int(ck["step"])
        if start >= cfg.steps:
            raise ValueError(f"{resume} is at step {start}; --steps must be larger to continue")
    model = model or from_pretrained(recompute=cfg.recompute)
    model.to(dev).train()
    tasks = [t for t in tr.tasks() if cfg.task_weights.get(t, 0) > 0]
    if not tasks:
        raise ValueError("no task has training episodes")
    w = np.array([cfg.task_weights[t] for t in tasks], dtype=np.float64)
    w /= w.sum()
    run_dir = new_run(name, {"model": "expr-icl (TabICLv2 architecture, from the released regressor weights)",
                             "features": f"panel {len(corpus.genes)} genes, episodes of {cfg.n_cols} columns",
                             "train_set": "expression episodes (RM30)", "n_params": model.n_params(),
                             "train_groups": tr.counts(), "eval_groups": ev.counts(), "tasks": tasks,
                             "train_config": asdict(cfg), "device": dev, "resumed_from": resume,
                             "start_step": start})  # fmt: skip
    ckpt_dir = Path(json.loads((run_dir / "config.json").read_text())["checkpoint_dir"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))  # fmt: skip
    if ck is not None:
        opt.load_state_dict(ck["optimizer"])
    # bf16 where the card has it; a 2080 Ti has not: fp16 there, with a loss scaler against underflow
    amp = cfg.amp and dev.startswith("cuda")
    from rna_age.train.runs import has_native_bf16

    amp_dtype = torch.bfloat16 if amp and has_native_bf16() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    rows: list[dict] = []
    rng = np.random.default_rng(cfg.seed)

    def flush():
        pl.DataFrame(rows).write_parquet(run_dir / "train_metrics.parquet")

    if evset:
        for r in quick_eval(model, evset, dev):
            rows.append({"step": 0, "split": "eval", **r})
    for step in range(start, cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        per_task: dict[str, list[float]] = {}
        for _ in range(cfg.accum):
            t = tasks[rng.choice(len(tasks), p=w)]
            eps = tr.sample_batch(t, cfg.micro_batch)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss = episode_loss(model, eps, dev)
            scaler.scale(loss / cfg.accum).backward()
            per_task.setdefault(t, []).append(float(loss.detach()))
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scaler.step(opt)  # skipped when the scaled gradients overflowed
        scaler.update()
        for t, ls in per_task.items():
            rows.append({"step": step + 1, "split": "train", "task": t, "metric": "loss", "value": float(np.mean(ls)),
                         "episodes": len(ls) * cfg.micro_batch})  # fmt: skip
        rows.append({"step": step + 1, "split": "train", "task": "all", "metric": "grad_norm", "value": gn,
                     "episodes": 0})  # fmt: skip
        if (step + 1) % cfg.log_every == 0:
            log.info("step %d lr %.2e %s gn %.2f (%.0f s)", step + 1, lr_at(step, cfg),
                     {t: round(float(np.mean(v)), 4) for t, v in per_task.items()}, gn, time.time() - t0)  # fmt: skip
        if evset and (step + 1) % cfg.eval_every == 0:
            ev_rows = quick_eval(model, evset, dev)
            rows += [{"step": step + 1, "split": "eval", **r} for r in ev_rows]
            log.info("eval step %d: %s", step + 1, {f"{r['task']}.{r['metric']}": round(r["value"], 3)
                                                    for r in ev_rows if r["metric"] in ("mae", "skill", "auc")})  # fmt: skip
            flush()
        if (step + 1) % cfg.ckpt_every == 0:
            save_checkpoint(ckpt_dir / f"step-{step + 1}.pt", model, opt, step + 1, cfg)
            save_checkpoint(ckpt_dir / "last.pt", model, opt, step + 1, cfg)
    if evset and cfg.steps % cfg.eval_every:
        rows += [{"step": cfg.steps, "split": "eval", **r} for r in quick_eval(model, evset, dev)]
    save_checkpoint(ckpt_dir / "last.pt", model, opt, cfg.steps, cfg)
    flush()
    conf = json.loads((run_dir / "config.json").read_text())
    conf["seconds"] = time.time() - t0
    (run_dir / "config.json").write_text(json.dumps(conf, indent=2, default=str))
    log.info("run %s: %d steps in %.0f s", run_dir.name, cfg.steps, time.time() - t0)
    return run_dir
