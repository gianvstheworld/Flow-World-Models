# Fork changes

Record of how this fork diverges from the upstream
[facebookresearch/Flow-World-Models](https://github.com/facebookresearch/Flow-World-Models)
release, and why each change was made.

This fork adapts FlowWM for **traversability** research — using the model's predicted
future DINOv3 features to estimate where a vehicle will be able to drive, rather than the
paper's object-detection and depth benchmarks — and for training on a **single GPU**
instead of the paper's 64-GPU setup.

Entries are newest-last. Each states the problem, the change, and its scope.

---

## Changes

### 1. Evaluator crashes when no detector is installed — *bug fix*

**Files:** `evaluators/evaluator.py` · **Commit:** `1e14c78`

The trainer degrades gracefully when `detectron2`/`detrex` are unavailable, catching the
import error and setting `self.detector = None`
(`trainer/trainer.py:1159-1174` and `1565-1580`). The evaluator did not honour that
contract: it indexed the detection outputs unconditionally.

```python
coco_predictions = downstream_outputs["coco_predictions"]   # KeyError
```

Any evaluation run without a detector therefore died with
`KeyError: 'coco_predictions'`. This affects **every user of the open-source release**,
because the DINO-DETR config and checkpoint the detector needs are not part of it — so
the graceful-degradation path is the only path available, and it was never exercised.

Fixed at both call sites (`DeterministicPredictorEvaluatorWaymo.accumulate` and
`FlowMatchingEvaluatorWaymoFastV2.accumulate`) using the defensive access already present
one line below for `coco_predictions_oracle`:

```python
coco_predictions = downstream_outputs.get("coco_predictions", [])
```

Downstream code already tolerates the empty list — `_evaluate_with_ranking` returns `{}`
on empty input. Detection metrics are simply absent rather than fatal.

*Scope:* upstream bug, not a fork-specific adaptation. Good candidate to contribute back.

---

### 2. Single-GPU training configs

**Files:** `configs/dinov3/flow_matching_rae_waymo/local3060/384/` · **Commit:** `75651a4`

Upstream ships configs sized for multi-GPU training (the paper uses 64 V100s at an
effective batch of 128). This adds a three-config chain for single-GPU work, composed via
the repo's `inherits:` mechanism on top of the standard
`base/384/dinov3_waymo_512_vits_16_384_flow_matching_rae_base.yaml`:

| Config | Purpose |
|---|---|
| `local_smoke.yaml` | Short run to verify the loop end to end and measure s/step |
| `local_eval.yaml` | Evaluation only, against an existing checkpoint |
| `local_overnight.yaml` | Long unattended run |

Deviations from the base config, and why:

- `use_mixed_precision: true` with `mixed_precision_dtype: bfloat16` — memory headroom
  on a 12 GB card. The dtype must be stated explicitly; see Change 4.
- `batch_size_per_gpu: 1` — activations over the 12,288 target tokens dominate memory
  and scale with batch size; higher values OOM on 12 GB.
- `n_predictions_for_evaluation: 1` (base: 3) — each prediction costs a full 50-step ODE
  sampling pass (~10 s/clip).
- Reduced `epochs` so the `CosineAnnealingLR` schedule
  (`T_max = epochs - warmup_epochs`, `main.py:341`) completes within the run budget
  rather than being interrupted mid-decay.

*Note:* the `local3060/` directory name is machine-specific and the configs contain
absolute dataset paths. Both are scheduled to be replaced with environment-variable
interpolation so the configs are portable across machines.

---

### 3. Unattended-run tooling

**Files:** `run_overnight.sh`, `make_report.py` · **Commit:** `75651a4`

`run_overnight.sh` wraps `torchrun` so that a long run survives terminal disconnection,
records GPU temperature/utilisation/memory throughout, and **always** generates a report
regardless of exit code — a run that fails at hour 6 should still produce a readable
account of what happened. It sets `WANDB_MODE=disabled` by default, since `main.py:106-110`
calls `wandb.init` unconditionally outside `--debug` and would otherwise block on an
unauthenticated machine.

`make_report.py` parses the training log into a markdown summary: loss/LR curve,
evaluation metrics, sampling latency, checkpoints and dump artifacts, tracebacks, and a
thermal summary.

*Scope:* additive. Nothing in the training path depends on either file.

---

### 4. Mixed precision pinned to bfloat16 — *divergence fix*

**Files:** `configs/.../local3060/384/local_smoke.yaml` · **Commit:** `c6427d4`

`trainer/trainer.py:119-121` reads `mixed_precision_dtype` with a default of
`"float16"`. Enabling `use_mixed_precision` without naming a dtype therefore selects
fp16, whose exponent range overflows at the base `lr: 1e-3`: gradient norms reach `inf`
within a few epochs and the model becomes irrecoverably `NaN` once LR warmup completes
and the peak learning rate is applied.

`bfloat16` carries fp32's exponent range, so the overflow cannot occur, and the
`GradScaler` disables itself for it (`trainer/trainer.py:122-126`). Support already
existed in `trainer/utils.py:50-55`.

Measured over one epoch, against the fp16 run it replaces: identical loss
(0.5987 vs 0.5903 at epoch 1), zero non-finite values (against 1,903 log lines
containing them), and ~7% slower per step (0.838 s vs 0.786 s) — a fair price for a
run that terminates.

---

### 5. Train-set evaluation pass made configurable

**Files:** `trainer/trainer.py`, `configs/.../local3060/384/local_smoke.yaml` · **Commit:** `678251b`

`run_evaluation` called `run_train_set_evaluation(n_samples=256)` with the count
hardcoded. That is a full 50-step ODE sampling pass over 256 *training* clips — around
45 minutes — executed before the validation pass and emitting no log output, so it
presents as an unexplained stall at every evaluation. At the local single-GPU scale it
cost more than ten times the validation pass it preceded, to measure memorisation of
data already being trained on.

Added `trainer.evaluation.train_eval_n_samples` (`0` skips the pass), defaulting to 256
so existing configs behave exactly as before, plus a log line naming the clip count and
how to disable it. Set to `0` in the local config chain.

---

### 6. Visualisation failures made non-fatal

**Files:** `trainer/trainer.py` · **Commit:** `0be78e3`

`utils/utils_visualisation.py` contains no exception handling anywhere in the file, and
the three `save_batch_and_model_outputs` call sites were unguarded.
`_append_pca_latents_to_canvas` fits a PCA on context latents and transforms predicted
latents through it, so a single non-finite prediction raises `ValueError` and ends the
entire job from inside a plotting call — after training and evaluation have already
succeeded.

All three call sites now route through `Trainer._save_visualisations`, which logs the
failure with its traceback and continues. Added
`trainer.evaluation.enable_visualization` (default `true`) to skip dumping outright.
Visualisation is a reporting side effect and must not be able to end a run.

---

### 7. Abort on sustained non-finite loss

**Files:** `trainer/trainer.py` · **Commit:** `996d12a`

`_check_loss_health` returned a boolean that the caller only logged before proceeding to
`.backward()` and the optimizer step. A diverged run therefore continued for as long as
it was scheduled, producing nothing while periodic checkpointing overwrote the last
healthy weights with unusable ones.

Added `_register_loss_health`, gating on the combined loss immediately before
`backward()` so divergence from any loss term is caught. Isolated non-finite steps are
tolerated — the `GradScaler` already skips those — and only a sustained run aborts,
after `trainer.training.max_consecutive_unhealthy_steps` (default 25). The error names
`mixed_precision_dtype` as the first thing to check, that being the known cause.

---

### 8. Report captures diverged epochs

**Files:** `make_report.py` · **Commit:** `675db25`

The epoch regex matched `loss_total=([\d.]+)`, which cannot match `nan`. Diverged epochs
were silently dropped, so a run that went `NaN` partway through and burned the remaining
epochs was reported as a clean, shorter run that "completed".

`loss_total` is now matched as `\S+`; any non-finite loss disqualifies a run from being
reported as completed; and the report names the divergence epoch, the last healthy
epoch, and warns that checkpoints written after it hold non-finite weights.

---

## Known upstream issues

Identified in the upstream code, not yet fixed here. Recorded so the findings are not
lost and so anyone hitting them knows they are known.

**1. No gradient accumulation.**
There is no accumulation logic in `trainer/`. On hardware that cannot fit a large batch,
the effective batch size is bounded by what fits in VRAM — far below the effective batch
of 128 the published `lr: 1e-3` was tuned for. In practice gradients clip on nearly every
step at batch 1.

**2. Checkpoint resume is incomplete.**
`save_checkpoint` (`trainer/trainer.py:256-287`) stores model weights only — no
optimizer, scheduler, or `GradScaler` state. Additionally, `train()` iterates
`for _ in range(self.epochs)` (`trainer/trainer.py:2294`), so resuming from a checkpoint
runs `epochs` *additional* epochs while the LR schedule restarts from warmup. Resume is
currently reliable only for `eval_only` runs.

**3. The DINOv3 backbone is never downloaded automatically.**
`models/models.py:48-55` passes `local_files_only=True` to both
`AutoImageProcessor.from_pretrained` and `AutoModel.from_pretrained`. On a machine with a
cold Hugging Face cache this fails rather than fetching the model. Because
`facebook/dinov3-vits16-pretrain-lvd1689m` is additionally a **gated** repository, first-time
setup on any new machine requires accepting the model licence, authenticating, and
pre-downloading the weights once with `local_files_only=False`.

**4. `misc.latent_size` is dead configuration.**
Declared as `[768, 16, 16]` in the base configs but referenced nowhere in the Python
source. The actual token grid is derived from `image_size // patch_size`
(`models/models.py:407-409`), giving 32×32 = 1024 patches per frame at 512×512 with
patch size 16 — so 4,096 context tokens and 12,288 target tokens. The config value is
misleading and should be ignored.

---

## Notes on the paper vs. the release

Two components described in the paper are not present in the open-source release. Neither
is a defect, but both are easy to look for and not find:

- **The object detector.** The DINO-DETR config and checkpoint used for the `AP_L`
  metrics are external. Without `detectron2`/`detrex` and those weights, detection
  metrics come back empty (see Change 1).
- **The depth head.** There is no depth-estimation code in the repository. The
  `configs/dinov3/flow_matching_rae_waymo/depth/*.yaml` files set `stage_2.params.depth`,
  which is the **number of DiT blocks in the predictor**, not a depth task. The closest
  implemented analogue of a dense per-patch downstream head is the heatmap predictor
  (`models/models.py:70-156`, trained via `TrainerHeatmap` at `trainer/trainer.py:838-1076`).
