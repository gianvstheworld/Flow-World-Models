#!/usr/bin/env python3
"""Generate the morning report for an unattended FlowWM run.

Reads the experiment's rank_0.log plus the wrapper's stdout capture and writes a
single markdown file summarising what ran, what it produced, and what to look at.
Designed to run *after* training regardless of exit status, so a crash still
produces a readable report.

Usage: python make_report.py <experiment_dir> <wrapper_log> <exit_code> <out.md>
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception as exc:  # keep the report generating no matter what
        return f"(failed: {exc})"


def main():
    exp_dir = Path(sys.argv[1])
    wrapper_log = Path(sys.argv[2])
    exit_code = sys.argv[3]
    out_path = Path(sys.argv[4])

    log_path = exp_dir / "logs" / "rank_0.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    wrapper = wrapper_log.read_text(errors="replace") if wrapper_log.exists() else ""

    # --- training curve -------------------------------------------------
    epoch_re = re.compile(
        r"\[TRAIN\] EPOCH (\d+) SUMMARY \| loss_total=([\d.]+) \| lr=([\d.e+-]+) \| "
        r"avg time per gradient step=([\d.]+)s \| steps=(\d+) \| duration=([\d.]+)s"
    )
    epochs = epoch_re.findall(log)

    # --- errors ---------------------------------------------------------
    tracebacks = re.findall(r"(?:\[rank0\]: )?Traceback \(most recent call last\)", wrapper)
    err_lines = [
        l for l in wrapper.splitlines()
        if re.search(r"Error|FAILED|Killed|OOM|out of memory|assert", l)
        and "error_file" not in l
    ]
    warn_lines = [l for l in log.splitlines() if " - WARNING - " in l]

    # --- latency / eval -------------------------------------------------
    lat = re.findall(r"encoder=([\d.]+) ms \| denoising=([\d.]+) ms", log)
    metrics_path = exp_dir / "dumps" / "metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None

    ckpts = sorted((exp_dir / "checkpoints").glob("*.pt")) if (exp_dir / "checkpoints").exists() else []
    dumps = sorted((exp_dir / "dumps").glob("*")) if (exp_dir / "dumps").exists() else []
    videos = list((exp_dir / "dumps").rglob("*.mp4")) if (exp_dir / "dumps").exists() else []
    canvases = list((exp_dir / "dumps").rglob("canvas_*.png")) if (exp_dir / "dumps").exists() else []

    # "Completed" requires a clean exit *and* at least one finished epoch -- a zero exit
    # with no epoch summaries means it died before doing any real work.
    ok = exit_code == "0" and not tracebacks and bool(epochs)

    L = []
    A = L.append
    A(f"# FlowWM overnight run — {'COMPLETED' if ok else 'FAILED'}")
    A("")
    A(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Exit code `{exit_code}`.")
    A("")

    # ---- headline ------------------------------------------------------
    A("## Bottom line")
    A("")
    if ok and epochs:
        first, last = epochs[0], epochs[-1]
        A(f"- Training **completed**: {len(epochs)} epoch summaries logged, "
          f"last was epoch {last[0]}.")
        A(f"- Loss went **{first[1]} (epoch {first[0]}) -> {last[1]} (epoch {last[0]})**.")
        A(f"- Wall-clock per epoch: {float(last[5])/60:.1f} min "
          f"({last[4]} steps at {last[3]}s/step).")
        A(f"- {len(ckpts)} checkpoint(s) written.")
    elif epochs:
        A(f"- Run **did not finish cleanly**, but {len(epochs)} epoch(s) completed first.")
        A(f"- Last logged epoch: {epochs[-1][0]}, loss {epochs[-1][1]}.")
        A("- See **Errors** below; checkpoints from completed epochs are still usable.")
    else:
        A("- Run **failed before completing an epoch**. See **Errors** below.")
    A("")
    A("> Reminder: this is a ~480-clip run, roughly 1% of the paper's training data. "
      "The loss value is evidence the machinery works, **not** a result comparable to "
      "the paper. Expect overfitting.")
    A("")

    # ---- training curve ------------------------------------------------
    A("## Training curve")
    A("")
    if epochs:
        A("| epoch | loss_total | lr | s/step | duration |")
        A("|---:|---:|---:|---:|---:|")
        for e in epochs:
            A(f"| {e[0]} | {e[1]} | {e[2]} | {e[3]} | {float(e[5])/60:.1f} min |")
    else:
        A("_No epoch summaries found in the log._")
    A("")

    # ---- eval ----------------------------------------------------------
    A("## Evaluation")
    A("")
    # The trainer logs eval results as a CSV header/value pair; prefer it over metrics.json
    # because it carries the per-horizon breakdown (l1_latents_t0 .. t11).
    hdr = re.findall(r"FINAL_EVAL_HEADERS,(.*)", log)
    val = re.findall(r"FINAL_EVAL_VALUES,(.*)", log)
    eval_rows = re.findall(r"\[EVAL\] epoch (\d+) \| L1 loss=([\d.]+) \| duration=([\d.]+)s", log)
    if eval_rows:
        A("| epoch | L1 latents | duration |")
        A("|---:|---:|---:|")
        for e in eval_rows:
            A(f"| {e[0]} | {e[1]} | {float(e[2])/60:.1f} min |")
        A("")
    if hdr and val:
        keys = hdr[-1].split(",")
        vals = val[-1].split(",")
        A("Final eval, per-horizon L1 (t0 = first predicted frame, t11 = 1.2 s ahead):")
        A("")
        A("| " + " | ".join(k for k in keys if k.startswith("l1_latents_t")) + " |")
        A("|" + "---:|" * sum(1 for k in keys if k.startswith("l1_latents_t")))
        A("| " + " | ".join(v for k, v in zip(keys, vals) if k.startswith("l1_latents_t")) + " |")
        A("")
        A("_Error rising left to right means later futures are harder to predict — expected._")
        A("")
    if metrics:
        A("```json")
        A(json.dumps(metrics, indent=2)[:4000])
        A("```")
    else:
        A("_No `dumps/metrics.json` found._")
    if lat:
        d = [float(x[1]) for x in lat]
        A("")
        A(f"Sampling latency: {len(d)} measurement(s), mean denoising "
          f"**{sum(d)/len(d)/1000:.1f} s** per generated future (50-step ODE).")
    A("")
    A("Detection metrics (AP_L) are empty **by design** — they need a detectron2/detrex "
      "checkpoint that is not part of the open-source release.")
    A("")

    # ---- artifacts -----------------------------------------------------
    A("## Artifacts")
    A("")
    A(f"Experiment dir: `{exp_dir}`")
    A("")
    A(f"- **Checkpoints** ({len(ckpts)}): " +
      (", ".join(f"`{c.name}`" for c in ckpts[:12]) + (" ..." if len(ckpts) > 12 else "")
       if ckpts else "_none_"))
    A(f"- **Dump dirs** ({len(dumps)}): " +
      (", ".join(f"`{d.name}`" for d in dumps[:8]) + (" ..." if len(dumps) > 8 else "")
       if dumps else "_none_"))
    A(f"- **Prediction videos** (.mp4): {len(videos)}")
    A(f"- **Canvas strips** (.png): {len(canvases)}")
    if videos:
        A("")
        A("Look at these first — they are the qualitative result:")
        for v in sorted(videos)[:5]:
            A(f"  - `{v}`")
    A("")
    A(f"- Full training log: `{log_path}`")
    A(f"- Wrapper stdout/stderr: `{wrapper_log}`")
    A("- **Setup record** (every decision made before launch, including the batch-size "
      "calibration and the evaluator patch): `OVERNIGHT_SETUP_LOG.md`")
    A("")

    # ---- errors --------------------------------------------------------
    A("## Errors and warnings")
    A("")
    if tracebacks or err_lines:
        A(f"**{len(tracebacks)} traceback(s)** in the wrapper log.")
        A("")
        A("```")
        for l in err_lines[:25]:
            A(l[:300])
        A("```")
        A("")
        A("Full context:")
        A("```")
        idx = wrapper.find("Traceback (most recent call last)")
        A(wrapper[idx:idx + 3000] if idx >= 0 else wrapper[-3000:])
        A("```")
    else:
        A("No tracebacks. Expected warnings that are **not** problems:")
    A("")
    for w in warn_lines[:6]:
        A(f"- `{w.split(' - WARNING - ')[-1][:200]}`")
    A("")

    # ---- environment ---------------------------------------------------
    A("## Environment")
    A("")
    A("```")
    A(sh("nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader"))
    A(f"disk free: {sh('df -h --output=avail /home/davi | tail -1')}")
    A("```")
    A("")

    A("## Suggested next steps")
    A("")
    if ok:
        A("1. Watch the `.mp4` predictions — do the generated futures look like plausible "
          "driving scenes, or do they smear? That is the qualitative read on whether "
          "flow matching in DINOv3 space held together at this data scale.")
        A("2. Compare the epoch-1 loss to the final epoch in the table above.")
        A("3. Decide whether to scale up data (more Waymo shards) or move to the "
          "traversability readout — the insertion point is `_apply_downstream_tasks` "
          "in `trainer/trainer.py` feeding `evaluator.accumulate`.")
    else:
        A("1. Read the **Errors** section — the traceback is quoted in full.")
        A("2. Checkpoints from any completed epochs are listed under **Artifacts** and "
          "can be evaluated with `local_eval.yaml` by pointing `trainer.checkpoint_path` "
          "at one of them.")
    A("")

    out_path.write_text("\n".join(L))
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
