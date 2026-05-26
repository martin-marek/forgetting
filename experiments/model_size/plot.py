from pathlib import Path
import json

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def runs(cache, sweep):
    rows = []
    for run_dir in sorted((cache / "sweeps" / sweep).iterdir()):
        if run_dir.is_dir():
            cfg, summary, meta = (read_json(run_dir / f"{x}.json") for x in ("config", "summary", "metadata"))
            rows.append({"run": run_dir.name, "state": str(meta.get("state", "")).lower(), "cfg": cfg, "summary": summary})
    return pd.DataFrame(rows)


def target_loss_reached(summary):
    return summary.get("stop_reason") == "target_loss" or bool(summary.get("target_loss_reached"))


def model_params(summary):
    return summary["n_param_nonembed"] + summary["n_param_embed"]


def model_width(cfg):
    return cfg["model.D"]


def main(
    cache="~/wandb/cache",
    pretrain_sweep="dns1uo8c", # "sdbjefda",
    finetune_sweep="919li0b9", # "919li0b9", "5mm6szbb", "kur3je79",
    out_dir=str(HERE),
    y_offset=2.75,
    lr=3.2e-4,
):
    cache, out_dir = Path(cache).expanduser(), Path(out_dir).expanduser()

    pre = runs(cache, pretrain_sweep).query("state == 'finished'").assign(
        intended_D=lambda x: x.cfg.map(model_width),
        actual_D=lambda x: x.cfg.map(model_width),
        actual_params_pre=lambda x: x.summary.map(model_params),
        pretraining_loss=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
    )[["run", "intended_D", "actual_D", "actual_params_pre", "pretraining_loss"]].rename(columns={"run": "pretrain_run"})

    ft = runs(cache, finetune_sweep).query("state in ['finished', 'running']").assign(
        intended_D=lambda x: x.cfg.map(model_width),
        load_actual_D=lambda x: x.cfg.map(lambda c: c["load_cfg.model.D"]),
        actual_params_ft=lambda x: x.summary.map(model_params),
        finetune_end=lambda x: x.summary.map(lambda s: s.get("prev_ntp")),
        downstream_end=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
        peak_lr=lambda x: x.cfg.map(lambda c: c.get("opt.peak_lr")),
        stop_reason=lambda x: x.summary.map(lambda s: s.get("stop_reason")),
        target_loss_reached=lambda x: x.summary.map(target_loss_reached),
    )
    ft["peak_lr"] = pd.to_numeric(ft["peak_lr"])
    # ft = ft[np.isclose(ft.peak_lr, lr)][["run", "state", "intended_D", "load_actual_D", "actual_params_ft", "finetune_end", "downstream_end", "peak_lr", "stop_reason", "target_loss_reached"]].rename(columns={"run": "finetune_run"})

    df = pre.merge(ft, on="intended_D")
    numeric = ["intended_D", "actual_D", "load_actual_D", "actual_params_pre", "actual_params_ft", "pretraining_loss", "finetune_end", "downstream_end"]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    df["model_size_m"] = (df.actual_params_pre / 1e6).round().astype(int)
    df["intended_model_size_m"] = ((12 * 6 * df.intended_D**2 + 2 * df.intended_D * 4096) / 1e6).round().astype(int)
    df["actual_params_match"] = df.actual_params_pre.eq(df.actual_params_ft)
    df["intended_matches_actual"] = df.model_size_m.eq(df.intended_model_size_m)
    df["finetune_start"] = df["pretraining_loss"]
    df = df.sort_values(["model_size_m", "intended_D"])
    if not df.intended_matches_actual.all():
        print("warning: intended sweep sizes do not match logged actual parameter counts")
    if not df.actual_params_match.all():
        print("warning: pretrain and finetune actual parameter counts differ")
    finished_df = df[df.state == "finished"]
    running_df = df[df.state == "running"]
    missed = len(finished_df) - int(finished_df.target_loss_reached.sum())
    if missed:
        reasons = finished_df.loc[~finished_df.target_loss_reached, "stop_reason"].fillna("unknown").value_counts().to_dict()
        print(f"plotting {missed} finetune runs that did not reach target loss as x markers: {reasons}")
    if len(running_df):
        print(f"plotting {len(running_df)} running finetune runs as triangles")

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    pre_plot = df.groupby("model_size_m", as_index=False)["pretraining_loss"].mean()
    target_df = finished_df[finished_df.target_loss_reached]
    missed_df = finished_df[~finished_df.target_loss_reached]
    best_idx = target_df.groupby("model_size_m")["finetune_end"].idxmin()
    best_ft = target_df.loc[best_idx].sort_values("model_size_m")
    min_loss = min(df.pretraining_loss.min(), df.finetune_end.min())
    max_loss = max(df.pretraining_loss.max(), df.finetune_end.max())
    if y_offset >= min_loss:
        y_offset = np.floor((min_loss - 0.02) * 100) / 100

    def offset_log(y):
        y = np.asarray(y)
        out = np.full_like(y, np.nan, dtype=float)
        np.log(y - y_offset, out=out, where=y > y_offset)
        return out

    def inv_offset_log(y):
        return np.exp(np.asarray(y)) + y_offset

    ax.plot(pre_plot.model_size_m, pre_plot.pretraining_loss, "s-", color="tab:blue", label="pretraining / finetune start")
    ax.scatter(target_df.model_size_m, target_df.finetune_end, s=28, alpha=0.65, color="tab:green", marker="o", label="finished, target loss reached")
    ax.scatter(missed_df.model_size_m, missed_df.finetune_end, s=42, alpha=0.85, color="tab:red", marker="x", label="finished, target loss not reached")
    ax.scatter(running_df.model_size_m, running_df.finetune_end, s=38, alpha=0.75, color="tab:orange", marker="^", label="running")
    ax.plot(best_ft.model_size_m, best_ft.finetune_end, "-", color="tab:green", label="best finished target-loss run")
    ax.set_xlabel("Model size")
    ax.set_ylabel(f"Pretraining loss")
    # ax.set_yscale("function", functions=(offset_log, inv_offset_log))
    ax.set_xscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
    ax.set_xticks(pre_plot.model_size_m)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}M"))
    y_ticks = np.array([3.32, 3.34, 3.36, 3.40, 3.50, 3.60, 3.80, 4.00, 4.20])
    y_ticks = y_ticks[(y_ticks > y_offset) & (y_ticks <= max_loss + 0.03)]
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
    ax.set_ylim((2.99, 3.41))
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"plot.{ext}", dpi=220)


if __name__ == "__main__":
    fire.Fire(main)
