import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CACHE = Path("~/wandb/cache").expanduser()
OUT = Path(__file__).with_name("plot.png")
OUT_PDF = OUT.with_suffix(".pdf")
PRETRAIN_SWEEP = "uqznpx3l"
FINETUNE_SWEEP = "0hy3kujo"
MIN_PRETRAIN_LR = 0
MAX_FINETUNE_LR = 1


def format_lr(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def load_sweep(sweep_id: str) -> pd.DataFrame:
    rows = []
    for run in sorted(path for path in (CACHE / "sweeps" / sweep_id).iterdir() if path.is_dir()):
        cfg = json.loads((run / "config.json").read_text())
        summ = json.loads((run / "summary.json").read_text())
        meta = json.loads((run / "metadata.json").read_text())
        rows.append(
            dict(
                run_id=run.name,
                state=str(meta.get("state", "")).lower(),
                ft_lr=cfg.get("opt.peak_lr"),
                pt_lr=cfg.get("load_cfg.opt.peak_lr", cfg.get("opt.peak_lr")),
                reg_method=cfg.get("reg.method") or "null",
                pretrain_run=Path(cfg.get("model.load_path") or "").name,
                prev_ntp=summ.get("prev_ntp"),
                finetune_steps=summ.get("_step"),
            )
        )
    return pd.DataFrame(rows)


def load_points() -> pd.DataFrame:
    pretrain = load_sweep(PRETRAIN_SWEEP)[["run_id", "pt_lr"]].rename(
        columns={"run_id": "pretrain_run", "pt_lr": "pt_lr_from_sweep"}
    )
    points = load_sweep(FINETUNE_SWEEP)
    points = points[points.state == "finished"].merge(pretrain, on="pretrain_run")
    points["pt_lr"] = points.pt_lr_from_sweep
    return points.query("pt_lr >= @MIN_PRETRAIN_LR and ft_lr <= @MAX_FINETUNE_LR").sort_values(
        ["pt_lr", "ft_lr"]
    )


def main() -> None:
    points = load_points()
    pt_lrs = sorted(points.pt_lr.unique())
    cmap = plt.get_cmap("plasma")
    colors = {lr: cmap(0.05 + 0.80 * i / max(1, len(pt_lrs) - 1)) for i, lr in enumerate(pt_lrs)}

    fig, axs = plt.subplots(
        1,
        4,
        figsize=(9.5, 2.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 0.08, 1]},
    )

    labels = {"null": "No regularization", "kl_pre": "KL regularization"}
    for ax, (reg_method, title) in zip(axs[:2], labels.items()):
        for pt_lr, group in points[points.reg_method == reg_method].groupby("pt_lr", sort=True):
            group = group.sort_values("ft_lr")
            ax.plot(group.ft_lr, group.prev_ntp, marker="o", lw=1.8, color=colors[pt_lr], label=format_lr(pt_lr))
        ax.set_xscale("log")
        ax.set_xlabel("Finetuning learning rate")
        ax.set_ylabel("Pretraining loss after finetuning")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
    axs[1].set_ylabel("")
    axs[1].legend(title="Pretraining LR", frameon=True, framealpha=0.9, fontsize=8, loc="upper left")

    axs[2].set_axis_off()
    axs[2].plot([0.5, 0.5], [0, 1], color="black", ls="--", lw=1.2, transform=axs[2].transAxes, clip_on=False)

    step_points = points[points.reg_method == "null"].copy()
    step_points["lr_times_steps"] = step_points.ft_lr * step_points.finetune_steps
    for pt_lr, group in step_points.groupby("pt_lr", sort=True):
        group = group.sort_values("ft_lr")
        baseline = group.iloc[0].lr_times_steps
        axs[3].axhline(baseline, color=colors[pt_lr], ls="--", lw=1.2, alpha=0.55, zorder=0)
        axs[3].plot(group.ft_lr, group.lr_times_steps, marker="o", lw=1.8, color=colors[pt_lr], label=format_lr(pt_lr))

    y_min, y_max = step_points.lr_times_steps.min(), step_points.lr_times_steps.max()
    y_pad = 0.05 * (y_max - y_min)

    axs[3].set_xscale("log")
    axs[3].set_ylim(y_min - y_pad, y_max + y_pad)
    axs[3].set_xlabel("Finetuning learning rate")
    axs[3].set_ylabel("Finetuning LR x num steps")
    # axs[3].set_title("No regularization", fontsize=11, fontweight="bold")
    axs[3].grid(True, which="both", alpha=0.25)

    fig.savefig(OUT, dpi=220)
    fig.savefig(OUT_PDF)
    print(f"{OUT}\n{OUT_PDF}")


if __name__ == "__main__":
    main()
