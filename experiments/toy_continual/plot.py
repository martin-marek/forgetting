import json
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter


SWEEP_ID = "tshfjvvu"
CACHE = Path("~/wandb/cache").expanduser()
PNG_PATH = Path(__file__).with_name("plot.png")
PDF_PATH = Path(__file__).with_name("plot.pdf")

TASKS = ("add", "reversal", "sort", "modadd")
PALETTE = sns.color_palette("deep", n_colors=len(TASKS))
PANELS = [
    ("null", "Without replay"),
    ("ntp_pre", "With self-generated replay"),
]


def format_k_steps(value, _):
    if value == 0:
        return "0"
    if value % 1000 == 0:
        return f"{int(value / 1000)}K"
    return ""


def reg_method_matches(actual, expected):
    if expected == "null":
        return actual in (None, "null")
    return actual == expected


def load_run(reg_method):
    matches = []
    sweep_dir = CACHE / "sweeps" / SWEEP_ID
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get("toy.tasks") != list(TASKS) or not reg_method_matches(config.get("reg.method"), reg_method):
            continue

        history_paths = sorted((run_dir / "history").glob("*.parquet"))
        if history_paths:
            matches.append((config_path.stat().st_mtime, history_paths, config))

    if not matches:
        raise FileNotFoundError(f"no cached run with reg.method={reg_method!r} under {sweep_dir}")

    _, history_paths, config = max(matches)
    history = pd.concat([pd.read_parquet(path) for path in history_paths], ignore_index=True)
    return history.sort_values("_step"), config


def plot_panel(ax, history, config):
    for idx, task in enumerate(TASKS):
        ax.plot(
            history["_step"],
            history[f"{task}_accuracy"],
            label=task,
            color=PALETTE[idx],
            linewidth=2.2,
        )

    x_offset = 0.02 * max(float(history["_step"].max()), 1.0)
    for idx, task in enumerate(TASKS):
        step = idx * int(config["toy.steps_per_task"])
        ax.axvline(step, color="0.6", linestyle="--", linewidth=1.8)
        ax.text(
            step + x_offset,
            0.97,
            task,
            rotation=90,
            va="top",
            ha="left",
            fontsize=11,
            color="0.15",
            fontfamily="monospace",
            bbox={"facecolor": "0.9", "edgecolor": "none", "boxstyle": "round,pad=0.18"},
        )

    ax.set_xlabel("Training step")
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(FuncFormatter(format_k_steps))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))


def main():
    runs = [load_run(reg_method) for reg_method, _ in PANELS]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True, constrained_layout=True)

    for ax, (_, title), (history, config) in zip(axes, PANELS, runs):
        plot_panel(ax, history, config)
        ax.set_title(title, fontsize=11, fontweight="bold")

    axes[0].set_ylabel("Eval accuracy")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    legend = axes[1].legend(title="Task", loc="lower right", frameon=True, handlelength=1.2)
    legend.get_title().set_fontfamily("monospace")
    for text in legend.get_texts():
        text.set_fontfamily("monospace")

    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    plt.close(fig)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
