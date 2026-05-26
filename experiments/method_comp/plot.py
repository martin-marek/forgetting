#!/usr/bin/env python3
"""Plot cached trajectories from the direct method-comparison sweeps."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CACHE = Path("~/wandb/cache").expanduser()
OUT_DIR = Path(__file__).resolve().parent
SWEEPS = ["na6n2q17", "qs2vknqw"]
XY = ["x_downstream_likelihood", "y_pretraining_likelihood"]
EARLY_STOP_X_TOL = 1e-5
METHOD_ORDER = {
    "standard finetuning": 0,
    "LoRA": 1,
    "KL (real data)": 2,
    "KL (self-generated)": 3,
    "NTP (real data)": 4,
    "NTP (self-generated)": 5,
}
TOP_DASH_SEGMENT = 0.8
TOP_DASH_OFFSETS = [0, 2, 1, 3]
TOP_LINESTYLES = [
    (offset * TOP_DASH_SEGMENT, (TOP_DASH_SEGMENT, TOP_DASH_SEGMENT * 3))
    for offset in TOP_DASH_OFFSETS
]


def label_for_config(config: dict) -> str:
    lora_rank = config.get("model.lora_rank")
    if lora_rank not in (None, "null", 0, "0"):
        return f"LoRA (rank {lora_rank})"

    method = config.get("reg.method")
    if method is None:
        return "standard finetuning"

    method_name = method.removesuffix("_pre").upper()
    source = "self-generated" if config.get("reg.synth") else "real data"
    return f"{method_name} ({source})"


def cached_history(run_dir: Path) -> pd.DataFrame:
    paths = sorted((run_dir / "history").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no cached history for {run_dir.name!r} under {run_dir / 'history'}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True, sort=False)


def load_run(run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text())
    frame = cached_history(run_dir)
    frame[["_step", "valid_ntp", "prev_ntp"]] = frame[["_step", "valid_ntp", "prev_ntp"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    points = (
        frame.dropna(subset=["valid_ntp", "prev_ntp"])
        .sort_values("_step", kind="mergesort")
        .assign(
            x_downstream_likelihood=lambda df: -df["valid_ntp"],
            y_pretraining_likelihood=lambda df: -df["prev_ntp"],
            run_id=run_dir.name,
            label=label_for_config(config),
        )
    )
    if points.empty:
        raise ValueError(f"cached history for {run_dir.name!r} has no valid_ntp/prev_ntp points")

    stop_idx = (
        points["x_downstream_likelihood"]
        >= points["x_downstream_likelihood"].max() - EARLY_STOP_X_TOL
    ).to_numpy().argmax()
    return points.iloc[: stop_idx + 1].copy()


def load_points() -> list[pd.DataFrame]:
    runs = []
    for sweep in SWEEPS:
        sweep_dir = CACHE / "sweeps" / sweep
        run_dirs = sorted(path for path in sweep_dir.iterdir() if path.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"no cached runs for sweep {sweep!r} under {sweep_dir}")
        runs.extend(load_run(run_dir) for run_dir in run_dirs)
    return sorted(runs, key=line_order)


def line_order(frame: pd.DataFrame) -> tuple[float, int]:
    label = frame["label"].iloc[0]
    return (-frame["y_pretraining_likelihood"].iloc[-1], METHOD_ORDER.get(label, 99))


def plot(runs: list[pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 2.5), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = [
        (
            frame,
            frame["label"].iloc[0],
            colors[idx % len(colors)],
            TOP_LINESTYLES[idx] if idx < len(TOP_LINESTYLES) else "-",
        )
        for idx, frame in enumerate(runs)
    ]
    for zorder, (frame, label, color, linestyle) in enumerate(reversed(styles), 2):
        ax.plot(
            frame["x_downstream_likelihood"],
            frame["y_pretraining_likelihood"],
            color=color,
            linestyle=linestyle,
            lw=2.0,
            alpha=0.9,
            dash_capstyle="butt",
            label=label,
            solid_capstyle="butt",
            zorder=zorder,
        )
        ax.scatter(
            frame["x_downstream_likelihood"].iloc[-1],
            frame["y_pretraining_likelihood"].iloc[-1],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=zorder + len(styles),
        )

    points = pd.concat(runs, ignore_index=True)
    pad = (points[XY].max() - points[XY].min()).clip(lower=0.02) * 0.04
    ax.set_xlim(
        points["x_downstream_likelihood"].min() - pad["x_downstream_likelihood"],
        points["x_downstream_likelihood"].max() + pad["x_downstream_likelihood"],
    )
    ax.set_ylim(
        points["y_pretraining_likelihood"].min() - pad["y_pretraining_likelihood"],
        points["y_pretraining_likelihood"].max() + pad["y_pretraining_likelihood"],
    )
    ax.set(
        xlabel="Nemotron-CC-Math log-likelihood",
        ylabel="FineWeb-Edu log-likelihood",
    )
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            linestyle=(0, (2.2, 1.4)) if isinstance(linestyle, tuple) else "-",
            lw=2.0,
            label=label,
            dash_capstyle="butt",
            solid_capstyle="butt",
        )
        for _, label, color, linestyle in styles
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8.0,
        handlelength=2.4,
        loc="best",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT_DIR / "plot.png", dpi=240)
    fig.savefig(OUT_DIR / "plot.pdf")


if __name__ == "__main__":
    plot(load_points())
