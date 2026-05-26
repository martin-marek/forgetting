import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CACHE = Path("~/wandb/cache").expanduser()
OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "plot.png"
PDF_PATH = OUT_DIR / "plot.pdf"
LR_SWEEP = "90tvxwwn"
REPLAY_SWEEP = "9myhwsbn"
REPLAY_SLICE = slice(1, 8)

LR_METHOD = "changing LR"
REPLAY_METHOD = "pretraining data replay"
METHODS = [
    (LR_METHOD, r"Reduce learning rate", LR_SWEEP),
    (REPLAY_METHOD, r"Increase replay", REPLAY_SWEEP),
]
DEEP = sns.color_palette("deep").as_hex()
COLORS = {LR_METHOD: DEEP[0], REPLAY_METHOD: DEEP[3]}
NUMERIC_COLUMNS = [
    "time_s",
    "prev_ntp",
    "train_loss",
    "valid_ntp",
    "target_loss",
    "opt_peak_lr",
    "reg_coeff",
]


def load_points() -> pd.DataFrame:
    rows = []
    for method, _, sweep_id in METHODS:
        for run_dir in sorted(path for path in (CACHE / "sweeps" / sweep_id).iterdir() if path.is_dir()):
            metadata = json.loads((run_dir / "metadata.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            rows.append(
                dict(
                    method=method,
                    run_id=run_dir.name,
                    state=metadata["state"].lower(),
                    time_s=summary.get("train_time_elapsed"),
                    prev_ntp=summary.get("prev_ntp"),
                    train_loss=summary.get("train_loss"),
                    valid_ntp=summary.get("valid_ntp"),
                    target_loss=config.get("stop.target_loss", config.get("opt.target_loss")),
                    opt_peak_lr=config.get("opt.peak_lr"),
                    reg_coeff=config.get("reg.coeff"),
                )
            )

    points = pd.DataFrame(rows)
    points[NUMERIC_COLUMNS] = points[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    loss_min = points[["train_loss", "valid_ntp"]].min(axis=1)
    points["target_reached"] = loss_min <= points.target_loss
    points["plotted"] = points.state == "finished"
    points["reason"] = points.apply(skip_reason, axis=1)
    points["hparam"] = points.apply(hparam, axis=1)
    points["order"] = points.reg_coeff
    points.loc[points.method == LR_METHOD, "order"] = -points.opt_peak_lr

    replay = (points.method == REPLAY_METHOD) & points.plotted
    kept = points.loc[replay].sort_values(["order", "run_id"]).iloc[REPLAY_SLICE].index
    hidden = replay & ~points.index.isin(kept)
    points.loc[hidden, ["plotted", "reason"]] = [False, "hidden by replay slice [1:8]"]
    return points.sort_values(["method", "order", "run_id"], ignore_index=True)


def skip_reason(row: pd.Series) -> str:
    if row.state != "finished":
        return f"state={row.state}"
    return ""


def hparam(row: pd.Series) -> str:
    if row.method == LR_METHOD:
        return f"lr={row.opt_peak_lr:.2g}"
    return f"replay={row.reg_coeff:.4g}"


def format_scientific(value: float) -> str:
    exponent = math.floor(math.log10(value))
    mantissa = value / (10**exponent)
    if math.isclose(mantissa, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        return rf"$10^{{{exponent}}}$"
    return rf"${f'{mantissa:.1f}'.rstrip('0').rstrip('.')}\times10^{{{exponent}}}$"


def point_label(row: pd.Series, anchor: bool = False) -> str:
    lr = format_scientific(row.opt_peak_lr)
    if anchor:
        return f"LR={lr}\nreplay=0"
    if row.method == LR_METHOD:
        return f"LR decreases to {lr}\nreplay=0"
    return f"LR={lr}\nreplay increases to {row.reg_coeff:.4g}"


def interpolate_time_at_prev_ntp(points: pd.DataFrame, prev_ntp: float) -> float:
    rows = list(points.sort_values("prev_ntp", ascending=False).itertuples(index=False))
    for high, low in zip(rows, rows[1:]):
        if high.prev_ntp >= prev_ntp >= low.prev_ntp:
            fraction = (prev_ntp - low.prev_ntp) / (high.prev_ntp - low.prev_ntp)
            log_time = math.log10(low.time_s) + fraction * (
                math.log10(high.time_s) - math.log10(low.time_s)
            )
            return 10**log_time


def annotate_points(ax: plt.Axes, plotted: pd.DataFrame) -> None:
    label_box = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.86,
    }
    arrow = {
        "arrowstyle": "-",
        "color": "#666666",
        "linewidth": 0.7,
        "shrinkA": 0,
        "shrinkB": 6,
    }
    lr_points = plotted[plotted.method == LR_METHOD].sort_values(["order", "run_id"])
    replay_points = plotted[plotted.method == REPLAY_METHOD].sort_values(["order", "run_id"])
    anchor = lr_points.iloc[0]
    ax.annotate(
        point_label(anchor, anchor=True),
        (anchor.time_s, anchor.prev_ntp),
        xytext=(18, -24),
        textcoords="offset points",
        fontsize=8.5,
        color="#222222",
        bbox=label_box,
        arrowprops=arrow,
        zorder=5,
    )

    blue_row = lr_points.iloc[
        (lr_points.time_s.map(math.log10) - 2).abs().argsort().iloc[0]
    ]
    red_time = interpolate_time_at_prev_ntp(replay_points, blue_row.prev_ntp)
    multiplier = blue_row.time_s / red_time
    ax.annotate(
        "",
        xy=(blue_row.time_s, blue_row.prev_ntp),
        xytext=(red_time, blue_row.prev_ntp),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#444444",
            "linewidth": 1.8,
            "shrinkA": 8,
            "shrinkB": 8,
        },
        zorder=4,
    )
    ax.text(
        math.sqrt(red_time * blue_row.time_s),
        blue_row.prev_ntp - 0.02,
        rf"$\approx${multiplier:.0f}$\times$ compute",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        bbox=label_box,
        zorder=5,
    )

    for points, xytext, color, kwargs in [
        (lr_points, (-14, 31), COLORS[LR_METHOD], {"ha": "right"}),
        (replay_points, (62, 30), COLORS[REPLAY_METHOD], {"va": "center"}),
    ]:
        if len(points) <= 1:
            continue
        row = points.iloc[-1]
        ax.annotate(
            point_label(row),
            (row.time_s, row.prev_ntp),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            bbox=label_box,
            arrowprops={**arrow, "color": color},
            zorder=5,
            **kwargs,
        )


def plot(points: pd.DataFrame) -> None:
    plotted = points[points.plotted]
    fig, ax = plt.subplots(figsize=(4.8, 2.7), constrained_layout=True)

    for method, label, _ in METHODS:
        method_points = plotted[plotted.method == method].sort_values(["order", "run_id"])
        ax.plot(
            method_points.time_s,
            method_points.prev_ntp,
            color=COLORS[method],
            linewidth=1.6,
            alpha=0.7,
            zorder=2,
        )
        all_target_reached = method_points.target_reached.all()
        for reached, suffix, face in [
            (True, "target reached", COLORS[method]),
            (False, "available so far", "white"),
        ]:
            subset = method_points[method_points.target_reached == reached]
            if subset.empty:
                continue
            ax.scatter(
                subset.time_s,
                subset.prev_ntp,
                s=20,
                marker="o",
                facecolors=face,
                edgecolors=COLORS[method],
                linewidths=1.7,
                label=label if all_target_reached else f"{label} ({suffix})",
                zorder=3,
            )

    annotate_points(ax, plotted)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Pretraining loss after finetuning")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    plt.close(fig)


def print_report(points: pd.DataFrame) -> None:
    for method, _, _ in METHODS:
        method_points = points[points.method == method]
        print(
            f"{method}: {method_points.plotted.sum()} plotted "
            f"({(method_points.state == 'finished').sum()}/{len(method_points)} finished)"
        )

    hidden = points[~points.plotted]
    print("\nNot plotted:")
    for row in hidden.itertuples(index=False):
        print(f"{row.method} {row.run_id} {row.hparam}: {row.reason}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_points()
    plot(points)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}\n")
    print_report(points)


if __name__ == "__main__":
    main()
