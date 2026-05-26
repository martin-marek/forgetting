#!/usr/bin/env python3
"""Render the 1D regression KL comparison from precomputed CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator

DATA_DIR = Path(__file__).with_name("plot_data")
PNG_PATH = Path(__file__).with_name("plot.png")
PDF_PATH = Path(__file__).with_name("plot.pdf")
DPI = 220
SPLIT_X = 0.12
DEEP = sns.color_palette("deep")
MODEL_COLOR = DEEP[0]
NEW_POINT_COLOR = DEEP[2]
OLD_POINT_COLOR = DEEP[1]
REFERENCE_COLOR = DEEP[7]
INK_COLOR = "#111827"
SPLIT_COLOR = "#E5E7EB"


def darken(color, factor: float = 0.58) -> tuple[float, float, float]:
    return tuple(channel * factor for channel in to_rgb(color))


def load_table(name: str) -> np.ndarray:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python {Path(__file__).with_name('compute.py')}` first.")
    return np.genfromtxt(path, delimiter=",", names=True)


def configure_figure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_comparison_panel(
    ax: plt.Axes,
    x: np.ndarray,
    ref_curve: np.ndarray,
    curve: np.ndarray,
    interval_low: np.ndarray | None,
    interval_high: np.ndarray | None,
    old_points: np.ndarray,
    new_points: np.ndarray,
    title: str,
    split_x: float,
) -> None:
    old_mask = x <= split_x

    ax.axvline(split_x, color=SPLIT_COLOR, lw=1.2, zorder=0)
    if interval_low is not None and interval_high is not None:
        ax.fill_between(
            x,
            interval_low,
            interval_high,
            color=MODEL_COLOR,
            alpha=0.17,
            zorder=1,
        )

    # ax.plot(
    #     x[old_mask],
    #     ref_curve[old_mask],
    #     color=REFERENCE_COLOR,
    #     lw=1.9,
    #     ls=(0, (4, 3)),
    #     alpha=0.8,
    #     zorder=2,
    #     solid_capstyle="round",
    # )
    ax.plot(x, curve, color=MODEL_COLOR, lw=3.2, zorder=3, solid_capstyle="round")

    ax.scatter(
        old_points["x"],
        old_points["y"],
        s=32,
        facecolors=OLD_POINT_COLOR,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.9,
        zorder=4,
    )
    ax.scatter(
        new_points["x"],
        new_points["y"],
        s=40,
        facecolors=NEW_POINT_COLOR,
        edgecolors="white",
        linewidths=0.55,
        alpha=0.9,
        zorder=4,
    )

    ax.text(
        0.25,
        0.97,
        "old data",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=darken(OLD_POINT_COLOR, 0.9),
    )
    ax.text(
        0.77,
        0.97,
        "new data",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=darken(NEW_POINT_COLOR, 0.9),
    )

    ax.set_title(title, pad=10)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.45, 2.55)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-2, 0, 2])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(False)
    ax.spines["left"].set_color(INK_COLOR)
    ax.spines["bottom"].set_color(INK_COLOR)
    ax.tick_params(axis="both", which="major", colors=INK_COLOR, length=3.5)
    ax.tick_params(axis="both", which="minor", colors=INK_COLOR, length=2.0)


def save_figure(
    curves: np.ndarray,
    old_points: np.ndarray,
    new_points: np.ndarray,
) -> None:
    configure_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True, sharey=True)
    plot_comparison_panel(
        axes[0],
        curves["x"],
        curves["old_curve"],
        curves["old_curve"],
        None,
        None,
        old_points,
        new_points,
        "Before finetuning",
        SPLIT_X,
    )
    plot_comparison_panel(
        axes[1],
        curves["x"],
        curves["old_curve"],
        curves["finetuned_curve"],
        curves["finetuned_q25"],
        curves["finetuned_q75"],
        old_points,
        new_points,
        "Finetuning on new data only",
        SPLIT_X,
    )
    plot_comparison_panel(
        axes[2],
        curves["x"],
        curves["old_curve"],
        curves["kl_finetuned_curve"],
        curves["kl_finetuned_q25"],
        curves["kl_finetuned_q75"],
        old_points,
        new_points,
        "Finetuning w/ KL regularization",
        SPLIT_X,
    )

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=DPI)
    fig.savefig(PDF_PATH)
    plt.close(fig)


def main() -> None:
    curves = load_table("curves.csv")
    old_points = load_table("old_points.csv")
    new_points = load_table("new_points.csv")
    save_figure(curves, old_points, new_points)

    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
