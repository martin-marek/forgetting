from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.optimize import curve_fit


PRETRAIN_SWEEP = "e75af14b"
# FINETUNE_SWEEP = "t7r7qwp4"
FINETUNE_SWEEPS = ("04yiml9k", "6eslz8t8") # "0wk7vxgz", "xc7vpxs6"
# FINETUNE_SWEEP = "mez7lpf4"
FINETUNE_OVERLAY_RUNS = [
    # "tyasxzhu",
    # "v3hkdtmz",
    # "2hxb6ibv",
    # "eo8jphsc",
    # "kjr0mca8",
    # "u9fakoo9",
    # "f46ad73a",
]
CACHE = Path("~/wandb/cache").expanduser()
OUT = Path(__file__).with_name("plot.png")

PARETO_Z = 2
PRETRAIN_Z = 3
FINETUNE_Z = 3
FINETUNE_OVERLAY_Z = 5
ZOOM_BOX_Z = 4
PRETRAIN_SUBSAMPLE_EVERY = 5
TRAJECTORY_LW = 1.8
TRAJECTORY_SCATTER_SIZE = 7
FINAL_SCATTER_SIZE = 22
START_SCATTER_SIZE = 72
START_MARKER_LW = 1.8
START_LABEL_FONT_SIZE = 9
START_LABEL_X_OFFSET = -5
START_LABEL_Y_OFFSET = 5.5
START_LABEL_BBOX = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", linewidth=0.7, alpha=0.78)
PRETRAIN_X_SHIFT = 0.12
PRETRAIN_Y_SHIFT = 0.12
PRETRAIN_X_LOWER_PAD = 0.04
PRETRAIN_Y_LOWER_PAD = 0.02
FRONTIER_POWER_BLEND = 0.3
FRONTIER_POWER_X_SHIFT = 0.05
FRONTIER_POWER_Y_SHIFT = 0.01


def truncated_cmap(name, vmin=0.0, vmax=1.0, n=256):
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(vmin, vmax, n))
    return LinearSegmentedColormap.from_list(f"{name}_{vmin:g}_{vmax:g}", colors)


def frontier_points(frontier):
    points = frontier.sort_values("mix_ntp")
    x = points.mix_ntp.to_numpy()
    y = points.valid_ntp.to_numpy()

    keep = np.r_[True, y[1:] < np.minimum.accumulate(y[:-1])]
    return x[keep], y[keep]


def power_frontier(x, scale, power, x0, y0):
    return y0 + scale / np.maximum(x - x0, 1e-9) ** power


def fit_power_frontier(x, y):
    x0 = x[0] - FRONTIER_POWER_X_SHIFT
    y0 = y[-1] - FRONTIER_POWER_Y_SHIFT
    popt, _ = curve_fit(
        lambda xx, scale, power: power_frontier(xx, scale, power, x0, y0),
        x,
        y,
        p0=[0.02, 1.0],
        bounds=([1e-9, 0.05], [10.0, 8.0]),
        maxfev=20000,
    )
    return popt[0], popt[1], x0, y0


def frontier_curve(frontier, x_values):
    x, y = frontier_points(frontier)
    linear_y = np.interp(x_values, x, y)
    power_y = power_frontier(x_values, *fit_power_frontier(x, y))
    return (1 - FRONTIER_POWER_BLEND) * linear_y + FRONTIER_POWER_BLEND * power_y


def frontier_path(frontier, xlim, ylim):
    x, _ = frontier_points(frontier)
    curve_x = np.linspace(x[0], x[-1], 700)
    curve_y = frontier_curve(frontier, curve_x)
    right_y = curve_y[-1]
    return (
        np.concatenate([[x[0]], curve_x, [x[-1], xlim[1]]]),
        np.concatenate([[ylim[1]], curve_y, [right_y, right_y]]),
    )


def shade_inaccessible(ax, frontier, xlim, ylim):
    frontier_x, frontier_y = frontier_path(frontier, xlim, ylim)
    lower_clip_pad = 0.5 * (ylim[1] - ylim[0])
    shade_floor = ylim[0] - lower_clip_pad
    style = dict(
        facecolor=(0.55, 0.55, 0.55, 0.14),
        edgecolor=(0.45, 0.45, 0.45, 0.32),
        hatch="////",
        linewidth=0,
        zorder=PARETO_Z,
    )
    poly_x = np.concatenate([[xlim[0]], frontier_x, [xlim[1], xlim[0]]])
    poly_y = np.concatenate([[ylim[1]], frontier_y, [shade_floor, shade_floor]])
    ax.fill(poly_x, poly_y, **style)


def padded_limits(x, y, x_min_pad=0.004, y_min_pad=0.02, pad_frac=0.08):
    x_pad = max((x.max() - x.min()) * pad_frac, x_min_pad)
    y_pad = max((y.max() - y.min()) * pad_frac, y_min_pad)
    return (x.min() - x_pad, x.max() + x_pad), (y.min() - y_pad, y.max() + y_pad)


def load_pretrain_points(cache):
    paths = sorted((cache / "sweeps" / PRETRAIN_SWEEP).glob("*/history/*.parquet"))
    frames = []
    for p in paths:
        cfg = json.loads((p.parents[1] / "config.json").read_text())
        frames.append(
            pd.read_parquet(p)
            .sort_values("_step")
            .assign(
                run=p.parents[1].name,
                mix_coeff=cfg["mix.coeff"],
            )
        )

    points = pd.concat(frames, ignore_index=True)
    frontier = points.groupby("mix_coeff", as_index=False).tail(1).sort_values("mix_coeff")
    return points, frontier


def load_finetune_points(cache):
    frames = []
    for sweep in FINETUNE_SWEEPS:
        paths = sorted((cache / "sweeps" / sweep).glob("*/history/*.parquet"))
        for p in paths:
            cfg = json.loads((p.parents[1] / "config.json").read_text())
            frames.append(
                pd.read_parquet(p)
                .sort_values("_step")
                .assign(
                    sweep=sweep,
                    run=p.parents[1].name,
                    load_idx=cfg["model.load_idx"],
                    reg_coeff=cfg["reg.coeff"],
                    peak_lr=cfg["opt.peak_lr"],
                )
            )

    points = pd.concat(frames, ignore_index=True)
    # Comment out the next line to show all runs from the second finetuning sweep.
    points = filter_second_finetune_sweep_to_low_lr_reg_coeff(points)
    return points


def filter_second_finetune_sweep_to_low_lr_reg_coeff(points):
    second_sweep = FINETUNE_SWEEPS[1]
    return points[(points.sweep != second_sweep)| ( 
        (points.sweep == second_sweep)
        & (points.reg_coeff == 1)
        & (points.peak_lr == 3e-4)
    )]


def load_finetune_overlay_points(cache):
    frames = []
    for run in FINETUNE_OVERLAY_RUNS:
        run_dir = cache / "runs" / run
        cfg = json.loads((run_dir / "config.json").read_text())
        frames.append(
            pd.read_parquet(run_dir / "history" / "scan_history.parquet")
            .sort_values("_step")
            .assign(
                run=run,
                load_idx=cfg.get("model.load_idx"),
                reg_coeff=cfg["reg.coeff"],
                reg_method=cfg.get("reg.method"),
            )
        )

    if not frames:
        return pd.DataFrame(columns=["_step", "valid_ntp", "prev_ntp", "run", "load_idx", "reg_coeff", "reg_method"])

    return pd.concat(frames, ignore_index=True)


def plot_pretraining(ax, points):
    norm = Normalize(points.mix_coeff.min(), points.mix_coeff.max())
    cmap = plt.get_cmap("viridis")
    ordered = points.sort_values(["mix_coeff", "_step"])

    for (_, _), traj in ordered.groupby(["mix_coeff", "run"], sort=True):
        color = cmap(norm(traj.mix_coeff.iloc[0]))
        ax.plot(traj.mix_ntp, traj.valid_ntp, color=color, lw=TRAJECTORY_LW, alpha=0.46, zorder=PRETRAIN_Z)

    plot_points = ordered[ordered.groupby("run").cumcount() % PRETRAIN_SUBSAMPLE_EVERY == 0]
    colors = cmap(norm(plot_points.mix_coeff.to_numpy()))
    colors[:, 3] = 0.08
    ax.scatter(
        plot_points.mix_ntp,
        plot_points.valid_ntp,
        s=TRAJECTORY_SCATTER_SIZE,
        color=colors,
        linewidths=0,
        zorder=PRETRAIN_Z + 0.1,
    )

    endpoints = ordered.groupby("run", as_index=False).tail(1)
    ax.scatter(
        endpoints.mix_ntp,
        endpoints.valid_ntp,
        c=endpoints.mix_coeff,
        cmap=cmap,
        norm=norm,
        s=FINAL_SCATTER_SIZE,
        edgecolor="black",
        lw=0.25,
        zorder=PRETRAIN_Z + 0.2,
    )
    return norm, cmap, len(plot_points)


def plot_finetuning(ax, finetune_points):
    norm = LogNorm(finetune_points.reg_coeff.min(), finetune_points.reg_coeff.max())
    cmap = truncated_cmap("plasma", vmax=0.86)

    second_sweep = FINETUNE_SWEEPS[1]
    for (_, _, _), traj in finetune_points.groupby(["sweep", "reg_coeff", "run"], sort=True):
        color = cmap(norm(traj.reg_coeff.iloc[0]))
        if traj.sweep.iloc[0] == second_sweep:
            ax.plot(
                traj.valid_ntp,
                traj.prev_ntp,
                color="black",
                lw=TRAJECTORY_LW + 1.0,
                alpha=0.78,
                zorder=FINETUNE_Z,
            )
        ax.plot(traj.valid_ntp, traj.prev_ntp, color=color, lw=TRAJECTORY_LW, alpha=0.78, zorder=FINETUNE_Z)

    endpoints = finetune_points.groupby("run", as_index=False).tail(1)
    ax.scatter(
        endpoints.valid_ntp,
        endpoints.prev_ntp,
        c=endpoints.reg_coeff,
        cmap=cmap,
        norm=norm,
        s=FINAL_SCATTER_SIZE,
        edgecolor="black",
        lw=0.25,
        zorder=FINETUNE_Z + 0.2,
    )

    starts = finetune_points.sort_values(["sweep", "run", "_step"]).groupby("sweep", as_index=False).head(1)
    ax.scatter(
        starts.valid_ntp,
        starts.prev_ntp,
        marker="x",
        s=START_SCATTER_SIZE,
        color="black",
        linewidths=START_MARKER_LW,
        zorder=FINETUNE_OVERLAY_Z + 0.4,
    )
    start_labels = ["overtrained\n(17K TPP)", "Chinchilla\n(20 TPP)"]
    for (_, start), label in zip(starts.sort_values("prev_ntp").iterrows(), start_labels):
        ax.annotate(
            label,
            xy=(start.valid_ntp, start.prev_ntp),
            xytext=(START_LABEL_X_OFFSET, START_LABEL_Y_OFFSET + 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=START_LABEL_FONT_SIZE,
            color="black",
            linespacing=0.9,
            bbox=START_LABEL_BBOX,
            zorder=FINETUNE_OVERLAY_Z + 0.5,
        )
    return norm, cmap


def plot_finetune_overlay(ax, overlay_points, norm, cmap):
    for _, traj in overlay_points.groupby("run", sort=False):
        has_regularizer = traj.reg_method.iloc[0] is not None
        color = cmap(norm(traj.reg_coeff.iloc[0])) if has_regularizer else "black"
        outline = "black" if has_regularizer else "white"

        ax.plot(
            traj.valid_ntp,
            traj.prev_ntp,
            color=outline,
            lw=TRAJECTORY_LW + 1.0,
            alpha=0.9,
            zorder=FINETUNE_OVERLAY_Z,
        )
        ax.plot(
            traj.valid_ntp,
            traj.prev_ntp,
            color=color,
            lw=TRAJECTORY_LW,
            alpha=0.98,
            zorder=FINETUNE_OVERLAY_Z + 0.1,
        )

        endpoint = traj.tail(1)
        ax.scatter(
            endpoint.valid_ntp,
            endpoint.prev_ntp,
            s=FINAL_SCATTER_SIZE + 18,
            color=[color],
            edgecolor=outline,
            lw=0.8,
            zorder=FINETUNE_OVERLAY_Z + 0.2,
        )


def label_colorbar_side(cbar, label, label_side, tick_side=None, labelpad=2):
    tick_side = tick_side or label_side
    cbar.set_label(label, labelpad=labelpad, fontsize=9)
    cbar.ax.yaxis.set_label_position(label_side)
    cbar.ax.yaxis.set_ticks_position(tick_side)
    cbar.ax.tick_params(axis="y", labelleft=tick_side == "left", labelright=tick_side == "right")
    cbar.ax.yaxis.label.set_linespacing(0.9)


def plain_tick(x, _):
    return f"{x:.1f}"


def compact_decimal_tick(x, _):
    return f"{x:g}"


def percentage_tick(x, _):
    return f"{x:.0%}"


def shifted_log_ticks(lim, offset, n=5):
    ticks = np.exp(np.linspace(np.log(lim[0] - offset), np.log(lim[1] - offset), n)) + offset
    return np.round(ticks, 2)


def shifted_log_functions(offset):
    return (
        lambda x: np.log(np.maximum(np.asarray(x) - offset, np.finfo(float).tiny)),
        lambda x: np.exp(np.asarray(x)) + offset,
    )


def set_pretrain_scale(ax, x_offset, y_offset):
    ax.set_xscale("function", functions=shifted_log_functions(x_offset))
    ax.set_yscale("function", functions=shifted_log_functions(y_offset))


def add_zoom_box(ax, xlim, ylim):
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    visible_xlim = (max(xlim[0], ax_xlim[0]), min(xlim[1], ax_xlim[1]))
    visible_ylim = (max(ylim[0], ax_ylim[0]), min(ylim[1], ax_ylim[1]))
    zoom_line_style = dict(
        color="0.15",
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=ZOOM_BOX_Z,
        clip_on=False,
    )

    ax.add_patch(
        Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            fill=False,
            edgecolor="0.15",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            zorder=ZOOM_BOX_Z,
        )
    )
    if xlim[0] <= ax_xlim[0] <= xlim[1] and visible_ylim[0] < visible_ylim[1]:
        if np.isclose(visible_ylim[0], ax_ylim[0]):
            ax.spines["left"].set_bounds(visible_ylim[1], ax_ylim[1])
        ax.plot([0, 0], visible_ylim, transform=ax.get_yaxis_transform(), **zoom_line_style)
    if ylim[0] <= ax_ylim[0] <= ylim[1] and visible_xlim[0] < visible_xlim[1]:
        if np.isclose(visible_xlim[0], ax_xlim[0]):
            ax.spines["bottom"].set_bounds(visible_xlim[1], ax_xlim[1])
        ax.plot(visible_xlim, [0, 0], transform=ax.get_xaxis_transform(), **zoom_line_style)


def add_zoom_target_outline(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="0.15",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=ZOOM_BOX_Z + 1,
            clip_on=False,
        )
    )


def add_zoom_connectors(fig, source_ax, target_ax, source_xlim, source_ylim, target_xlim, target_ylim):
    connectors = [
        ConnectionPatch(
            xyA=(target_xlim[1], min(source_ylim[1], target_ylim[1])),
            xyB=(0, 1),
            coordsA=source_ax.transData,
            coordsB=target_ax.transAxes,
            color="0.45",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.65,
            zorder=ZOOM_BOX_Z - 0.2,
        ),
        ConnectionPatch(
            xyA=(target_xlim[1], 0),
            xyB=(0, 0),
            coordsA=source_ax.get_xaxis_transform(),
            coordsB=target_ax.transAxes,
            color="0.45",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.65,
            zorder=ZOOM_BOX_Z - 0.2,
        ),
    ]
    for connector in connectors:
        connector.set_clip_on(False)
        fig.add_artist(connector)


def main():
    pretrain_points, frontier = load_pretrain_points(CACHE)
    finetune_points = load_finetune_points(CACHE)
    finetune_overlay_points = load_finetune_overlay_points(CACHE)

    finetune_zoom_frames = [
        frontier[["mix_ntp", "valid_ntp"]].rename(columns={"mix_ntp": "spanish_ntp", "valid_ntp": "english_ntp"}),
        finetune_points[["valid_ntp", "prev_ntp"]].rename(columns={"valid_ntp": "spanish_ntp", "prev_ntp": "english_ntp"}),
    ]
    if not finetune_overlay_points.empty:
        finetune_zoom_frames.append(
            finetune_overlay_points[["valid_ntp", "prev_ntp"]].rename(
                columns={"valid_ntp": "spanish_ntp", "prev_ntp": "english_ntp"}
            )
        )
    finetune_zoom_xy = pd.concat(
        finetune_zoom_frames,
        ignore_index=True,
    )
    finetune_xlim, finetune_ylim = padded_limits(finetune_zoom_xy.spanish_ntp, finetune_zoom_xy.english_ntp)
    pretrain_xlim, pretrain_ylim = padded_limits(pretrain_points.mix_ntp, pretrain_points.valid_ntp)
    pretrain_x_offset = pretrain_points.mix_ntp.min() - PRETRAIN_X_SHIFT
    pretrain_y_offset = pretrain_points.valid_ntp.min() - PRETRAIN_Y_SHIFT
    pretrain_xlim = (pretrain_points.mix_ntp.min() - PRETRAIN_X_LOWER_PAD, pretrain_xlim[1])
    pretrain_ylim = (pretrain_points.valid_ntp.min() - PRETRAIN_Y_LOWER_PAD, pretrain_ylim[1])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, wspace=0.01)
    finetune_ylim = (finetune_ylim[0], 3.86)
    pretrain_ax, finetune_ax = axes

    panels = [
        (pretrain_ax, pretrain_xlim, pretrain_ylim, "Pretraining on mixed text"),
        (finetune_ax, finetune_xlim, finetune_ylim, "Finetuning on Spanish text"),
    ]
    for ax, xlim, ylim, title in panels:
        frontier_x, frontier_y = frontier_path(frontier, xlim, ylim)
        shade_inaccessible(ax, frontier, xlim, ylim)
        ax.plot(
            frontier_x,
            frontier_y,
            "--",
            color="0.35",
            lw=2.8,
            zorder=PARETO_Z + 0.2,
        )
        if ax is finetune_ax:
            ax.scatter(
                frontier.mix_ntp,
                frontier.valid_ntp,
                s=FINAL_SCATTER_SIZE,
                color="0.55",
                edgecolor="0.25",
                linewidth=0.6,
                zorder=PARETO_Z + 0.3,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)

    pretrain_norm, pretrain_cmap, plotted_pretrain_points = plot_pretraining(pretrain_ax, pretrain_points)
    finetune_norm, finetune_cmap = plot_finetuning(finetune_ax, finetune_points)
    plot_finetune_overlay(finetune_ax, finetune_overlay_points, finetune_norm, finetune_cmap)

    pretrain_ax.set(
        xlabel="Spanish loss",
        ylabel="",
    )
    set_pretrain_scale(pretrain_ax, pretrain_x_offset, pretrain_y_offset)
    add_zoom_box(pretrain_ax, finetune_xlim, finetune_ylim)
    add_zoom_connectors(fig, pretrain_ax, finetune_ax, pretrain_xlim, pretrain_ylim, finetune_xlim, finetune_ylim)
    add_zoom_target_outline(finetune_ax)
    pretrain_ax.xaxis.set_major_formatter(FuncFormatter(plain_tick))
    pretrain_ax.yaxis.set_major_formatter(FuncFormatter(plain_tick))
    pretrain_ax.set_xticks(shifted_log_ticks(pretrain_xlim, pretrain_x_offset))
    pretrain_ax.set_yticks(shifted_log_ticks(pretrain_ylim, pretrain_y_offset))
    pretrain_ax.xaxis.set_minor_formatter(NullFormatter())
    pretrain_ax.yaxis.set_minor_formatter(NullFormatter())
    finetune_ax.set(
        xlabel="Spanish loss",
        ylabel="English loss",
    )

    finetune_sm = plt.cm.ScalarMappable(norm=finetune_norm, cmap=finetune_cmap)
    finetune_sm.set_array([])
    finetune_cbar = fig.colorbar(
        finetune_sm,
        ax=finetune_ax,
        location="right",
        pad=0.02,
        fraction=0.035,
        aspect=40,
    )
    finetune_cbar.ax.yaxis.set_major_formatter(FuncFormatter(compact_decimal_tick))
    label_colorbar_side(finetune_cbar, "Regularization coefficient", "right")

    pretrain_sm = plt.cm.ScalarMappable(norm=pretrain_norm, cmap=pretrain_cmap)
    pretrain_sm.set_array([])
    pretrain_cbar = fig.colorbar(
        pretrain_sm,
        ax=pretrain_ax,
        location="left",
        pad=0.02,
        fraction=0.035,
        aspect=40,
    )
    pretrain_cbar.set_ticks(np.linspace(pretrain_norm.vmin, pretrain_norm.vmax, 6))
    pretrain_cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_tick))
    label_colorbar_side(pretrain_cbar, "Fraction of Spanish text", "left", tick_side="left", labelpad=6)

    fig.savefig(OUT, dpi=280)
    print(
        f"{OUT}\n"
        f"{len(frontier)} frontier runs, "
        f"{plotted_pretrain_points}/{len(pretrain_points)} plotted pretrain points, "
        f"{finetune_points.run.nunique()} finetune runs, "
        f"{len(finetune_points)} finetune points, "
        f"{finetune_overlay_points.run.nunique()} overlay runs, "
        f"{len(finetune_overlay_points)} overlay points"
    )


if __name__ == "__main__":
    main()
