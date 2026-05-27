import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent

PRETRAIN_LENGTH_PRETRAIN_SWEEP = "0zjpdd9g"
PRETRAIN_LENGTH_FINETUNE_SWEEP = "2ecqgvb5"
MODEL_SIZE_PRETRAIN_SWEEP = "dns1uo8c"
MODEL_SIZE_FINETUNE_SWEEP = "919li0b9"


def read_json(path):
    return json.loads(path.read_text())


def load_sweep(cache, sweep):
    rows = []
    for run_dir in sorted((cache / "sweeps" / sweep).iterdir()):
        if not run_dir.is_dir():
            continue
        rows.append(
            {
                "run": run_dir.name,
                "state": str(read_json(run_dir / "metadata.json").get("state", "")).lower(),
                "cfg": read_json(run_dir / "config.json"),
                "summary": read_json(run_dir / "summary.json"),
            }
        )
    return pd.DataFrame(rows)


def target_loss_reached(summary):
    return summary.get("stop_reason") == "target_loss" or bool(summary.get("target_loss_reached"))


def model_params(summary):
    return summary["n_param_nonembed"] + summary["n_param_embed"]


def pretrain_tpp(cfg):
    return cfg["stop.tokens_per_param"]


def loaded_pretrain_tpp(cfg):
    return cfg["load_cfg.stop.tokens_per_param"]


def model_width(cfg):
    return cfg["model.D"]


def pretrain_length_points(cache, pretrain_sweep, finetune_sweep):
    pre = (
        load_sweep(cache, pretrain_sweep)
        .query("state == 'finished'")
        .assign(
            tpp=lambda x: x.cfg.map(pretrain_tpp),
            pretraining_loss=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
        )[["run", "tpp", "pretraining_loss"]]
        .rename(columns={"run": "pretrain_run"})
    )
    ft = (
        load_sweep(cache, finetune_sweep)
        .query("state in ['finished', 'running']")
        .assign(
            tpp=lambda x: x.cfg.map(loaded_pretrain_tpp),
            finetune_end=lambda x: x.summary.map(lambda s: s.get("prev_ntp")),
            stop_reason=lambda x: x.summary.map(lambda s: s.get("stop_reason")),
            target_loss_reached=lambda x: x.summary.map(target_loss_reached),
        )[["run", "state", "tpp", "finetune_end", "stop_reason", "target_loss_reached"]]
        .rename(columns={"run": "finetune_run"})
    )

    df = pre.merge(ft, on="tpp")
    return numeric_sort(df, ["tpp", "pretraining_loss", "finetune_end"], "tpp")


def model_size_points(cache, pretrain_sweep, finetune_sweep):
    pre = (
        load_sweep(cache, pretrain_sweep)
        .query("state == 'finished'")
        .assign(
            width=lambda x: x.cfg.map(model_width),
            params=lambda x: x.summary.map(model_params),
            pretraining_loss=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
        )[["run", "width", "params", "pretraining_loss"]]
        .rename(columns={"run": "pretrain_run"})
    )
    ft = (
        load_sweep(cache, finetune_sweep)
        .query("state in ['finished', 'running']")
        .assign(
            width=lambda x: x.cfg.map(model_width),
            finetune_end=lambda x: x.summary.map(lambda s: s.get("prev_ntp")),
            stop_reason=lambda x: x.summary.map(lambda s: s.get("stop_reason")),
            target_loss_reached=lambda x: x.summary.map(target_loss_reached),
        )[["run", "state", "width", "finetune_end", "stop_reason", "target_loss_reached"]]
        .rename(columns={"run": "finetune_run"})
    )

    df = pre.merge(ft, on="width")
    df = numeric_sort(df, ["width", "params", "pretraining_loss", "finetune_end"], "params")
    df["model_size_m"] = (df.params / 1e6).round().astype(int)
    return df


def numeric_sort(df, numeric_cols, sort_col):
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=numeric_cols).sort_values(sort_col)


def plot_slice(ax, df, x_col, xlabel, colors, ylim=None):
    target = df[(df.state == "finished") & df.target_loss_reached]
    pre = df.groupby(x_col, as_index=False).pretraining_loss.mean().sort_values(x_col)
    best = target.loc[target.groupby(x_col).finetune_end.idxmin()].sort_values(x_col)

    y_values = pd.concat([df.pretraining_loss, df.finetune_end]).dropna()
    y_pad = (y_values.max() - y_values.min()) * 0.06
    y_bottom = y_values.min() - y_pad
    y_top = y_values.max() + y_pad
    if ylim is not None:
        y_bottom = min(y_bottom, ylim[0] - y_pad)
        y_top = max(y_top, ylim[1] + y_pad)

    ax.fill_between(
        pre[x_col],
        y_bottom,
        pre.pretraining_loss,
        facecolor=colors["after_pretraining"],
        edgecolor=colors["after_pretraining_hatch"],
        alpha=0.34,
        hatch="////",
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        best[x_col],
        pre.pretraining_loss,
        best.finetune_end,
        facecolor=colors["forgetting"],
        edgecolor=colors["forgetting_hatch"],
        alpha=0.28,
        hatch="\\\\\\\\",
        linewidth=0,
        zorder=2,
    )
    ax.fill_between(
        best[x_col],
        best.finetune_end,
        y_top,
        facecolor=colors["after_finetuning"],
        edgecolor=colors["after_finetuning_hatch"],
        alpha=0.28,
        hatch="////",
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        pre[x_col],
        pre.pretraining_loss,
        lw=1.9,
        marker="o",
        markersize=4.2,
        color=colors["pretrain_line"],
        markerfacecolor=colors["pretrain_line"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=4,
    )
    ax.plot(
        best[x_col],
        best.finetune_end,
        lw=1.9,
        marker="o",
        markersize=4.2,
        color=colors["finetune_line"],
        markerfacecolor=colors["finetune_line"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=4,
    )

    ax.set_xlabel(xlabel)
    ax.set_xlim(pre[x_col].min(), pre[x_col].max())
    ax.set_ylim(ylim or (y_bottom, y_top))
    ax.grid(alpha=0.25, linewidth=0.8)
    return pre, best


def label_regions(ax, colors, *, include_forgetting=True, positions=None):
    box = {
        "boxstyle": "round,pad=0.28",
        "facecolor": "white",
        "edgecolor": "0.72",
        "linewidth": 0.8,
        "alpha": 0.94,
    }
    positions = positions or {
        "after_finetuning": (0.60, 0.6),
        "forgetting": (0.13, 0.4),
        "before_finetuning": (0.55, 0.08),
    }
    labels = [
        (*positions["after_finetuning"], "after finetuning", colors["after_finetuning"]),
    ]
    if include_forgetting:
        labels.append((*positions["forgetting"], "forgetting", colors["forgetting"]))
    labels.append((*positions["before_finetuning"], "before finetuning", colors["after_pretraining"]))
    for x, y, text, color in labels:
        ax.text(
            x,
            y,
            text,
            color=color,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=box,
            zorder=5,
        )


def boundary_y_at(x, boundary, x_col, y_col, x_transform=lambda values: values):
    xs = boundary[x_col].to_numpy(dtype=float)
    ys = boundary[y_col].to_numpy(dtype=float)
    return float(np.interp(x_transform(x), x_transform(xs), ys))


def label_forgetting_gap(ax, pre, best, x_col, colors, *, x, x_transform=lambda values: values):
    y_pre = boundary_y_at(x, pre, x_col, "pretraining_loss", x_transform)
    y_best = boundary_y_at(x, best, x_col, "finetune_end", x_transform)
    y_low, y_high = sorted((y_pre, y_best))
    arrow_pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    y_mid = (y_low + y_high) / 2

    ax.annotate(
        "",
        xy=(x, y_high - arrow_pad),
        xytext=(x, y_low + arrow_pad),
        arrowprops={
            "arrowstyle": "<->",
            "color": colors["forgetting_hatch"],
            "lw": 1.6,
            "mutation_scale": 10,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )
    ax.annotate(
        "forgetting",
        xy=(x, y_mid),
        xytext=(7, 0),
        textcoords="offset points",
        color=colors["forgetting"],
        ha="left",
        va="center",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.72",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
        zorder=6,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="~/wandb/cache")
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--pretrain-length-pretrain-sweep", default=PRETRAIN_LENGTH_PRETRAIN_SWEEP)
    parser.add_argument("--pretrain-length-finetune-sweep", default=PRETRAIN_LENGTH_FINETUNE_SWEEP)
    parser.add_argument("--model-size-pretrain-sweep", default=MODEL_SIZE_PRETRAIN_SWEEP)
    parser.add_argument("--model-size-finetune-sweep", default=MODEL_SIZE_FINETUNE_SWEEP)
    return parser.parse_args()


def main():
    args = parse_args()
    cache = Path(args.cache).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["hatch.linewidth"] = 0.65
    colors = {
        "after_pretraining": "#4c72b0",
        "after_pretraining_hatch": "#2d5b96",
        "forgetting": "#df6b68",
        "forgetting_hatch": "#b53a3a",
        "after_finetuning": "#6fbd75",
        "after_finetuning_hatch": "#3c944a",
        "pretrain_line": "#4c72b0",
        "finetune_line": "#2f8f45",
    }

    pretrain_df = pretrain_length_points(
        cache,
        args.pretrain_length_pretrain_sweep,
        args.pretrain_length_finetune_sweep,
    )
    model_df = model_size_points(
        cache,
        args.model_size_pretrain_sweep,
        args.model_size_finetune_sweep,
    )

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.4), constrained_layout=True)

    model_pre, model_best = plot_slice(axs[0], model_df, "model_size_m", "Model size", colors, ylim=(2.93, 3.35))
    axs[0].set_title("Larger models forget less", fontsize=10, fontweight="bold")
    axs[0].set_ylabel("Pretraining loss")
    axs[0].set_xscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
    model_ticks = sorted(model_df.model_size_m.unique())
    axs[0].set_xticks(model_ticks)
    axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}M"))
    label_regions(axs[0], colors, include_forgetting=False)
    label_forgetting_gap(
        axs[0],
        model_pre,
        model_best,
        "model_size_m",
        colors,
        x=7,
        x_transform=lambda values: values**0.5,
    )

    plot_slice(axs[1], pretrain_df, "tpp", "Tokens per parameter (TPP)", colors)
    axs[1].set_title("Longer pretraining\nincreases forgetting", fontsize=10, fontweight="bold")
    axs[1].set_xscale("log")
    axs[1].set_xticks([10, 100, 1000, 10000])
    axs[1].xaxis.set_minor_locator(mticker.NullLocator())
    axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    # label_regions(
    #     axs[1],
    #     colors,
    #     include_forgetting=False,
    #     positions={
    #         "after_finetuning": (0.66, 0.63),
    #         "before_finetuning": (0.29, 0.09),
    #     },
    # )

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"plot_simple.{ext}", dpi=400)
    print(out_dir / "plot_simple.png")
    print(out_dir / "plot_simple.pdf")


if __name__ == "__main__":
    main()
