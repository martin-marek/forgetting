import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


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


def plot_slice(ax, df, x_col, xlabel, colors, marker="o"):
    finished = df[df.state == "finished"]
    running = df[df.state == "running"]
    target = finished[finished.target_loss_reached]
    missed = finished[~finished.target_loss_reached]

    pre = df.groupby(x_col, as_index=False).pretraining_loss.mean().sort_values(x_col)
    ax.plot(
        pre[x_col],
        pre.pretraining_loss,
        marker=marker,
        lw=1.9,
        color=colors["pretrain"],
        label="end of pretraining",
    )
    if not target.empty:
        ax.scatter(
            target[x_col],
            target.finetune_end,
            s=28,
            alpha=0.68,
            color=colors["finished"],
            label="target loss reached",
        )
        best = target.loc[target.groupby(x_col).finetune_end.idxmin()].sort_values(x_col)
        ax.plot(
            best[x_col],
            best.finetune_end,
            lw=1.9,
            color=colors["finished"],
            label="best finished target-loss run",
        )
    if not missed.empty:
        ax.scatter(
            missed[x_col],
            missed.finetune_end,
            s=46,
            alpha=0.86,
            marker="x",
            color=colors["missed"],
            label="target loss not reached",
        )
    if not running.empty:
        ax.scatter(
            running[x_col],
            running.finetune_end,
            s=42,
            alpha=0.78,
            marker="^",
            color=colors["running"],
            label="running",
        )

    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    return {"finished": not target.empty, "missed": not missed.empty, "running": not running.empty}


def legend_handles(colors, has):
    handles = []
    if has["finished"]:
        handles.extend(
            [
                Line2D([0], [0], color=colors["finished"], marker="o", lw=0, label="target loss reached"),
            ]
        )
    if has["missed"]:
        handles.append(
            Line2D([0], [0], color=colors["missed"], marker="x", lw=0, markersize=7, label="target loss not reached")
        )
    if has["running"]:
        handles.append(Line2D([0], [0], color=colors["running"], marker="^", lw=0, markersize=7, label="running"))
    handles.append(
        Line2D([0], [0], color=colors["pretrain"], marker="o", lw=1.9, label="end of pretraining")
    )
    return handles


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

    palette = sns.color_palette('deep')
    colors = {
        "pretrain": palette[0],
        "running": palette[1],
        "finished": palette[2],
        "missed": palette[3],
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

    fig, axs = plt.subplots(1, 2, figsize=(8.3, 2.6), constrained_layout=True)

    left_has = plot_slice(axs[0], pretrain_df, "tpp", "Tokens per parameter (TPP)", colors, marker="o")
    axs[0].set_title("Longer pretraining increases forgetting", fontsize=10, fontweight="bold")
    axs[0].set_ylabel("Pretraining loss")
    axs[0].set_xscale("log")
    axs[0].set_xticks([10, 100, 1000, 10000])
    axs[0].xaxis.set_minor_locator(mticker.NullLocator())
    axs[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))

    right_has = plot_slice(axs[1], model_df, "model_size_m", "Model size", colors, marker="s")
    axs[1].set_title("Larger models forget less", fontsize=10, fontweight="bold")
    axs[1].set_xscale("function", functions=(lambda x: x**0.5, lambda x: x**2))
    model_ticks = sorted(model_df.model_size_m.unique())
    axs[1].set_xticks(model_ticks)
    axs[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}M"))
    axs[1].set_ylim((2.99, 3.41))

    has = {key: left_has[key] or right_has[key] for key in left_has}
    axs[0].legend(handles=legend_handles(colors, has), frameon=True, loc="upper right", fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"plot_orig.{ext}", dpi=220)
    print(out_dir / "plot_orig.png")
    print(out_dir / "plot_orig.pdf")


if __name__ == "__main__":
    main()
