import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


matplotlib.use("Agg")
import matplotlib.pyplot as plt


CACHE = Path("~/wandb/cache").expanduser()


def model_params(summary: dict) -> int:
    return summary["n_param_nonembed"] + summary["n_param_embed"]


def progress_curve(history: pd.DataFrame) -> pd.DataFrame:
    curve = history[["valid_ntp", "train_tokens_seen", "_step"]].copy()
    curve["loss"] = pd.to_numeric(curve.valid_ntp)
    curve["progress"] = pd.to_numeric(curve.train_tokens_seen).ffill().bfill()
    return (
        curve.dropna(subset=["loss", "progress", "_step"])
        .sort_values("_step")
        .drop_duplicates("_step", keep="last")[["loss", "progress"]]
    )


def load_pretrain() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    curves = {}

    for run_dir in sorted(path for path in (CACHE / "sweeps" / "smy460xs").iterdir() if path.is_dir()):
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        history = pd.concat(
            [pd.read_parquet(path) for path in sorted((run_dir / "history").glob("*.parquet"))],
            ignore_index=True,
        )
        rows.append(
            dict(
                pretrain_run=run_dir.name,
                tpp=config["data.tokens_per_param"],
                n_params=model_params(summary),
                n_param_nonembed=summary["n_param_nonembed"],
                pretrain_loss=summary["valid_ntp"],
            )
        )
        curves[run_dir.name] = progress_curve(history)

    return pd.DataFrame(rows), curves


def load_finetune() -> pd.DataFrame:
    rows = []

    for run_dir in sorted(path for path in (CACHE / "sweeps" / "d5xm4du4").iterdir() if path.is_dir()):
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        reg_method = config.get("reg.method") or "null"
        rows.append(
            dict(
                finetune_run=run_dir.name,
                pretrain_run=Path(config["model.load"]).name,
                reg_method=reg_method,
                regularized=reg_method != "null",
                final_pretrain_loss=summary["prev_ntp"],
                target_loss=config["opt.target_loss"],
            )
        )

    return pd.DataFrame(rows)


def compute_multiplier(curve: pd.DataFrame, target_loss: float) -> float:
    total_progress = curve.progress.max()
    hits = curve[curve.loss <= target_loss]
    if hits.empty:
        return 1.0
    return np.clip(hits.iloc[0].progress / total_progress, 0, 1)


def load_points() -> pd.DataFrame:
    pretrain, curves = load_pretrain()
    points = load_finetune().merge(pretrain, on="pretrain_run")

    points["forgetting_loss_delta"] = points.final_pretrain_loss - points.pretrain_loss
    points["forgetting_loss_delta_for_plot"] = points.forgetting_loss_delta.clip(lower=0)
    points["compute_multiplier"] = [
        compute_multiplier(curves[run], loss)
        for run, loss in zip(points.pretrain_run, points.final_pretrain_loss, strict=True)
    ]
    return points.sort_values(["regularized", "n_params", "tpp"])


def format_si(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def pivot_grid(points: pd.DataFrame, value_col: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return (
        points.groupby(["n_params", "tpp"], as_index=False)[value_col]
        .median()
        .pivot(index="tpp", columns="n_params", values=value_col)
        .reindex(index=ys, columns=xs)
        .to_numpy(dtype=float)
    )


def smooth_grid(grid: np.ndarray, strength: float = 0.8) -> np.ndarray:
    if strength <= 0:
        return grid

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()

    finite = np.isfinite(grid)
    padded_values = np.pad(np.where(finite, grid, 0.0), 1, mode="edge")
    padded_weights = np.pad(finite.astype(float), 1, mode="edge")

    smoothed = np.zeros_like(grid, dtype=float)
    weights = np.zeros_like(grid, dtype=float)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            smoothed += weight * padded_values[dy : dy + grid.shape[0], dx : dx + grid.shape[1]]
            weights += weight * padded_weights[dy : dy + grid.shape[0], dx : dx + grid.shape[1]]

    result = grid.copy()
    valid = finite & (weights > 0)
    result[valid] = (1 - strength) * grid[valid] + strength * smoothed[valid] / weights[valid]
    return result


def bilinear_sample(grid: np.ndarray, x: float, y: float) -> float:
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, grid.shape[1] - 1), min(y0 + 1, grid.shape[0] - 1)
    dx, dy = x - x0, y - y0

    values = np.array([grid[y0, x0], grid[y0, x1], grid[y1, x0], grid[y1, x1]])
    if not np.all(np.isfinite(values)):
        return np.nan

    top = values[0] * (1 - dx) + values[1] * dx
    bottom = values[2] * (1 - dx) + values[3] * dx
    return top * (1 - dy) + bottom * dy


def contour_levels(grid: np.ndarray) -> np.ndarray:
    values = grid[np.isfinite(grid)]
    levels = [
        bilinear_sample(grid, t * (grid.shape[1] - 1), t * (grid.shape[0] - 1))
        for t in np.linspace(0, 1, 9)[1:-1]
    ]
    levels = np.array([level for level in levels if np.isfinite(level)])

    if len(np.unique(np.round(levels, 8))) < 7:
        levels = np.linspace(values.min(), values.max(), 9)[1:-1]

    return np.array(sorted(np.unique(np.round(levels, 8))))


def plot_grid(
    points: pd.DataFrame,
    value_col: str,
    stem: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    xs = np.array(sorted(points.n_params.unique()), dtype=int)
    ys = np.array(sorted(points.tpp.unique()), dtype=float)
    pretrain_grid = pivot_grid(
        points.drop_duplicates("pretrain_run"), "pretrain_loss", xs, ys
    )
    pretrain_levels = contour_levels(pretrain_grid)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True, sharey=True, layout='constrained')
    panels = [
        ("No regularization", points[~points.regularized]),
        ("KL regularization", points[points.regularized]),
    ]

    image = None
    for ax, (panel_title, panel_points) in zip(axes, panels):
        image = ax.imshow(
            smooth_grid(pivot_grid(panel_points, value_col, xs, ys)),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=Normalize(vmin, vmax, clip=True),
        )
        contours = ax.contour(
            np.ma.masked_invalid(pretrain_grid),
            levels=pretrain_levels,
            colors="black",
            linewidths=0.8,
            alpha=0.8,
        )
        ax.clabel(contours, contours.levels, inline=True, fmt="%.2f", fontsize=7)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("Model parameters")
        ax.set_xticks(np.arange(len(xs))[::2], [format_si(x) for x in xs[::2]], rotation=0, ha="center")
        ax.set_xticks(np.arange(len(xs))[1::2], minor=True)
        ax.set_yticks(np.arange(len(ys))[::2], [f"{y:g}" for y in ys[::2]])
        ax.set_yticks(np.arange(len(ys))[1::2], minor=True)
        ax.tick_params(axis="both", labelsize=8)

    axes[0].set_ylabel("Tokens per parameter")
    fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.025, fraction=0.05).set_label(
        colorbar_label
    )

    png_path = Path(__file__).with_name(f"{stem}.png")
    pdf_path = Path(__file__).with_name(f"{stem}.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    points = load_points()

    plot_grid(
        points,
        value_col="forgetting_loss_delta_for_plot",
        stem="loss_delta",
        colorbar_label="Forgetting (compute multipler)",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=points.forgetting_loss_delta_for_plot.max(),
    )
    plot_grid(
        points,
        value_col="compute_multiplier",
        stem="compute_multiplier",
        colorbar_label="Compute multiplier",
        cmap="RdYlGn",
        vmin=0.48,
        vmax=0.92,
    )

    print(f"{len(points)} finetune runs matched to {points.pretrain_run.nunique()} pretrain runs")


if __name__ == "__main__":
    main()
