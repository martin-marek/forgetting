#!/usr/bin/env python3
"""Plot validation accuracy against average chat MCQ accuracy for selected W&B runs."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares


BASELINE = "standard finetuning"
CACHE = Path("~/wandb/cache").expanduser()
SWEEP = "oeg7ap9w"
PLOT_SPECS = [ # showing the best config for each method
    ("KL (Olmo data)", "kl_pre", False, 1),
    ("KL (self-generated)", "kl_pre", True, 1),
    ("NTP (Olmo data)", "ntp_pre", False, 1),
    ("NTP (self-generated)", "ntp_pre", True, 1),
]
X = "eval/valid/accuracy"
Y = "__derived_y"
CHAT_MCQ_RE = re.compile(r"^chat_mcq/.+/accuracy$")
OUTPUTS = [Path(__file__).with_suffix(ext) for ext in (".png", ".pdf")]

EARLY_STOP_X_TOL = 0.0001
X_MAX_MARGIN = 0.00001
X_NOISE = 0.00025
Y_NOISE = 0.002
CURVE_STOP_FROM_END = 3
BASELINE_CURVE_STOP_FROM_END = 2
CURVE_POINTS = 2000
BOOTSTRAP_ITER = 100
BOOTSTRAP_SEED = 0
CI_PERCENTILES = (2.5, 97.5)
Y_BOTTOM_LIMIT_MARGIN_FRAC = 0.05
Y_TOP_LIMIT_MARGIN_FRAC = 0.04
X_AXIS_RIGHT_MARGIN = 0.01
X_AXIS_LEFT_PAD_FRAC = 0.03
X_AXIS_RIGHT_PAD_FRAC = 0.012
X_AXIS_FOCUS_TICKS = np.array([0.74, 0.81, 0.84, 0.86, 0.87])
SELF_GENERATED_DASHES = (0, (2.2, 1.4))
LEGEND_HANDLE_LENGTH = 3.2

plt.rcParams["path.simplify"] = False


def cfg_value(cfg, dotted_key):
    value = cfg.get(dotted_key, cfg)
    if value is cfg:
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
    return None if isinstance(value, str) and value.lower() in {"null", "none"} else value


def read_json(path):
    return json.loads(path.read_text())


def sweep_runs(cache=CACHE, sweep=SWEEP):
    rows = []
    for run_dir in sorted((cache / "sweeps" / sweep).iterdir()):
        if not run_dir.is_dir():
            continue
        cfg, meta = read_json(run_dir / "config.json"), read_json(run_dir / "metadata.json")
        rows.append(
            {
                "run": meta.get("run_id", run_dir.name),
                "run_dir": run_dir,
                "created_at": meta.get("created_at") or "",
                "reg_method": cfg_value(cfg, "reg.method"),
                "reg_synth": cfg_value(cfg, "reg.synth"),
                "reg_coeff": cfg_value(cfg, "reg.coeff"),
            }
        )
    return pd.DataFrame(rows).sort_values(["created_at", "run"]).reset_index(drop=True)


def require_one(df, mask, description):
    matches = df[mask]
    if matches.empty:
        raise ValueError(f"sweep {SWEEP!r} has no {description}")
    return matches.iloc[0]


def selected_runs():
    df = sweep_runs()
    if df.empty:
        raise FileNotFoundError(f"no cached runs for sweep {SWEEP!r} under {CACHE / 'sweeps' / SWEEP}")

    runs = [(BASELINE, require_one(df, df.reg_method.isna(), "non-regularized runs").run_dir)]
    for label, method, synth, coeff in PLOT_SPECS:
        row = require_one(
            df,
            (df.reg_method == method) & (df.reg_synth == synth) & (df.reg_coeff == coeff),
            f"run matching reg.method={method!r}, reg.synth={synth!r}, reg.coeff={coeff!r}",
        )
        runs.append((label, row.run_dir))

    print(f"loaded sweep {SWEEP}: {len(df)} runs, plotting {len(runs)}")
    return runs


def cached_history(run_dir):
    paths = sorted((run_dir / "history").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no cached history for {run_dir.name!r} under {run_dir}")
    return pd.concat(map(pd.read_parquet, paths), ignore_index=True, sort=False)


def load_points():
    frames = []
    for label, run_dir in selected_runs():
        frame = cached_history(run_dir)
        frame["run"] = label
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)
    chat_cols = [col for col in df if CHAT_MCQ_RE.match(col)]
    df[[X, *chat_cols]] = df[[X, *chat_cols]].apply(pd.to_numeric, errors="coerce")
    df[Y] = df[chat_cols].mean(axis=1, skipna=True)

    runs = []
    for label, points in df.dropna(subset=[X, Y]).groupby("run", sort=False):
        points = points.sort_values("_step")
        stop = (points[X] >= points[X].max() - EARLY_STOP_X_TOL).to_numpy().argmax()
        runs.append((label, points.iloc[: stop + 1].copy()))
    return runs


def shared_drop_shape(x, x0, x_max, p, q):
    base = np.power(np.clip((x - x0) / (x_max - x0), 0, 1), p)
    return base if abs(q) < 1e-8 else np.expm1(q * base) / np.expm1(q)


def predict(x, y0, x0, x_max, shared_params, drop):
    return y0 - drop * shared_drop_shape(x, x0, x_max, *shared_params)


def density_weights(x):
    if len(x) < 2:
        return np.ones_like(x)
    order = np.argsort(x)
    sx = x[order]
    widths = np.empty_like(sx)
    widths[0], widths[-1] = (sx[1] - sx[0]) / 2, (sx[-1] - sx[-2]) / 2
    if len(sx) > 2:
        widths[1:-1] = (sx[2:] - sx[:-2]) / 2
    weights = np.sqrt(np.maximum(widths, widths[widths > 0].min()) / widths.mean())
    weights = np.clip(weights, 0.25, 4.0)
    weights[[0, -1]] = np.maximum(weights[[0, -1]], 1.0)
    weights /= weights.mean()
    out = np.empty_like(weights)
    out[order] = weights
    return out


def sort_for_initial(points):
    return points.sort_values("_step" if "_step" in points else X)


def shared_anchor(runs):
    initial = [sort_for_initial(points).iloc[0] for _, points in runs]
    return min(row[X] for row in initial), np.mean([row[Y] for row in initial])


def bootstrap_sample(points, rng):
    points = sort_for_initial(points)
    first, rest = points.iloc[[0]], points.iloc[1:]
    if rest.empty:
        return first.copy()
    return pd.concat([first, rest.iloc[rng.integers(0, len(rest), len(rest))]]).copy()


def curve_grid(x0, x_stop, x_anchor):
    tx = np.linspace(-np.log(x_anchor - x0), -np.log(x_anchor - x_stop), CURVE_POINTS)
    return x_anchor - np.exp(-tx)


def fit_curves(runs, grids=None, quiet=False):
    labels = [label for label, _ in runs]
    x0, y0 = shared_anchor(runs)
    points = {label: df.sort_values(X) for label, df in runs}
    xs = {label: df[X].to_numpy(float) for label, df in points.items()}
    ys = {label: df[Y].to_numpy(float) for label, df in points.items()}
    x_max = max(x.max() for x in xs.values()) + X_MAX_MARGIN
    x_anchor = min(0.999, x_max - X_MAX_MARGIN + X_AXIS_RIGHT_MARGIN)

    params0 = np.array([1.0, 0.0, *[max(1e-8, y0 - ys[label].min()) for label in labels]])
    latent0 = np.concatenate([xs[label] for label in labels])
    n_params = len(params0)
    offsets = np.cumsum([0, *[len(xs[label]) for label in labels]])
    x_slices = {label: slice(offsets[i], offsets[i + 1]) for i, label in enumerate(labels)}
    drop_indices = {label: 2 + i for i, label in enumerate(labels)}
    lower = np.r_[[0.05, -8.0], np.zeros(len(labels)), np.full(len(latent0), x0)]
    upper = np.r_[[8.0, 8.0], np.ones(len(labels)), np.full(len(latent0), x_max - 1e-12)]

    def residuals(theta):
        out = []
        latent_x = theta[n_params:]
        for label in labels:
            true_x = latent_x[x_slices[label]]
            weights = np.sqrt(density_weights(xs[label]))
            y_hat = predict(true_x, y0, x0, x_max, theta[:2], theta[drop_indices[label]])
            out.extend((weights * (true_x - xs[label]) / X_NOISE, weights * (y_hat - ys[label]) / Y_NOISE))
        return np.concatenate(out)

    result = least_squares(residuals, np.r_[params0, latent0], bounds=(lower, upper), max_nfev=50000)
    if not result.success:
        raise RuntimeError(f"Fit failed: {result.message}")

    curves = {}
    for label in labels:
        stop_from_end = BASELINE_CURVE_STOP_FROM_END if label == BASELINE else CURVE_STOP_FROM_END
        x_line = grids[label] if grids is not None else curve_grid(x0, np.sort(xs[label])[-stop_from_end], x_anchor)
        curves[label] = (x_line, predict(x_line, y0, x0, x_max, result.x[:2], result.x[drop_indices[label]]))

    if not quiet:
        print(
            f"shared monotone fit: y0={y0:.4f}, x0={x0:.4f}, x_max={x_max:.4f}, "
            f"p={result.x[0]:.4f}, q={result.x[1]:.4f}, x_noise={X_NOISE:g}, y_noise={Y_NOISE:g}"
        )
    return curves


def bootstrap_intervals(runs, curves, bootstrap_iter):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grids = {label: x_line for label, (x_line, _) in curves.items()}
    samples = {label: [] for label, _ in runs}

    for _ in range(bootstrap_iter):
        resampled = [(label, bootstrap_sample(points, rng)) for label, points in runs]
        try:
            sample_curves = fit_curves(resampled, grids=grids, quiet=True)
        except RuntimeError:
            continue
        for label, (_, y_line) in sample_curves.items():
            samples[label].append(y_line)

    intervals = {}
    for label, y_samples in samples.items():
        if not y_samples:
            raise RuntimeError(f"No successful bootstrap fits for {label}")
        intervals[label] = np.percentile(np.stack(y_samples), CI_PERCENTILES, axis=0)
    print(f"bootstrap CI: samples={bootstrap_iter}, seed={BOOTSTRAP_SEED}")
    return intervals


def percent_formatter(x, _):
    pct = 100 * x
    return f"{pct:.0f}%" if abs(pct - round(pct)) < 1e-8 else f"{pct:.1f}%"


def ordered_curve_labels(curves):
    order_x = min(x_line[-1] for x_line, _ in curves.values())
    return sorted(curves, key=lambda label: np.interp(order_x, *curves[label]), reverse=True)


def set_right_expanded_xaxis(ax, runs):
    x = np.concatenate([points[X].to_numpy(float) for _, points in runs])
    x_anchor = min(0.999, x.max() + X_AXIS_RIGHT_MARGIN)

    def forward(values):
        return -np.log(np.clip(x_anchor - np.asarray(values), np.finfo(float).tiny, None))

    def inverse(values):
        return x_anchor - np.exp(-np.asarray(values))

    tx_min, tx_max = forward(x.min()), forward(x.max())
    tx_span = tx_max - tx_min
    left_limit = inverse(tx_min - X_AXIS_LEFT_PAD_FRAC * tx_span)
    right_limit = inverse(tx_max + X_AXIS_RIGHT_PAD_FRAC * tx_span)
    ax.set_xscale("function", functions=(forward, inverse))
    ax.set_xlim(left_limit, min(x_anchor - 1e-6, right_limit))
    ax.set_xticks(X_AXIS_FOCUS_TICKS[(X_AXIS_FOCUS_TICKS >= left_limit) & (X_AXIS_FOCUS_TICKS <= right_limit)])


def dash_segments(ax, x, y, linewidth):
    points = np.column_stack([x, y])
    display = ax.transData.transform(points)
    distance = np.r_[0, np.cumsum(np.hypot(*np.diff(display, axis=0).T))] * 72 / ax.figure.dpi
    scale = linewidth if plt.rcParams["lines.scale_dashes"] else 1.0
    offset, (on, off) = SELF_GENERATED_DASHES
    starts = np.arange(offset * scale, distance[-1], (on + off) * scale)
    return [
        np.column_stack([np.interp([start, *distance[(distance > start) & (distance < end)], end], distance, values) for values in (x, y)])
        for start in starts
        for end in [min(start + on * scale, distance[-1])]
    ]


def plot_runs(ax, runs, curves, intervals=None):
    labels = ordered_curve_labels(curves)
    colors = dict(zip(labels, sns.color_palette("deep", len(labels))))
    for label, points in runs:
        ax.scatter(points[X], points[Y], color=colors[label], alpha=0.5, s=28, edgecolors="none", label="_nolegend_")

    for label in labels:
        x_line, y_line = curves[label]
        if intervals is not None:
            ax.fill_between(x_line, *intervals[label], color=colors[label], alpha=0.17, linewidth=0, label="_nolegend_", zorder=0.5)
        line_kwargs = {
            "color": colors[label],
            "linewidth": 3.5,
            "label": label,
            "solid_capstyle": "round",
            "solid_joinstyle": "round",
            "dash_capstyle": "butt",
            "dash_joinstyle": "round",
        }
        linestyle = SELF_GENERATED_DASHES if label == "KL (self-generated)" else "-"
        if label == "KL (self-generated)":
            segments = dash_segments(ax, x_line, y_line, line_kwargs["linewidth"])
            ax.add_collection(LineCollection(segments, colors=[line_kwargs["color"]], linewidths=line_kwargs["linewidth"], capstyle="butt", label="_nolegend_"))
            ax.plot([], [], linestyle=linestyle, **line_kwargs)
        else:
            ax.plot(x_line, y_line, linestyle=linestyle, **line_kwargs)


def set_axis_labels(ax):
    ax.set_xlabel("Verilog next-token accuracy")
    ax.set_ylabel("Avg. benchmark accuracy", labelpad=24)
    ax.text(
        -0.14,
        0.47,
        "MMLU + ARC Challenge + CommonsenseQA",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        clip_on=False,
    )


def main():
    if BOOTSTRAP_ITER is not None and BOOTSTRAP_ITER < 1:
        raise ValueError("BOOTSTRAP_ITER must be None or a positive integer")

    runs = load_points()
    curves = fit_curves(runs)
    intervals = bootstrap_intervals(runs, curves, BOOTSTRAP_ITER) if BOOTSTRAP_ITER is not None else None
    y_values = [points[Y].to_numpy(float) for _, points in runs]
    if intervals is not None:
        y_values.extend(y for interval in intervals.values() for y in interval)
    y_min, y_max = np.concatenate(y_values).min(), np.concatenate(y_values).max()

    fig, ax = plt.subplots(figsize=(5.3, 3.1), constrained_layout=True)
    set_axis_labels(ax)
    set_right_expanded_xaxis(ax, runs)
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.set_ylim(
        y_min - (y_max - y_min) * Y_BOTTOM_LIMIT_MARGIN_FRAC,
        y_max + (y_max - y_min) * Y_TOP_LIMIT_MARGIN_FRAC,
    )
    fig.canvas.draw()
    plot_runs(ax, runs, curves, intervals)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, handlelength=LEGEND_HANDLE_LENGTH)
    for path in OUTPUTS:
        fig.savefig(path, dpi=200 if path.suffix == ".png" else None)
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
