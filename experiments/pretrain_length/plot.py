from pathlib import Path
import json

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def runs(cache, sweep):
    rows = []
    for run_dir in sorted((cache / "sweeps" / sweep).iterdir()):
        if run_dir.is_dir():
            cfg, summary, meta = (read_json(run_dir / f"{x}.json") for x in ("config", "summary", "metadata"))
            rows.append({"run": run_dir.name, "state": str(meta.get("state", "")).lower(), "cfg": cfg, "summary": summary})
    return pd.DataFrame(rows)


def pretrain_tpp(cfg):
    return cfg["stop.tokens_per_param"]


def loaded_pretrain_tpp(cfg):
    return cfg["load_cfg.stop.tokens_per_param"]


def loaded_pretrain_run(cfg):
    load_path = cfg.get("model.load_path")
    if not load_path:
        return None
    return Path(load_path).name


def target_loss_reached(summary):
    return summary.get("stop_reason") == "target_loss" or bool(summary.get("target_loss_reached"))


def model_params(summary):
    return summary["n_param_nonembed"] + summary["n_param_embed"]


def tpp_label(value):
    return f"{value:g}"


def main(
    cache="~/wandb/cache",
    pretrain_sweep="0zjpdd9g",
    finetune_sweep="2ecqgvb5",
    out_dir=str(HERE),
):
    cache, out_dir = Path(cache).expanduser(), Path(out_dir).expanduser()

    pre = runs(cache, pretrain_sweep).query("state == 'finished'").assign(
        tpp=lambda x: x.cfg.map(pretrain_tpp),
        actual_params_pre=lambda x: x.summary.map(model_params),
        pretraining_loss=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
    )[["run", "tpp", "actual_params_pre", "pretraining_loss"]].rename(columns={"run": "pretrain_run"})
    pre["tpp"] = pd.to_numeric(pre["tpp"]).astype(float)

    ft = runs(cache, finetune_sweep).query("state in ['finished', 'running']").assign(
        tpp=lambda x: x.cfg.map(loaded_pretrain_tpp),
        loaded_pretrain_run=lambda x: x.cfg.map(loaded_pretrain_run),
        actual_params_ft=lambda x: x.summary.map(model_params),
        finetune_end=lambda x: x.summary.map(lambda s: s.get("prev_ntp")),
        downstream_end=lambda x: x.summary.map(lambda s: s.get("valid_ntp")),
        stop_reason=lambda x: x.summary.map(lambda s: s.get("stop_reason")),
        target_loss_reached=lambda x: x.summary.map(target_loss_reached),
    )[["run", "state", "tpp", "loaded_pretrain_run", "actual_params_ft", "finetune_end", "downstream_end", "stop_reason", "target_loss_reached"]].rename(columns={"run": "finetune_run"})
    ft["tpp"] = pd.to_numeric(ft["tpp"]).astype(float)

    df = pre.merge(ft, on="tpp")
    numeric = ["tpp", "actual_params_pre", "actual_params_ft", "pretraining_loss", "finetune_end", "downstream_end"]
    df[numeric] = df[numeric].apply(pd.to_numeric)
    df["actual_params_match"] = df.actual_params_pre.eq(df.actual_params_ft)
    df["load_path_matches_pretrain_run"] = df.pretrain_run.eq(df.loaded_pretrain_run)
    df["forgetting_delta"] = df.finetune_end - df.pretraining_loss
    df = df.sort_values(["tpp", "finetune_end"])
    if not df.actual_params_match.all():
        print("warning: pretrain and finetune actual parameter counts differ")
    if not df.load_path_matches_pretrain_run.all():
        print("warning: finetune model.load_path does not match merged pretrain run")
    finished_df = df[df.state == "finished"]
    running_df = df[df.state == "running"]
    missed = len(finished_df) - int(finished_df.target_loss_reached.sum())
    if missed:
        reasons = finished_df.loc[~finished_df.target_loss_reached, "stop_reason"].fillna("unknown").value_counts().to_dict()
        print(f"plotting {missed} finetune runs that did not reach target loss as x markers: {reasons}")
    if len(running_df):
        print(f"plotting {len(running_df)} running finetune runs as blue circles")

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    pre_plot = df.groupby("tpp", as_index=False)["pretraining_loss"].mean()
    target_df = finished_df[finished_df.target_loss_reached]
    missed_df = finished_df[~finished_df.target_loss_reached]
    best_idx = target_df.groupby("tpp")["finetune_end"].idxmin()
    best_ft = target_df.loc[best_idx].sort_values("tpp")
    ax.plot(pre_plot.tpp, pre_plot.pretraining_loss, "o-", color="tab:blue", label="pretraining / finetune start")
    ax.scatter(target_df.tpp, target_df.finetune_end, s=28, alpha=0.65, color="tab:green", label="finetune end runs")
    ax.scatter(missed_df.tpp, missed_df.finetune_end, s=42, alpha=0.85, color="tab:red", marker="x", label="target loss not reached")
    ax.scatter(running_df.tpp, running_df.finetune_end, s=34, alpha=0.75, color="tab:blue", label="running finetune runs")
    ax.plot(best_ft.tpp, best_ft.finetune_end, "o-", color="tab:green", label="best finetune end")
    ax.set_xlabel("Tokens per parameter (TPP)")
    ax.set_ylabel("pretraining validation loss")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: tpp_label(x)))
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"plot.{ext}", dpi=220)
    print(df.to_string(index=False))


if __name__ == "__main__":
    fire.Fire(main)
