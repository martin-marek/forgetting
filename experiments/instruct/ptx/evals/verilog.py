"""Functional Verilog accuracy on VerilogEval v2 spec-to-RTL.

Samples RTL from the in-training model for each benchmark spec, then compiles
and simulates each candidate against the golden testbench with Icarus Verilog.
A sample passes iff the testbench reports "Mismatches: 0 in N samples" with no
compile error, degenerate-sensitivity warning, or timeout (the same criteria
as the official sv-iv-analyze script).

One-time setup:
    git clone https://github.com/NVlabs/verilog-eval ~/verilog-eval
    # iverilog v12 required (apt's v11 is too old, v13 unsupported upstream):
    git clone --depth 1 --branch v12-branch https://github.com/steveicarus/iverilog
    cd iverilog && sh autoconf.sh && ./configure --prefix=$HOME/.local && make -j && make install

Standalone usage (from the ptx directory):
    python -m evals.verilog selftest   # CPU-only: golden refs must pass 100%
    python -m evals.verilog eval --model alpindale/Llama-3.2-1B-Instruct --tp_size 2
"""
import json
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data import CHAT_TEMPLATE_KWARGS
from models.sampling import sample

# Prompt format must match training pairs built by data.process_pyranet.
INSTRUCTION = "Write the following Verilog program. {spec}"

PASS_RE = re.compile(r"^Mismatches: 0 in \d+ samples$", re.MULTILINE)
# Warnings sv-iv-analyze counts as failures: the DUT compiled but a
# combinational block has no sensitivities, so it would never trigger.
DEGENERATE_RES = (
    re.compile(r"always_comb process has no sensitivities"),
    re.compile(r"found no sensitivities so it will never trigger"),
)
FENCE_RE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
MODULE_RE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.DOTALL)
MODULE_DECL_RE = re.compile(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)")

WORKDIR_ROOT = "/dev/shm/ptx/verilog_eval"
COMPILE_TIMEOUT = 30

# Problems whose golden reference cannot pass its own testbench under the
# official criteria (found by `python -m evals.verilog selftest`): the first
# three run into the testbench's internal TIMEOUT guard by design, the last
# binds testbench ports the reference module does not declare.
SKIP_PROBLEMS = {
    "Prob082_lfsr32",
    "Prob099_m2014_q6c",
    "Prob141_count_clock",
    "Prob156_review2015_fancytimer",
}


def load_problems(data_dir):
    data_dir = Path(data_dir).expanduser()
    prompt_paths = sorted(data_dir.glob("Prob*_prompt.txt"))
    if not prompt_paths:
        raise FileNotFoundError(
            f"No VerilogEval problems found under {data_dir}. Setup: "
            "git clone https://github.com/NVlabs/verilog-eval ~/verilog-eval"
        )
    problems = []
    for path in prompt_paths:
        stem = path.name.removesuffix("_prompt.txt")
        if stem in SKIP_PROBLEMS:
            continue
        test_path = path.with_name(stem + "_test.sv")
        ref_path = path.with_name(stem + "_ref.sv")
        if not (test_path.exists() and ref_path.exists()):
            raise FileNotFoundError(f"Missing test/ref for {stem} in {data_dir}")
        problems.append({
            "name": stem,
            "spec": path.read_text().strip(),
            "test": test_path,
            "ref": ref_path,
        })
    return problems


def extract_code(text):
    """Best-effort Verilog extraction: fenced code blocks that contain a module
    (chatty instruct-style output), else bare module...endmodule spans (raw
    PyraNet-style output). Returns None if no module is found."""
    fenced = [block for block in FENCE_RE.findall(text) if "module" in block]
    spans = MODULE_RE.findall("\n".join(fenced) if fenced else text)
    return "\n\n".join(spans) if spans else None


def rename_top_module(code):
    """The testbench instantiates the DUT as TopModule; if the sample declares
    exactly one module under another name, rename it (specs ask for TopModule,
    but PyraNet-tuned models tend to reuse names from the spec text)."""
    decls = MODULE_DECL_RE.findall(code)
    if len(decls) == 1 and decls[0] != "TopModule":
        code = re.sub(rf"\bmodule\s+{re.escape(decls[0])}\b", "module TopModule", code, count=1)
    return code


def check_one(code, problem, workdir, cfg):
    """Compile sample+test+ref with iverilog and run the simulation; classify
    like scripts/sv-iv-analyze. Returns flags plus a short log tail."""
    result = {"name": problem["name"], "extracted": code is not None,
              "compiled": False, "passed": False, "timed_out": False, "log": ""}
    if code is None:
        result["log"] = "no verilog extracted"
        return result
    iverilog = Path(cfg.iverilog).expanduser()
    vvp = iverilog.parent / "vvp"
    wd = Path(workdir) / problem["name"]
    wd.mkdir(parents=True, exist_ok=True)
    sample_sv = wd / "sample.sv"
    sample_sv.write_text(code)
    binary = wd / "out.vvp"
    try:
        comp = subprocess.run(
            [str(iverilog), "-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb",
             "-o", str(binary), str(sample_sv), str(problem["test"]), str(problem["ref"])],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT, cwd=wd)
    except subprocess.TimeoutExpired:
        result.update(timed_out=True, log="iverilog compile timeout")
        return result
    compile_log = (comp.stdout or "") + (comp.stderr or "")
    if comp.returncode != 0:
        result["log"] = compile_log[-2000:]
        return result
    result["compiled"] = True
    try:
        # ulimit via a shell instead of preexec_fn: preexec_fn forces os.fork(),
        # which can deadlock inside a multithreaded JAX training process.
        sim = subprocess.run(
            ["bash", "-c", 'ulimit -v 4194304; exec "$0" "$1"', str(vvp), str(binary)],
            capture_output=True, text=True, timeout=cfg.sim_timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        result.update(timed_out=True, log="simulation timeout")
        return result
    sim_log = (sim.stdout or "") + (sim.stderr or "")
    result["log"] = (compile_log + sim_log)[-2000:]
    result["passed"] = (
        sim.returncode == 0
        and "TIMEOUT" not in sim_log
        and not any(r.search(compile_log) for r in DEGENERATE_RES)
        and bool(PASS_RE.search(sim_log))
    )
    return result


_prompt_cache = None


def build_buffer(tokenizer, cfg):
    """Fixed [B, T] sampling buffer of chat-templated prompts (pad elsewhere).
    Cached: the shape must stay constant across a run so sample() jits once."""
    global _prompt_cache
    if _prompt_cache is None:
        pad_id = tokenizer.pad_token_id
        problems, prompt_ids, num_dropped = [], [], 0
        for problem in load_problems(cfg.data_dir):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION.format(spec=problem["spec"])}],
                tokenize=True, return_dict=False, add_generation_prompt=True,
                **CHAT_TEMPLATE_KWARGS)
            if len(ids) > cfg.max_prompt_tokens:
                num_dropped += 1
                continue
            # sample() treats buffer slots equal to pad_id as writable, so a
            # pad_id inside a prompt would be silently overwritten mid-prompt.
            assert pad_id not in ids, (
                f"prompt for {problem['name']} contains pad_token_id={pad_id}; "
                "use a reserved pad id (e.g. 128004 for Llama 3)")
            problems.append(problem)
            prompt_ids.append(ids)
            if cfg.num_problems is not None and len(problems) >= cfg.num_problems:
                break
        if num_dropped:
            print(f"verilog eval: dropped {num_dropped} prompts over {cfg.max_prompt_tokens} tokens")
        if not problems:
            raise ValueError("verilog eval: no problems left after prompt-length filtering")
        T = max(len(ids) for ids in prompt_ids) + cfg.max_new_tokens
        B = -(-len(problems) // 8) * 8  # filler rows keep B a multiple of the data axis
        buf = np.full((B, T), pad_id, dtype=np.int32)
        prompt_lens = np.zeros(B, dtype=np.int32)
        for i, ids in enumerate(prompt_ids + [prompt_ids[0]] * (B - len(problems))):
            buf[i, :len(ids)] = ids
            prompt_lens[i] = len(ids)
        _prompt_cache = (problems, buf, prompt_lens)
    return _prompt_cache


def score_completions(tokenizer, problems, tokens, prompt_lens, cfg, run_dir=None, step=0):
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    completions, gen_lens = [], []
    for i in range(len(problems)):
        gen = np.asarray(tokens[i, int(prompt_lens[i]):])
        end = next((j for j, t in enumerate(gen) if int(t) in stop_ids), len(gen))
        gen_lens.append(end)
        completions.append(tokenizer.decode(gen[:end], skip_special_tokens=True))
    codes = [code and rename_top_module(code) for code in map(extract_code, completions)]
    workdir = Path(WORKDIR_ROOT) / uuid.uuid4().hex
    try:
        with ThreadPoolExecutor(max_workers=min(32, len(problems))) as pool:
            results = list(pool.map(lambda pc: check_one(pc[1], pc[0], workdir, cfg),
                                    zip(problems, codes)))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if run_dir is not None:
        out_dir = Path(run_dir) / "verilog"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"step_{step}.jsonl", "w") as f:
            for completion, result in zip(completions, results):
                f.write(json.dumps({**result, "completion": completion}) + "\n")
    return results, np.asarray(gen_lens)


_last_step, _num_calls = None, 0


def run_eval(model, cfg, step, run_dir=None):
    """In-loop entry point; returns {} on cadence-skipped triggers. Counts
    distinct steps so the EMA second call at the same step is not
    double-counted (and stays consistent: both run or both skip)."""
    global _last_step, _num_calls
    if step != _last_step:
        _last_step = step
        _num_calls += 1
    if (_num_calls - 1) % cfg.every != 0:
        return {}
    t_start = time.perf_counter()
    problems, buf, prompt_lens = build_buffer(model.tokenizer, cfg)
    tokens = sample(jax.random.key(step), model, jnp.asarray(buf), temperature=cfg.temperature)
    tokens = np.asarray(jax.device_get(tokens))
    t_sample = time.perf_counter() - t_start
    results, gen_lens = score_completions(
        model.tokenizer, problems, tokens, prompt_lens, cfg, run_dir=run_dir, step=step)
    n = len(results)
    return {
        "verilog/pass_rate": sum(r["passed"] for r in results) / n,
        "verilog/compile_rate": sum(r["compiled"] for r in results) / n,
        "verilog/extract_rate": sum(r["extracted"] for r in results) / n,
        "verilog/timeout_rate": sum(r["timed_out"] for r in results) / n,
        "verilog/gen_tokens_mean": float(gen_lens.mean()),
        "verilog/sample_seconds": t_sample,
        "verilog/seconds": time.perf_counter() - t_start,
    }


DEFAULTS = {
    "data_dir": "~/verilog-eval/dataset_spec-to-rtl",
    "iverilog": "~/.local/bin/iverilog",
    "every": 2,
    "num_problems": None,
    "temperature": 0.01,
    "max_prompt_tokens": 896,
    "max_new_tokens": 512,
    "sim_timeout": 30,
}


def _selftest(cfg):
    """No model, no TPU: golden refs (renamed to TopModule) must pass 100%;
    an empty TopModule stub must pass ~0%. Validates iverilog, flags, the
    pass criteria, timeouts, and the thread pool."""
    problems = load_problems(cfg.data_dir)
    workdir = Path(WORKDIR_ROOT) / f"selftest_{uuid.uuid4().hex}"
    stub = "module TopModule();\nendmodule\n"
    t_start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            refs = list(pool.map(
                lambda p: check_one(rename_top_module(p["ref"].read_text()), p, workdir / "refs", cfg),
                problems))
            stubs = list(pool.map(
                lambda p: check_one(stub, p, workdir / "stubs", cfg), problems))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    elapsed = time.perf_counter() - t_start
    ref_failures = [r for r in refs if not r["passed"]]
    stub_passes = [r for r in stubs if r["passed"]]
    print(f"golden refs passed: {len(refs) - len(ref_failures)}/{len(refs)} (want all)")
    for r in ref_failures[:10]:
        print(f"  FAIL {r['name']}: {r['log'][-300:]!r}")
    print(f"empty stubs passed: {len(stub_passes)}/{len(stubs)} (want 0)")
    for r in stub_passes[:10]:
        print(f"  PASS {r['name']} (unexpected)")
    print(f"selftest wall clock: {elapsed:.1f}s")
    return 1 if (ref_failures or stub_passes) else 0


def _eval_checkpoint(cfg, model_id, tp_size):
    import models

    model = models.load(model_id, tp_size=tp_size)
    problems, _, prompt_lens = build_buffer(model.tokenizer, cfg)
    lens = prompt_lens[:len(problems)]
    print(f"problems: {len(problems)}, prompt tokens min/mean/max: "
          f"{lens.min()}/{lens.mean():.0f}/{lens.max()}")
    metrics = run_eval(model, cfg, step=0, run_dir=Path(WORKDIR_ROOT).parent / "verilog_eval_cli")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return 0


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="VerilogEval functional accuracy")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_self = sub.add_parser("selftest", help="CPU-only harness check (no model)")
    p_eval = sub.add_parser("eval", help="run the eval once against a checkpoint")
    p_eval.add_argument("--model", default="alpindale/Llama-3.2-1B-Instruct")
    p_eval.add_argument("--tp_size", type=int, default=1)
    arg_types = {"data_dir": str, "iverilog": str, "every": int, "num_problems": int,
                 "temperature": float, "max_prompt_tokens": int, "max_new_tokens": int,
                 "sim_timeout": int}
    for p in (p_self, p_eval):
        for key, value in DEFAULTS.items():
            p.add_argument(f"--{key}", type=arg_types[key], default=value)
    args = vars(parser.parse_args())
    mode, model_id, tp_size = args.pop("mode"), args.pop("model", None), args.pop("tp_size", None)
    cfg = OmegaConf.create({**args, "every": 1})
    raise SystemExit(_selftest(cfg) if mode == "selftest" else _eval_checkpoint(cfg, model_id, tp_size))
