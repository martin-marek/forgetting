"""Minimal OpenAI-compatible API server for ptx-models.

Usage:
python ptx-models/server.py --model_id=Qwen/Qwen3-0.6B --max_seq_len=512 --batch_size=512 --tp_size=8
python ptx-models/server.py --model_id=Qwen/Qwen3-4B-Instruct-2507 --max_seq_len=1024 --batch_size=256 --tp_size=8
"""
import asyncio, threading, time, uuid, numpy as np, jax
from fastapi import FastAPI, HTTPException
from jax.sharding import NamedSharding, AxisType
from pydantic import BaseModel
import uvicorn

from . import load
from .sampling import sample

app = FastAPI()
model, args = None, None
pending = []  # [(req, asyncio.Future)]
BATCH_TIMEOUT = 2.0  # seconds of no new requests before flushing a partial batch


class Request(BaseModel):
    model: str = ""
    prompt: str | None = None
    messages: list[dict] | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    seed: int | None = None


def _tokenize(req):
    if req.messages is not None:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        messages = [{"role": m["role"], "content": m["content"]} for m in req.messages]
        return model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=False,
        )
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    return model.tokenizer(req.prompt)["input_ids"]


def _run_batch(items):
    if _mesh is not None:
        jax.set_mesh(_mesh)
    B, T = args["batch_size"], args["max_seq_len"]
    pad_id = model.tokenizer.pad_token_id
    seed = items[0][0].seed
    temperature = items[0][0].temperature

    # tokenize and build buffer
    buf = np.full([B, T], pad_id, dtype=np.int32)
    prompt_lens = []
    for i, (req, _) in enumerate(items):
        tids = _tokenize(req)
        buf[i, :len(tids)] = tids
        prompt_lens.append(len(tids))

    # sample
    key = jax.random.key(seed if seed is not None else int(time.time() * 1e6) % (2**31))
    t0 = time.time()
    out = np.asarray(sample(key, model, buf, temperature=max(temperature, 1e-2)))
    print(f"Batch: {len(items)}/{B} requests, {time.time()-t0:.1f}s")

    # extract results
    stop_ids = {model.tokenizer.eos_token_id, pad_id}
    results = []
    for i, (req, _) in enumerate(items):
        gen = out[i, prompt_lens[i]:]
        end = next((j for j, t in enumerate(gen) if int(t) in stop_ids), len(gen))
        finish = "stop" if end < len(gen) else "length"
        gen = gen[:end]
        if req.max_tokens and len(gen) > req.max_tokens:
            gen, finish = gen[:req.max_tokens], "length"
        results.append((gen.tolist(), prompt_lens[i], len(gen), finish))
    return results


async def _batch_consumer():
    """Background task: drains pending requests and runs batches off the event loop."""
    while True:
        if not pending:
            await asyncio.sleep(0.01)
            continue
        # Wait for batch to fill or for request flow to settle
        while len(pending) < args["batch_size"]:
            prev = len(pending)
            await asyncio.sleep(BATCH_TIMEOUT)
            if len(pending) == prev:
                break
        batch = pending[:args["batch_size"]]
        del pending[:args["batch_size"]]
        try:
            results = await asyncio.to_thread(_run_batch, batch)
        except Exception as exc:
            for _, fut in batch:
                fut.set_exception(exc)
            continue
        for (_, fut), res in zip(batch, results):
            fut.set_result(res)


async def generate(req):
    fut = asyncio.get_event_loop().create_future()
    pending.append((req, fut))
    return await fut


@app.on_event("startup")
async def _start_consumer():
    asyncio.create_task(_batch_consumer())


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": args["model_id"], "object": "model", "owned_by": "ptx"}]}


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def completions(req: Request):
    gen_ids, n_prompt, n_gen, finish = await generate(req)
    text = model.tokenizer.decode(gen_ids, skip_special_tokens=True)
    is_chat = req.messages is not None
    choice = {"message": {"role": "assistant", "content": text}} if is_chat else {"text": text}
    return {
        "id": f"{'chatcmpl' if is_chat else 'cmpl'}-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion" if is_chat else "text_completion",
        "created": int(time.time()),
        "model": args["model_id"],
        "choices": [{**choice, "index": 0, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen},
    }


_server = None
_mesh = None  # sampling mesh (cached for JIT)
_train_mesh = None
_warmed_up = False


def _reshard_weights(weights, mesh):
    return jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, x.sharding.spec)), weights)


def _warmup(batch_size, max_seq_len):
    global _warmed_up
    if _warmed_up:
        return
    print("Warmup (JIT compile)...")
    buf = np.full([batch_size, max_seq_len], model.tokenizer.eos_token_id, dtype=np.int32)
    sample(jax.random.key(0), model, buf).block_until_ready()
    print("Warmup done")
    _warmed_up = True


def start(mdl, batch_size, max_seq_len, port=8000, model_id="custom"):
    """Start server in background thread. Blocks until ready, then returns."""
    global model, args, _server, _mesh, _train_mesh
    # switch to sampling mesh (TP-only), reuse across calls for JIT cache
    _train_mesh = jax.tree.leaves(mdl.weights)[0].sharding.mesh
    if _mesh is None:
        n_devices = jax.device_count()
        _mesh = jax.make_mesh((1, n_devices), ('data', 'model'), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(_mesh)
    mdl.weights = _reshard_weights(mdl.weights, _mesh)
    model = mdl
    args = {"batch_size": batch_size, "max_seq_len": max_seq_len, "model_id": model_id}
    _warmup(batch_size, max_seq_len)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    _server = uvicorn.Server(config)
    threading.Thread(target=_server.run, daemon=True).start()
    while not _server.started:
        time.sleep(0.1)


def stop():
    global _server
    if _server:
        _server.should_exit = True
        _server = None
    # restore training mesh
    if _train_mesh is not None:
        jax.set_mesh(_train_mesh)
        model.weights = _reshard_weights(model.weights, _train_mesh)


def main(model_id="Qwen/Qwen3-0.6B", hf_ckpt_dir="/dev/shm/ptx/weights", max_seq_len=1024, batch_size=1, tp_size=1, port=8000):
    global model, args
    args = {k: v for k, v in locals().items()}
    print(f'Loading model... ({model_id}, batch_size={batch_size}, max_seq_len={max_seq_len}, tp_size={tp_size})')
    model = load(model_id=model_id, hf_ckpt_dir=hf_ckpt_dir, tp_size=tp_size)
    print("Model loaded.")
    _warmup(batch_size, max_seq_len)
    uvicorn.run(app, host="0.0.0.0", port=port) # log_level="warning"


if __name__ == "__main__":
    # Standalone usage: python -m models.server ...
    import fire
    fire.Fire(main)
