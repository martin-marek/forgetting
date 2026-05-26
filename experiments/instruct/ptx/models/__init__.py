import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import llama, qwen


LOADERS = {
    "llama": llama.load,
    "qwen3": qwen.load,
}


def _load_config(model_id, hf_ckpt_dir):
    for root in (Path(model_id).expanduser(), Path(hf_ckpt_dir).expanduser() / model_id):
        path = root / "config.json"
        if path.exists():
            return json.loads(path.read_text())
    path = hf_hub_download(repo_id=model_id, filename="config.json")
    return json.loads(Path(path).read_text())


def load(model_id="Qwen/Qwen3-0.6B-Base", hf_ckpt_dir="/dev/shm/ptx/weights", *args, **kwargs):
    cfg = _load_config(model_id, hf_ckpt_dir)
    model_type = cfg.get("model_type")
    if model_type not in LOADERS:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"Unsupported model_type={model_type!r}; supported types: {supported}")
    return LOADERS[model_type](model_id, hf_ckpt_dir, *args, **kwargs)
