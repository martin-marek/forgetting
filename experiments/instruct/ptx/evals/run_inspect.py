from pathlib import Path


def eval_inspect(model, log_dir, batch_size=512, seq_len=1024, limit=1024):
    import shutil
    import tempfile

    import httpx
    from inspect_ai import eval_set
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_evals.ifeval import ifeval
    from inspect_evals.mmlu import mmlu_0_shot
    from inspect_evals.truthfulqa import truthfulqa

    from evals.inspect.lenient_choice import lenient_choice
    from models import server

    server.start(model, batch_size=batch_size, max_seq_len=seq_len)
    try:
        http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=None), timeout=600)
        inspect_model = get_model(
            "openai/custom",
            base_url="http://localhost:8000/v1",
            api_key="key",
            config=GenerateConfig(temperature=0),
            http_client=http_client,
        )
        with tempfile.TemporaryDirectory() as tmp_log_dir:
            mmlu = mmlu_0_shot(cot=True)
            mmlu.scorer = [lenient_choice()]
            for i, sample in enumerate(mmlu.dataset):
                sample.id = i + 1
            tqa = truthfulqa()
            tqa.scorer = [lenient_choice()]
            _, logs = eval_set(
                [ifeval(), tqa, mmlu],
                model=inspect_model,
                max_connections=2048,
                limit=limit,
                log_dir=tmp_log_dir,
            )
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_log_dir, str(log_dir), dirs_exist_ok=True)
        results = {}
        for log in logs:
            task_name = log.eval.task
            for k, v in log.results.scores[0].metrics.items():
                results[f"inspect/{task_name}/{k}"] = v.value
        return results
    finally:
        server.stop()
