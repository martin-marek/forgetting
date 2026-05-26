import queue
import json
import re
import threading

import numpy as np
from datasets import load_dataset


CHAT_TEMPLATE_KWARGS = dict(enable_thinking=False)


def _tokenize_chat(tokenizer, messages):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
    # Discover assistant turn delimiters by probing the chat template.
    # Template '' and 'X' as assistant, find where they diverge:
    #   empty = [BOS? | im_assist... | boundary | im_end ...]
    #   probe = [BOS? | im_assist... | boundary | X | im_end ...]
    #                                              ^ split
    # The boundary token (e.g. '>\n') can merge with content via BPE,
    # so we match im_assist without it and skip 1 extra token.
    empty = tokenizer.apply_chat_template([{'role': 'assistant', 'content': ''}], tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
    probe = tokenizer.apply_chat_template([{'role': 'assistant', 'content': 'X'}], tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
    split = next(i for i, (a, b) in enumerate(zip(empty, probe)) if a != b)
    start = 1 if empty[0] == tokenizer.bos_token_id else 0
    # Some templates, e.g. Llama 3 Instruct, inject a system turn even when
    # probing a bare assistant message. Only the final assistant header is the
    # delimiter we want to match inside real conversations.
    if tokenizer.eos_token_id is not None:
        prev_end = max((i for i in range(start, split - 1) if empty[i] == tokenizer.eos_token_id), default=None)
        if prev_end is not None:
            start = prev_end + 1
    im_assist = empty[start : split - 1]
    im_end = empty[split]
    base = tokenizer.apply_chat_template([{'role': 'user', 'content': ''}], tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
    gen = tokenizer.apply_chat_template([{'role': 'user', 'content': ''}], add_generation_prompt=True, tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
    gen_split = next((i for i, (a, b) in enumerate(zip(base, gen)) if a != b), len(base))
    prefixes = []
    for prefix in (gen[gen_split:-1], im_assist):
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    prefixes.sort(key=len, reverse=True)

    # create mask
    is_assistant = np.zeros(len(tokens), dtype=bool)
    i = 0
    while i < len(tokens):
        prefix = next((prefix for prefix in prefixes if tokens[i:i+len(prefix)] == prefix), None)
        if prefix is not None:
            i += len(prefix) + 1  # +1 to skip boundary token
            while True:
                is_assistant[i] = True
                if (i == len(tokens)-1) or (tokens[i] == im_end): break
                i += 1
        i += 1

    return tokens, is_assistant


def tokenize_text(text, tokenizer, seq_len):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > seq_len: return None
    mask = np.ones(len(tokens), dtype=bool)
    tokens = np.pad(tokens, pad_width=(0, seq_len-len(tokens)))
    mask = np.pad(mask, pad_width=(0, seq_len-len(mask)))
    return dict(tokens=tokens, mask=mask)


def tokenize_chat(messages, tokenizer, seq_len):
    tokens, mask = _tokenize_chat(tokenizer, messages)
    if len(tokens) > seq_len: return None
    assert mask.sum() > 0, (messages, mask)
    tokens = np.pad(tokens, pad_width=(0, seq_len-len(tokens)))
    mask = np.pad(mask, pad_width=(0, seq_len-len(mask)))
    return dict(tokens=tokens, mask=mask)


def tokenize_qa_answer_only(question, answer, tokenizer, seq_len, include_eos=True):
    question = question.strip()
    answer = answer.strip()
    if not answer:
        return None

    prompt = f'Q: {question}\nA: '
    text = prompt + answer
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > seq_len:
        return None

    # Find where answer tokens begin using a probe token to avoid BPE boundary issues.
    empty = tokenizer.encode(prompt, add_special_tokens=True)
    probe = tokenizer.encode(prompt + 'X', add_special_tokens=True)
    split = next((i for i, (a, b) in enumerate(zip(empty, probe)) if a != b), None)
    if split is None:
        return None

    mask = np.zeros(len(tokens), dtype=bool)
    end = len(tokens)
    eos_token_id = tokenizer.eos_token_id
    if not include_eos and eos_token_id is not None and tokens[-1] == eos_token_id:
        end -= 1
    if split >= end:
        return None
    mask[split:end] = True
    if mask.sum() == 0:
        return None

    tokens = np.pad(tokens, pad_width=(0, seq_len-len(tokens)))
    mask = np.pad(mask, pad_width=(0, seq_len-len(mask)))
    return dict(tokens=tokens, mask=mask)


def tokenize_text_batch(texts, tokenizer, add_eos=False):
    tokens_batch = tokenizer(texts, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
        tokens_batch = [tokens if tokens and tokens[-1] == eos_token_id else tokens + [eos_token_id] for tokens in tokens_batch]
    return tokens_batch


def tokenize_text_rows(texts, tokenizer):
    return dict(tokens=tokenize_text_batch(texts, tokenizer, add_eos=True))


def process_enron_email(sample, tokenizer, seq_len):
    """Next-token SFT on raw email text (no chat template)."""
    return tokenize_text(sample['email'], tokenizer, seq_len)


def process_instruct(sample, *args, **kwargs):
    return tokenize_chat(sample['messages'], *args, **kwargs)


def _clean_pyranet_description(metadata):
    desc = metadata.get('description')
    if not isinstance(desc, str):
        return None
    desc = desc.strip()
    if len(desc) < 16:
        return None
    return desc


def _clean_pyranet_code(code):
    if not isinstance(code, str):
        return None
    code = code.strip()
    if not code:
        return None
    if code[0] in ("[", "'", '"'):
        return None
    if re.search(r"\bmodule\b", code) is None or re.search(r"\bendmodule\b", code) is None:
        return None
    if "\\n" in code and "\n" not in code:
        return None
    return code


def process_pyranet(sample, *args, **kwargs):
    try:
        metadata = json.loads(sample['description'])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get('compile_status') != "No error!":
        return None
    desc = _clean_pyranet_description(metadata)
    code = _clean_pyranet_code(sample.get('code'))
    if desc is None or code is None:
        return None
    messages = [
        {'role': 'user', 'content': f"Write the following Verilog program. {desc}"},
        {'role': 'assistant', 'content': code},
    ]
    return tokenize_chat(messages, *args, **kwargs)


def process_preference(sample, tokenizer, seq_len):
    def strip(msgs):
        out = [{'role': m['role'], 'content': m['content']} for m in msgs]
        if any(m['content'] is None for m in out):
            return None
        return out
    chosen_msgs, rejected_msgs = strip(sample['chosen']), strip(sample['rejected'])
    if chosen_msgs is None or rejected_msgs is None:
        return None
    chosen = tokenize_chat(chosen_msgs, tokenizer, seq_len)
    rejected = tokenize_chat(rejected_msgs, tokenizer, seq_len)
    if chosen is None or rejected is None:
        return None
    return dict(
        chosen_tokens=chosen['tokens'],   chosen_mask=chosen['mask'],
        rejected_tokens=rejected['tokens'], rejected_mask=rejected['mask'],
    )


def process_medical_qa(sample, *args, **kwargs):
    messages = [
        {"role": "user", "content": sample['Question']},
        {"role": "assistant", "content": sample['Response']}
    ]
    return tokenize_chat(messages, *args, **kwargs)


def flatten_enron_qa(hf_dataset):
    """Flatten EnronQA so each row is a single Q→A pair (not grouped by email)."""
    flattened = []
    for record in hf_dataset:
        questions = record['rephrased_questions']
        answers = [a[0] for a in record['alternate_answers']]
        for q, a in zip(questions, answers):
            flattened.append({'question': q, 'answer': a})
    return flattened


def process_qa_answer_only(sample, tokenizer, seq_len):
    return tokenize_qa_answer_only(sample['question'], sample['answer'], tokenizer, seq_len)


def _shuffle_rows(source, seed):
    if hasattr(source, 'shuffle'):
        yield from source.shuffle(seed=seed)
        return
    for i in np.random.default_rng(seed).permutation(len(source)):
        yield source[int(i)]


def _map_rows(rows, row_map_fn, tokenizer, seq_len):
    for row in rows:
        row = row_map_fn(row, tokenizer=tokenizer, seq_len=seq_len)
        if row is not None:
            yield row


def _batch_rows(rows, batch_size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield {k: np.asarray([x[k] for x in batch]) for k in batch[0]}
            batch.clear()


def _prefetch_rows(rows, queue_size):
    q = queue.Queue(maxsize=queue_size)
    done = object()

    def worker():
        try:
            for row in rows:
                q.put(row)
        except Exception as e:
            q.put(e)
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        row = q.get()
        if row is done:
            return
        if isinstance(row, Exception):
            raise row
        yield row


def _pack_rows(rows, seq_len):
    tokens_buf, mask_buf, segment_buf = [], [], []
    segment_id = 1

    def emit(n_tokens):
        pad = seq_len - n_tokens
        return dict(
            tokens=np.asarray(tokens_buf[:n_tokens] + ([0] * pad), dtype=np.int32),
            mask=np.asarray(mask_buf[:n_tokens] + ([False] * pad), dtype=bool),
            segment_ids=np.asarray(segment_buf[:n_tokens] + ([0] * pad), dtype=np.int32),
        )

    for row in rows:
        tokens = row["tokens"]
        if not tokens:
            continue
        mask = row.get("mask")
        if mask is None:
            mask = [True] * len(tokens)
        tokens_buf.extend(tokens)
        mask_buf.extend(mask)
        segment_buf.extend([segment_id] * len(tokens))
        segment_id += 1
        while len(tokens_buf) >= seq_len:
            yield emit(seq_len)
            del tokens_buf[:seq_len], mask_buf[:seq_len], segment_buf[:seq_len]
    if tokens_buf:
        yield emit(len(tokens_buf))


def _split_source(source, num_tokens_valid, seq_len, seed):
    n_valid = num_tokens_valid // seq_len
    if n_valid == 0:
        return source, None
    if hasattr(source, 'train_test_split'):
        split = source.train_test_split(test_size=n_valid, seed=seed)
        train, valid = split['train'], split['test']
    else:
        indices = np.random.default_rng(seed).permutation(len(source))
        valid = [source[int(i)] for i in indices[:n_valid]]
        train = [source[int(i)] for i in indices[n_valid:]]
    return train, valid


def _stream_valid_batches(hf_path, subset, split, tokenizer, seq_len, batch_size, num_tokens_valid):
    n_valid_rows = num_tokens_valid // seq_len
    if n_valid_rows == 0:
        return None, 0
    raw_rows, n_skip, n_tokens = [], 0, 0
    for sample in load_dataset(hf_path, subset, split=split, streaming=True, cache_dir="/dev/shm/ptx/huggingface/datasets"):
        tokens = tokenize_text_batch([sample["text"]], tokenizer, add_eos=True)[0]
        raw_rows.append({"tokens": tokens})
        n_skip += 1
        n_tokens += len(tokens)
        if n_tokens >= n_valid_rows * seq_len:
            break
    valid_rows = [row for _, row in zip(range(n_valid_rows), _pack_rows(raw_rows, seq_len))]
    return list(_batch_rows(valid_rows, batch_size)), n_skip


def load(name, split, batch_size, seq_len, tokenizer, seed=0, num_tokens_valid=0):
    """Returns (source_len, make_train, make_valid), where make_valid is optional."""

    # get dataset
    hf_path = name
    subset = None
    streaming = False
    ds_transform_fn = None
    match name:
        case 'allenai/dolmino-mix-1124':
            subset, streaming = 'dclm', True
        case 'allenai/tulu-3-sft-olmo-2-mixture-0225':
            row_map_fn = process_instruct
        case 'allenai/Dolci-Instruct-SFT':
            row_map_fn = process_instruct
        case 'bnadimi/PyraNet-Verilog':
            row_map_fn = process_pyranet
        case 'allenai/olmo-2-0425-1b-preference-mix':
            row_map_fn = process_preference
        case 'FreedomIntelligence/medical-o1-reasoning-SFT':
            row_map_fn, subset = process_medical_qa, 'en'
        case 'enron_emails':
            row_map_fn, hf_path = process_enron_email, 'MichaelR207/enron_qa_0922'
        case 'enron_qa_base':
            row_map_fn, ds_transform_fn, hf_path = process_qa_answer_only, flatten_enron_qa, 'MichaelR207/enron_qa_0922'
        case 'bio_qa':
            row_map_fn, hf_path = process_qa_answer_only, 'sqvareinch/synthetic-biography-qa-v2'
        case _:
            raise ValueError(f'Unknown dataset: {name}')

    # streaming datasets (too large to download)
    if streaming:
        valid_batches, n_skip = (None, 0) if num_tokens_valid == 0 else _stream_valid_batches(hf_path, subset, split, tokenizer, seq_len, batch_size, num_tokens_valid)
        def make_epoch(epoch):
            ds = load_dataset(hf_path, subset, split=split, streaming=True, cache_dir="/dev/shm/ptx/huggingface/datasets")
            if n_skip:
                ds = ds.skip(n_skip)
            ds = ds.shuffle(seed=seed + epoch, buffer_size=1024)
            ds = ds.map(
                tokenize_text_rows,
                input_columns='text',
                batched=True,
                batch_size=512,
                remove_columns=list(ds.features or next(iter(ds))),
                fn_kwargs=dict(tokenizer=tokenizer),
            )
            return _prefetch_rows(_batch_rows(_pack_rows(iter(ds), seq_len), batch_size), 64)
        make_valid = None if not valid_batches else lambda epoch: iter(valid_batches)
        return None, make_epoch, make_valid

    # load dataset
    hf_dataset = load_dataset(hf_path, subset, split=split, cache_dir="/dev/shm/ptx/huggingface/datasets")

    # optional dataset-level transformation
    if ds_transform_fn is not None:
        source = ds_transform_fn(hf_dataset)
        print(f"Transformed {len(hf_dataset)} records into {len(source)} records")
    else:
        source = hf_dataset

    def make_epoch(source):
        return lambda epoch: _batch_rows(_map_rows(_shuffle_rows(source, seed + epoch), row_map_fn, tokenizer, seq_len), batch_size)

    source, source_valid = _split_source(source, num_tokens_valid, seq_len, seed)
    train_epoch = make_epoch(source)
    make_valid = None if source_valid is None else make_epoch(source_valid)
    return len(source), train_epoch, make_valid


if __name__ == '__main__':
    # print samples
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
    source_len, make_epoch, _ = load('enron_qa_base', 'train[:5]', batch_size=1, seq_len=512, tokenizer=tokenizer)
    print(f'{source_len=}')
    for batch in make_epoch(0):
        for tokens, mask in zip(batch['tokens'], batch['mask']):
            print(mask.mean())
