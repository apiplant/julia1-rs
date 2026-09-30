"""Python reference + baseline benchmark for the original Julia-1 runtime.

  python bench/py_bench.py export                     # rows -> bench/data/*.jsonl
  python bench/py_bench.py reference --device cpu     # logits -> bench/data/ref-cpu.json
  python bench/py_bench.py bench --device cpu --threads 4

Uses the unmodified FastEngine from the checkpoint directory (JULIA_CHECKPOINT,
default ../../ai/Julia-1). The broken system flash_attn wheel is masked so that
transformers falls back to SDPA, which is what the Julia runtime requests anyway.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.modules['flash_attn'] = None  # see module docstring
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CHECKPOINT = Path(os.environ.get('JULIA_CHECKPOINT', ROOT.parent.parent / '..' / 'ai' / 'Julia-1')).resolve()


def export(_args):
    import pyarrow.parquet as pq
    cases = pq.read_table(DATA / 'typed-test.parquet').to_pylist()
    rows = []
    for case in cases:
        state = json.loads(case['state'])
        for question in json.loads(case['questions']).values():
            kind = question['type']
            criteria = question.get('criteria')
            if criteria is None and kind == 'noul':
                criteria = {'false': 'false', 'true': 'true'}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ['false', 'true'] if kind == 'noul' else list(criteria)
            rows.append(dict(state=state, question=question['instructions'], type=kind,
                             options=[criteria[k] for k in keys]))
    with (DATA / 'typed.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Long-context rows: concatenated real states, multilingual text, then a choice.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(CHECKPOINT / 'tokenizer' / 'tokenizer.json'))
    texts = [c['state'] for c in cases]
    tail = 'Preciso trocar minha senha urgentemente — ¿puedes ayudarme? 你好。'
    long_rows = []
    for target_tokens in (1_000, 4_000, 7_600):  # state tokens; head+options add ~40
        state, i = [], 0
        while len(tok.encode('\n'.join(state + [texts[i % len(texts)], tail]), add_special_tokens=False).ids) < target_tokens:
            state.append(texts[i % len(texts)])
            i += 1
        state.append(tail)
        long_rows.append(dict(state='\n'.join(state), question='Which team should handle this request?',
                              type='choice', options=['Billing and payment disputes', 'Shipping and delivery',
                                                      'Account access and login', 'Agent observability']))
    with (DATA / 'long.jsonl').open('w') as f:
        for row in long_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{len(rows)} typed rows, {len(long_rows)} long rows')


def load_rows(name):
    return [json.loads(line) for line in (DATA / name).open()]


def engine_for(args):
    os.environ['JULIA_CPU_THREADS'] = str(args.threads)  # julia.cuda.configure() applies it
    sys.path.insert(0, str(CHECKPOINT))
    from julia import load_model
    return load_model(str(CHECKPOINT), device=args.device, strict_encoding=True,
                      max_length=8192, head_length=512, batch_size=args.batch_size)


def reference(args):
    engine = engine_for(args)
    out = {}
    for name in ('typed.jsonl', 'long.jsonl'):
        rows = load_rows(name)
        out[name] = dict(logits=engine.logits(rows),
                         tokens=[len(x['ids']) for x in engine._encode(rows)],
                         ids=[x['ids'] for x in engine._encode(rows[:50])] if name == 'typed.jsonl'
                         else [x['ids'] for x in engine._encode(rows)])
    path = DATA / f'ref-{args.device}.json'
    path.write_text(json.dumps(out))
    print('wrote', path)


def sync(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()


def bench(args):
    engine = engine_for(args)
    rows = load_rows('typed.jsonl')
    long_rows = load_rows('long.jsonl')
    result = dict(impl='python', device=args.device, threads=args.threads, batch_size=args.batch_size)
    for row in rows[:20]:
        engine.predict([row])
    engine.predict(rows[:64])
    sync(args.device)

    # 1) single-request latency (one request per forward, like reproduce_typed.py)
    n = args.single
    engine.clear_cache()
    lat = []
    for row in rows[:n]:
        t = time.perf_counter()
        engine.predict([row])
        sync(args.device)
        lat.append(time.perf_counter() - t)
    result['single'] = dict(n=n, mean_ms=1e3 * statistics.mean(lat), p50_ms=1e3 * statistics.median(lat),
                            p99_ms=1e3 * sorted(lat)[int(0.99 * (n - 1))], rows_per_s=n / sum(lat))

    # 2) batched throughput over all 2,000 rows (engine micro-batches by length)
    times = []
    for _ in range(args.repeats):
        engine.clear_cache()
        t = time.perf_counter()
        engine.predict(rows)
        sync(args.device)
        times.append(time.perf_counter() - t)
    best = min(times)
    result['batch'] = dict(n=len(rows), best_s=best, rows_per_s=len(rows) / best)

    # 3) long context single requests
    result['long'] = []
    for row in long_rows:
        tokens = len(engine._encode([row])[0]['ids'])
        engine.predict([row])
        sync(args.device)
        ts = []
        for _ in range(args.long_repeats):
            engine.clear_cache()
            t = time.perf_counter()
            engine.predict([row])
            sync(args.device)
            ts.append(time.perf_counter() - t)
        result['long'].append(dict(tokens=tokens, best_ms=1e3 * min(ts)))
    print(json.dumps(result))
    if args.output:
        with open(args.output, 'a') as f:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['export', 'reference', 'bench'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('JULIA_CPU_THREADS', '4')))
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--single', type=int, default=200)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--long-repeats', type=int, default=3)
    ap.add_argument('--output')
    args = ap.parse_args()
    dict(export=export, reference=reference, bench=bench)[args.mode](args)
