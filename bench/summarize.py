"""Render bench/results/*.jsonl (py_bench.py + `julia1 bench` lines) as a Markdown table."""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else 'bench/results/final.jsonl')]
configs = {}
for r in rows:
    key = (r['device'], r['threads'] if r['device'] == 'cpu' else None)
    configs.setdefault(key, {})[r['impl']] = r


def fmt(x, unit):
    return f'{x:,.1f} {unit}' if x < 100 else f'{x:,.0f} {unit}'


print('| Config | Metric | Python | Rust | Speedup |')
print('| --- | --- | ---: | ---: | ---: |')
for (device, threads), pair in configs.items():
    if 'python' not in pair or 'rust' not in pair:
        continue
    py, rs = pair['python'], pair['rust']
    name = device.upper() if threads is None else f'CPU {threads} threads'
    metrics = [
        ('single request, mean', py['single']['mean_ms'], rs['single']['mean_ms'], 'ms', False),
        ('2,000 rows, batch 16', py['batch']['rows_per_s'], rs['batch']['rows_per_s'], 'rows/s', True),
    ] + [(f"{a['tokens']:,}-token request", a['best_ms'], b['best_ms'], 'ms', False)
         for a, b in zip(py['long'], rs['long'])]
    for label, a, b, unit, higher in metrics:
        speed = b / a if higher else a / b
        print(f'| {name} | {label} | {fmt(a, unit)} | {fmt(b, unit)} | **{speed:.2f}×** |')
