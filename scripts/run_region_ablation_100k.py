#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/yqr/work')
DEFAULT_INDEX = ROOT / 'data/gorgeous/wiki1m/M100_R64_L128'
DEFAULT_BUILD = ROOT / 'data/gorgeous/release_replica_stats'
DEFAULT_LAYOUT_ROOT = ROOT / 'data/gorgeous/experiments/region_ablation_controls_20260714'
DEFAULT_OUT_ROOT = ROOT / 'data/gorgeous/experiments'
MODES = ['history', 'dedup_only', 'reorder_only', 'dedup_and_reorder']
ROTATIONS = [
    ['history', 'dedup_only', 'reorder_only', 'dedup_and_reorder'],
    ['reorder_only', 'dedup_and_reorder', 'history', 'dedup_only'],
    ['dedup_and_reorder', 'history', 'dedup_only', 'reorder_only'],
]
METRICS = ['Recall','QPS','MeanLatency','P50','P95','P99','GraphIO','EmbIO','ReplicaRedundancy',
           'CrossPageRedundancy','DuplicateReplicasPerIO','PagesWithAnyDuplicateRatio',
           'FullyRedundantPageRatio','wall_clock_runtime','CPU_utilization','peak_memory']
RESULT_RE = re.compile(r'^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+).*?\s+(\d+)\s+([0-9.]+)\s*$')
REPLICA_RE = re.compile(r'Replica redundancy stats L=(\d+).*replica_duplicate_rate=([0-9.eE+-]+).*cross_page_duplicate_rate=([0-9.eE+-]+).*duplicates_per_graph_io=([0-9.eE+-]+).*pages_with_any_duplicate_rate=([0-9.eE+-]+).*fully_redundant_page_rate=([0-9.eE+-]+)')
TIME_CPU_RE = re.compile(r'Percent of CPU this job got: ([0-9.]+)%')
TIME_RSS_RE = re.compile(r'Maximum resident set size \(kbytes\): (\d+)')

def read_partition(path):
    data = Path(path).read_bytes(); off = 0
    C, n_pages, nd = struct.unpack_from('QQQ', data, off); off += 24
    pages = []
    for _ in range(n_pages):
        (sz,) = struct.unpack_from('I', data, off); off += 4
        vals = list(struct.unpack_from(f'{sz}I', data, off)) if sz else []
        off += 4 * sz
        pages.append(vals)
    mapping = list(struct.unpack_from(f'{nd}I', data, off))
    return C, n_pages, nd, pages, mapping

def owner_replica_signature(pages):
    return sorted((p[0], tuple(p[1:])) for p in pages if p)

def page_order(pages):
    return tuple(p[0] if p else 0xffffffff for p in pages)

def graph_meta(path):
    p = Path(path)
    with p.open('rb') as f:
        head = f.read(4096)
    vals = struct.unpack_from('II', head, 0)
    meta_n, meta_dim = vals
    u64s = struct.unpack_from('9Q', head, 8) if meta_n == 9 else struct.unpack_from('11Q', head, 0)
    return (p.stat().st_size, meta_n, meta_dim, u64s[:6])

def load_summary_rows(layout_root):
    with (layout_root / 'layout_summary.csv').open() as f:
        return {r['mode']: r for r in csv.DictReader(f)}

def preflight(layout_root):
    rows = load_summary_rows(layout_root)
    parts = {m: layout_root / m / '_part.bin' for m in MODES}
    indexes = {m: layout_root / m / '_graph_rep.index' for m in MODES}
    for m in MODES:
        for p in [parts[m], indexes[m], layout_root / m / 'packing_summary.csv', layout_root / m / 'mapping.tsv']:
            if not p.exists():
                raise SystemExit(f'missing required file: {p}')
    parsed = {m: read_partition(parts[m]) for m in MODES}
    pages = {m: parsed[m][3] for m in MODES}
    checks = []
    checks.append(('history_reorder_only_packing_summary', rows['history']['parsed_owner_replica_hash'] == rows['reorder_only']['parsed_owner_replica_hash']))
    checks.append(('dedup_only_dedup_and_reorder_packing_summary', rows['dedup_only']['parsed_owner_replica_hash'] == rows['dedup_and_reorder']['parsed_owner_replica_hash']))
    checks.append(('history_dedup_only_page_order', page_order(pages['history']) == page_order(pages['dedup_only'])))
    checks.append(('reorder_only_dedup_and_reorder_page_order', page_order(pages['reorder_only']) == page_order(pages['dedup_and_reorder'])))
    base_meta = (parsed['history'][0], parsed['history'][1], parsed['history'][2])
    checks.append(('partition_node_page_count_consistent', all((parsed[m][0], parsed[m][1], parsed[m][2]) == base_meta for m in MODES)))
    idx_meta = graph_meta(indexes['history'])
    checks.append(('graph_index_metadata_consistent', all(graph_meta(indexes[m]) == idx_meta for m in MODES)))
    # Same page capacity and every graph-rep page has owner plus replicas bounded by C.
    checks.append(('graph_page_format_consistent', all(all(1 <= len(p) <= parsed[m][0] for p in pages[m]) for m in MODES)))
    return checks

def run_cmd(cmd, cwd, stdout_log, time_log, cpu_affinity):
    full = []
    if cpu_affinity:
        full += ['taskset', '-c', cpu_affinity]
    full += ['/usr/bin/time', '-v', '-o', str(time_log)] + [str(x) for x in cmd]
    started = time.time()
    with open(stdout_log, 'w') as out:
        p = subprocess.run(full, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - started
    if p.returncode != 0:
        raise SystemExit(f'command failed ({p.returncode}); see {stdout_log}')
    return wall

def parse_one(stdout_log, time_log, layout, repeat, L):
    row = None; red = {}
    for line in Path(stdout_log).read_text(errors='ignore').splitlines():
        m = RESULT_RE.match(line)
        if m:
            row = {
                'layout': layout, 'repeat': repeat, 'L': int(m.group(1)), 'BW': int(m.group(2)),
                'QPS': float(m.group(3)), 'MeanLatency': int(m.group(4)), 'P50': int(m.group(5)),
                'P95': int(m.group(6)), 'P99': int(m.group(7)), 'P999': int(m.group(8)),
                'GraphIO': float(m.group(9)), 'EmbIO': float(m.group(10)),
                'peak_memory': int(m.group(11)), 'Recall': float(m.group(12)),
            }
        m = REPLICA_RE.search(line)
        if m:
            red[int(m.group(1))] = {
                'ReplicaRedundancy': float(m.group(2)), 'CrossPageRedundancy': float(m.group(3)),
                'DuplicateReplicasPerIO': float(m.group(4)), 'PagesWithAnyDuplicateRatio': float(m.group(5)),
                'FullyRedundantPageRatio': float(m.group(6)),
            }
    if row is None:
        raise SystemExit(f'could not parse result row from {stdout_log}')
    row.update(red.get(row['L'], {}))
    txt = Path(time_log).read_text(errors='ignore') if Path(time_log).exists() else ''
    cm = TIME_CPU_RE.search(txt); rm = TIME_RSS_RE.search(txt)
    row['CPU_utilization'] = float(cm.group(1)) if cm else 0.0
    if rm:
        row['peak_memory'] = max(row.get('peak_memory', 0), int(int(rm.group(1)) / 1024))
    return row

def mean_std(rows):
    out = []
    for layout in MODES:
        for L in [50, 100]:
            group = [r for r in rows if r['layout'] == layout and r['L'] == L]
            if not group: continue
            base = {'layout': layout, 'L': L, 'BW': 8}
            for metric in METRICS:
                vals = [float(r.get(metric, 0.0)) for r in group]
                base[f'{metric}_mean'] = statistics.mean(vals)
                base[f'{metric}_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                base[f'{metric}_min'] = min(vals)
                base[f'{metric}_max'] = max(vals)
            out.append(base)
    return out

def comparison(summary):
    by = {(r['layout'], r['L']): r for r in summary}
    rows = []
    for layout in MODES:
        if layout == 'history': continue
        for L in [50, 100]:
            h = by[('history', L)]; r = by[(layout, L)]
            rows.append({
                'layout': layout, 'L': L, 'BW': 8,
                'QPS_change_pct': pct(r['QPS_mean'], h['QPS_mean']),
                'MeanLatency_change_pct': pct(r['MeanLatency_mean'], h['MeanLatency_mean']),
                'P99_change_pct': pct(r['P99_mean'], h['P99_mean']),
                'GraphIO_change_pct': pct(r['GraphIO_mean'], h['GraphIO_mean']),
                'Recall_change_pp': r['Recall_mean'] - h['Recall_mean'],
                'ReplicaRedundancy_change_pp': (r['ReplicaRedundancy_mean'] - h['ReplicaRedundancy_mean']) * 100.0,
                'DuplicateReplicasPerIO_change_pct': pct(r['DuplicateReplicasPerIO_mean'], h['DuplicateReplicasPerIO_mean']),
            })
    return rows

def pct(v, b):
    return 0.0 if b == 0 else (v - b) * 100.0 / b

def write_csv(path, rows, fields=None):
    if not rows: return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def plot_metric(summary, metric, ylabel, out):
    for L in [50, 100]:
        labels = MODES
        means = [next(r[f'{metric}_mean'] for r in summary if r['layout']==m and r['L']==L) for m in labels]
        stds = [next(r[f'{metric}_std'] for r in summary if r['layout']==m and r['L']==L) for m in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_ylabel(ylabel); ax.set_title(f'{ylabel} L={L}, BW=8')
        ax.tick_params(axis='x', rotation=15); ax.grid(axis='y', alpha=.25)
        fig.tight_layout(); fig.savefig(out / f'{metric.lower()}_L{L}.png', dpi=180); plt.close(fig)

def make_report(out, summary, comp, checks, repeat, order_text, cache_method, cpu_affinity):
    by = {(r['layout'], r['L']): r for r in summary}
    lines = ['# Co-access Region Ablation Report', '', f'- repeats: {repeat}', f'- order: {order_text}',
             f'- cache handling: {cache_method}', f'- CPU affinity: {cpu_affinity or "not pinned"}', '', '## Consistency Checks']
    for name, ok in checks:
        lines.append(f'- {name}: {"PASS" if ok else "FAIL"}')
    lines += ['', '## Key Results']
    for L in [50,100]:
        lines.append(f'### L={L}')
        h = by[('history', L)]
        for layout in MODES:
            r = by[(layout, L)]
            lines.append(f'- {layout}: QPS {r["QPS_mean"]:.2f}±{r["QPS_std"]:.2f}, Mean {r["MeanLatency_mean"]:.0f}, P99 {r["P99_mean"]:.0f}, GraphIO {r["GraphIO_mean"]:.2f}, Recall {r["Recall_mean"]:.2f}, ReplicaRedundancy {r["ReplicaRedundancy_mean"]:.3f}')
    lines += ['', '## Answers']
    def c(layout,L,field):
        return next(r for r in comp if r['layout']==layout and r['L']==L)[field]
    for L in [50,100]:
        lines.append(f'- L={L}: dedup_only GraphIO vs history: {c("dedup_only",L,"GraphIO_change_pct"):.2f}%')
        lines.append(f'- L={L}: reorder_only QPS vs history: {c("reorder_only",L,"QPS_change_pct"):.2f}%, MeanLatency: {c("reorder_only",L,"MeanLatency_change_pct"):.2f}%')
        lines.append(f'- L={L}: dedup_and_reorder QPS vs history: {c("dedup_and_reorder",L,"QPS_change_pct"):.2f}%, GraphIO: {c("dedup_and_reorder",L,"GraphIO_change_pct"):.2f}%')
        single_sum_qps = c('dedup_only',L,'QPS_change_pct') + c('reorder_only',L,'QPS_change_pct')
        combined_qps = c('dedup_and_reorder',L,'QPS_change_pct')
        lines.append(f'- L={L}: combined QPS gain {combined_qps:.2f}% vs single-gain sum {single_sum_qps:.2f}%')
        lines.append(f'- L={L}: ReplicaRedundancy change dedup_only {c("dedup_only",L,"ReplicaRedundancy_change_pp"):.2f} pp, reorder_only {c("reorder_only",L,"ReplicaRedundancy_change_pp"):.2f} pp, combined {c("dedup_and_reorder",L,"ReplicaRedundancy_change_pp"):.2f} pp')
        lines.append(f'- L={L}: P99 change dedup_only {c("dedup_only",L,"P99_change_pct"):.2f}%, reorder_only {c("reorder_only",L,"P99_change_pct"):.2f}%, combined {c("dedup_and_reorder",L,"P99_change_pct"):.2f}%')
    lines.append('')
    lines.append('Interpretation should prioritize changes that are consistent across L=50 and L=100 and larger than run-to-run standard deviation in summary_mean_std.csv.')
    (out/'ablation_report.md').write_text('\n'.join(lines)+'\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--layout_root', type=Path, default=DEFAULT_LAYOUT_ROOT)
    ap.add_argument('--out_dir', type=Path, default=None)
    ap.add_argument('--index_dir', type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--build_dir', type=Path, default=DEFAULT_BUILD)
    ap.add_argument('--cpu_affinity', default='0-7')
    args = ap.parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = args.out_dir or (DEFAULT_OUT_ROOT / f'region_ablation_100k_{ts}')
    out.mkdir(parents=True, exist_ok=False)
    logs = out/'logs'; logs.mkdir(); figs = out/'figures'; figs.mkdir()

    checks = preflight(args.layout_root)
    write_csv(out/'preflight_checks.csv', [{'check':n,'passed':ok} for n,ok in checks])
    for n, ok in checks: print(f'{n}: {"PASS" if ok else "FAIL"}')
    if not all(ok for _, ok in checks): raise SystemExit('preflight failed')

    search = args.build_dir / 'tests/search_disk_index'
    graph_rep_dir = args.index_dir / 'GRAPH_CACHE_INDEX'
    active_index = graph_rep_dir / '_graph_rep.index'
    active_part = graph_rep_dir / '_partition.bin'
    backup_index = out / 'active_before_graph_rep.index'
    backup_part = out / 'active_before_partition.bin'
    shutil.copy2(active_index, backup_index); shutil.copy2(active_part, backup_part)

    rows = []
    cache_method = 'sync before each run; no drop_caches; O_DIRECT search binary'
    order_lines = []
    try:
        for rep in range(1, args.repeat+1):
            order = ROTATIONS[(rep-1) % len(ROTATIONS)]
            order_lines.append(f'repeat {rep}: ' + ' -> '.join(order))
            for layout in order:
                mode_dir = args.layout_root / layout
                shutil.copy2(mode_dir/'_graph_rep.index', active_index)
                shutil.copy2(mode_dir/'_part.bin', active_part)
                reorder = 1 if layout in ('reorder_only','dedup_and_reorder') else 0
                for L in [50, 100]:
                    subprocess.run(['sync'], check=False)
                    stdout_log = logs / f'{layout}_repeat{rep}_L{L}.log'
                    time_log = logs / f'{layout}_repeat{rep}_L{L}.time.log'
                    cmd = [search, '--data_type','float','--dist_fn','l2',
                           '--index_path_prefix', str(args.index_dir)+'/', '--pq_path_prefix', str(ROOT/'data/gorgeous/wiki1m/PQ/C4')+'/',
                           '--query_file', ROOT/'data/ann-wiki-1m/query.public.100K.fbin', '--gt_file', ROOT/'data/ann-wiki-1m/groundtruth.public.100K.ibin',
                           '-K','10','--result_path', args.index_dir/'result/result', '--num_nodes_to_cache','0',
                           '-T','8','-L',str(L),'-W','8','--mem_L','0','--sector_len','4096',
                           '--mem_index_path', ROOT/'data/gorgeous/wiki1m/MEM_INDEX/MEM_R_24_L_128_ALPHA_1.2_RANDOM_RATE0.005/',
                           '--mem_sample_path', ROOT/'data/gorgeous/wiki1m/MEM_SAMPLE/SAMPLE_RATE_0.005/',
                           '--use_page_search','1','--use_ratio','0.3','--pq_ratio','0.9','--disk_file_path', args.index_dir/'_disk.index',
                           '--graph_rep_index_prefix', str(graph_rep_dir)+'/', '--disk_graph_prefix', str(args.index_dir/'GRAPH')+'/',
                           '--deco_impl','1','--use_graph_rep_index','1','--enable_region_physical_reorder',str(reorder),
                           '--collect_transition_trace','0','--mem_graph_use_ratio','0.0','--mem_emb_use_ratio','0.0','--emb_search_ratio','0.4']
                    wall = run_cmd(cmd, ROOT, stdout_log, time_log, args.cpu_affinity)
                    row = parse_one(stdout_log, time_log, layout, rep, L)
                    row['wall_clock_runtime'] = wall
                    rows.append(row)
                    write_csv(out/'raw_results.csv', rows)
    finally:
        shutil.copy2(backup_index, active_index); shutil.copy2(backup_part, active_part)

    raw_fields = ['layout','repeat','L','BW','Recall','QPS','MeanLatency','P50','P95','P99','GraphIO','EmbIO',
                  'ReplicaRedundancy','CrossPageRedundancy','DuplicateReplicasPerIO','PagesWithAnyDuplicateRatio',
                  'FullyRedundantPageRatio','wall_clock_runtime','CPU_utilization','peak_memory']
    write_csv(out/'raw_results.csv', rows, raw_fields)
    summary = mean_std(rows)
    summary_fields = list(summary[0].keys())
    write_csv(out/'summary_mean_std.csv', summary, summary_fields)
    comp = comparison(summary)
    write_csv(out/'comparison_vs_history.csv', comp)
    for metric,ylabel in [('QPS','QPS'),('MeanLatency','Mean latency'),('P99','P99'),('GraphIO','Mean Graph I/O'),('Recall','Recall'),('ReplicaRedundancy','Replica redundancy'),('DuplicateReplicasPerIO','Duplicate replicas per Graph I/O')]:
        plot_metric(summary, metric, ylabel, figs)
    make_report(out, summary, comp, checks, args.repeat, '\n'.join(order_lines), cache_method, args.cpu_affinity)
    print(f'out_dir={out}')

if __name__ == '__main__':
    main()
