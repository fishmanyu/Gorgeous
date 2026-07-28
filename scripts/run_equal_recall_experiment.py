#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_region_ablation_100k as base

ROOT = Path('/home/yqr/work')
MODES = base.MODES
ROTATIONS = base.ROTATIONS
DEFAULT_INDEX = base.DEFAULT_INDEX
DEFAULT_BUILD = base.DEFAULT_BUILD
DEFAULT_LAYOUT_ROOT = base.DEFAULT_LAYOUT_ROOT
DEFAULT_OUT_ROOT = base.DEFAULT_OUT_ROOT

SCAN_FIELDS = ['layout', 'L', 'Recall', 'QPS', 'MeanLatency', 'P99', 'GraphIO']
RAW_FIELDS = ['target_recall', 'layout', 'repeat', 'selected_L', 'Recall', 'QPS', 'MeanLatency', 'P95', 'P99',
              'GraphIO', 'EmbIO', 'ReplicaRedundancy', 'DuplicateReplicasPerIO']
SUMMARY_METRICS = ['Recall', 'QPS', 'MeanLatency', 'P95', 'P99', 'GraphIO', 'EmbIO',
                   'ReplicaRedundancy', 'DuplicateReplicasPerIO']


def read_fbin_prefix(src, dst, count):
    with open(src, 'rb') as f:
        n, d = struct.unpack('ii', f.read(8))
        if count > n:
            raise SystemExit(f'requested {count} queries but source has only {n}')
        data = f.read(count * d * 4)
    with open(dst, 'wb') as f:
        f.write(struct.pack('ii', count, d))
        f.write(data)


def read_ibin_prefix(src, dst, count):
    with open(src, 'rb') as f:
        n, k = struct.unpack('II', f.read(8))
        if count > n:
            raise SystemExit(f'requested {count} GT rows but source has only {n}')
        data = f.read(count * k * 4)
    with open(dst, 'wb') as f:
        f.write(struct.pack('II', count, k))
        f.write(data)


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def run_search(args, out, layout, L, query_file, gt_file, phase, repeat=0, target=None):
    graph_rep_dir = args.index_dir / 'GRAPH_CACHE_INDEX'
    active_index = graph_rep_dir / '_graph_rep.index'
    active_part = graph_rep_dir / '_partition.bin'
    mode_dir = args.layout_root / layout
    shutil.copy2(mode_dir / '_graph_rep.index', active_index)
    shutil.copy2(mode_dir / '_part.bin', active_part)
    reorder = 1 if layout in ('reorder_only', 'dedup_and_reorder') else 0

    subprocess.run(['sync'], check=False)
    logs = out / 'logs'
    tag = f'{phase}_{layout}_L{L}'
    if target is not None:
        tag = f'{phase}_target{target:g}_{layout}_repeat{repeat}_L{L}'
    stdout_log = logs / f'{tag}.log'
    time_log = logs / f'{tag}.time.log'
    search = args.build_dir / 'tests/search_disk_index'
    cmd = [search, '--data_type', 'float', '--dist_fn', 'l2',
           '--index_path_prefix', str(args.index_dir) + '/',
           '--pq_path_prefix', str(ROOT / 'data/gorgeous/wiki1m/PQ/C4') + '/',
           '--query_file', query_file, '--gt_file', gt_file,
           '-K', '10', '--result_path', args.index_dir / 'result/result',
           '--num_nodes_to_cache', '0', '-T', str(args.threads), '-L', str(L), '-W', str(args.beamwidth),
           '--mem_L', '0', '--sector_len', '4096',
           '--mem_index_path', ROOT / 'data/gorgeous/wiki1m/MEM_INDEX/MEM_R_24_L_128_ALPHA_1.2_RANDOM_RATE0.005/',
           '--mem_sample_path', ROOT / 'data/gorgeous/wiki1m/MEM_SAMPLE/SAMPLE_RATE_0.005/',
           '--use_page_search', '1', '--use_ratio', '0.3', '--pq_ratio', '0.9',
           '--disk_file_path', args.index_dir / '_disk.index',
           '--graph_rep_index_prefix', str(graph_rep_dir) + '/',
           '--disk_graph_prefix', str(args.index_dir / 'GRAPH') + '/',
           '--deco_impl', '1', '--use_graph_rep_index', '1',
           '--enable_region_physical_reorder', str(reorder),
           '--collect_transition_trace', '0',
           '--mem_graph_use_ratio', '0.0', '--mem_emb_use_ratio', '0.0', '--emb_search_ratio', '0.4']
    wall = base.run_cmd(cmd, ROOT, stdout_log, time_log, args.cpu_affinity)
    row = base.parse_one(stdout_log, time_log, layout, repeat, L)
    row['wall_clock_runtime'] = wall
    return row


def nearest_row(rows, layout, target):
    group = [r for r in rows if r['layout'] == layout]
    return min(group, key=lambda r: (abs(float(r['Recall']) - target), abs(int(r['L']) - 80)))


def select_targets(scan_rows, requested_targets):
    selected = []
    notes = []
    for requested in requested_targets:
        chosen_target = requested
        tolerance = 0.10
        ok = False
        for tol in (0.10, 0.20):
            picks = [nearest_row(scan_rows, m, requested) for m in MODES]
            if all(abs(float(p['Recall']) - requested) <= tol for p in picks):
                chosen_target = requested
                tolerance = tol
                ok = True
                break
        if not ok:
            candidates = sorted({round(float(r['Recall']), 2) for r in scan_rows})
            best = None
            for cand in candidates:
                picks = [nearest_row(scan_rows, m, cand) for m in MODES]
                max_diff = max(abs(float(p['Recall']) - cand) for p in picks)
                # Keep the fallback near the requested target first; spread only breaks ties.
                score = (abs(cand - requested), max_diff)
                if best is None or score < best[0]:
                    best = (score, cand, picks, max_diff)
            chosen_target = best[1]
            tolerance = max(0.20, best[3])
            notes.append(f'requested {requested:.2f} moved to common 10k target {chosen_target:.2f}; max 10k spread {best[3]:.3f} pp')
        picks = [nearest_row(scan_rows, m, chosen_target) for m in MODES]
        for p in picks:
            selected.append({
                'target_recall': chosen_target,
                'layout': p['layout'],
                'selected_L': int(p['L']),
                'observed_recall_10k': float(p['Recall']),
                'requested_recall': requested,
                'tolerance_pp': tolerance,
            })
    return selected, notes


def summarize(raw_rows):
    out = []
    keys = sorted({(float(r['target_recall']), r['layout'], int(r['selected_L'])) for r in raw_rows})
    for target, layout, selected_L in keys:
        group = [r for r in raw_rows if float(r['target_recall']) == target and r['layout'] == layout and int(r['selected_L']) == selected_L]
        row = {'target_recall': target, 'layout': layout, 'selected_L': selected_L, 'runs': len(group)}
        for metric in SUMMARY_METRICS:
            vals = [float(r[metric]) for r in group]
            row[f'{metric}_mean'] = statistics.mean(vals)
            row[f'{metric}_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f'{metric}_min'] = min(vals)
            row[f'{metric}_max'] = max(vals)
        out.append(row)
    return out


def comparison(summary, tolerances):
    by = {(float(r['target_recall']), r['layout']): r for r in summary}
    rows = []
    for target in sorted({float(r['target_recall']) for r in summary}):
        h = by[(target, 'history')]
        tol = tolerances[target]
        for layout in MODES:
            if layout == 'history':
                continue
            r = by[(target, layout)]
            recall_diff = r['Recall_mean'] - h['Recall_mean']
            target_diff = abs(r['Recall_mean'] - target)
            history_target_diff = abs(h['Recall_mean'] - target)
            rows.append({
                'target_recall': target,
                'layout': layout,
                'selected_L': r['selected_L'],
                'history_L': h['selected_L'],
                'valid_equal_recall': target_diff <= tol and history_target_diff <= tol and abs(recall_diff) <= tol,
                'QPS_change_pct': base.pct(r['QPS_mean'], h['QPS_mean']),
                'MeanLatency_change_pct': base.pct(r['MeanLatency_mean'], h['MeanLatency_mean']),
                'P99_change_pct': base.pct(r['P99_mean'], h['P99_mean']),
                'GraphIO_change_pct': base.pct(r['GraphIO_mean'], h['GraphIO_mean']),
                'Recall_difference_pp': recall_diff,
            })
    return rows


def plot_equal_recall(summary, out):
    figs = out / 'figures'
    for target in sorted({float(r['target_recall']) for r in summary}):
        rows = {r['layout']: r for r in summary if float(r['target_recall']) == target}
        for metric, ylabel in [('QPS', 'QPS'), ('MeanLatency', 'Mean latency'), ('P99', 'P99'),
                               ('GraphIO', 'Graph I/O'), ('Recall', 'Actual Recall')]:
            labels = MODES
            means = [rows[m][f'{metric}_mean'] for m in labels]
            stds = [rows[m][f'{metric}_std'] for m in labels]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, means, yerr=stds, capsize=4)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} target Recall {target:.2f}, BW=8')
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', alpha=.25)
            fig.tight_layout()
            fig.savefig(figs / f'{metric.lower()}_target{target:.2f}.png', dpi=180)
            plt.close(fig)


def make_report(out, selected, summary, comp, notes, repeat, scan_ls, cache_method, cpu_affinity):
    by = {(float(r['target_recall']), r['layout']): r for r in summary}
    lines = ['# Equal Recall Region Ablation Report', '',
             f'- repeats: {repeat}',
             f'- scan L values: {",".join(str(x) for x in scan_ls)}',
             f'- cache handling: {cache_method}',
             f'- CPU affinity: {cpu_affinity or "not pinned"}',
             '- comparison rule: actual Recall must be within selected tolerance and layout-history difference must be within tolerance',
             '']
    if notes:
        lines += ['## Target Selection Notes'] + [f'- {n}' for n in notes] + ['']
    lines += ['## Selected Parameters']
    for row in selected:
        lines.append(f'- target {float(row["target_recall"]):.2f}, {row["layout"]}: L={row["selected_L"]}, 10k Recall={float(row["observed_recall_10k"]):.2f}')
    lines += ['', '## 100k Results']
    for target in sorted({float(r['target_recall']) for r in summary}):
        lines.append(f'### Target Recall {target:.2f}')
        for layout in MODES:
            r = by[(target, layout)]
            lines.append(f'- {layout}: L={r["selected_L"]}, Recall {r["Recall_mean"]:.2f}±{r["Recall_std"]:.3f}, QPS {r["QPS_mean"]:.2f}±{r["QPS_std"]:.2f}, Mean {r["MeanLatency_mean"]:.0f}, P99 {r["P99_mean"]:.0f}, GraphIO {r["GraphIO_mean"]:.2f}, EmbIO {r["EmbIO_mean"]:.2f}, ReplicaRedundancy {r["ReplicaRedundancy_mean"]:.3f}')
    lines += ['', '## Answers']
    for target in sorted({float(r['target_recall']) for r in summary}):
        cs = {r['layout']: r for r in comp if float(r['target_recall']) == target}
        valid = all(r['valid_equal_recall'] in (True, 'True') for r in cs.values())
        lines.append(f'- target {target:.2f}: equal-Recall validity vs history: {"PASS" if valid else "CHECK"}')
        for layout in ['dedup_only', 'reorder_only', 'dedup_and_reorder']:
            r = cs[layout]
            lines.append(f'- target {target:.2f}: {layout} vs history: QPS {r["QPS_change_pct"]:.2f}%, MeanLatency {r["MeanLatency_change_pct"]:.2f}%, P99 {r["P99_change_pct"]:.2f}%, GraphIO {r["GraphIO_change_pct"]:.2f}%, Recall diff {r["Recall_difference_pp"]:.3f} pp')
        single_qps = cs['dedup_only']['QPS_change_pct'] + cs['reorder_only']['QPS_change_pct']
        combined_qps = cs['dedup_and_reorder']['QPS_change_pct']
        lines.append(f'- target {target:.2f}: combined QPS gain {combined_qps:.2f}% vs single-gain sum {single_qps:.2f}%')
    lines += ['', 'Interpretation: prefer effects that are valid equal-Recall comparisons and larger than the three-run standard deviation. No layout or search algorithm was changed in this phase.']
    (out / 'equal_recall_report.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--layout_root', type=Path, default=DEFAULT_LAYOUT_ROOT)
    ap.add_argument('--out_dir', type=Path, default=None)
    ap.add_argument('--index_dir', type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--build_dir', type=Path, default=DEFAULT_BUILD)
    ap.add_argument('--cpu_affinity', default='0-7')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--beamwidth', type=int, default=8)
    ap.add_argument('--subset_size', type=int, default=10000)
    ap.add_argument('--scan_Ls', default='40,45,50,55,60,70,75,80,90,100,110,120')
    ap.add_argument('--targets', default='96.0,98.3')
    ap.add_argument('--reuse_scan_dir', type=Path, default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = args.out_dir or (DEFAULT_OUT_ROOT / f'equal_recall_ablation_{ts}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    (out / 'figures').mkdir()

    checks = base.preflight(args.layout_root)
    write_csv(out / 'preflight_checks.csv', [{'check': n, 'passed': ok} for n, ok in checks])
    for name, ok in checks:
        print(f'{name}: {"PASS" if ok else "FAIL"}')
    if not all(ok for _, ok in checks):
        raise SystemExit('preflight failed')

    query_100k = ROOT / 'data/ann-wiki-1m/query.public.100K.fbin'
    gt_100k = ROOT / 'data/ann-wiki-1m/groundtruth.public.100K.ibin'
    query_10k = out / 'query.public.10K.fbin'
    gt_10k = out / 'groundtruth.public.10K.ibin'
    if args.reuse_scan_dir:
        shutil.copy2(args.reuse_scan_dir / 'query.public.10K.fbin', query_10k)
        shutil.copy2(args.reuse_scan_dir / 'groundtruth.public.10K.ibin', gt_10k)
        shutil.copy2(args.reuse_scan_dir / 'recall_scan_10k.csv', out / 'recall_scan_10k.csv')
    else:
        read_fbin_prefix(query_100k, query_10k, args.subset_size)
        read_ibin_prefix(gt_100k, gt_10k, args.subset_size)

    graph_rep_dir = args.index_dir / 'GRAPH_CACHE_INDEX'
    active_index = graph_rep_dir / '_graph_rep.index'
    active_part = graph_rep_dir / '_partition.bin'
    backup_index = out / 'active_before_graph_rep.index'
    backup_part = out / 'active_before_partition.bin'
    shutil.copy2(active_index, backup_index)
    shutil.copy2(active_part, backup_part)

    scan_ls = [int(x) for x in args.scan_Ls.split(',') if x.strip()]
    requested_targets = [float(x) for x in args.targets.split(',') if x.strip()]
    cache_method = 'sync before each run; no drop_caches; O_DIRECT search binary'
    scan_rows = []
    full_rows = []
    selected = []
    notes = []
    try:
        if args.reuse_scan_dir:
            with open(out / 'recall_scan_10k.csv') as f:
                scan_rows = [{**r, 'L': int(r['L']), 'Recall': float(r['Recall']), 'QPS': float(r['QPS']),
                              'MeanLatency': float(r['MeanLatency']), 'P99': float(r['P99']), 'GraphIO': float(r['GraphIO'])}
                             for r in csv.DictReader(f)]
        else:
            for L in scan_ls:
                for layout in MODES:
                    row = run_search(args, out, layout, L, query_10k, gt_10k, phase='scan')
                    scan_row = {k: row[k] for k in SCAN_FIELDS}
                    scan_rows.append(scan_row)
                    write_csv(out / 'recall_scan_10k.csv', scan_rows, SCAN_FIELDS)

        selected, notes = select_targets(scan_rows, requested_targets)
        selected_fields = ['target_recall', 'layout', 'selected_L', 'observed_recall_10k']
        write_csv(out / 'selected_equal_recall_params.csv', selected, selected_fields)
        tolerances = {float(r['target_recall']): float(r['tolerance_pp']) for r in selected}

        for rep in range(1, args.repeat + 1):
            order = ROTATIONS[(rep - 1) % len(ROTATIONS)]
            for target in sorted(tolerances):
                target_rows = {r['layout']: r for r in selected if float(r['target_recall']) == target}
                for layout in order:
                    L = int(target_rows[layout]['selected_L'])
                    row = run_search(args, out, layout, L, query_100k, gt_100k, phase='full',
                                     repeat=rep, target=target)
                    full_rows.append({
                        'target_recall': target,
                        'layout': layout,
                        'repeat': rep,
                        'selected_L': L,
                        'Recall': row['Recall'],
                        'QPS': row['QPS'],
                        'MeanLatency': row['MeanLatency'],
                        'P95': row['P95'],
                        'P99': row['P99'],
                        'GraphIO': row['GraphIO'],
                        'EmbIO': row['EmbIO'],
                        'ReplicaRedundancy': row.get('ReplicaRedundancy', 0.0),
                        'DuplicateReplicasPerIO': row.get('DuplicateReplicasPerIO', 0.0),
                    })
                    write_csv(out / 'equal_recall_raw.csv', full_rows, RAW_FIELDS)
    finally:
        shutil.copy2(backup_index, active_index)
        shutil.copy2(backup_part, active_part)

    summary = summarize(full_rows)
    write_csv(out / 'equal_recall_summary.csv', summary)
    tolerances = {float(r['target_recall']): float(r['tolerance_pp']) for r in selected}
    comp = comparison(summary, tolerances)
    write_csv(out / 'equal_recall_vs_history.csv', comp)
    plot_equal_recall(summary, out)
    make_report(out, selected, summary, comp, notes, args.repeat, scan_ls, cache_method, args.cpu_affinity)
    print(f'out_dir={out}')


if __name__ == '__main__':
    main()
