#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/yqr/work')
DEFAULT_EXP = ROOT / 'data/gorgeous/experiments/region_layout_100k_20260713'
DEFAULT_INDEX = ROOT / 'data/gorgeous/wiki1m/M100_R64_L128'
DEFAULT_BUILD = ROOT / 'data/gorgeous/release_replica_stats'
DEFAULT_REGIONS = ROOT / 'data/gorgeous/logs/regions_stage2_size4/regions.tsv'
DEFAULT_SCORE = ROOT / 'data/gorgeous/logs/transition_scores_later64_1k.tsv'

RESULT_RE = re.compile(r'^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+).*?\s+([0-9.]+)\s*$')
REPLICA_RE = re.compile(r'Replica redundancy stats L=(\d+).*replica_duplicate_rate=([0-9.eE+-]+).*duplicates_per_graph_io=([0-9.eE+-]+)')


def run(cmd, cwd, log_path=None):
    print('+ ' + ' '.join(map(str, cmd)), flush=True)
    started = time.time()
    if log_path:
        with open(log_path, 'w') as f:
            p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise SystemExit(f'command failed ({p.returncode}): {cmd}')
    print(f'  done in {time.time() - started:.1f}s', flush=True)


def parse_search_log(path, policy):
    rows = []
    replica = {}
    for line in Path(path).read_text(errors='ignore').splitlines():
        m = RESULT_RE.match(line)
        if m:
            rows.append({
                'layout': policy,
                'L': int(m.group(1)),
                'BW': int(m.group(2)),
                'QPS': float(m.group(3)),
                'MeanLatency': int(m.group(4)),
                'P99': int(m.group(5)),
                'GraphIO': float(m.group(6)),
                'EmbIO': float(m.group(7)),
                'Recall': float(m.group(8)),
            })
        m = REPLICA_RE.search(line)
        if m:
            replica[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    for row in rows:
        red, dup_per_io = replica.get(row['L'], (0.0, 0.0))
        row['ReplicaRedundancy'] = red
        row['DuplicateReplicasPerIO'] = dup_per_io
    return rows


def write_csv(rows, path):
    fields = ['layout', 'L', 'BW', 'Recall', 'QPS', 'MeanLatency', 'P99', 'GraphIO',
              'ReplicaRedundancy', 'DuplicateReplicasPerIO']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_metric(rows, metric, ylabel, path):
    layouts = []
    for r in rows:
        if r['layout'] not in layouts:
            layouts.append(r['layout'])
    ls = sorted({r['L'] for r in rows})
    width = 0.24
    x = list(range(len(ls)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, layout in enumerate(layouts):
        vals = []
        for L in ls:
            match = next((r for r in rows if r['layout'] == layout and r['L'] == L), None)
            vals.append(match[metric] if match else 0)
        offsets = [v + (idx - (len(layouts)-1)/2) * width for v in x]
        ax.bar(offsets, vals, width=width, label=layout)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L={L}' for L in ls])
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', type=Path, default=DEFAULT_EXP)
    ap.add_argument('--index_dir', type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--build_dir', type=Path, default=DEFAULT_BUILD)
    ap.add_argument('--regions', type=Path, default=DEFAULT_REGIONS)
    ap.add_argument('--transition_scores', type=Path, default=DEFAULT_SCORE)
    ap.add_argument('--run_queries', action='store_true')
    ap.add_argument('--query_file', type=Path, default=ROOT/'data/ann-wiki-1m/query.public.100K.fbin')
    ap.add_argument('--gt_file', type=Path, default=ROOT/'data/ann-wiki-1m/groundtruth.public.100K.ibin')
    args = ap.parse_args()

    exp = args.experiment_dir
    layouts = exp / 'layouts'
    logs = exp / 'logs'
    figs = exp / 'figures'
    for d in (layouts, logs, figs):
        d.mkdir(parents=True, exist_ok=True)

    partitioner = args.build_dir / 'graph_partition/partitioner'
    relayout = args.build_dir / 'tests/utils/index_relayout_free_mem'
    search = args.build_dir / 'tests/search_disk_index'
    base_index = args.index_dir / '_disk.index'
    graph_rep_dir = args.index_dir / 'GRAPH_CACHE_INDEX'
    graph_rep_prefix = str(graph_rep_dir) + '/'
    active_graph_rep = graph_rep_dir / '_graph_rep.index'
    active_partition = graph_rep_dir / '_partition.bin'
    pq_prefix = str(ROOT / 'data/gorgeous/wiki1m/PQ/C4') + '/'

    region_part = layouts / 'region_history_region_layout_part.bin'
    region_summary = layouts / 'packing_summary.csv'
    region_prefix = layouts / 'region_history_region_layout'
    region_index = layouts / 'region_history_region_layout_part_tmp.index'

    if not (layouts/'new_layout_graph_rep.index').exists() or not (layouts/'new_layout_partition.bin').exists():
        run([partitioner, '--data_type', 'float', '--index_file', base_index,
             '--gp_file', region_part, '--thread_nums', '8', '--lock_nums', '0',
             '--block_size', '1', '--ldg_times', '16', '--use_disk', '1', '--visual', '0',
             '--cut', '4096', '--scale', '0', '--mode', '3', '--in_sector_len', '4096',
             '--out_sector_len', '4096', '--packing_policy', 'region_history',
             '--transition_score_file', args.transition_scores, '--replica_limit', '0',
             '--region_file', args.regions, '--packing_summary_file', region_summary,
             '--enable_region_layout', '1'], ROOT, logs/'region_partitioner.log')

        run([relayout, base_index, region_part, 'float', '3', '4096', '4096', '1'], ROOT, logs/'region_relayout.log')
        shutil.copy2(region_index, layouts/'new_layout_graph_rep.index')
        shutil.copy2(region_part, layouts/'new_layout_partition.bin')
    else:
        print(f'use existing new_layout={layouts / "new_layout_graph_rep.index"}')

    if not args.run_queries:
        print(f'new_layout={layouts / "new_layout_graph_rep.index"}')
        print(f'packing_summary={region_summary}')
        return

    backups = []
    if active_graph_rep.exists():
        backup = layouts / 'active_before_region_layout_graph_rep.index'
        shutil.copy2(active_graph_rep, backup)
        backups.append((backup, active_graph_rep))
    if active_partition.exists():
        backup = layouts / 'active_before_region_layout_partition.bin'
        shutil.copy2(active_partition, backup)
        backups.append((backup, active_partition))

    candidates = [
        ('random', graph_rep_dir/'GP_TIMES_16_LOCK_0_CUT4096/_part_tmp.index', graph_rep_dir/'GP_TIMES_16_LOCK_0_CUT4096/_part.bin', 0),
        ('history', graph_rep_dir/'GP_TIMES_16_LOCK_0_CUT4096_PACK_history_RL_0/_part_tmp.index', graph_rep_dir/'GP_TIMES_16_LOCK_0_CUT4096_PACK_history_RL_0/_part.bin', 0),
        ('region', layouts/'new_layout_graph_rep.index', layouts/'new_layout_partition.bin', 1),
    ]

    all_rows = []
    for name, index_file, partition_file, enable_region in candidates:
        if not index_file.exists() or not partition_file.exists():
            print(f'skip {name}: missing {index_file} or {partition_file}')
            continue
        shutil.copy2(index_file, active_graph_rep)
        shutil.copy2(partition_file, active_partition)
        search_log = logs / f'{name}_search.log'
        replica_csv = ROOT / 'logs/replica_redundancy_stats.csv'
        if replica_csv.exists():
            replica_csv.unlink()
        run([search, '--data_type', 'float', '--dist_fn', 'l2',
             '--index_path_prefix', str(args.index_dir) + '/', '--pq_path_prefix', pq_prefix,
             '--query_file', args.query_file, '--gt_file', args.gt_file, '-K', '10',
             '--result_path', args.index_dir/'result/result', '--num_nodes_to_cache', '0',
             '-T', '8', '-L', '50', '100', '-W', '8', '--mem_L', '0',
             '--sector_len', '4096', '--mem_index_path', ROOT/'data/gorgeous/wiki1m/MEM_INDEX/MEM_R_24_L_128_ALPHA_1.2_RANDOM_RATE0.005/',
             '--mem_sample_path', ROOT/'data/gorgeous/wiki1m/MEM_SAMPLE/SAMPLE_RATE_0.005/', '--use_page_search', '1',
             '--use_ratio', '0.3', '--pq_ratio', '0.9', '--disk_file_path', base_index,
             '--graph_rep_index_prefix', graph_rep_prefix, '--disk_graph_prefix', str(args.index_dir/'GRAPH') + '/',
             '--deco_impl', '1', '--use_graph_rep_index', '1', '--enable_region_layout', str(enable_region),
             '--collect_transition_trace', '0', '--mem_graph_use_ratio', '0.0', '--mem_emb_use_ratio', '0.0',
             '--emb_search_ratio', '0.4'], ROOT, search_log)
        if replica_csv.exists():
            shutil.copy2(replica_csv, logs / f'{name}_replica_redundancy_stats.csv')
        all_rows.extend(parse_search_log(search_log, name))

    write_csv(all_rows, exp/'comparison.csv')
    plot_metric(all_rows, 'GraphIO', 'Graph IO', figs/'graph_io.png')
    plot_metric(all_rows, 'MeanLatency', 'Mean Latency', figs/'latency.png')
    plot_metric(all_rows, 'ReplicaRedundancy', 'Replica redundancy', figs/'replica_redundancy.png')

    for backup, target in backups:
        shutil.copy2(backup, target)
    print(f'comparison_csv={exp / "comparison.csv"}')
    print(f'figures={figs}')

if __name__ == '__main__':
    main()
