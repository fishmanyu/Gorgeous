#!/usr/bin/env python3
import argparse
import csv
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/yqr/work')
DEFAULT_INDEX = ROOT / 'data/gorgeous/wiki1m/M100_R64_L128'
DEFAULT_BUILD = ROOT / 'data/gorgeous/release_replica_stats'
DEFAULT_REGIONS = ROOT / 'data/gorgeous/logs/regions_stage2_size4/regions.tsv'
DEFAULT_SCORE = ROOT / 'data/gorgeous/logs/transition_scores_later64_1k.tsv'
DEFAULT_OUT = ROOT / 'data/gorgeous/experiments/region_ablation_controls_20260714'
OLD_HISTORY_PART = DEFAULT_INDEX / 'GRAPH_CACHE_INDEX/GP_TIMES_16_LOCK_0_CUT4096_PACK_history_RL_0/_part.bin'

MODES = {
    'history': (0, 0),
    'dedup_only': (1, 0),
    'reorder_only': (0, 1),
    'dedup_and_reorder': (1, 1),
}

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
INF = 0xffffffff

def fnv_mix(h, v):
    h ^= int(v) & 0xffffffffffffffff
    h = (h * FNV_PRIME) & 0xffffffffffffffff
    return h

def run(cmd, cwd, log):
    print('+ ' + ' '.join(map(str, cmd)), flush=True)
    t = time.time()
    with open(log, 'w') as f:
        p = subprocess.run([str(x) for x in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise SystemExit(f'command failed: {cmd}; see {log}')
    print(f'  done in {time.time() - t:.1f}s', flush=True)

def read_partition(path):
    data = Path(path).read_bytes()
    off = 0
    C, n_pages, nd = struct.unpack_from('QQQ', data, off)
    off += 24
    pages = []
    for _ in range(n_pages):
        (sz,) = struct.unpack_from('I', data, off)
        off += 4
        vals = list(struct.unpack_from(f'{sz}I', data, off)) if sz else []
        off += 4 * sz
        pages.append(vals)
    mapping = list(struct.unpack_from(f'{nd}I', data, off))
    return C, n_pages, nd, pages, mapping

def owner_replica_hash(pages):
    entries = [(p[0], p) for p in pages if p]
    entries.sort(key=lambda x: x[0])
    h = FNV_OFFSET
    for owner, page in entries:
        h = fnv_mix(h, owner)
        h = fnv_mix(h, len(page))
        for v in page[1:]:
            h = fnv_mix(h, v)
    return h

def mapping_hash(mapping):
    h = FNV_OFFSET
    for v in mapping:
        h = fnv_mix(h, v)
    return h

def write_mapping(path, mapping):
    with open(path, 'w') as f:
        f.write('node_id\tphysical_page_id\n')
        for i, p in enumerate(mapping):
            f.write(f'{i}\t{p}\n')

def summary_row(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--index_dir', type=Path, default=DEFAULT_INDEX)
    ap.add_argument('--build_dir', type=Path, default=DEFAULT_BUILD)
    ap.add_argument('--regions', type=Path, default=DEFAULT_REGIONS)
    ap.add_argument('--transition_scores', type=Path, default=DEFAULT_SCORE)
    ap.add_argument('--old_history_part', type=Path, default=OLD_HISTORY_PART)
    args = ap.parse_args()

    if args.out_dir.exists():
        stamp = time.strftime('%H%M%S')
        args.out_dir = args.out_dir.with_name(args.out_dir.name + '_' + stamp)
    args.out_dir.mkdir(parents=True)
    logs = args.out_dir / 'logs'
    logs.mkdir()

    partitioner = args.build_dir / 'graph_partition/partitioner'
    relayout = args.build_dir / 'tests/utils/index_relayout_free_mem'
    base_index = args.index_dir / '_disk.index'

    rows = []
    for name, (dedup, reorder) in MODES.items():
        mode_dir = args.out_dir / name
        mode_dir.mkdir()
        part = mode_dir / '_part.bin'
        summary = mode_dir / 'packing_summary.csv'
        cmd = [partitioner, '--data_type', 'float', '--index_file', base_index,
               '--gp_file', part, '--thread_nums', '8', '--lock_nums', '0',
               '--block_size', '1', '--ldg_times', '16', '--use_disk', '1',
               '--visual', '0', '--cut', '4096', '--scale', '0', '--mode', '3',
               '--in_sector_len', '4096', '--out_sector_len', '4096',
               '--packing_policy', 'history', '--transition_score_file', args.transition_scores,
               '--replica_limit', '0', '--packing_summary_file', summary,
               '--enable_region_replica_dedup', str(dedup),
               '--enable_region_physical_reorder', str(reorder)]
        if dedup or reorder:
            cmd.extend(['--region_file', args.regions])
        run(cmd, ROOT, logs / f'{name}.partitioner.log')

        run([relayout, base_index, part, 'float', '3', '4096', '4096', str(reorder)],
            ROOT, logs / f'{name}.relayout.log')
        tmp_index = mode_dir / '_part_tmp.index'
        graph_rep = mode_dir / '_graph_rep.index'
        if tmp_index.exists():
            tmp_index.rename(graph_rep)

        C, n_pages, nd, pages, mapping = read_partition(part)
        write_mapping(mode_dir / 'mapping.tsv', mapping)
        row = summary_row(summary)
        row.update({
            'mode': name,
            'part_file': str(part),
            'graph_rep_index': str(graph_rep),
            'mapping_file': str(mode_dir / 'mapping.tsv'),
            'parsed_owner_replica_hash': str(owner_replica_hash(pages)),
            'parsed_mapping_hash': str(mapping_hash(mapping)),
        })
        rows.append(row)

    fields = ['mode', 'packing_policy', 'replica_dedup', 'physical_reorder', 'region_file',
              'total_slots', 'filled_slots', 'page_duplicate_slots', 'duplicate_candidates_skipped',
              'replacement_selected', 'fallback_duplicates', 'transition_score_loss',
              'Region_redundancy_before', 'Region_redundancy_after', 'owner_replica_hash',
              'node_to_page_hash', 'parsed_owner_replica_hash', 'parsed_mapping_hash',
              'part_file', 'graph_rep_index', 'mapping_file']
    with open(args.out_dir / 'layout_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    checks = []
    by = {r['mode']: r for r in rows}
    if args.old_history_part.exists():
        _, _, _, old_pages, old_mapping = read_partition(args.old_history_part)
        checks.append(('history_matches_old_history_replica_hash', by['history']['parsed_owner_replica_hash'] == str(owner_replica_hash(old_pages))))
    checks.append(('reorder_only_matches_history_replica_hash', by['reorder_only']['parsed_owner_replica_hash'] == by['history']['parsed_owner_replica_hash']))
    checks.append(('dedup_and_reorder_matches_dedup_only_replica_hash', by['dedup_and_reorder']['parsed_owner_replica_hash'] == by['dedup_only']['parsed_owner_replica_hash']))
    checks.append(('history_mapping_unchanged_vs_dedup_only', by['history']['parsed_mapping_hash'] == by['dedup_only']['parsed_mapping_hash']))
    checks.append(('reorder_only_mapping_changed_vs_history', by['reorder_only']['parsed_mapping_hash'] != by['history']['parsed_mapping_hash']))
    checks.append(('dedup_and_reorder_mapping_changed_vs_dedup_only', by['dedup_and_reorder']['parsed_mapping_hash'] != by['dedup_only']['parsed_mapping_hash']))

    with open(args.out_dir / 'consistency_checks.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['check', 'passed'])
        w.writerows(checks)
    for name, ok in checks:
        print(f'{name}: {"PASS" if ok else "FAIL"}')
    if not all(ok for _, ok in checks):
        raise SystemExit('one or more consistency checks failed')
    print(f'out_dir={args.out_dir}')

if __name__ == '__main__':
    main()
