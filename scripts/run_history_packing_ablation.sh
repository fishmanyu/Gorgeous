#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-wiki1m}"
BUILD_TYPE="${BUILD_TYPE:-release}"
TRANSITION_SCORE_FILE="${TRANSITION_SCORE_FILE:-/home/yqr/work/data/gorgeous/logs/transition_scores.tsv}"
REPLICA_LIMIT="${REPLICA_LIMIT:-0}"
COLLECT_TRACE_FOR_IO="${COLLECT_TRACE_FOR_IO:-0}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/yqr/work/data/gorgeous/experiments/history_packing_$(date +%Y%m%d_%H%M%S)}"

CONFIG_LOCAL="${SCRIPT_DIR}/config_local.sh"
CONFIG_BACKUP="${EXPERIMENT_ROOT}/config_local.before.sh"

mkdir -p "${EXPERIMENT_ROOT}/logs" "${EXPERIMENT_ROOT}/layouts"
cp "${CONFIG_LOCAL}" "${CONFIG_BACKUP}"

restore_config() {
  if [ -f "${CONFIG_BACKUP}" ]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_LOCAL}"
  fi
}
trap restore_config EXIT

case "${DATASET}" in
  wiki1m)
    DATASET_FUNC="dataset_wiki1m"
    PREFIX_NAME="wiki1m"
    ;;
  sift|sift_learn)
    DATASET_FUNC="dataset_sift_learn"
    PREFIX_NAME="sift_learn"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; use wiki1m or sift_learn." >&2
    exit 1
    ;;
esac

python3 - "${CONFIG_LOCAL}" "${DATASET_FUNC}" "${COLLECT_TRACE_FOR_IO}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
dataset_func = sys.argv[2]
collect_trace = sys.argv[3]
text = path.read_text()
text = re.sub(r'(?m)^dataset_(sift_learn|wiki1m)\s*$', f'{dataset_func}', text)
if 'COLLECT_TRANSITION_TRACE=' in text:
    text = re.sub(r'(?m)^COLLECT_TRANSITION_TRACE=.*$', f'COLLECT_TRANSITION_TRACE={collect_trace}', text)
else:
    text += f'\nCOLLECT_TRANSITION_TRACE={collect_trace}\n'
path.write_text(text)
PY

# shellcheck source=/dev/null
source "${CONFIG_LOCAL}"

MEM_TOPK="${MEM_TOPK:-}"

INDEX_PREFIX_PATH="${PREFIX}/M${M}_R${R}_L${BUILD_L}/"
GRAPH_REP_INDEX_PATH="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}GRAPH_CACHE_INDEX/"
GRAPH_PATH="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}GRAPH/"
BASE_GP_NAME="GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}"
RANDOM_LAYOUT_DIR="${GRAPH_REP_INDEX_PATH}${BASE_GP_NAME}"
HISTORY_LAYOUT_DIR="${GRAPH_REP_INDEX_PATH}${BASE_GP_NAME}_PACK_history_RL_${REPLICA_LIMIT}"
ACTIVE_GRAPH_REP="${GRAPH_REP_INDEX_PATH}_graph_rep.index"
ACTIVE_PARTITION="${GRAPH_REP_INDEX_PATH}_partition.bin"
SEARCH_LOG="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}search/search_K${K}_CACHE${CACHE}_BW${BM_LIST[0]}_T${T_LIST[0]}_MEML${MEM_L}_MEMK${MEM_TOPK}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log"

if [ "${PREFIX}" != "${PREFIX_NAME}" ]; then
  echo "Config dataset mismatch: expected ${PREFIX_NAME}, got ${PREFIX}" >&2
  exit 1
fi

required_files=(
  "${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}_disk.index"
  "${GRAPH_PATH}_disk_graph.index"
  "${TRANSITION_SCORE_FILE}"
)

for f in "${required_files[@]}"; do
  if [ ! -f "${f}" ]; then
    echo "Missing required file: ${f}" >&2
    exit 1
  fi
done

save_existing_layout_dir() {
  local dir="$1"
  local policy="$2"
  if [ -d "${dir}" ]; then
    local backup="${dir}.before_history_ablation_$(date +%Y%m%d_%H%M%S)"
    echo "Saving existing ${policy} layout dir: ${dir} -> ${backup}"
    mv "${dir}" "${backup}"
    echo "${backup}" > "${EXPERIMENT_ROOT}/layouts/${policy}_previous_layout_dir.txt"
  fi
}

activate_layout() {
  local dir="$1"
  local policy="$2"
  if [ ! -f "${dir}/_part_tmp.index" ] || [ ! -f "${dir}/_part.bin" ]; then
    echo "Layout files missing for ${policy}: ${dir}" >&2
    exit 1
  fi
  mkdir -p "${GRAPH_REP_INDEX_PATH}"
  if [ -f "${ACTIVE_GRAPH_REP}" ]; then
    cp -a "${ACTIVE_GRAPH_REP}" "${EXPERIMENT_ROOT}/layouts/active_before_${policy}_graph_rep.index"
  fi
  if [ -f "${ACTIVE_PARTITION}" ]; then
    cp -a "${ACTIVE_PARTITION}" "${EXPERIMENT_ROOT}/layouts/active_before_${policy}_partition.bin"
  fi
  cp "${dir}/_part_tmp.index" "${ACTIVE_GRAPH_REP}"
  cp "${dir}/_part.bin" "${ACTIVE_PARTITION}"
  {
    echo "policy=${policy}"
    echo "layout_dir=${dir}"
    echo "active_graph_rep=${ACTIVE_GRAPH_REP}"
    echo "active_partition=${ACTIVE_PARTITION}"
    ls -lh "${dir}/_part_tmp.index" "${dir}/_part.bin" "${ACTIVE_GRAPH_REP}" "${ACTIVE_PARTITION}"
    sha256sum "${dir}/_part_tmp.index" "${dir}/_part.bin" "${ACTIVE_GRAPH_REP}" "${ACTIVE_PARTITION}"
  } > "${EXPERIMENT_ROOT}/layouts/${policy}_layout_manifest.txt"
}

run_layout() {
  local policy="$1"
  local score_file="$2"
  local layout_dir="$3"
  local env_args=(PACKING_POLICY="${policy}" REPLICA_LIMIT="${REPLICA_LIMIT}")
  if [ "${policy}" = "history" ]; then
    env_args+=(TRANSITION_SCORE_FILE="${score_file}")
  else
    env_args+=(TRANSITION_SCORE_FILE="")
  fi

  echo "Generating ${policy} graph-replicated layout..."
  env "${env_args[@]}" bash run_benchmark.sh "${BUILD_TYPE}" gr_layout 2>&1 | tee "${EXPERIMENT_ROOT}/logs/${policy}_gr_layout.driver.log"
  if [ -f "${layout_dir}/_part.bin.log" ]; then
    cp "${layout_dir}/_part.bin.log" "${EXPERIMENT_ROOT}/logs/${policy}_gr_layout.partitioner.log"
  fi
  if [ -f "${layout_dir}/relayout.log" ]; then
    cp "${layout_dir}/relayout.log" "${EXPERIMENT_ROOT}/logs/${policy}_gr_layout.relayout.log"
  fi
}

run_search() {
  local policy="$1"
  echo "Running search for ${policy} layout..."
  bash run_benchmark.sh "${BUILD_TYPE}" search knn 2>&1 | tee "${EXPERIMENT_ROOT}/logs/${policy}_search.driver.log"
  if [ -f "${SEARCH_LOG}" ]; then
    cp "${SEARCH_LOG}" "${EXPERIMENT_ROOT}/logs/${policy}_search.log"
  else
    echo "Expected search log not found: ${SEARCH_LOG}" >&2
    exit 1
  fi
  if [ "${COLLECT_TRANSITION_TRACE}" -eq 1 ]; then
    local trace_file="${DATA_DIR}/gorgeous/logs/search_trace.csv"
    if [ -f "${trace_file}" ]; then
      cp "${trace_file}" "${EXPERIMENT_ROOT}/logs/${policy}_search_trace.csv"
    else
      echo "Expected trace file not found: ${trace_file}" >&2
      exit 1
    fi
  fi
}

extract_metrics() {
  python3 - "${EXPERIMENT_ROOT}/logs/random_search.log" "${EXPERIMENT_ROOT}/logs/history_search.log" "${EXPERIMENT_ROOT}/comparison.csv" "${EXPERIMENT_ROOT}/comparison.md" <<'PY'
import csv
import re
import sys
from pathlib import Path

logs = [('random', Path(sys.argv[1])), ('history', Path(sys.argv[2]))]
csv_out = Path(sys.argv[3])
md_out = Path(sys.argv[4])
fields = ['policy', 'L', 'BW', 'QPS', 'Mean Latency', 'P999 Latency', 'Graph IO', 'Emb IO', 'Ext Cmp', 'PQ Cmp', 'Mem(MB)', 'Recall@10']

def parse(path):
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        parts = line.split()
        if len(parts) >= 18 and parts[0].isdigit() and parts[1].isdigit():
            rows.append({
                'L': parts[0],
                'BW': parts[1],
                'QPS': parts[2],
                'Mean Latency': parts[3],
                'P999 Latency': parts[4],
                'Graph IO': parts[5],
                'Emb IO': parts[6],
                'Ext Cmp': parts[7],
                'PQ Cmp': parts[8],
                'Mem(MB)': parts[16],
                'Recall@10': parts[17],
            })
    return rows

all_rows = []
by_policy = {}
for policy, path in logs:
    rows = parse(path)
    by_policy[policy] = rows
    for row in rows:
        out = {'policy': policy}
        out.update(row)
        all_rows.append(out)

with csv_out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_rows)

md = []
md.append('# History Packing Ablation\\n')
md.append('| ' + ' | '.join(fields) + ' |')
md.append('| ' + ' | '.join(['---'] + ['---:'] * (len(fields) - 1)) + ' |')
for row in all_rows:
    md.append('| ' + ' | '.join(str(row.get(k, '')) for k in fields) + ' |')

md.append('\\n## Judgement\\n')
random_rows = {r['L']: r for r in by_policy.get('random', [])}
history_rows = {r['L']: r for r in by_policy.get('history', [])}
for L in sorted(set(random_rows) & set(history_rows), key=lambda x: int(x)):
    r = random_rows[L]
    h = history_rows[L]
    def f(row, key): return float(row[key])
    md.append(f'### L={L}\\n')
    md.append(f'- Graph IO lowered: {f(h, "Graph IO") < f(r, "Graph IO")} ({r["Graph IO"]} -> {h["Graph IO"]})')
    md.append(f'- QPS improved: {f(h, "QPS") > f(r, "QPS")} ({r["QPS"]} -> {h["QPS"]})')
    md.append(f'- Mean latency lowered: {f(h, "Mean Latency") < f(r, "Mean Latency")} ({r["Mean Latency"]} -> {h["Mean Latency"]})')
    md.append(f'- P999 latency lowered: {f(h, "P999 Latency") < f(r, "P999 Latency")} ({r["P999 Latency"]} -> {h["P999 Latency"]})')
    md.append(f'- Recall@10 unchanged: {abs(f(h, "Recall@10") - f(r, "Recall@10")) < 0.01} ({r["Recall@10"]} -> {h["Recall@10"]})\\n')

md_out.write_text('\\n'.join(md) + '\\n')
PY
}

{
  echo "experiment_root=${EXPERIMENT_ROOT}"
  echo "dataset=${DATASET}"
  echo "transition_score_file=${TRANSITION_SCORE_FILE}"
  echo "replica_limit=${REPLICA_LIMIT}"
  echo "index_prefix=${INDEX_PREFIX_PATH}"
  echo "query_file=${QUERY_FILE}"
  echo "gt_file=${GT_FILE}"
  echo "K=${K}"
  echo "LS=${LS}"
  echo "BM_LIST=${BM_LIST[*]}"
  echo "T_LIST=${T_LIST[*]}"
  echo "CACHE=${CACHE}"
  echo "MEM_GRAPH_USE_RATIO=${MEM_GRAPH_USE_RATIO}"
  echo "MEM_EMB_USE_RATIO=${MEM_EMB_USE_RATIO}"
  echo "USE_PAGE_SEARCH=${USE_PAGE_SEARCH}"
  echo "PQ_FILTER_RATIO=${PQ_FILTER_RATIO}"
  echo "COLLECT_TRANSITION_TRACE=${COLLECT_TRANSITION_TRACE}"
  echo "COLLECT_TRACE_FOR_IO=${COLLECT_TRACE_FOR_IO}"
} | tee "${EXPERIMENT_ROOT}/experiment_config.txt"

save_existing_layout_dir "${RANDOM_LAYOUT_DIR}" random
save_existing_layout_dir "${HISTORY_LAYOUT_DIR}" history

run_layout random "" "${RANDOM_LAYOUT_DIR}"
activate_layout "${RANDOM_LAYOUT_DIR}" random
run_search random

run_layout history "${TRANSITION_SCORE_FILE}" "${HISTORY_LAYOUT_DIR}"
activate_layout "${HISTORY_LAYOUT_DIR}" history
run_search history

extract_metrics

echo "Experiment complete: ${EXPERIMENT_ROOT}"
echo "Comparison: ${EXPERIMENT_ROOT}/comparison.md"
