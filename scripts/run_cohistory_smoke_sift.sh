#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
OUT_ROOT="${OUT_ROOT:-/home/yqr/work/data/gorgeous/experiments/cohistory_smoke_sift_$(date +%Y%m%d_%H%M%S)}"
SCORE_FILE="${TRANSITION_SCORE_FILE:-/home/yqr/work/data/gorgeous/logs/transition_scores_later.tsv}"
CONFIG=config_local.sh
mkdir -p "${OUT_ROOT}/logs"
cp "${CONFIG}" "${OUT_ROOT}/config_local.before.sh"
restore() { cp "${OUT_ROOT}/config_local.before.sh" "${CONFIG}"; }
trap restore EXIT
python3 - <<'PY'
from pathlib import Path
import re
p=Path('config_local.sh')
s=p.read_text()
s=re.sub(r'(?m)^dataset_(sift_learn|wiki1m)\s*$', 'dataset_sift_learn', s)
s=re.sub(r'(?m)^COLLECT_TRANSITION_TRACE=.*$', 'COLLECT_TRANSITION_TRACE=0', s)
p.write_text(s)
PY
source ./config_local.sh
INDEX_PREFIX_PATH="${PREFIX}/M${M}_R${R}_L${BUILD_L}/"
SEARCH_LOG="${DATA_DIR}/gorgeous/${INDEX_PREFIX_PATH}search/search_K${K}_CACHE${CACHE}_BW${BM_LIST[0]}_T${T_LIST[0]}_MEML${MEM_L}_MEMK${MEM_TOPK:-}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log"
run_one() {
  local policy="$1"
  echo "=== ${policy} gr_layout ==="
  PACKING_POLICY="${policy}" TRANSITION_SCORE_FILE="${SCORE_FILE}" COHISTORY_WINDOW=2 COHISTORY_DUP_PENALTY=0.1 bash run_benchmark.sh release gr_layout 2>&1 | tee "${OUT_ROOT}/logs/${policy}_gr_layout.driver.log"
  echo "=== ${policy} search ==="
  bash run_benchmark.sh release search knn 2>&1 | tee "${OUT_ROOT}/logs/${policy}_search.driver.log"
  cp "${SEARCH_LOG}" "${OUT_ROOT}/logs/${policy}_search.log"
}
run_one random
run_one history
run_one cohistory
python3 - "${OUT_ROOT}" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
fields=['policy','L','BW','QPS','Mean Latency','P999 Latency','Graph IO','Emb IO','Recall@10']
rows=[]
for policy in ['random','history','cohistory']:
    path=root/'logs'/f'{policy}_search.log'
    for line in path.read_text(errors='replace').splitlines():
        p=line.split()
        if len(p)>=17 and p[0].isdigit() and p[1].isdigit():
            rows.append({'policy':policy,'L':p[0],'BW':p[1],'QPS':p[2],'Mean Latency':p[3],'P999 Latency':p[4],'Graph IO':p[5],'Emb IO':p[6],'Recall@10':p[17] if len(p)>17 else ''})
md=['# cohistory smoke test on SIFT\n','| '+' | '.join(fields)+' |','| '+' | '.join(['---']+['---:']*(len(fields)-1))+' |']
for r in rows:
    md.append('| '+' | '.join(r.get(f,'') for f in fields)+' |')
(root/'comparison.md').write_text('\n'.join(md)+'\n')
print(root/'comparison.md')
PY
