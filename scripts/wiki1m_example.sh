#!/bin/bash
set -e

# make sure you modify DATA_DIR in config_dataset.sh
source config_dataset.sh

CUR_PWD=$PWD
echo "Current scripts dir: ${CUR_PWD}"

cd ${DATA_DIR}

# Download Wiki1M data.
# Dataset: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
if [ ! -d "${DATA_DIR}/ann-wiki-1m" ]; then
  echo "【yqr】Installing git-lfs if needed..."
  sudo apt update
  sudo apt install -y git-lfs

  git lfs install

  echo "【yqr】Cloning Wiki1M dataset..."
  git clone https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
else
  echo "【yqr】Wiki1M dataset already exists."
fi

echo "【yqr】Download Wiki1M data done."

# Build Gorgeous code.
cd ${CUR_PWD}
# bash run_benchmark.sh release
echo "【yqr】Build code done."

# Wiki1M is already in .fbin/.ibin format.
# So no fvecs_to_bin conversion is needed.
# Expected files:
#   ${DATA_DIR}/ann-wiki-1m/base.1M.fbin
#   ${DATA_DIR}/ann-wiki-1m/query.public.100K.fbin
#   ${DATA_DIR}/ann-wiki-1m/groundtruth.public.100K.ibin

if [ ! -f "${DATA_DIR}/ann-wiki-1m/base.1M.fbin" ]; then
  echo "Missing base file: ${DATA_DIR}/ann-wiki-1m/base.1M.fbin"
  exit 1
fi

if [ ! -f "${DATA_DIR}/ann-wiki-1m/query.public.100K.fbin" ]; then
  echo "Missing query file: ${DATA_DIR}/ann-wiki-1m/query.public.100K.fbin"
  exit 1
fi

if [ ! -f "${DATA_DIR}/ann-wiki-1m/groundtruth.public.100K.ibin" ]; then
  echo "Missing groundtruth file: ${DATA_DIR}/ann-wiki-1m/groundtruth.public.100K.ibin"
  exit 1
fi

echo "【yqr】Wiki1M files check done."

# Optional: recompute groundtruth.
# Usually unnecessary because Wiki1M already provides groundtruth.public.100K.ibin.
# If you want to recompute it, uncomment the following command.
#
# ${DATA_DIR}/gorgeous/release/tests/utils/compute_groundtruth \
#   --data_type float \
#   --dist_fn l2 \
#   --base_file ${DATA_DIR}/ann-wiki-1m/base.1M.fbin \
#   --query_file ${DATA_DIR}/ann-wiki-1m/query.public.100K.fbin \
#   --gt_file ${DATA_DIR}/ann-wiki-1m/computed_gt_1000.bin \
#   --K 1000
#
# If recomputed, also update GT_FILE in config_dataset.sh:
#   GT_FILE=${DATA_DIR}/ann-wiki-1m/computed_gt_1000.bin

echo "【yqr】Groundtruth ready."

# Run Gorgeous pipeline.
cd ${CUR_PWD}

bash run_benchmark.sh release build
echo "【yqr】Build graph done."

bash run_benchmark.sh release build_mem
echo "【yqr】Build memory graph done."

# For Starling baseline only:
# bash run_benchmark.sh release gp

bash run_benchmark.sh release split_graph
echo "【yqr】Split graph done."

bash run_benchmark.sh release gr_layout
echo "【yqr】Graph replicated layout done."

bash run_benchmark.sh release search knn
echo "【yqr】Search done."
