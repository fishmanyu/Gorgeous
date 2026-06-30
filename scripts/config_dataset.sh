#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

# your path
DATA_DIR="/home/yqr/work/data"

dataset_sift_learn() {
  BASE_PATH=${DATA_DIR}/sift1m/sift_learn.fbin
  QUERY_FILE=${DATA_DIR}/sift1m/sift_query.fbin
  GT_FILE=${DATA_DIR}/sift1m/computed_gt_1000.bin
  PREFIX=sift_learn
  DATA_TYPE=float
  DIST_FN=l2
  K=10
  DATA_DIM=128
  DATA_N=100000
  SECTOR_LEN=4096     # default SECTOR_LEN for disk page (for build DiskANN graph)
  GR_SECTOR_LEN=4096  # the SECTOR_LEN for graph replicated layout
  N_PQ_CODE=4         # represent PQ dimension. 4 represents 4 dim compressed to one PQ byte
                      # Usually, for single-modal dataset, N_PQ_CODE should be 4 for optimal performance.
                      # For multi-modal dataset, N_PQ_CODE should be 2 for optimal performance.
}

dataset_wiki1m() {
  BASE_PATH=${DATA_DIR}/ann-wiki-1m/base.1M.fbin
  QUERY_FILE=${DATA_DIR}/ann-wiki-1m/query.public.100K.fbin
  GT_FILE=${DATA_DIR}/ann-wiki-1m/groundtruth.public.100K.ibin
  PREFIX=wiki1m
  DATA_TYPE=float
  DIST_FN=l2
  K=10
  DATA_DIM=256
  DATA_N=1000000
  SECTOR_LEN=4096
  GR_SECTOR_LEN=4096
  N_PQ_CODE=4
}