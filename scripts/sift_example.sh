# make sure you modify the DATA_DIR in config_dataset.sh
source config_dataset.sh
CUR_PWD=$PWD
echo "$PWD"

cd ${DATA_DIR}
# download data.
# wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz FTP 被网络/NAT/防火墙干扰下载失败了，http和curl下载都失败了，换成从镜像下载
# tar -xf sift.tar.gz
# 以下内容只执行一次
# sudo apt install -y git-lfs
# git lfs install
# git clone https://huggingface.co/datasets/qbo-odp/sift1m
cd sift1m
echo "【yqr】Download sift1m data done."

# make sure you have already built the code, here are sample steps:
cd $CUR_PWD
bash run_benchmark.sh release
echo "【yqr】Build code done."
${DATA_DIR}/gorgeous/release/tests/utils/fvecs_to_bin ${DATA_DIR}/sift1m/sift_query.fvecs ${DATA_DIR}/sift1m/sift_query.fbin
${DATA_DIR}/gorgeous/release/tests/utils/fvecs_to_bin ${DATA_DIR}/sift1m/sift_learn.fvecs ${DATA_DIR}/sift1m/sift_learn.fbin
${DATA_DIR}/gorgeous/release/tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${DATA_DIR}/sift1m/sift_learn.fbin --query_file  
${DATA_DIR}/sift1m/sift_query.fbin --gt_file ${DATA_DIR}/sift1m/computed_gt_1000.bin --K 1000
echo "【yqr】Compute ground truth done."

# To test the code: (just using the default config_local.sh is enough)
cd $CUR_PWD
bash run_benchmark.sh release build
echo "【yqr】Build graph done."
bash run_benchmark.sh release build_mem
echo "【yqr】Build memory graph done."
# bash run_benchmark.sh release gp      # for test Starling only
bash run_benchmark.sh release split_graph
echo "【yqr】Split graph done."
bash run_benchmark.sh release gr_layout
echo "【yqr】Graph replicated layout done."
bash run_benchmark.sh release search knn
echo "【yqr】Search done."
