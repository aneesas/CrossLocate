#!/bin/bash
set -u
BASE=https://cphoto.fit.vutbr.cz/crosslocate/dataset
ROOT=/home/aneesa/src/CrossLocate
cd "$ROOT" || exit 1
dl() { curl -fsSL --retry 5 --retry-delay 5 -C - -o "$2/$(basename "$1")" "$1" \
       && echo "OK   $(basename "$1")" || echo "FAIL $1"; }

mkdir -p pretrained_models/original_to_depth_swiss_15_3

echo "--- meta ---"
for s in train val test; do
  dl "$BASE/meta_structures/sparse_dataset/GeoPose3K_v2_original_to_depth_swiss_${s}.npy" datasets_meta/Sparse
done

echo "--- checkpoint ---"
M=GeoImRet_origToDepthSwiss_res500_negs5_qReuse3-model-164
for f in "$M.data-00000-of-00001" "$M.index" "$M.meta" \
         15_1_train_origToDepthSwiss_res500_negs5_qReuse3.py; do
  dl "$BASE/models/original_to_depth_swiss_15_3/$f" pretrained_models/original_to_depth_swiss_15_3
done

echo "--- database_depth (6G) ---"
dl "$BASE/datasets/sparse_dataset/sparse_database/database_depth.tar.gz" datasets/Sparse
echo "--- extracting ---"
tar -xzf datasets/Sparse/database_depth.tar.gz -C datasets/Sparse \
  && echo "OK   extract database_depth" || echo "FAIL extract database_depth"
echo "=== DONE ==="
