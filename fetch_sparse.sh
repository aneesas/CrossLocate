#!/bin/bash
set -u
BASE=https://cphoto.fit.vutbr.cz/crosslocate/dataset
ROOT=/home/aneesa/src/CrossLocate
cd "$ROOT" || exit 1

dl() { # url destdir
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$2/$(basename "$1")" "$1" \
    && echo "OK   $(basename "$1")" || echo "FAIL $1"
}

mkdir -p datasets/Sparse datasets_meta/Sparse pretrained_models/original_to_segments_swiss_13_5

echo "--- meta structures ---"
for s in train val test; do
  dl "$BASE/meta_structures/sparse_dataset/GeoPose3K_v2_original_to_segments_swiss_${s}.npy" datasets_meta/Sparse
done

echo "--- pretrained checkpoint ---"
M=GeoImRet_origToSegmentsSwiss_res500_negs5_qReuse3-model-213
for f in "$M.data-00000-of-00001" "$M.index" "$M.meta" checkpoint \
         13_5_train_origToSegmentsSwiss_res500_negs5_qReuse3.py; do
  dl "$BASE/models/original_to_segments_swiss_13_5/$f" pretrained_models/original_to_segments_swiss_13_5
done

echo "--- dataset tarballs ---"
dl "$BASE/datasets/sparse_dataset/sparse_queries/query_original_result.tar.gz" datasets/Sparse
dl "$BASE/datasets/sparse_dataset/sparse_database/database_segments.tar.gz"    datasets/Sparse

echo "--- extracting ---"
for t in query_original_result database_segments; do
  echo "extracting $t"
  tar -xzf "datasets/Sparse/$t.tar.gz" -C datasets/Sparse && echo "OK   extract $t" || echo "FAIL extract $t"
done

echo "=== DONE ==="
