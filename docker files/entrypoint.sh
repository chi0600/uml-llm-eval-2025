#!/usr/bin/env bash
# Ensure Triton’s autotune cache dir exists (after any mounts)
mkdir -p "$TRITON_CACHE_DIR" && chmod -R 777 "$TRITON_CACHE_DIR"
set -e

for script in evaluate_fewshots_feature4.py \
              evaluate_fewshots_feature5.py \
              evaluate_fewshots_feature3.py; do
  echo "▶ Running $script"
  python /workspace/$script
done
