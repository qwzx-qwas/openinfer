#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/accuracy/run_qwen35_gdn_stage7.sh ARTIFACT_DIR MODEL_DIR [LOG_DIR]

Arguments:
  ARTIFACT_DIR  Bundle containing qwen35_4b_candidate/ and operator_hv48/
  MODEL_DIR     Absolute path to Qwen3.5-4B weights
  LOG_DIR       Optional evidence directory (default: docs/private/gdn-stage7/logs)

Run this inside docker/dev.sh shell on one RTX 5090 / SM120 GPU.  The script
validates the pinned bundle, records the environment, then stops at the first
failed host, Triton, Hv32, Hv48, or model-diagnostic gate.
EOF
}

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="$(realpath "$1")"
model_dir="$(realpath "$2")"
log_dir="${3:-$repo_root/docs/private/gdn-stage7/logs}"
mkdir -p "$log_dir"
log_dir="$(realpath "$log_dir")"

case "$model_dir" in
  /*) ;;
  *) echo "MODEL_DIR must resolve to an absolute path" >&2; exit 2 ;;
esac

candidate_manifest="$artifact_dir/qwen35_4b_candidate/manifest.json"
hv48_manifest="$artifact_dir/operator_hv48/manifest.json"
for required in \
  "$artifact_dir/bundle.json" \
  "$candidate_manifest" \
  "$artifact_dir/qwen35_4b_candidate/kernel.ptx" \
  "$hv48_manifest" \
  "$artifact_dir/operator_hv48/kernel.ptx" \
  "$model_dir/config.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Stage 7 input: $required" >&2
    exit 2
  fi
done

cd "$repo_root"

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$log_dir/$name.log"
}

date -Iseconds | tee "$log_dir/date.txt"
run_logged nvidia-smi nvidia-smi
gpu_compute_cap="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | sed -n '1p' \
    | tr -d '[:space:]'
)"
printf '%s\n' "$gpu_compute_cap" | tee "$log_dir/compute-capability.txt"
if [[ "$gpu_compute_cap" != "12.0" ]]; then
  echo "Stage 7 requires device 0 to be SM120 / compute capability 12.0; got $gpu_compute_cap" >&2
  exit 1
fi
run_logged nvcc nvcc --version
run_logged rustc rustc -Vv
run_logged openinfer-commit git rev-parse HEAD
run_logged worktree git status --short
run_logged submodules git submodule status --recursive
run_logged artifact-sha256 sha256sum \
  "$artifact_dir/qwen35_4b_candidate/kernel.ptx" \
  "$artifact_dir/operator_hv48/kernel.ptx"
cp "$artifact_dir/bundle.json" "$log_dir/bundle.json"
cp "$candidate_manifest" "$log_dir/qwen35_4b_candidate-manifest.json"
cp "$hv48_manifest" "$log_dir/operator_hv48-manifest.json"

run_logged artifact-validation \
  python3 openinfer-kernels/tools/flashinfer_gdn/artifact_contract.py \
  validate-bundle "$artifact_dir" \
  --flashinfer-dir openinfer-kernels/third_party/flashinfer

export OPENINFER_CUDA_SM=120
export OPENINFER_TEST_MODEL_PATH="$model_dir"

for host_filter in \
  gdn_prefill_test_contract \
  gdn_prepare_test_contract \
  gdn_stage6_test_contract \
  gdn_stage7_test_support \
  flashinfer_gdn::tests; do
  log_name="host-${host_filter//:/-}"
  run_logged "$log_name" \
    cargo test --release -p openinfer-qwen35 --features qwen35 --lib \
    "$host_filter" -- --nocapture
done

run_logged triton-stepwise-baseline \
  cargo test --release -p openinfer-qwen35 --features qwen35 --lib \
  recurrent::tests::gdn_chunkwise_prefill_matches_stepwise_decode_at_48_value_heads \
  -- --exact --nocapture

run_logged operator-hv32 \
  env OPENINFER_GDN_STAGE3_MANIFEST="$candidate_manifest" \
  cargo test --release -p openinfer-qwen35 --features qwen35 --lib \
  flashinfer_gdn::tests::sm120_launch_smoke_covers_alias_separate_and_dynamic_t \
  -- --ignored --exact --nocapture

run_logged operator-hv48 \
  env OPENINFER_GDN_STAGE3_MANIFEST="$hv48_manifest" \
  cargo test --release -p openinfer-qwen35 --features qwen35 --lib \
  flashinfer_gdn::tests::sm120_launch_smoke_covers_alias_separate_and_dynamic_t \
  -- --ignored --exact --nocapture

# This final test reports full-model hidden/recurrent/conv max-abs values.  It
# is a Stage 7 diagnostic only; Stage 8's unchanged HF fixtures remain the
# authority for model-level acceptance.
run_logged full-model-diagnostic \
  env OPENINFER_GDN_STAGE3_MANIFEST="$candidate_manifest" \
  cargo test --release -p openinfer-qwen35 --features qwen35 \
  --test gdn_prefill_candidate \
  full_model_seam_compares_named_backends_at_boundary_lengths \
  -- --ignored --exact --nocapture

echo "Stage 7 harness completed; evidence: $log_dir"
