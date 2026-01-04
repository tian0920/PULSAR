#!/usr/bin/env bash
# Shared helpers for FedPerm shell entrypoints. Logic is kept identical to the
# previous inline implementations to avoid behaviour changes.

# Compute rounds to skip based on a save interval (0 = save every round).
compute_skip_rounds() {
  local total="$1"
  local interval="$2"
  local skip=""
  if [ "${interval}" -le 0 ]; then
    echo ""
    return
  fi
  local gap=$((interval + 1))
  for ((r=1; r<=total; r++)); do
    if [ "${r}" -eq "${total}" ]; then
      continue  # 最后一轮不跳过
    fi
    if [ $(((r - 1) % gap)) -ne 0 ]; then
      skip+="$((r-1)),"
    fi
  done
  echo "${skip%,}"
}

# Decide whether to run eval for a checkpoint; respects SAVE_INTERVAL and LAST_ROUND.
maybe_run_eval() {
  local ckpt_dir="$1"
  local stage="$2"
  local log_file="$3"
  shift 3
  local summary_paths=("$@")

  if [ "${CURRENT_ROUND:-0}" -gt 0 ] && [ "${SAVE_INTERVAL:-0}" -gt 0 ]; then
    local gap=$((SAVE_INTERVAL + 1))
    if [ "${CURRENT_ROUND}" -ne "${LAST_ROUND}" ] && [ $(((CURRENT_ROUND - 1) % gap)) -ne 0 ]; then
      echo "Skip eval ${stage} at round ${CURRENT_ROUND} due to SAVE_INTERVAL=${SAVE_INTERVAL}"
      return
    fi
  fi

  for summary_file in "${summary_paths[@]}"; do
    if [ -s "${summary_file}" ]; then
      # treat "{}" or empty summary as missing to avoid skipping real eval
      if [ "$(wc -c <"${summary_file}")" -le 3 ]; then
        echo "Found placeholder summary (<=3 bytes), will re-evaluate: ${summary_file}"
        continue
      fi
      echo "Skip eval ${stage}, summary already exists: ${summary_file}"
      return
    fi
  done

  run_eval "${ckpt_dir}" "${stage}" 2>&1 | tee "${log_file}"
}

# Check whether the final round directory exists and is non-empty.
training_complete() {
  local dir="$1"
  local final_round="$2"
  local final_dir="${dir}/round_${final_round}"
  if [ -d "${final_dir}" ] && [ -n "$(ls -A "${final_dir}")" ]; then
    return 0
  fi
  return 1
}

# Drop intermediate checkpoints according to SAVE_INTERVAL rules.
prune_rounds() {
  local base_dir="$1"
  local total_rounds="$2"
  local interval="$3"
  if [ "${interval}" -le 0 ]; then
    return
  fi
  local gap=$((interval + 1))
  for ((r=1; r<=total_rounds; r++)); do
    if [ "${r}" -eq "${total_rounds}" ]; then
      continue  # 保留最后一轮
    fi
    if [ $(((r - 1) % gap)) -ne 0 ]; then
      local dir="${base_dir}/round_${r}"
      if [ -d "${dir}" ]; then
        echo "Prune checkpoint ${dir} (SAVE_INTERVAL=${interval})"
        rm -rf "${dir}"
      fi
    fi
  done
}

# Run eval with optional forget indices injection.
run_eval_common() {
  local ckpt_dir="$1"
  local stage="$2"
  local forget_idx_override="${3:-}"
  local out_dir="${ckpt_dir}/eval/${stage}"
  local extra_args=()
  # 可选：强制切换 eval 组（例如 muse）
  if [ -n "${EVAL_GROUP:-}" ]; then
    extra_args+=(eval="${EVAL_GROUP}")
  fi
  if [ "${USE_CLIENT_FORGET_EVAL:-false}" = "true" ]; then
    local forget_idx_path="${forget_idx_override}"
    if [ -z "${forget_idx_path}" ] && [ -n "${COMMON_SPLIT_DIR:-}" ] && [ -n "${REVOKER_CLIENT_IDX:-}" ]; then
      forget_idx_path="${COMMON_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
    fi
    if [ -n "${forget_idx_path}" ] && [ -f "${forget_idx_path}" ]; then
      extra_args+=(
        ++eval.tofu.forget_indices_path="${forget_idx_path}"
        ++eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
      )
    fi
  fi
  mkdir -p "${out_dir}"
  python src/eval.py \
    experiment="${CONFIG_EVAL}" \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    task_name="${stage}" \
    model.model_args.pretrained_model_name_or_path="${ckpt_dir}" \
    paths.output_dir="${out_dir}" \
    hydra.run.dir="${out_dir}" \
    "${extra_args[@]}"
}

# Client split sync (full script variant: baseline may be empty, current must exist).
sync_client_splits_full() {
  local src_dir="$1"
  if [ ! -d "${src_dir}" ]; then
    echo "⚠️ 未找到客户端划分目录：${src_dir}"
    return
  fi
  if [ -z "$(ls -A "${COMMON_SPLIT_DIR}")" ]; then
    echo "初始化公共 client_splits 到 ${COMMON_SPLIT_DIR}"
    if [ "${src_dir}" != "${COMMON_SPLIT_DIR}" ]; then
      cp -r "${src_dir}/." "${COMMON_SPLIT_DIR}/"
    else
      echo "源目录与目标目录相同且当前为空，请确认上游步骤已生成划分"
    fi
  fi
  python - "$COMMON_SPLIT_DIR" "$src_dir" <<'PY'
import json, os, sys, shutil
base, current = sys.argv[1], sys.argv[2]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

for fname in os.listdir(current):
    cpath = os.path.join(current, fname)
    bpath = os.path.join(base, fname)
    if os.path.isfile(cpath):
        if not os.path.exists(bpath):
            shutil.copy2(cpath, bpath)
            continue
        if fname.endswith(".json") and load_json(bpath) != load_json(cpath):
            print(f"Mismatch in {fname}: overwrite current with baseline")
            shutil.copy2(bpath, cpath)

# 生成/更新全局索引
retain_all, forget_all = [], []
def add_indices(payload, fname):
    if isinstance(payload, dict):
        retain_all.extend(payload.get("retain_indices", []))
        forget_all.extend(payload.get("forget_indices", []))
        return
    if isinstance(payload, list):
        # Historical files store plain lists for retain/forget splits
        if "retain" in fname:
            retain_all.extend(payload)
        elif "forget" in fname:
            forget_all.extend(payload)
        else:
            retain_all.extend(payload)
        return
    raise SystemExit(f"Unexpected payload type for {fname}: {type(payload).__name__}")

for fname in os.listdir(base):
    if fname.startswith("client") and fname.endswith("_indices.json"):
        payload = load_json(os.path.join(base, fname))
        add_indices(payload, fname)
retain_all = sorted(set(retain_all))
forget_all = sorted(set(forget_all))
with open(os.path.join(base, "global_retain_indices.json"), "w") as f:
    json.dump(retain_all, f)
with open(os.path.join(base, "global_forget_indices.json"), "w") as f:
    json.dump(forget_all, f)
PY
}

# Client split sync (retain script variant: baseline must contain all current files).
sync_client_splits_retain() {
  local src_dir="$1"
  if [ ! -d "${src_dir}" ]; then
    echo "⚠️ 未找到客户端划分目录：${src_dir}"
    return
  fi
  if [ -z "$(ls -A "${COMMON_SPLIT_DIR}")" ]; then
    echo "初始化公共 client_splits 到 ${COMMON_SPLIT_DIR}"
    cp -r "${src_dir}/." "${COMMON_SPLIT_DIR}/"
  fi
  python - "$COMMON_SPLIT_DIR" "$src_dir" <<'PY'
import json, os, sys, shutil
base, current = sys.argv[1], sys.argv[2]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

for fname in os.listdir(base):
    bpath = os.path.join(base, fname)
    cpath = os.path.join(current, fname)
    if not os.path.exists(cpath):
        shutil.copy2(bpath, cpath)
        continue
    if fname.endswith(".json") and load_json(bpath) != load_json(cpath):
        print(f"Mismatch in {fname}: overwrite current with baseline")
        shutil.copy2(bpath, cpath)
print("client_splits match baseline.")

# 生成/更新全局索引文件
retain_all, forget_all = [], []
def add_indices(payload, fname):
    if isinstance(payload, dict):
        retain_all.extend(payload.get("retain_indices", []))
        forget_all.extend(payload.get("forget_indices", []))
        return
    if isinstance(payload, list):
        if "retain" in fname:
            retain_all.extend(payload)
        elif "forget" in fname:
            forget_all.extend(payload)
        else:
            retain_all.extend(payload)
        return
    raise SystemExit(f"Unexpected payload type for {fname}: {type(payload).__name__}")

for fname in os.listdir(base):
    if fname.startswith("client") and fname.endswith("_indices.json"):
        payload = load_json(os.path.join(base, fname))
        add_indices(payload, fname)
retain_all = sorted(set(retain_all))
forget_all = sorted(set(forget_all))
with open(os.path.join(base, "global_retain_indices.json"), "w") as f:
    json.dump(retain_all, f)
with open(os.path.join(base, "global_forget_indices.json"), "w") as f:
    json.dump(forget_all, f)
PY
}
