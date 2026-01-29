#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedProx}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fl-method)
      FL_METHOD="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done
# 支持多个 FL 方法（逗号分隔），默认只有一个
IFS=',' read -ra FL_METHODS <<< "${FL_METHOD}"
LAUNCHER=(python)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
echo "Single-GPU mode; using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 是否运行+保存所有 pre 评测，默认关闭以节省时间
ENABLE_PRE_EVAL="${ENABLE_PRE_EVAL:-false}"

########################################
MODELS=("Llama-3.2-3B-Instruct-lora")

FORGET_SPLITS=("forget10")
HOLDOUT_SPLITS=("holdout10")
RETAIN_SPLITS=("retain90")
USE_CLIENT_FORGET_EVAL=true  # 是否在 forget_utility 中使用目标客户端的 forget 子集
ASSIGN_FORGET_TO_REVOKER_ONLY=true   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/TOFU/fedperm_only"
  OUT_ROOT="saves/TOFU/fedperm_only"
else
  LOG_ROOT="logs/TOFU/fedperm"
  OUT_ROOT="saves/TOFU/fedperm"
fi

REVOKER_CLIENT_IDX=0

GLOBAL_ROUNDS=20
GLOBAL_ROUNDS=20
SAVE_INTERVAL=0              # 0 表示每轮保存；1 表示间隔一轮保存（1,3,5...），以此类推

SEED=42

CONFIG_TRAIN="train-lora.yaml"
CONFIG_EVAL="eval/tofu/default.yaml"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"
########################################
# 工具函数
########################################

run_eval() {
  run_eval_common "$1" "$2" "${FORGET_IDX_PATH:-}"
}

eval_rounds() {
  local base_dir="$1"
  local prefix="$2"
  LAST_ROUND=${GLOBAL_ROUNDS}
  for ((r=1; r<=GLOBAL_ROUNDS; r++)); do
    CURRENT_ROUND=${r}
    local post="${base_dir}/round_${r}"
    if [ -d "${post}" ]; then
      local post_summary_paths=(
        "${post}/eval/${prefix}_round${r}_post_agg/TOFU_SUMMARY.json"
        "${base_dir}/eval_rounds/round_${r}/server_post_agg/TOFU_SUMMARY.json"
      )
      maybe_run_eval "${post}" "${prefix}_round${r}_post_agg" \
        "${LOG_ROOT}/${prefix}_round${r}_post_agg.txt" \
        "${post_summary_paths[@]}"
    else
      echo "⚠️ 缺少聚合后检查点：${post}"
    fi
    local pre_round=$((r-1))
    if [ ${pre_round} -ge 1 ]; then
      if [ "${ENABLE_PRE_EVAL}" != "true" ]; then
        if [ -z "${PRE_EVAL_SKIP_LOGGED:-}" ]; then
          echo "Skip all pre evals because ENABLE_PRE_EVAL=${ENABLE_PRE_EVAL}"
          PRE_EVAL_SKIP_LOGGED=true
        fi
        continue
      fi
      local pre="${base_dir}/round_${pre_round}"
      if [ -d "${pre}" ]; then
        local pre_summary_paths=(
          "${pre}/eval/${prefix}_round${r}_pre_agg/TOFU_SUMMARY.json"
          "${base_dir}/eval_rounds/round_${r}/server_pre_agg/TOFU_SUMMARY.json"
        )
        maybe_run_eval "${pre}" "${prefix}_round${r}_pre_agg" \
          "${LOG_ROOT}/${prefix}_round${r}_pre_agg.txt" \
          "${pre_summary_paths[@]}"
      fi
    fi
  done
}

sync_client_splits() {
  sync_client_splits_retain "$@"
}

########################################
# retain
########################################

for CURRENT_FL_METHOD in "${FL_METHODS[@]}"; do
  echo "===== FL_METHOD=${CURRENT_FL_METHOD} ====="

  for MODEL_CFG in "${MODELS[@]}"; do
    echo "=========== MODEL=${MODEL_CFG} ==========="

    for idx in "${!FORGET_SPLITS[@]}"; do
      FORGET_SPLIT="${FORGET_SPLITS[$idx]}"
      HOLDOUT_SPLIT="${HOLDOUT_SPLITS[$idx]}"
      RETAIN_SPLIT="${RETAIN_SPLITS[$idx]}"

      # 唯一的客户端划分目录，按 ASSIGN_FORGET_TO_REVOKER_ONLY 区分：fedperm / fedperm_only
      COMMON_SPLIT_DIR="${OUT_ROOT}/client_splits/${RETAIN_SPLIT}_${FORGET_SPLIT}_seed${SEED}"
      mkdir -p "${COMMON_SPLIT_DIR}"
      export FEDPERM_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      RETAIN_IDX_FILE="${COMMON_SPLIT_DIR}/global_retain_indices.json"
      FORGET_IDX_FILE="${COMMON_SPLIT_DIR}/global_forget_indices.json"
      SPLITS_READY=false
     if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
       SPLITS_READY=true
       echo "Found existing global indices, reuse: ${COMMON_SPLIT_DIR}"
     fi

      echo ">>> SPLITS retain=${RETAIN_SPLIT} forget=${FORGET_SPLIT} holdout=${HOLDOUT_SPLIT}"

      ############################
      # 1) retain-only 训练
      ############################
      RETAIN_TASK="retain_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
      RETAIN_DIR="${OUT_ROOT}/${RETAIN_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      FORGET_IDX_PATH="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
      mkdir -p "${RETAIN_DIR}"

      if training_complete "${RETAIN_DIR}" "${GLOBAL_ROUNDS}"; then
        echo "Skip retain-only training, final round exists: ${RETAIN_DIR}/round_${GLOBAL_ROUNDS}"
      else
        EXTRA_EVAL_ARGS=()
        SAVE_SPLITS_FOR_EVAL=true
        if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
          EXTRA_EVAL_ARGS+=(
            eval.tofu.forget_indices_path="${FORGET_IDX_PATH}"
            eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
          )
        fi
        # 若公共划分已存在，直接复用，不再生成新划分
        if [ "${SPLITS_READY}" = true ]; then
          SAVE_SPLITS_FOR_EVAL=false
          EXTRA_EVAL_ARGS+=(
            +trainer.method_args.retain_indices_path="${RETAIN_IDX_FILE}"
            +trainer.method_args.forget_indices_path="${FORGET_IDX_FILE}"
            trainer.method_args.save_client_splits_for_eval=false
          )
        fi
        # 计算本阶段跳过的评测轮次（传递给 Trainer）
        SKIP_ROUNDS=$(compute_skip_rounds "${GLOBAL_ROUNDS}" "${SAVE_INTERVAL}")
        FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
        FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
      "${LAUNCHER[@]}" src/federated_train.py \
        --config-name="${CONFIG_TRAIN}" \
        experiment=finetune/tofu/default \
        eval=tofu \
        model="${MODEL_CFG}" \
        task_name="${RETAIN_TASK}" \
        trainer=FederatedFinetune \
          data/datasets@data.train=TOFU_QA_full \
          data.train.TOFU_QA_full.args.hf_args.name=full \
          trainer.method_args.train_on="retain" \
          trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
          trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
          trainer.method_args.aggregation_strategy="${CURRENT_FL_METHOD}" \
          trainer.method_args.aggregation_strategy="${CURRENT_FL_METHOD}" \
          trainer.method_args.global_rounds="${GLOBAL_ROUNDS}" \
          trainer.method_args.save_client_splits_for_eval="${SAVE_SPLITS_FOR_EVAL}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          trainer.method_args.revoker_client_idx="${REVOKER_CLIENT_IDX}" \
          trainer.args.output_dir="${RETAIN_DIR}" \
          trainer.args.save_strategy="epoch" \
          paths.output_dir="${RETAIN_DIR}" \
          trainer.args.seed="${SEED}" \
          "${EXTRA_EVAL_ARGS[@]}" \
          2>&1 | tee "${LOG_ROOT}/${RETAIN_TASK}.txt"
        if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
          SPLITS_READY=true
        fi
      fi

      if [ "${SPLITS_READY}" != true ]; then
        echo "缺少公共划分文件：${COMMON_SPLIT_DIR}，请确保已生成全局索引后再继续。" >&2
        exit 1
      fi

      # 根据保存频次裁剪 checkpoint
      prune_rounds "${RETAIN_DIR}" "${GLOBAL_ROUNDS}" "${SAVE_INTERVAL}"

      eval_rounds "${RETAIN_DIR}" "${RETAIN_TASK}"

    done

  done

done

echo "全部完成。输出：${OUT_ROOT}，日志：${LOG_ROOT}"
