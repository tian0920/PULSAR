#!/usr/bin/env bash
# FedPerm pipeline for MUSE (full -> unlearn). Retain-only阶段可选，默认不跑。
# Splits assumed:
#   retain: retain1
#   forget: forget
#   holdout: holdout1
# 如需更换 split，修改下方变量即可。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAvg}"
DEFAULT_UNLEARN_CLS=("GradAscent" "GradDiff" "NPO" "SimNPO")
UNLEARN_CLS=("${DEFAULT_UNLEARN_CLS[@]}")
LAUNCHER=(python)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Single-GPU mode; using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fl-method)
      FL_METHOD="$2"
      shift 2
      ;;
    --unlearn-cls)
      IFS=',' read -ra UNLEARN_CLS <<< "$2"
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
IFS=',' read -ra FL_METHODS <<< "${FL_METHOD}"

########################################
# 配置
########################################
MODEL_CFG="Llama-3.2-1B-Instruct-lora"
RETAIN_SPLITS=("retain1")
FORGET_SPLITS=("forget")
DATASET_HANDLER="pretrain"
DATASET_PATH="muse-bench/MUSE-News"
DATASET_NAME="train"
TEXT_KEY="text"


ASSIGN_FORGET_TO_REVOKER_ONLY=true   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/MUSE/fedperm_only"
  OUT_ROOT="saves/MUSE/fedperm_only"
else
  LOG_ROOT="logs/MUSE/fedperm"
  OUT_ROOT="saves/MUSE/fedperm"
fi

# 强制评测使用 muse 组，避免 eval=tofu 时注入失败
EVAL_GROUP="muse"

ENABLE_PERMUTATION=false   # 神经元转置

FULL_GLOBAL_ROUNDS=5     # full阶段的通信轮次
FULL_SAVE_INTERVAL=0

UNLEARN_GLOBAL_ROUNDS=3  # unlearn + 恢复的总轮次
UNLEARN_SAVE_INTERVAL=0
NORMAL_BEFORE_FORGET=0
FORGET_ROUNDS=2
RECOVERY_START=2

UNLEARN_EPOCHS=1         # filter 训练次数
FILTER_ALPHA=1           # θ_g^t-(θ_g^{t-1}-alpha*θ_unlearn)

TARGET_PREV_CE_ENABLED=true   # 目标客户端本地训练双CE开关
TARGET_PREV_CE_WEIGHT=1.0     # θ_unlearn的微调项系数
SEED=42

CONFIG_TRAIN="train-lora.yaml"
CONFIG_EVAL="eval/muse/default"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

########################################
# 工具
########################################
run_eval() {
  run_eval_common "$@"
}

########################################
# 主流程：外层 FL 方法，unlearn 阶段内层方法循环
########################################

for CURRENT_FL_METHOD in "${FL_METHODS[@]}"; do
  echo "===== FL_METHOD=${CURRENT_FL_METHOD} ====="

  for idx in "${!RETAIN_SPLITS[@]}"; do
    RETAIN_SPLIT="${RETAIN_SPLITS[$idx]}"
    FORGET_SPLIT="${FORGET_SPLITS[$idx]}"
    echo ">>> SPLITS retain=${RETAIN_SPLIT} forget=${FORGET_SPLIT}"

    # 每组 split 各自维护 client_splits，避免相互污染
    COMMON_SPLIT_DIR="saves/client_splits/${RETAIN_SPLIT}_${FORGET_SPLIT}_seed${SEED}"
    mkdir -p "${COMMON_SPLIT_DIR}"
    export FEDPERM_SPLIT_DIR="${COMMON_SPLIT_DIR}"
    RETAIN_IDX_FILE="${COMMON_SPLIT_DIR}/global_retain_indices.json"
    FORGET_IDX_FILE="${COMMON_SPLIT_DIR}/global_forget_indices.json"

    ########################################
    # 阶段 1：full 联邦训练（MUSE）
    ########################################
    FULL_TASK="full_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
    FULL_DIR="${OUT_ROOT}/${FULL_TASK}"
    CLIENT_SPLIT_DIR="${FULL_DIR}/client_splits"
    mkdir -p "${FULL_DIR}"

    if training_complete "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}"; then
      echo "Skip full training, final round exists: ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}"
    else
      EXTRA_EVAL_ARGS=()
      if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
        EXTRA_EVAL_ARGS+=(
          +trainer.method_args.retain_indices_path="${RETAIN_IDX_FILE}"
          +trainer.method_args.forget_indices_path="${FORGET_IDX_FILE}"
          trainer.method_args.save_client_splits_for_eval=false
        )
      fi
      SKIP_ROUNDS=$(compute_skip_rounds "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}")
      FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
      FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
      "${LAUNCHER[@]}" src/federated_train.py \
        --config-name="${CONFIG_TRAIN}" \
        experiment=finetune/tofu/default \
        eval=muse \
        model="${MODEL_CFG}" \
        task_name="${FULL_TASK}" \
        trainer=FederatedPermTrainer \
        trainer.method_args.enable_revocation=false \
        trainer.method_args.global_rounds="${FULL_GLOBAL_ROUNDS}" \
        trainer.method_args.train_on="full" \
        trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
        trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
        trainer.method_args.save_client_splits_for_eval=true \
        trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
        +trainer.method_args.retain_dataset_handler="${DATASET_HANDLER}" \
        +trainer.method_args.retain_dataset_hf_args.path="${DATASET_PATH}" \
        +trainer.method_args.retain_dataset_hf_args.name="${DATASET_NAME}" \
        +trainer.method_args.retain_dataset_hf_args.split="${RETAIN_SPLIT}" \
        +trainer.method_args.retain_text_key="${TEXT_KEY}" \
        +trainer.method_args.forget_dataset_handler="${DATASET_HANDLER}" \
        +trainer.method_args.forget_dataset_hf_args.path="${DATASET_PATH}" \
        +trainer.method_args.forget_dataset_hf_args.name="${DATASET_NAME}" \
        +trainer.method_args.forget_dataset_hf_args.split="${FORGET_SPLIT}" \
        +trainer.method_args.forget_text_key="${TEXT_KEY}" \
        data/datasets@data.train=MUSE_retain \
        data.train.MUSE_retain.args.hf_args.name="${DATASET_NAME}" \
        data.train.MUSE_retain.args.hf_args.split="${RETAIN_SPLIT}" \
        trainer.args.output_dir="${FULL_DIR}" \
        trainer.args.save_strategy="epoch" \
        paths.output_dir="${FULL_DIR}" \
        trainer.args.seed="${SEED}" \
        "${EXTRA_EVAL_ARGS[@]}" \
        2>&1 | tee "${LOG_ROOT}/${FULL_TASK}.txt"
    fi

    # 保存 filter 基线（full 阶段最后一轮 server pre）
    FILTER_BASE_DIR=""
    PRE_BASE_ROUND=$((FULL_GLOBAL_ROUNDS - 1))
    if [ ${PRE_BASE_ROUND} -ge 1 ]; then
      PRE_BASE_SRC="${FULL_DIR}/round_${PRE_BASE_ROUND}"
      FILTER_BASE_DIR="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}_server_pre_base"
      if [ -d "${PRE_BASE_SRC}" ] && [ ! -d "${FILTER_BASE_DIR}" ]; then
        echo "Preserve filter base from ${PRE_BASE_SRC} to ${FILTER_BASE_DIR}"
        cp -r "${PRE_BASE_SRC}" "${FILTER_BASE_DIR}"
      fi
    fi

    prune_rounds "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"
    eval_final_round "${FULL_DIR}" "${FULL_TASK}"
    sync_client_splits "${FULL_DIR}"

    ########################################
    # 阶段 2：基于 full 模型的 FedPerm 遗忘
    ########################################
    for UNLEARN_METHOD in "${UNLEARN_CLS[@]}"; do
      echo "------- UNLEARN_METHOD=${UNLEARN_METHOD} -------"
      FEDPERM_TASK="unlearn_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}_${UNLEARN_METHOD}_filter${UNLEARN_EPOCHS}_CE${TARGET_PREV_CE_WEIGHT}"
      FEDPERM_DIR="${OUT_ROOT}/${FEDPERM_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      mkdir -p "${FEDPERM_DIR}"

      if training_complete "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}"; then
        echo "Skip FedPerm unlearn, final round exists: ${FEDPERM_DIR}/round_${UNLEARN_GLOBAL_ROUNDS}"
      else
        EXTRA_EVAL_ARGS=()
        SKIP_ROUNDS=$(compute_skip_rounds "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}")
        FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
        FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
      "${LAUNCHER[@]}" src/federated_train.py \
        --config-name="${CONFIG_TRAIN}" \
        experiment=finetune/tofu/default \
        eval=muse \
          model="${MODEL_CFG}" \
          task_name="${FEDPERM_TASK}" \
          trainer=FederatedPermTrainer \
          trainer.method_args.enable_revocation=true \
          trainer.method_args.global_rounds="${UNLEARN_GLOBAL_ROUNDS}" \
          trainer.method_args.train_on="full" \
          trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
          trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
          trainer.method_args.unlearn_epochs="${UNLEARN_EPOCHS}" \
          trainer.method_args.normal_rounds_before_forget="${NORMAL_BEFORE_FORGET}" \
          trainer.method_args.forget_round_count="${FORGET_ROUNDS}" \
          trainer.method_args.recovery_round_start="${RECOVERY_START}" \
          trainer.method_args.filter_alpha="${FILTER_ALPHA}" \
          trainer.method_args.save_client_splits_for_eval=true \
          trainer.method_args.retain_indices_path="${RETAIN_IDX_FILE}" \
          trainer.method_args.forget_indices_path="${FORGET_IDX_FILE}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          trainer.method_args.target_prev_ce_enabled="${TARGET_PREV_CE_ENABLED}" \
          trainer.method_args.target_prev_ce_weight="${TARGET_PREV_CE_WEIGHT}" \
          +trainer.method_args.unlearn_base_full_rounds="${FULL_GLOBAL_ROUNDS}" \
          +trainer.method_args.filter_base_model_path="${FILTER_BASE_DIR}" \
          +trainer.method_args.round_offset="${FULL_GLOBAL_ROUNDS}" \
          +trainer.method_args.enable_permutation="${ENABLE_PERMUTATION}" \
          +trainer.method_args.retain_dataset_handler="${DATASET_HANDLER}" \
          +trainer.method_args.retain_dataset_hf_args.path="${DATASET_PATH}" \
          +trainer.method_args.retain_dataset_hf_args.name="${DATASET_NAME}" \
          +trainer.method_args.retain_dataset_hf_args.split="${RETAIN_SPLIT}" \
          +trainer.method_args.retain_text_key="${TEXT_KEY}" \
          +trainer.method_args.forget_dataset_handler="${DATASET_HANDLER}" \
          +trainer.method_args.forget_dataset_hf_args.path="${DATASET_PATH}" \
          +trainer.method_args.forget_dataset_hf_args.name="${DATASET_NAME}" \
          +trainer.method_args.forget_dataset_hf_args.split="${FORGET_SPLIT}" \
          +trainer.method_args.forget_text_key="${TEXT_KEY}" \
          data/datasets@data.train=MUSE_retain \
          data.train.MUSE_retain.args.hf_args.name="${DATASET_NAME}" \
          data.train.MUSE_retain.args.hf_args.split="${RETAIN_SPLIT}" \
          model.model_args.pretrained_model_name_or_path="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}" \
          trainer.args.output_dir="${FEDPERM_DIR}" \
          trainer.args.save_strategy="epoch" \
          paths.output_dir="${FEDPERM_DIR}" \
          trainer.args.seed="$((SEED + FULL_GLOBAL_ROUNDS))" \
          "${EXTRA_EVAL_ARGS[@]}" \
          2>&1 | tee "${LOG_ROOT}/${FEDPERM_TASK}.txt"
      fi

      prune_rounds "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}"
      eval_rounds "${FEDPERM_DIR}" "${FEDPERM_TASK}"

    done
  done

done

echo "=========== MUSE pipeline done =========="
echo "输出目录：${OUT_ROOT}"
echo "日志目录：${LOG_ROOT}"
