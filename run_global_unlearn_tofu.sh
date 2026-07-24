#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAdagrad,FedAdam,FedAvg,FedAvgM,FedYogi,FedProx}" # FedAdagrad, FedAdam, FedAvg, FedAvgM, FedYogi, FedProx
DEFAULT_UNLEARN_CLS=("SimNPO" "GradAscent")  # "NPO" "SimNPO" "GradAscent" "GradDiff"
MODELS=("Llama-3.2-1B-Instruct-lora" "Qwen2.5-1.5B-Instruct-lora")  # Qwen2.5-1.5B-Instruct-lora, Llama-3.2-1B-Instruct-lora

UNLEARN_CLS=("${DEFAULT_UNLEARN_CLS[@]}")
LAUNCHER=(python)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Single-GPU mode; using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# 禁用 SDPA/flash，避免多卡下的 kernel 问题
export PYTORCH_SDPA_ENABLE_FLASH=0
export PYTORCH_SDPA_ENABLE_MEM_EFFICIENT=0
export PYTORCH_SDPA_ENABLE_HEURISTIC=0

# 是否运行+保存所有 pre 评测，默认关闭以节省时间
ENABLE_PRE_EVAL="${ENABLE_PRE_EVAL:-false}"
export ENABLE_PRE_EVAL
# Target 客户端额外「当轮全部 source forget 合并」评测（与 full/retain 脚本一致）
export EVAL_TARGET_ROUND_FORGET_MERGE="${EVAL_TARGET_ROUND_FORGET_MERGE:-true}"
PARTICIPATION_RATIO="${PARTICIPATION_RATIO:-1}"
MIN_PARTICIPATING_CLIENTS="${MIN_PARTICIPATING_CLIENTS:-1}"

PARTICIPATION_ARGS=(
  ++trainer.method_args.participation_ratio="${PARTICIPATION_RATIO}"
  ++trainer.method_args.min_participating_clients="${MIN_PARTICIPATING_CLIENTS}"
)

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
# 支持多个 FL 方法（逗号分隔），默认只有一个
IFS=',' read -ra FL_METHODS <<< "${FL_METHOD}"

########################################
# 公共配置
########################################
FORGET_SPLITS=("forget10")
HOLDOUT_SPLITS=("holdout10")
RETAIN_SPLITS=("retain90")
USE_CLIENT_FORGET_EVAL=true   # forget_utility 是否用源客户端的 forget indices
ASSIGN_FORGET_TO_REVOKER_ONLY=true   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/TOFU/global_unlearn_only"
  OUT_ROOT="saves/TOFU/global_unlearn_only"
else
  LOG_ROOT="logs/TOFU/global_unlearn"
  OUT_ROOT="saves/TOFU/global_unlearn"
fi

TARGET_CLIENT_IDX=0
REVOKER_CLIENT_IDX=0

# full 阶段
FULL_GLOBAL_ROUNDS=10
FULL_SAVE_INTERVAL=0     # full 阶段：0 每轮保存并逐轮评测，>0 间隔保存

# global unlearn 阶段
UNLEARN_GLOBAL_ROUNDS=1
UNLEARN_SAVE_INTERVAL=0  # unlearn 阶段：同上
UNLEARN_Filter_EPOCHS="${UNLEARN_Filter_EPOCHS:-1}"

SEED=42

CONFIG_TRAIN="train-lora.yaml"
CONFIG_EVAL="eval/tofu/default.yaml"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

########################################
# 工具函数
########################################

run_eval() {
  run_eval_common "$@"
}

eval_rounds() {
  eval_rounds_common "$@"
}

# 确认按 SAVE_INTERVAL 需要保留的 round_* 是否都存在且非空
check_rounds_exist() {
  local base_dir="$1"
  local total_rounds="$2"
  local interval="$3"
  local gap=$((interval + 1))
  local missing=()
  for ((r=1; r<=total_rounds; r++)); do
    local should_exist=false
    if [ "${interval}" -le 0 ]; then
      should_exist=true
    elif [ "${r}" -eq "${total_rounds}" ] || [ $(((r - 1) % gap)) -eq 0 ]; then
      should_exist=true
    fi
    if [ "${should_exist}" = true ]; then
      local dir="${base_dir}/round_${r}"
      if [ ! -d "${dir}" ] || [ -z "$(ls -A "${dir}")" ]; then
        missing+=("${dir}")
      fi
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "❌ 缺少应保留的轮次目录：${missing[*]}"
    return 1
  fi
  return 0
}

########################################
# 主流程
########################################

for CURRENT_FL_METHOD in "${FL_METHODS[@]}"; do
  echo "===== FL_METHOD=${CURRENT_FL_METHOD} ====="

  for MODEL_CFG in "${MODELS[@]}"; do
    echo "=========== MODEL=${MODEL_CFG} ==========="

    for idx in "${!FORGET_SPLITS[@]}"; do
      FORGET_SPLIT="${FORGET_SPLITS[$idx]}"
      HOLDOUT_SPLIT="${HOLDOUT_SPLITS[$idx]}"
      RETAIN_SPLIT="${RETAIN_SPLITS[$idx]}"

      echo ">>> SPLITS retain=${RETAIN_SPLIT} forget=${FORGET_SPLIT} holdout=${HOLDOUT_SPLIT}"

      # 唯一划分目录，按 ASSIGN_FORGET_TO_REVOKER_ONLY 区分
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

      ########################################
      # 阶段 1：full 数据联邦训练（默认每轮保存+评测）
      ########################################
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}
      FULL_TASK="full_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
      FULL_DIR="${OUT_ROOT}/${FULL_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      mkdir -p "${FULL_DIR}"

      FULL_DONE=false
      if training_complete "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}"; then
        FULL_DONE=true
        if ! check_rounds_exist "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"; then
          echo "❌ full 已完成但缺少按 SAVE_INTERVAL 应该存在的轮次；请删除 ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS} 后重新跑 full 阶段" >&2
          exit 1
        fi
        echo "Skip full training, final round exists: ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}"
      fi

      if [ "${FULL_DONE}" != true ]; then
        EXTRA_EVAL_ARGS=()
        SAVE_SPLITS_FOR_EVAL=true
        if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
          EXTRA_EVAL_ARGS+=(
            ++eval.tofu.forget_indices_path="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
            ++eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
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

        SKIP_ROUNDS=$(compute_skip_rounds "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}")
        FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
        FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
        "${LAUNCHER[@]}" src/federated_train.py \
          --config-name="${CONFIG_TRAIN}" \
          experiment=finetune/tofu/default \
          model="${MODEL_CFG}" \
          task_name="${FULL_TASK}" \
          trainer=FederatedFinetune \
          trainer.method_args.global_rounds="${FULL_GLOBAL_ROUNDS}" \
          trainer.method_args.aggregation_strategy="${CURRENT_FL_METHOD}" \
          trainer.method_args.train_on="full" \
          trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
          trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
          trainer.method_args.save_client_splits_for_eval="${SAVE_SPLITS_FOR_EVAL}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          "${PARTICIPATION_ARGS[@]}" \
          data/datasets@data.train=TOFU_QA_full \
          data.train.TOFU_QA_full.args.hf_args.name=full \
          trainer.args.output_dir="${FULL_DIR}" \
          trainer.args.save_strategy="epoch" \
          paths.output_dir="${FULL_DIR}" \
          trainer.args.seed="${SEED}" \
          "${EXTRA_EVAL_ARGS[@]}" \
          2>&1 | tee "${LOG_ROOT}/${FULL_TASK}.txt"
        if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
          SPLITS_READY=true
        fi
      fi

      # 根据保存频次裁剪 checkpoint（仅当 SAVE_INTERVAL>0 时删中间轮）
      prune_rounds "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}

      eval_rounds "${FULL_DIR}" "${FULL_TASK}" "${FULL_GLOBAL_ROUNDS}"

      ########################################
      # 阶段 2：global unlearn（client0遗忘 -> 本地微调 -> 全局聚合）
      ########################################
      if [ "${SPLITS_READY}" != true ]; then
        echo "缺少公共划分文件：${COMMON_SPLIT_DIR}，请先生成全局索引后再运行遗忘阶段。" >&2
        exit 1
      fi
      SAVE_INTERVAL=${UNLEARN_SAVE_INTERVAL}
      for UNLEARN_METHOD in "${UNLEARN_CLS[@]}"; do
        echo "------- UNLEARN_METHOD=${UNLEARN_METHOD} -------"
        UNLEARN_TASK="global_unlearn_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}_${UNLEARN_METHOD}_${UNLEARN_Filter_EPOCHS}"
        UNLEARN_DIR="${OUT_ROOT}/${UNLEARN_TASK}"
        CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
        mkdir -p "${UNLEARN_DIR}"

        if training_complete "${UNLEARN_DIR}" "${UNLEARN_GLOBAL_ROUNDS}"; then
          echo "Skip global unlearn, final round exists: ${UNLEARN_DIR}/round_${UNLEARN_GLOBAL_ROUNDS}"
        else
          EXTRA_EVAL_ARGS=()
          if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
            EXTRA_EVAL_ARGS+=(
              ++eval.tofu.forget_indices_path="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
              ++eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
            )
          fi

          SKIP_ROUNDS=$(compute_skip_rounds "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}")
          FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
          FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
          "${LAUNCHER[@]}" src/federated_train.py \
            --config-name="${CONFIG_TRAIN}" \
            experiment=finetune/tofu/default \
            model="${MODEL_CFG}" \
            task_name="${UNLEARN_TASK}" \
            trainer=FederatedUnlearningTrainer \
            trainer.method_args.global_rounds="${UNLEARN_GLOBAL_ROUNDS}" \
            trainer.method_args.unlearn_epochs="${UNLEARN_Filter_EPOCHS}" \
            trainer.method_args.unlearn_trainer_cls="${UNLEARN_METHOD}" \
            trainer.method_args.aggregation_strategy="${CURRENT_FL_METHOD}" \
            +trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
            +trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
            +trainer.method_args.save_client_splits_for_eval=false \
            +trainer.method_args.retain_indices_path="${RETAIN_IDX_FILE}" \
            +trainer.method_args.forget_indices_path="${FORGET_IDX_FILE}" \
            +trainer.method_args.init_all_clients_from_unlearn=true \
            trainer.method_args.target_client_idx="${TARGET_CLIENT_IDX}" \
            +trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
            "${PARTICIPATION_ARGS[@]}" \
            data/datasets@data.train=TOFU_QA_full \
            data.train.TOFU_QA_full.args.hf_args.name=full \
            model.model_args.pretrained_model_name_or_path="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}" \
            trainer.args.output_dir="${UNLEARN_DIR}" \
            trainer.args.save_strategy="epoch" \
            paths.output_dir="${UNLEARN_DIR}" \
            trainer.args.seed="${SEED}" \
            "${EXTRA_EVAL_ARGS[@]}" \
            2>&1 | tee "${LOG_ROOT}/${UNLEARN_TASK}.txt"
        fi

        # 根据保存频次裁剪 checkpoint（保留最后一轮）
        prune_rounds "${UNLEARN_DIR}" "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}"
        SAVE_INTERVAL=${UNLEARN_SAVE_INTERVAL}

        eval_rounds "${UNLEARN_DIR}" "${UNLEARN_TASK}" "${UNLEARN_GLOBAL_ROUNDS}"

      done

    done

  done

done

echo "=========== 全流程完成 ==========="
echo "输出目录：${OUT_ROOT}"
echo "日志目录：${LOG_ROOT}"
