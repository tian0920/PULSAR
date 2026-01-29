#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAvgM}"
DEFAULT_UNLEARN_CLS=("GradAscent" "GradDiff")  # "NPO" "SimNPO" "GradAscent" "GradDiff"
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

# 两阶段脚本：
#   1) 基于 full 数据完成联邦训练，保存最后一轮的 server/client pre/post 评测
#   2) 以上述 full 模型为起点，按 run_fedperm_only 的遗忘+恢复流程跑 FedPerm，并保留每一轮评测

########################################
# 公共配置
########################################
MODELS=("Llama-3.2-3B-Instruct-lora")  # Qwen2.5-1.5B-Instruct-lora, Llama-3.2-1B-Instruct-lora

FORGET_SPLITS=("forget10")
HOLDOUT_SPLITS=("holdout10")
RETAIN_SPLITS=("retain90")
USE_CLIENT_FORGET_EVAL=true   # forget_utility 是否用源客户端的 forget indices
ASSIGN_FORGET_TO_REVOKER_ONLY=true   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/TOFU/fedperm_only"
  OUT_ROOT="saves/TOFU/fedperm_only"
else
  LOG_ROOT="logs/TOFU/fedperm"
  OUT_ROOT="saves/TOFU/fedperm"
fi

TARGET_CLIENT_IDX=1
REVOKER_CLIENT_IDX=0
CLIENT_C_IDX=2
ENABLE_PERMUTATION=false    # 神经元转置

# full 阶段
FULL_GLOBAL_ROUNDS=10
FULL_SAVE_INTERVAL=0     # full 阶段：0 每轮保存并逐轮评测，>0 间隔保存

# 遗忘+恢复阶段（沿用 run_fedperm_only 的编排）
UNLEARN_GLOBAL_ROUNDS=1
NORMAL_BEFORE_FORGET=0
FORGET_ROUNDS=1
RECOVERY_START=1
UNLEARN_SAVE_INTERVAL=0  # unlearn 阶段：同上

UNLEARN_EPOCHS=1   # 过滤器训练轮次
FILTER_ALPHA=1     # θ_g^t-(θ_g^{t-1}-alpha*θ_unlearn)

# 遗忘阶段目标客户端B本地LOSS正则锚点/双CE开关
TARGET_ANCHOR_REG_ENABLED=false
TARGET_ANCHOR_REG_LAMBDA=1

TARGET_PREV_CE_ENABLED=true   # 目标客户端本地训练双CE开关
TARGET_PREV_CE_WEIGHT=1     # θ_unlearn的微调项系数

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

sync_client_splits() {
  sync_client_splits_full "$@"
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

      # 唯一划分目录，按 ASSIGN_FORGET_TO_REVOKER_ONLY 区分：fedperm / fedperm_only
      COMMON_SPLIT_DIR="${OUT_ROOT}/client_splits/${RETAIN_SPLIT}_${FORGET_SPLIT}_seed${SEED}"
      mkdir -p "${COMMON_SPLIT_DIR}"
      export FEDPERM_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      RETAIN_IDX_FILE="${COMMON_SPLIT_DIR}/global_retain_indices.json"
      FORGET_IDX_FILE="${COMMON_SPLIT_DIR}/global_forget_indices.json"
      SPLITS_READY=false
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

      PRE_BASE_ROUND=$((FULL_GLOBAL_ROUNDS - 1))
      PRE_BASE_SRC="${FULL_DIR}/round_${PRE_BASE_ROUND}"
      PREV_TARGET_PATH="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}_target_client_${TARGET_CLIENT_IDX}"
      FORCE_FULL_RERUN=false
      FULL_DONE=false

      if training_complete "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}"; then
        FULL_DONE=true
        if [ "${TARGET_PREV_CE_ENABLED}" = "true" ]; then
          if [ ! -d "${PREV_TARGET_PATH}" ] || [ -z "$(ls -A "${PREV_TARGET_PATH}")" ]; then
            echo "⚠️ 缺少 prev target checkpoint：${PREV_TARGET_PATH}，删除最后一轮以触发重跑 full 阶段"
            rm -rf "${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}"
            FORCE_FULL_RERUN=true
            FULL_DONE=false
          fi
        fi
        if [ "${FULL_DONE}" = true ]; then
          if ! check_rounds_exist "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"; then
            echo "❌ full 已完成但缺少按 SAVE_INTERVAL 应该存在的轮次；请删除 ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS} 后重新跑 full 阶段" >&2
            exit 1
          fi
          if [ ! -d "${PRE_BASE_SRC}" ]; then
            echo "❌ full 已完成但缺少中间轮检查点：${PRE_BASE_SRC}；请删除 ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS} 后重新跑 full 阶段（FULL_SAVE_INTERVAL=0 确保每轮保存）" >&2
            exit 1
          fi
          echo "Skip full training, final round exists: ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}"
        fi
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
          trainer.method_args.target_client_idx="${TARGET_CLIENT_IDX}" \
          trainer.method_args.save_target_client_checkpoint=true \
          trainer.method_args.save_client_splits_for_eval="${SAVE_SPLITS_FOR_EVAL}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          trainer.method_args.revoker_client_idx="${REVOKER_CLIENT_IDX}" \
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

      # 预留 filter 基线：取“最后一轮开始时”服务器下发的模型 = 上一轮聚合产物
      FILTER_BASE_DIR=""
      if [ ${PRE_BASE_ROUND} -ge 1 ]; then
        FILTER_BASE_DIR="${FULL_DIR}/round_${PRE_BASE_ROUND}_server_post"
        if [ ! -d "${PRE_BASE_SRC}" ]; then
          echo "❌ 缺少 filter 基线源目录：${PRE_BASE_SRC}；请先重新跑 full 阶段生成该轮检查点" >&2
          exit 1
        fi
        if [ ! -d "${FILTER_BASE_DIR}" ] || [ -z "$(ls -A "${FILTER_BASE_DIR}")" ]; then
          echo "Preserve filter base (start of last round) from ${PRE_BASE_SRC} to ${FILTER_BASE_DIR}"
          mkdir -p "${FILTER_BASE_DIR}"
          cp -r "${PRE_BASE_SRC}/." "${FILTER_BASE_DIR}/"
        fi
        if [ -z "$(ls -A "${FILTER_BASE_DIR}")" ]; then
          echo "❌ filter 基线目录仍为空：${FILTER_BASE_DIR}" >&2
          exit 1
        fi
      fi

      # 根据保存频次裁剪 checkpoint（仅当 SAVE_INTERVAL>0 时删中间轮）
      prune_rounds "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}

      eval_rounds "${FULL_DIR}" "${FULL_TASK}" "${FULL_GLOBAL_ROUNDS}"
      ########################################
      # 阶段 2：基于 full 模型的 FedPerm 遗忘 + 恢复（每轮评测）
      ########################################
      if [ "${SPLITS_READY}" != true ]; then
        echo "缺少公共划分文件：${COMMON_SPLIT_DIR}，请先生成全局索引后再运行遗忘阶段。" >&2
        exit 1
      fi
      SAVE_INTERVAL=${UNLEARN_SAVE_INTERVAL}
      for UNLEARN_METHOD in "${UNLEARN_CLS[@]}"; do
        echo "------- UNLEARN_METHOD=${UNLEARN_METHOD} -------"
        FEDPERM_TASK="unlearn_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}_${UNLEARN_METHOD}_filter${UNLEARN_EPOCHS}_CE${TARGET_PREV_CE_WEIGHT}_Alpha${FILTER_ALPHA}"
        FEDPERM_DIR="${OUT_ROOT}/${FEDPERM_TASK}"
        CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
        mkdir -p "${FEDPERM_DIR}"

        if training_complete "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}"; then
          echo "Skip FedPerm unlearn, final round exists: ${FEDPERM_DIR}/round_${UNLEARN_GLOBAL_ROUNDS}"
        else
          EXTRA_EVAL_ARGS=()
          if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
            EXTRA_EVAL_ARGS+=(
              ++eval.tofu.forget_indices_path="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
              ++eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
            )
          fi

          SKIP_ROUNDS=$(compute_skip_rounds "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}")
          TARGET_PREV_CE_ARGS=()
          if [ "${TARGET_PREV_CE_ENABLED}" = "true" ]; then
            TARGET_PREV_CE_ARGS+=(trainer.method_args.prev_target_model_path="${PREV_TARGET_PATH}")
          fi
          FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
          FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
          "${LAUNCHER[@]}" src/federated_train.py \
            --config-name="${CONFIG_TRAIN}" \
            experiment=finetune/tofu/default \
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
            trainer.method_args.unlearn_trainer_cls="${UNLEARN_METHOD}" \
            trainer.method_args.aggregation_strategy="${CURRENT_FL_METHOD}" \
            trainer.method_args.save_client_splits_for_eval=false \
            +trainer.method_args.retain_indices_path="${RETAIN_IDX_FILE}" \
            +trainer.method_args.forget_indices_path="${FORGET_IDX_FILE}" \
            trainer.method_args.target_client_idx="${TARGET_CLIENT_IDX}" \
            trainer.method_args.revoker_client_idx="${REVOKER_CLIENT_IDX}" \
            trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
            trainer.method_args.eval_client_c_idx="${CLIENT_C_IDX}" \
            trainer.method_args.target_anchor_reg_enabled="${TARGET_ANCHOR_REG_ENABLED}" \
            trainer.method_args.target_anchor_reg_lambda="${TARGET_ANCHOR_REG_LAMBDA}" \
            trainer.method_args.target_prev_ce_enabled="${TARGET_PREV_CE_ENABLED}" \
            trainer.method_args.target_prev_ce_weight="${TARGET_PREV_CE_WEIGHT}" \
            +trainer.method_args.filter_base_model_path="${FILTER_BASE_DIR}" \
            "${TARGET_PREV_CE_ARGS[@]}" \
            +trainer.method_args.enable_permutation="${ENABLE_PERMUTATION}" \
            data/datasets@data.train=TOFU_QA_full \
            data.train.TOFU_QA_full.args.hf_args.name=full \
            model.model_args.pretrained_model_name_or_path="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}" \
            trainer.args.output_dir="${FEDPERM_DIR}" \
            trainer.args.save_strategy="epoch" \
            paths.output_dir="${FEDPERM_DIR}" \
            trainer.args.seed="${SEED}" \
            "${EXTRA_EVAL_ARGS[@]}" \
            2>&1 | tee "${LOG_ROOT}/${FEDPERM_TASK}.txt"
        fi

        # 根据保存频次裁剪 checkpoint（保留最后一轮）
        prune_rounds "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}"
        SAVE_INTERVAL=${UNLEARN_SAVE_INTERVAL}

        eval_rounds "${FEDPERM_DIR}" "${FEDPERM_TASK}"

      done

    done

  done

done

echo "=========== 全流程完成 ==========="
echo "输出目录：${OUT_ROOT}"
echo "日志目录：${LOG_ROOT}"
