#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAvg}"
DEFAULT_UNLEARN_CLS=("GradAscent" "GradDiff" "NPO" "SimNPO")
UNLEARN_CLS=("${DEFAULT_UNLEARN_CLS[@]}")
# 单进程多 GPU：无需 torchrun，内部线程池按 FEDPERM_CLIENT_DEVICES 分配 GPU
LAUNCHER=(python)
CLIENT_DEVICES="${CLIENT_DEVICES:-}"
if [ -n "${CLIENT_DEVICES}" ]; then
  export FEDPERM_CLIENT_DEVICES="${CLIENT_DEVICES}"
  echo "Using single-process multi-GPU with devices: ${FEDPERM_CLIENT_DEVICES}"
else
  echo "Using single-process launcher (set CLIENT_DEVICES=\"0,1,...\" to pin GPUs)"
fi

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
MODELS=("Llama-3.2-1B-Instruct-lora")

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
FULL_GLOBAL_ROUNDS=3
FULL_SAVE_INTERVAL=0     # full 阶段：0 每轮保存，1 表示间隔一轮保存（1,3,5...），以此类推

# 遗忘+恢复阶段（沿用 run_fedperm_only 的编排）
UNLEARN_GLOBAL_ROUNDS=4
NORMAL_BEFORE_FORGET=0
FORGET_ROUNDS=2
RECOVERY_START=2
UNLEARN_SAVE_INTERVAL=0  # unlearn 阶段：同上

UNLEARN_EPOCHS=1   # 过滤器训练轮次
FILTER_ALPHA=1     # θ_g^t-(θ_g^{t-1}-alpha*θ_unlearn)

# 遗忘阶段目标客户端B本地LOSS正则锚点/双CE开关
TARGET_ANCHOR_REG_ENABLED=false
TARGET_ANCHOR_REG_LAMBDA=1

TARGET_PREV_CE_ENABLED=true   # 目标客户端本地训练双CE开关
TARGET_PREV_CE_WEIGHT=1.0     # θ_unlearn的微调项系数

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

maybe_run_eval() {
  local ckpt_dir="$1"
  local stage="$2"
  local log_file="$3"
  shift 3
  local summary_paths=("$@")

  if [ "${CURRENT_ROUND:-0}" -gt 0 ] && [ "${SAVE_INTERVAL:-0}" -gt 0 ]; then
    local gap=$((SAVE_INTERVAL + 1))
    if [ "${CURRENT_ROUND}" -ne "${LAST_ROUND}" ] && [ $(( (CURRENT_ROUND - 1) % gap )) -ne 0 ]; then
      echo "Skip eval ${stage} at round ${CURRENT_ROUND} due to SAVE_INTERVAL=${SAVE_INTERVAL}"
      return
    fi
  fi

  for summary_file in "${summary_paths[@]}"; do
    if [ -s "${summary_file}" ]; then
      echo "Skip eval ${stage}, summary already exists: ${summary_file}"
      return
    fi
  done

  run_eval "${ckpt_dir}" "${stage}" 2>&1 | tee "${log_file}"
}

sync_client_splits() {
  sync_client_splits_full "$@"
}

eval_final_round() {
  local base_dir="$1"
  local prefix="$2"
  local r="${FULL_GLOBAL_ROUNDS}"
  LAST_ROUND=${r}
  local post="${base_dir}/round_${r}"
  CURRENT_ROUND=${r}
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
}

eval_rounds() {
  local base_dir="$1"
  local prefix="$2"
  LAST_ROUND=${UNLEARN_GLOBAL_ROUNDS}
  for ((r=1; r<=UNLEARN_GLOBAL_ROUNDS; r++)); do
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
      if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
        SPLITS_READY=true
        echo "Found existing global indices, reuse: ${COMMON_SPLIT_DIR}"
      fi
      SPLITS_READY=false
      if [ -f "${RETAIN_IDX_FILE}" ] && [ -f "${FORGET_IDX_FILE}" ]; then
        SPLITS_READY=true
        echo "Found existing global indices, reuse: ${COMMON_SPLIT_DIR}"
      fi

      ########################################
      # 阶段 1：full 数据联邦训练（只保留最后一轮评测）
      ########################################
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}
      FULL_TASK="full_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
      FULL_DIR="${OUT_ROOT}/${FULL_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      mkdir -p "${FULL_DIR}"

      if training_complete "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}"; then
        echo "Skip full training, final round exists: ${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}"
      else
        EXTRA_EVAL_ARGS=()
        SAVE_SPLITS_FOR_EVAL=true
        if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
          EXTRA_EVAL_ARGS+=(
            eval.tofu.forget_indices_path="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
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

        SKIP_ROUNDS=$(compute_skip_rounds "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}")
        FEDPERM_SKIP_EVAL_ROUNDS="${SKIP_ROUNDS}" \
        FEDPERM_SKIP_SAVE_ROUNDS="${SKIP_ROUNDS}" \
        "${LAUNCHER[@]}" src/federated_train.py \
          --config-name="${CONFIG_TRAIN}" \
          experiment=finetune/tofu/default \
          model="${MODEL_CFG}" \
          task_name="${FULL_TASK}" \
          trainer=FederatedPermTrainer \
          trainer.method_args.enable_revocation=false \
          trainer.method_args.global_rounds="${FULL_GLOBAL_ROUNDS}" \
          trainer.method_args.train_on="full" \
          trainer.method_args.retain_split_name="${RETAIN_SPLIT}" \
          trainer.method_args.forget_split_name="${FORGET_SPLIT}" \
          trainer.method_args.save_client_splits_for_eval="${SAVE_SPLITS_FOR_EVAL}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          trainer.method_args.revoker_client_idx="${REVOKER_CLIENT_IDX}" \
          trainer.method_args.target_client_idx="${TARGET_CLIENT_IDX}" \
          trainer.method_args.eval_client_c_idx="${CLIENT_C_IDX}" \
          trainer.method_args.normal_rounds_before_forget="${FULL_GLOBAL_ROUNDS}" \
          trainer.method_args.forget_round_count=0 \
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
      PRE_BASE_ROUND=$((FULL_GLOBAL_ROUNDS - 1))
      if [ ${PRE_BASE_ROUND} -ge 1 ]; then
        PRE_BASE_SRC="${FULL_DIR}/round_${PRE_BASE_ROUND}"
        FILTER_BASE_DIR="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}_server_pre"
        if [ -d "${PRE_BASE_SRC}" ]; then
          if [ ! -d "${FILTER_BASE_DIR}" ] || [ -z "$(ls -A "${FILTER_BASE_DIR}")" ]; then
            echo "Preserve filter base (start of last round) from ${PRE_BASE_SRC} to ${FILTER_BASE_DIR}"
            mkdir -p "${FILTER_BASE_DIR}"
            # 仅拷贝内容，避免产生 round_x 嵌套目录
            cp -r "${PRE_BASE_SRC}/." "${FILTER_BASE_DIR}/"
          fi
        else
          echo "⚠️ 缺少 filter 基线源目录：${PRE_BASE_SRC}"
        fi
      fi

      # 根据保存频次裁剪 checkpoint（保留最后一轮）
      prune_rounds "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${FULL_SAVE_INTERVAL}"
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}

      eval_final_round "${FULL_DIR}" "${FULL_TASK}"
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
        FEDPERM_TASK="unlearn_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}_${UNLEARN_METHOD}_filter${UNLEARN_EPOCHS}_CE${TARGET_PREV_CE_WEIGHT}"
        FEDPERM_DIR="${OUT_ROOT}/${FEDPERM_TASK}"
        CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
        mkdir -p "${FEDPERM_DIR}"

        if training_complete "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}"; then
          echo "Skip FedPerm unlearn, final round exists: ${FEDPERM_DIR}/round_${UNLEARN_GLOBAL_ROUNDS}"
        else
          EXTRA_EVAL_ARGS=()
          if [ "${USE_CLIENT_FORGET_EVAL}" = "true" ]; then
            EXTRA_EVAL_ARGS+=(
              eval.tofu.forget_indices_path="${CLIENT_SPLIT_DIR}/client${REVOKER_CLIENT_IDX}_forget_indices.json"
              eval.tofu.forget_dataset_name="${FORGET_SPLIT}"
            )
          fi

          SKIP_ROUNDS=$(compute_skip_rounds "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}")
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
            trainer.method_args.prev_target_model_path="${FULL_DIR}/round_${FULL_GLOBAL_ROUNDS}_target_client_${TARGET_CLIENT_IDX}" \
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
