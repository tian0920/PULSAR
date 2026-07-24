#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAvg,FedProx,FedAdam,FedAWA,FedLWS}" # FedAvg,FedAvgM,FedYogi,FedProx,FedAdam,FedAdagrad
DEFAULT_UNLEARN_CLS=("GradAscent" "NPO" "SimNPO" "GradDiff" "CEU" "PDU" "UNDIAL" "WGA")  # "NPO" "SimNPO" "GradAscent" "GradDiff"
UNLEARN_CLS=("${DEFAULT_UNLEARN_CLS[@]}")
LAUNCHER=(python)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES}"
# 禁用 SDPA/flash，避免多卡下的 kernel 问题
export PYTORCH_SDPA_ENABLE_FLASH=0
export PYTORCH_SDPA_ENABLE_MEM_EFFICIENT=0
export PYTORCH_SDPA_ENABLE_HEURISTIC=0

PARALLEL_CLIENT_WORKERS="${PARALLEL_CLIENT_WORKERS:-0}"
PARALLEL_FILTER_WORKERS="${PARALLEL_FILTER_WORKERS:-0}"
CLIENT_PARALLEL_DEVICES="${CLIENT_PARALLEL_DEVICES:-}"
FILTER_PARALLEL_DEVICES="${FILTER_PARALLEL_DEVICES:-}"
PARTICIPATION_RATIO="${PARTICIPATION_RATIO:-0.1}"
MIN_PARTICIPATING_CLIENTS="${MIN_PARTICIPATING_CLIENTS:-3}"

COMMON_PARALLEL_ARGS=(
  ++trainer.method_args.parallel_client_workers="${PARALLEL_CLIENT_WORKERS}"
)
if [ -n "${CLIENT_PARALLEL_DEVICES}" ]; then
  COMMON_PARALLEL_ARGS+=(
    "++trainer.method_args.client_parallel_devices=[${CLIENT_PARALLEL_DEVICES}]"
  )
fi

FEDPERM_PARALLEL_ARGS=(
  "${COMMON_PARALLEL_ARGS[@]}"
  ++trainer.method_args.parallel_filter_workers="${PARALLEL_FILTER_WORKERS}"
)
if [ -n "${FILTER_PARALLEL_DEVICES}" ]; then
  FEDPERM_PARALLEL_ARGS+=(
    "++trainer.method_args.filter_parallel_devices=[${FILTER_PARALLEL_DEVICES}]"
  )
fi

PARTICIPATION_ARGS=(
  ++trainer.method_args.participation_ratio="${PARTICIPATION_RATIO}"
  ++trainer.method_args.min_participating_clients="${MIN_PARTICIPATING_CLIENTS}"
)

# 是否运行+保存所有 pre 评测，默认关闭以节省时间
ENABLE_PRE_EVAL="${ENABLE_PRE_EVAL:-false}"

# Target 客户端在常规 post_train 评测外，额外用「当轮已采样全部 source 的 forget 合并」再评一次；
# 输出：eval_rounds/round_*/client{n}_post_train__forget_merge_round_sources/TOFU_SUMMARY.json
# 关闭：EVAL_TARGET_ROUND_FORGET_MERGE=false
export EVAL_TARGET_ROUND_FORGET_MERGE="${EVAL_TARGET_ROUND_FORGET_MERGE:-false}"

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
MODELS=("Llama-3.2-1B-Instruct-lora")  # Qwen2.5-1.5B-Instruct-lora, Llama-3.2-1B-Instruct-lora

FORGET_SPLITS=("forget10")
HOLDOUT_SPLITS=("holdout10")
RETAIN_SPLITS=("retain90")
# NUM_CLIENTS="${NUM_CLIENTS:-100}"
# NUM_SOURCE_CLIENTS="${NUM_SOURCE_CLIENTS:-2}"
# NUM_TARGETS_PER_SOURCE="${NUM_TARGETS_PER_SOURCE:-2}"
# SOURCE_CLIENT_INDICES="${SOURCE_CLIENT_INDICES:-0,5}"      # 指定 "0,3,7"
# SOURCE_TO_TARGETS="${SOURCE_TO_TARGETS:-{0:[2,4],5:[3,7]}}"  ## {0:[1,2,4],3:[4,5,8],7:[3,6,8]}

NUM_CLIENTS="${NUM_CLIENTS:-100}"
# 以下为全局 source 候选池与数据划分（fedperm_topology）；每轮实际映射由 Trainer 写入 participation.json
NUM_SOURCE_CLIENTS="${NUM_SOURCE_CLIENTS:-10}"
NUM_TARGETS_PER_SOURCE="${NUM_TARGETS_PER_SOURCE:-2}"
SOURCE_CLIENT_INDICES="${SOURCE_CLIENT_INDICES:-0,10,20,30,40,50,60,70,80,90}"      # 指定 "0,3,7"
SOURCE_TO_TARGETS="${SOURCE_TO_TARGETS:-{0:[3,6],10:[13,16],20:[23,26],30:[33,36],40:[43,46],50:[53,56],60:[63,66],70:[73,76],80:[83,86],90:[93,96]}}"  ## {0:[1,2,4],3:[4,5,8],7:[3,6,8]}"
MIN_SOURCE_CLIENTS_PER_ROUND="${MIN_SOURCE_CLIENTS_PER_ROUND:-2}"
TARGETS_PER_SOURCE_PER_ROUND="${TARGETS_PER_SOURCE_PER_ROUND:-2}"

RETAIN_CLIENT_WEIGHTS="${RETAIN_CLIENT_WEIGHTS:-}"  # {0: 2, 1: 1, 2: 1, 3: 0.5}
FORGET_SOURCE_WEIGHTS="${FORGET_SOURCE_WEIGHTS:-}"  # [0, 3, 7]
ASSIGN_FORGET_TO_REVOKER_ONLY=true   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/TOFU/fedperm_only"
  OUT_ROOT="saves/TOFU/fedperm_only"
else
  LOG_ROOT="logs/TOFU/fedperm"
  OUT_ROOT="saves/TOFU/fedperm"
fi

ENABLE_PERMUTATION=false    # 神经元转置

TOPOLOGY_ARGS=(
  ++trainer.method_args.num_clients="${NUM_CLIENTS}"
)
if [ -n "${NUM_SOURCE_CLIENTS}" ]; then
  TOPOLOGY_ARGS+=(
    ++trainer.method_args.num_source_clients="${NUM_SOURCE_CLIENTS}"
  )
fi
if [ -n "${NUM_TARGETS_PER_SOURCE}" ]; then
  TOPOLOGY_ARGS+=(
    ++trainer.method_args.num_targets_per_source="${NUM_TARGETS_PER_SOURCE}"
  )
fi
if [ -n "${SOURCE_CLIENT_INDICES}" ]; then
  TOPOLOGY_ARGS+=(
    "++trainer.method_args.source_client_indices=[${SOURCE_CLIENT_INDICES}]"
  )
fi
if [ -n "${SOURCE_TO_TARGETS}" ]; then
  TOPOLOGY_ARGS+=(
    "++trainer.method_args.source_to_targets=${SOURCE_TO_TARGETS}"
  )
fi
if [ -n "${RETAIN_CLIENT_WEIGHTS}" ]; then
  TOPOLOGY_ARGS+=(
    "++trainer.method_args.retain_client_weights=${RETAIN_CLIENT_WEIGHTS}"
  )
fi
if [ -n "${FORGET_SOURCE_WEIGHTS}" ]; then
  TOPOLOGY_ARGS+=(
    "++trainer.method_args.forget_source_weights=${FORGET_SOURCE_WEIGHTS}"
  )
fi

ROUND_TOPOLOGY_ARGS=(
  ++trainer.method_args.min_source_clients_per_round="${MIN_SOURCE_CLIENTS_PER_ROUND}"
  ++trainer.method_args.targets_per_source_per_round="${TARGETS_PER_SOURCE_PER_ROUND}"
)

# full 阶段
FULL_GLOBAL_ROUNDS=10
FULL_SAVE_INTERVAL=0     # full 阶段：0 每轮保存并逐轮评测，>0 间隔保存
PRE_BASE_OFFSET="${PRE_BASE_OFFSET:-1}"  # 取最后几轮前的基线（默认 1）
PRE_BASE_OFFSETS="1"  # 逗号分隔，遍历多个 offset  ,2,3,4

# 遗忘+恢复阶段（沿用 run_fedperm_only 的编排）
UNLEARN_GLOBAL_ROUNDS=1
NORMAL_BEFORE_FORGET=0
FORGET_ROUNDS=1
RECOVERY_START=1
UNLEARN_SAVE_INTERVAL=0  # unlearn 阶段：同上

UNLEARN_EPOCHS=1   # 过滤器训练轮次
FILTER_ALPHA=1     # θ_g^t-(θ_g^{t-1}-alpha*θ_unlearn)
LEARNING_RATE="${LEARNING_RATE:-8e-5}"  ## 8e-5

# 遗忘阶段目标客户端B本地LOSS正则锚点/双CE开关

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

resolve_target_clients() {
  local split_dir="$1"
  SOURCE_TO_TARGETS="${SOURCE_TO_TARGETS}" \
  SOURCE_CLIENT_INDICES="${SOURCE_CLIENT_INDICES}" \
  python3 - "$split_dir" <<'PY'
import json
import os
import sys
import ast

split_dir = sys.argv[1]
topology = None
env_mapping = os.environ.get("SOURCE_TO_TARGETS", "").strip()
if env_mapping:
    try:
        topology = json.loads(env_mapping)
    except Exception:
        try:
            topology = ast.literal_eval(env_mapping)
        except Exception:
            topology = None

if topology is None:
    path = os.path.join(split_dir, "fedperm_topology.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            payload = json.load(f)
        topology = payload.get("source_to_targets")

targets = set()
if topology:
    for values in topology.values():
        for value in values:
            targets.add(int(value))

if not targets:
    raise SystemExit("No target clients resolved from source_to_targets")

print(",".join(str(v) for v in sorted(targets)))
PY
}

build_prev_target_model_paths_expr() {
  local full_dir="$1"
  local full_round="$2"
  local target_csv="$3"
  python3 - "$full_dir" "$full_round" "$target_csv" <<'PY'
import sys

full_dir, full_round, target_csv = sys.argv[1:4]
targets = [int(x) for x in target_csv.split(",") if x.strip()]
parts = []
for target in targets:
    path = f"{full_dir}/round_{full_round}_target_client_{target}"
    parts.append(f'{target}:"{path}"')
print("{" + ",".join(parts) + "}")
PY
}

check_target_teacher_dirs() {
  local full_dir="$1"
  local full_round="$2"
  local target_csv="$3"
  local missing=()
  IFS=',' read -ra TARGETS <<< "${target_csv}"
  for target_idx in "${TARGETS[@]}"; do
    [ -z "${target_idx}" ] && continue
    local dir="${full_dir}/round_${full_round}_target_client_${target_idx}"
    if [ ! -d "${dir}" ] || [ -z "$(ls -A "${dir}")" ]; then
      missing+=("${dir}")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "❌ 缺少 prev target checkpoints: ${missing[*]}"
    return 1
  fi
  return 0
}

########################################
# 主流程
########################################

IFS=',' read -ra PRE_BASE_OFFSET_LIST <<< "${PRE_BASE_OFFSETS}"
for CURRENT_FL_METHOD in "${FL_METHODS[@]}"; do
  echo "===== FL_METHOD=${CURRENT_FL_METHOD} ====="

  for MODEL_CFG in "${MODELS[@]}"; do
    echo "=========== MODEL=${MODEL_CFG} ==========="

    for idx in "${!FORGET_SPLITS[@]}"; do
      FORGET_SPLIT="${FORGET_SPLITS[$idx]}"
      HOLDOUT_SPLIT="${HOLDOUT_SPLITS[$idx]}"
      RETAIN_SPLIT="${RETAIN_SPLITS[$idx]}"

      echo ">>> SPLITS retain=${RETAIN_SPLIT} forget=${FORGET_SPLIT} holdout=${HOLDOUT_SPLIT}"

      for PRE_BASE_OFFSET in "${PRE_BASE_OFFSET_LIST[@]}"; do
        echo ">>> PRE_BASE_OFFSET=${PRE_BASE_OFFSET}"

      # 唯一划分目录，按 ASSIGN_FORGET_TO_REVOKER_ONLY 区分：fedperm / fedperm_only
      COMMON_SPLIT_DIR="${OUT_ROOT}/client_splits/${RETAIN_SPLIT}_${FORGET_SPLIT}_seed${SEED}"
      mkdir -p "${COMMON_SPLIT_DIR}"
      export FEDPERM_SPLIT_DIR="${COMMON_SPLIT_DIR}"

      ########################################
      # 阶段 1：full 数据联邦训练（默认每轮保存+评测）
      ########################################
      SAVE_INTERVAL=${FULL_SAVE_INTERVAL}
      FULL_TASK="full_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
      FULL_DIR="${OUT_ROOT}/${FULL_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      mkdir -p "${FULL_DIR}"

      PRE_BASE_ROUND=$((FULL_GLOBAL_ROUNDS - PRE_BASE_OFFSET))
      PRE_BASE_SRC="${FULL_DIR}/round_${PRE_BASE_ROUND}"
      TARGET_CLIENTS_CSV="$(resolve_target_clients "${COMMON_SPLIT_DIR}")"
      FORCE_FULL_RERUN=false
      FULL_DONE=false

      if training_complete "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}"; then
        FULL_DONE=true
        if [ "${TARGET_PREV_CE_ENABLED}" = "true" ]; then
          if ! check_target_teacher_dirs "${FULL_DIR}" "${FULL_GLOBAL_ROUNDS}" "${TARGET_CLIENTS_CSV}"; then
            echo "⚠️ 缺少 prev target checkpoints，删除最后一轮以触发重跑 full 阶段"
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
          trainer.method_args.save_target_client_checkpoint=true \
          trainer.method_args.save_target_client_final_only=true \
          trainer.method_args.save_client_splits_for_eval="${SAVE_SPLITS_FOR_EVAL}" \
          trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
          "${PARTICIPATION_ARGS[@]}" \
          "${TOPOLOGY_ARGS[@]}" \
          "${ROUND_TOPOLOGY_ARGS[@]}" \
          "${COMMON_PARALLEL_ARGS[@]}" \
          data/datasets@data.train=TOFU_QA_full \
          data.train.TOFU_QA_full.args.hf_args.name=full \
          trainer.args.output_dir="${FULL_DIR}" \
          trainer.args.save_strategy="epoch" \
          paths.output_dir="${FULL_DIR}" \
          trainer.args.seed="${SEED}" \
          trainer.args.learning_rate="${LEARNING_RATE}" \
          "${EXTRA_EVAL_ARGS[@]}" \
          2>&1 | tee "${LOG_ROOT}/${FULL_TASK}.txt"
      fi

      # 预留 unlearn 起点：确保 PRE_BASE_ROUND 的检查点存在
      if [ ${PRE_BASE_ROUND} -ge 1 ]; then
        if [ ! -d "${PRE_BASE_SRC}" ] || [ -z "$(ls -A "${PRE_BASE_SRC}")" ]; then
          echo "❌ 缺少 unlearn 起点目录：${PRE_BASE_SRC}；请先重新跑 full 阶段生成该轮检查点" >&2
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
      TARGET_CLIENTS_CSV="$(resolve_target_clients "${COMMON_SPLIT_DIR}")"
      SAVE_INTERVAL=${UNLEARN_SAVE_INTERVAL}
      for UNLEARN_METHOD in "${UNLEARN_CLS[@]}"; do
        echo "------- UNLEARN_METHOD=${UNLEARN_METHOD} -------"
        FEDPERM_TASK="unlearn_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}_${UNLEARN_METHOD}_filter${UNLEARN_EPOCHS}_CE${TARGET_PREV_CE_WEIGHT}_Alpha${FILTER_ALPHA}_Base${PRE_BASE_OFFSET}"
        FEDPERM_DIR="${OUT_ROOT}/${FEDPERM_TASK}"
        CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
        mkdir -p "${FEDPERM_DIR}"

        if training_complete "${FEDPERM_DIR}" "${UNLEARN_GLOBAL_ROUNDS}"; then
          echo "Skip FedPerm unlearn, final round exists: ${FEDPERM_DIR}/round_${UNLEARN_GLOBAL_ROUNDS}"
        else
          EXTRA_EVAL_ARGS=()

          SKIP_ROUNDS=$(compute_skip_rounds "${UNLEARN_GLOBAL_ROUNDS}" "${UNLEARN_SAVE_INTERVAL}")
          TARGET_PREV_CE_ARGS=()
          if [ "${TARGET_PREV_CE_ENABLED}" = "true" ]; then
            PREV_TARGET_PATHS_EXPR="$(build_prev_target_model_paths_expr "${FULL_DIR}" "${PRE_BASE_ROUND}" "${TARGET_CLIENTS_CSV}")"
            TARGET_PREV_CE_ARGS+=("++trainer.method_args.prev_target_model_paths=${PREV_TARGET_PATHS_EXPR}")
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
            trainer.method_args.assign_forget_to_revoker_only="${ASSIGN_FORGET_TO_REVOKER_ONLY}" \
            trainer.method_args.target_prev_ce_enabled="${TARGET_PREV_CE_ENABLED}" \
            trainer.method_args.target_prev_ce_weight="${TARGET_PREV_CE_WEIGHT}" \
            "${PARTICIPATION_ARGS[@]}" \
            "${TOPOLOGY_ARGS[@]}" \
            "${ROUND_TOPOLOGY_ARGS[@]}" \
            "${FEDPERM_PARALLEL_ARGS[@]}" \
            +trainer.method_args.unlearn_base_full_rounds="${FULL_GLOBAL_ROUNDS}" \
            +trainer.method_args.unlearn_base_offset="${PRE_BASE_OFFSET}" \
            +trainer.method_args.unlearn_base_full_dir="${FULL_DIR}" \
            +trainer.method_args.round_offset="${FULL_GLOBAL_ROUNDS}" \
            +trainer.method_args.data_partition_seed="${SEED}" \
            "${TARGET_PREV_CE_ARGS[@]}" \
            +trainer.method_args.enable_permutation="${ENABLE_PERMUTATION}" \
            data/datasets@data.train=TOFU_QA_full \
            data.train.TOFU_QA_full.args.hf_args.name=full \
            model.model_args.pretrained_model_name_or_path="${FULL_DIR}/round_${PRE_BASE_ROUND}" \
            trainer.args.output_dir="${FEDPERM_DIR}" \
            trainer.args.save_strategy="epoch" \
            paths.output_dir="${FEDPERM_DIR}" \
            trainer.args.seed="$((SEED + FULL_GLOBAL_ROUNDS))" \
            trainer.args.learning_rate="${LEARNING_RATE}" \
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

done

echo "=========== 全流程完成 ==========="
echo "输出目录：${OUT_ROOT}"
echo "日志目录：${LOG_ROOT}"
