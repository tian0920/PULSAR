#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/scripts/lib/fedperm_common.sh"

# 可选：可通过参数或环境变量覆盖的配置
FL_METHOD="${FL_METHOD:-FedAvg,FedAdam,FedAdagrad}" # FedAvgM, FedAvg, FedYogi, FedProx, FedAdagrad, FedAdam，FedNova
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES}"

PARALLEL_CLIENT_WORKERS="${PARALLEL_CLIENT_WORKERS:-0}"
CLIENT_PARALLEL_DEVICES="${CLIENT_PARALLEL_DEVICES:-}"
PARTICIPATION_RATIO="${PARTICIPATION_RATIO:-0.1}"
MIN_PARTICIPATING_CLIENTS="${MIN_PARTICIPATING_CLIENTS:-3}"
RETAIN_PARALLEL_ARGS=(
  ++trainer.method_args.parallel_client_workers="${PARALLEL_CLIENT_WORKERS}"
)
if [ -n "${CLIENT_PARALLEL_DEVICES}" ]; then
  RETAIN_PARALLEL_ARGS+=(
    "++trainer.method_args.client_parallel_devices=[${CLIENT_PARALLEL_DEVICES}]"
  )
fi

PARTICIPATION_ARGS=(
  ++trainer.method_args.participation_ratio="${PARTICIPATION_RATIO}"
  ++trainer.method_args.min_participating_clients="${MIN_PARTICIPATING_CLIENTS}"
)

# 是否运行+保存所有 pre 评测，默认关闭以节省时间
ENABLE_PRE_EVAL="${ENABLE_PRE_EVAL:-false}"

# Target 客户端额外 forget 合并评测（与 source/授权客户端当轮合并集一致），见 eval_routing.ROUND_FORGET_MERGE_STAGE_SUFFIX
# 关闭：EVAL_TARGET_ROUND_FORGET_MERGE=false
export EVAL_TARGET_ROUND_FORGET_MERGE="${EVAL_TARGET_ROUND_FORGET_MERGE:-false}"

########################################
MODELS=("Llama-3.2-1B-Instruct-lora")  # Qwen2.5-1.5B-Instruct-lora, Llama-3.2-1B-Instruct-lora

FORGET_SPLITS=("forget10")
HOLDOUT_SPLITS=("holdout10")
RETAIN_SPLITS=("retain90")
NUM_CLIENTS="${NUM_CLIENTS:-100}"
# 全局 source 候选池与数据划分；每轮动态 source→target 见 eval_rounds/round_*/participation.json
NUM_SOURCE_CLIENTS="${NUM_SOURCE_CLIENTS:-10}"
NUM_TARGETS_PER_SOURCE="${NUM_TARGETS_PER_SOURCE:-2}"
SOURCE_CLIENT_INDICES="${SOURCE_CLIENT_INDICES:-0,10,20,30,40,50,60,70,80,90}"      # 指定 "0,3,7"
SOURCE_TO_TARGETS="${SOURCE_TO_TARGETS:-{0:[3,6],10:[13,16],20:[23,26],30:[33,36],40:[43,46],50:[53,56],60:[63,66],70:[73,76],80:[83,86],90:[93,96]}}"  ## {0:[1,2,4],3:[4,5,8],7:[3,6,8]}
MIN_SOURCE_CLIENTS_PER_ROUND="${MIN_SOURCE_CLIENTS_PER_ROUND:-2}"
TARGETS_PER_SOURCE_PER_ROUND="${TARGETS_PER_SOURCE_PER_ROUND:-2}"
RETAIN_CLIENT_WEIGHTS="${RETAIN_CLIENT_WEIGHTS:-}"
FORGET_SOURCE_WEIGHTS="${FORGET_SOURCE_WEIGHTS:-}"

LEARNING_RATE="${LEARNING_RATE:-8e-4}"  ## 8e-5

# retain-only 默认按 source-client 拓扑分配 forget，支持大规模部分客户端参与。
# 如需回退旧行为，可显式传 ASSIGN_FORGET_TO_REVOKER_ONLY=true。
ASSIGN_FORGET_TO_REVOKER_ONLY="${ASSIGN_FORGET_TO_REVOKER_ONLY:-true}"   # 数据集划分方式
if [ "${ASSIGN_FORGET_TO_REVOKER_ONLY}" = "true" ]; then
  LOG_ROOT="logs/TOFU/fedperm_only"
  OUT_ROOT="saves/TOFU/fedperm_only"
else
  LOG_ROOT="logs/TOFU/fedperm"
  OUT_ROOT="saves/TOFU/fedperm"
fi

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

GLOBAL_ROUNDS=5
SAVE_INTERVAL=0              # 0 表示每轮保存；1 表示间隔一轮保存（1,3,5...），以此类推

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

      # 共享客户端划分目录；默认按 source-client 拓扑切分 retain/forget。
      COMMON_SPLIT_DIR="${OUT_ROOT}/client_splits/${RETAIN_SPLIT}_${FORGET_SPLIT}_seed${SEED}"
      mkdir -p "${COMMON_SPLIT_DIR}"
      export FEDPERM_SPLIT_DIR="${COMMON_SPLIT_DIR}"

      echo ">>> SPLITS retain=${RETAIN_SPLIT} forget=${FORGET_SPLIT} holdout=${HOLDOUT_SPLIT}"

      ############################
      # 1) retain-only 训练
      ############################
      RETAIN_TASK="retain_${CURRENT_FL_METHOD}_${MODEL_CFG}_${RETAIN_SPLIT}"
      RETAIN_DIR="${OUT_ROOT}/${RETAIN_TASK}"
      CLIENT_SPLIT_DIR="${COMMON_SPLIT_DIR}"
      mkdir -p "${RETAIN_DIR}"

      if training_complete "${RETAIN_DIR}" "${GLOBAL_ROUNDS}"; then
        echo "Skip retain-only training, final round exists: ${RETAIN_DIR}/round_${GLOBAL_ROUNDS}"
      else
        EXTRA_EVAL_ARGS=()
        SAVE_SPLITS_FOR_EVAL=true
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
          "${PARTICIPATION_ARGS[@]}" \
          "${TOPOLOGY_ARGS[@]}" \
          "${ROUND_TOPOLOGY_ARGS[@]}" \
          "${RETAIN_PARALLEL_ARGS[@]}" \
          trainer.args.output_dir="${RETAIN_DIR}" \
          trainer.args.save_strategy="epoch" \
          paths.output_dir="${RETAIN_DIR}" \
          trainer.args.seed="${SEED}" \
          trainer.args.learning_rate="${LEARNING_RATE}" \
          "${EXTRA_EVAL_ARGS[@]}" \
          2>&1 | tee "${LOG_ROOT}/${RETAIN_TASK}.txt"
      fi

      # 根据保存频次裁剪 checkpoint
      prune_rounds "${RETAIN_DIR}" "${GLOBAL_ROUNDS}" "${SAVE_INTERVAL}"

      eval_rounds "${RETAIN_DIR}" "${RETAIN_TASK}"

    done

  done

done

echo "全部完成。输出：${OUT_ROOT}，日志：${LOG_ROOT}"
