#!/usr/bin/env bash
set -e

CONFIG_NAME=train-lora.yaml
FINETUNE_EXP=finetune/tofu/default.yaml
EVAL_EXP=eval/tofu/default.yaml
SEED=42

FORGET_SPLIT=forget_0
RETAIN_SPLIT=retain_0
HOLDOUT_SPLIT=holdout_0

RETAIN_TASK_NAME=retain_tofu
FEDAVG_TASK_NAME=fedavg_tofu
FEDPERM_TASK_NAME=fedperm_tofu

########################################
# 1. Retain 训练（可选，用于 TOFU 评估基线）
########################################
python src/federated_train.py \
  --config-name=${CONFIG_NAME} \
  experiment=${FINETUNE_EXP} \
  task_name=${RETAIN_TASK_NAME} \
  trainer=FederatedFinetune \
  trainer.args.output_dir=saves/retain/${RETAIN_TASK_NAME} \
  paths.output_dir=saves/retain/${RETAIN_TASK_NAME} \
  trainer.args.seed=${SEED}

########################################
# 2. FedAvg 联邦微调基线
########################################
python src/federated_train.py \
  --config-name=${CONFIG_NAME} \
  experiment=${FINETUNE_EXP} \
  task_name=${FEDAVG_TASK_NAME} \
  trainer=FederatedFinetune \
  trainer.args.output_dir=saves/finetune/${FEDAVG_TASK_NAME} \
  paths.output_dir=saves/finetune/${FEDAVG_TASK_NAME} \
  trainer.args.seed=${SEED}

########################################
# 3. FedPerm（你的方法）
########################################
python src/federated_train.py \
  --config-name=${CONFIG_NAME} \
  experiment=${FINETUNE_EXP} \
  task_name=${FEDPERM_TASK_NAME} \
  trainer=FederatedPermTrainer \
  trainer.args.output_dir=saves/finetune/${FEDPERM_TASK_NAME} \
  paths.output_dir=saves/finetune/${FEDPERM_TASK_NAME} \
  trainer.args.seed=${SEED}

########################################
# 4. Retain 评估（TOFU）
########################################
python src/eval.py \
  experiment=${EVAL_EXP} \
  forget_split=${FORGET_SPLIT} \
  holdout_split=${HOLDOUT_SPLIT} \
  task_name=${RETAIN_TASK_NAME} \
  model.model_args.pretrained_model_name_or_path=saves/retain/${RETAIN_TASK_NAME} \
  paths.output_dir=saves/eval/${RETAIN_TASK_NAME} \
  hydra.run.dir=saves/eval/${RETAIN_TASK_NAME}

########################################
# 5. FedAvg 评估（TOFU）
########################################
python src/eval.py \
  experiment=${EVAL_EXP} \
  forget_split=${FORGET_SPLIT} \
  holdout_split=${HOLDOUT_SPLIT} \
  task_name=${FEDAVG_TASK_NAME} \
  model.model_args.pretrained_model_name_or_path=saves/finetune/${FEDAVG_TASK_NAME} \
  retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json \
  paths.output_dir=saves/eval/${FEDAVG_TASK_NAME}/${FORGET_SPLIT} \
  hydra.run.dir=saves/eval/${FEDAVG_TASK_NAME}/${FORGET_SPLIT}

########################################
# 6. FedPerm 评估（TOFU）
########################################
python src/eval.py \
  experiment=${EVAL_EXP} \
  forget_split=${FORGET_SPLIT} \
  holdout_split=${HOLDOUT_SPLIT} \
  task_name=${FEDPERM_TASK_NAME} \
  model.model_args.pretrained_model_name_or_path=saves/finetune/${FEDPERM_TASK_NAME} \
  retain_logs_path=saves/eval/${RETAIN_TASK_NAME}/TOFU_EVAL.json \
  paths.output_dir=saves/eval/${FEDPERM_TASK_NAME}/${FORGET_SPLIT} \
  hydra.run.dir=saves/eval/${FEDPERM_TASK_NAME}/${FORGET_SPLIT}