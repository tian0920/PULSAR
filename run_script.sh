#!/bin/bash
set -ex

forget_split_list=(
  "forget01"
  "forget05"
  "forget10"
)

retain_split_list=(
  "retain99"
  "retain95"
  "retain90"
)

holdout_split_list=(
  "holdout01"
  "holdout05"
  "holdout10"
)

FL_Method="FedAvg"
Model_Size="3B"

# Make Directory for Logs
mkdir -p logs/retain
mkdir -p logs/retain/eval

# 获取组合长度
num_splits=${#forget_split_list[@]}

for ((i=0; i<$num_splits; i++)); do
  forget_split=${forget_split_list[$i]}
  retain_split=${retain_split_list[$i]}
  holdout_split=${holdout_split_list[$i]}

  retain_task_name="${Model_Size}_${FL_Method}_Retain_${retain_split}"

  # Step 1: Retain Training
  python src/federated_train.py \
    --config-name=train-lora.yaml \
    experiment=finetune/tofu/default \
    task_name=${retain_task_name} \
    datasets@data.train=TOFU_QA_retain \
    data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
    2>&1 | tee logs/retain/${retain_task_name}.txt

  echo "[Step 1]✅"

  # Step 2: Evaluation
  python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    task_name=${retain_task_name} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${retain_task_name} \
    2>&1 | tee logs/retain/eval/${retain_task_name}_Eval.txt

  echo "[Step 2]✅"

  # Step 3: Fine-Tune
  finetune_task_name = "${Model_Size}_${FL_Method}_Finetune"
  python src/federated_train.py \
    --config-name=train-lora.yaml \
    experiment=finetune/tofu/default.yaml \
    task_name=${finetune_task_name} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full
    2>&1 | tee logs/finetune/${finetune_task_name}.txt

  # Step 4: Evaluation
  eval_task_name = "${Model_Size}_${FL_Method}_Finetune_${forget_split}_Eval"

  python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    task_name= ${eval_task_name}\
    model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
    retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
    paths.output_dir=saves/eval/${finetune_task_name}/evals_${forget_split}
    2>&1 | tee logs/finetune/eval/${eval_task_name}.txt


  # Step 5: Unlearning
  unlearn_cls=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
  )

  default_yaml="configs/experiment/unlearn/tofu/default.yaml"
  trainer_yaml="configs/trainer/FederatedUnlearningTrainer.yaml"

  # 顺序尝试每个方法 ===
  for AfterUnlearn in "${unlearn_cls[@]}"; do
    echo -e "\n==============================="
    echo ">>> 当前目标方法: $AfterUnlearn"
    echo "==============================="

    # 提取 default.yaml 中 override /trainer 的方法（允许前面有空格）
    trainer_line=$(grep '\- override /trainer:' "$default_yaml")
    if [[ -z "$trainer_line" ]]; then
      echo "❌ 错误：未找到 override /trainer 行"
      exit 1
    fi
    BeforeUnlearn=$(echo "$trainer_line" | awk -F": " '{print $2}' | xargs)  # xargs去掉空格
    echo "[INFO] default.yaml 中 BeforeUnlearn: $BeforeUnlearn"

    # 提取 FederatedUnlearningTrainer.yaml 中的 unlearn_trainer_cls
    current_cls=$(grep "^[[:space:]]*unlearn_trainer_cls:" "$trainer_yaml" | awk -F'"' '{print $2}' | xargs)
    echo "[INFO] FederatedUnlearningTrainer.yaml 中 unlearn_trainer_cls: $current_cls"

    # 如果两个配置一致，则替换为 AfterUnlearn
    if [[ "$BeforeUnlearn" == "$current_cls" ]]; then
      echo "[ACTION] 配置一致，执行替换为 $AfterUnlearn"

      # 替换 default.yaml 中 override /trainer
      sed -i "s|^\([[:space:]]*- override /trainer:\).*|\1 $AfterUnlearn|" "$default_yaml"

      # 替换 trainer_yaml 中的 unlearn_trainer_cls（带引号）
      sed -i "s|^\([[:space:]]*unlearn_trainer_cls:\).*|\1 \"$AfterUnlearn\"|" "$trainer_yaml"

      echo "✅ 替换完成：$BeforeUnlearn → $AfterUnlearn"

      # === Step 4: Unlearning 训练 ===
      python src/federated_train.py \
        --config-name=unlearn-lora.yaml \
        experiment=unlearn/tofu/default \
        forget_split=${forget_split}\
        retain_split=${retain_split} \
        trainer=FederatedUnlearningTrainer \
        task_name=FedAvg_${AfterUnlearn}_Unlearn_${forget_split} \
        retain_logs_path=
        model.model_args.pretrained_model_name_or_path=saves/finetune/FedAvg_LoRA_Finetune \
        2>&1 | tee logs/FedAvg_LoRA_${AfterUnlearn}_Unlearn_Split95.txt

      # === Step 5: Unlearning 评估 ===
      python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=FedAvg_${AfterUnlearn}_Unlearn_Eval_${forget_split} \
        retain_logs_path=
        model.model_args.pretrained_model_name_or_path=saves/unlearn/FedAvg_LoRA_${AfterUnlearn}_Unlearn_Split95 \
        2>&1 | tee logs/FedAvg_LoRA_${AfterUnlearn}_Unlearn_Eval_Split95.txt
    else
      echo "⚠️ 配置不一致（$BeforeUnlearn ≠ $current_cls），跳过 $AfterUnlearn"
    fi
  done

  echo -e "\n✅ 全部运行完成。"

done