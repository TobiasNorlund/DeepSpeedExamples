#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

module load buildenv-gcccuda/11.4-system-nsc1
module load Anaconda/2021.05-nsc1 && conda activate /proj/nlg/users/x_tobno/envs/DeepSpeed-Chat


# DeepSpeed Team
#ACTOR_MODEL_PATH=$1
#CRITIC_MODEL_PATH=$2
#ACTOR_ZERO_STAGE=$3
#CRITIC_ZERO_STAGE=$4
OUTPUT=output-gpt-sw3

mkdir -p $OUTPUT

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path /proj/nlg/gpt-sw3/gpt-sw3-1.3b-instruct/ \
   --critic_model_name_or_path ../step2_reward_model_finetuning/output/ \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 128 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 0 \
   --critic_zero_stage 0 \
   --enable_ema \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT
