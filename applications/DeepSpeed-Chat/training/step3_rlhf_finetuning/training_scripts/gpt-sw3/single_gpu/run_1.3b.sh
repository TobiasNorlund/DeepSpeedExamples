#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

module load buildenv-gcccuda/11.4-system-nsc1
module load Anaconda/2021.05-nsc1 && conda activate /proj/nlg/users/x_tobno/envs/DeepSpeed-Chat

# DeepSpeed Team
ACTOR_MODEL_PATH="/proj/nlg/gpt-sw3/gpt-sw3-1.3b-instruct"
CRITIC_MODEL_PATH="../step2_reward_model_finetuning/runs/2023-08-28T16_27_34"
#CRITIC_MODEL_PATH="../step2_reward_model_finetuning/runs/2023-08-21T11_28_37"
ACTOR_ZERO_STAGE=0
CRITIC_ZERO_STAGE=0
OUTPUT=runs/$(date '+%Y-%m-%dT%H_%M_%S')
#if [ "$OUTPUT" == "" ]; then
#    OUTPUT=./output
#fi
#if [ "$ACTOR_ZERO_STAGE" == "" ]; then
#    ACTOR_ZERO_STAGE=0
#fi
#if [ "$CRITIC_ZERO_STAGE" == "" ]; then
#    CRITIC_ZERO_STAGE=0
#fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --num_train_epochs 10 \
   --deepspeed --actor_lora_dim 128 --actor_lora_module_name "transformer.h" --enable_hybrid_engine --actor_gradient_checkpointing --disable_actor_dropout \
   --enable_tensorboard --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT | tee $OUTPUT/training.log
