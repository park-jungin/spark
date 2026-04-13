STEP=21690
TASK=goal
SAVE_ROUTER_OVERLAY_VIDEOS=${SAVE_ROUTER_OVERLAY_VIDEOS:-True}
ROUTER_OVERLAY_ALPHA=${ROUTER_OVERLAY_ALPHA:-0.7}

export PYTHONPATH="/mnt/hdd2/vla/ours:/mnt/hdd2/vla/VLA-Adapter/LIBERO:${PYTHONPATH}"
export LIBERO_CONFIG_PATH="/mnt/hdd2/vla/ours/log/rollout_validation/.libero"

nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/hdd2/vla/ours/ckpts/training_results/${TASK}-suit/${TASK}-suit--${STEP}_chkpt \
  --base_model_path /mnt/hdd2/vla/ours/ckpts/pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b-hf \
  --openvla_baseline False \
  --task_suite_name libero_${TASK} \
  --num_trials_per_task 50 \
  --max_tasks -1 \
  --center_crop True \
  --save_rollout_videos False \
  --save_router_overlay_videos "${SAVE_ROUTER_OVERLAY_VIDEOS}" \
  --router_overlay_alpha "${ROUTER_OVERLAY_ALPHA}" \
  --local_log_dir /mnt/hdd2/vla/ours/log/rollout_validation \
  --seed 7 \
  --use_wandb False \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_dino_only_vision True \
  --use_siglip_only_vision False \
  --use_knowledge_router True \
  --knowledge_router_target_keep_ratio 0.7 \
  --knowledge_router_min_keep_tokens 250 \
  --vision_lora True \
  --train_vla_projector False \
  --visual_path_mode dual \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --load_in_8bit False \
  --load_in_4bit False \
  --run_id_note eval-${TASK}-${STEP} \
  > /mnt/hdd2/vla/ours/log/libero_eval_${TASK}_${STEP}.log 2>&1 &
