TASK=spatial

BATCH_SIZE=16
NPROC_PER_NODE=3
GRAD_ACCUM=1
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM ))


declare -A LR=(
  [spatial]=5e-5
  [object]=5e-5
  [goal]=5e-5
  [10]=1e-5
)

declare -A NS=(
  [spatial]=52970
  [object]=66984
  [goal]=52042
  [10]=101469
)

LEARNING_RATE=${LR[$TASK]:-5e-5}
N_SAMPLES=${NS[$TASK]:-0}

if (( N_SAMPLES > 0 )); then
  # ceil(N_SAMPLES / GLOBAL_BATCH)
  SAVE_FREQ=$(( (N_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
else
  SAVE_FREQ=2000
fi

RUN_ID=${TASK}-suit \
LOCAL_RANKS_FILTER=0 \
NPROC_PER_NODE=${NPROC_PER_NODE} \
SHUFFLE_BUFFER_SIZE=10000 \
QUIET_NON_MAIN_RANKS=True \
VLA_PATH="/mnt/hdd2/vla/ours/ckpts/pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b-hf" \
VGGT_PATH="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt" \
DATA_ROOT_DIR="/mnt/hdd2/vla/ours/data/libero" \
DATASET_NAME="libero_${TASK}_no_noops" \
LEARNING_RATE=${LEARNING_RATE} \
LOG_DIR="/mnt/hdd2/vla/ours/log" \
OPENVLA_BASELINE=False \
USE_SIGLIP_ONLY_VISION=False \
USE_DINO_ONLY_VISION=True \
USE_VGGT_ALIGNMENT=True \
VISION_LORA=True \
ALIGN_ALL_LAYERS=True \
USE_L1_REGRESSION=True \
VISUAL_PATH_MODE=dual \
VLA_ALIGNMENT_BRANCH=dino \
TRAIN_VLA_PROJECTOR=False \
RESTRICT_LORA_TO_VISION=False \
USE_KNOWLEDGE_ROUTER=True \
KNOWLEDGE_ROUTER_WARMUP_STEPS=${SAVE_FREQ} \
SAVE_FREQ="${SAVE_FREQ}" \
SAVE_LATEST_CHECKPOINT_ONLY=False \
USE_VAL_SET=False \
VAL_TIME_LIMIT=-1 \
USE_ROLLOUT_VAL=True \
ROLLOUT_NUM_TRIALS_PER_TASK=50 \
ROLLOUT_MAX_TASKS=-1 \
ROLLOUT_VAL_FREQ=${SAVE_FREQ} \
ROLLOUT_TASK_SUITE_NAME="libero_${TASK}" \
bash scripts/run_finetune_align_detached.sh
