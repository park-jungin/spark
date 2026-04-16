#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_ROLLOUT_LIBERO_PYTHONPATH=""
if [[ -d "${ROOT_DIR}/../VLA-Adapter/LIBERO/libero/libero" ]]; then
  DEFAULT_ROLLOUT_LIBERO_PYTHONPATH="${ROOT_DIR}/../VLA-Adapter/LIBERO"
fi

# Force local code resolution first (avoid picking another installed `prismatic` package).
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ===== User-overridable settings (via env vars) =====
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOCAL_RANKS_FILTER="${LOCAL_RANKS_FILTER:-0}"       # Show stdout/stderr only from these local ranks (e.g. 0 or 0,1)
QUIET_NON_MAIN_RANKS="${QUIET_NON_MAIN_RANKS:-True}" # If True, suppress stdout/stderr from non-main ranks in Python
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
VGGT_PATH="${VGGT_PATH:-https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt}"
AUTO_DOWNLOAD_VGGT="${AUTO_DOWNLOAD_VGGT:-False}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-data/libero/}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-5000}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-ckpts/training_results/}"
RUN_ID="${RUN_ID:-attn-align-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/log}"
RESUME="${RESUME:-False}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"

ALIGN_ALL_LAYERS="${ALIGN_ALL_LAYERS:-True}"
USE_VGGT_ALIGNMENT="${USE_VGGT_ALIGNMENT:-True}"
OPENVLA_BASELINE="${OPENVLA_BASELINE:-False}"
VLA_LAYERS_ALIGN="${VLA_LAYERS_ALIGN:--1}"
VGGT_LAYERS_ALIGN="${VGGT_LAYERS_ALIGN:--1}"
ALIGN_LOSS_TYPE="${ALIGN_LOSS_TYPE:-l1}"           # mse | l1 | kl
ALIGN_LOSS_COEFF="${ALIGN_LOSS_COEFF:-0.5}"
ATTN_ALIGN_HIDDEN_DIM="${ATTN_ALIGN_HIDDEN_DIM:-512}" # Compatibility arg; aligner now operates in VGGT dim
ATTN_ALIGN_TEMPERATURE="${ATTN_ALIGN_TEMPERATURE:-1.0}"
USE_SIGLIP_ONLY_VISION="${USE_SIGLIP_ONLY_VISION:-False}"
USE_DINO_ONLY_VISION="${USE_DINO_ONLY_VISION:-True}"
VLA_ALIGNMENT_BRANCH="${VLA_ALIGNMENT_BRANCH:-dino}" # auto | siglip | dino
TRAIN_VLA_PROJECTOR="${TRAIN_VLA_PROJECTOR:-False}"
FREEZE_BASE_VISUAL_PATH="${FREEZE_BASE_VISUAL_PATH:-True}"
VISUAL_PATH_MODE="${VISUAL_PATH_MODE:-dual}"       # dual | base_only | expert_only | base_only_separate | expert_only_separate
USE_KNOWLEDGE_ROUTER="${USE_KNOWLEDGE_ROUTER:-True}"
KNOWLEDGE_ROUTER_NUM_HEADS="${KNOWLEDGE_ROUTER_NUM_HEADS:-8}"
KNOWLEDGE_ROUTER_HIDDEN_DIM="${KNOWLEDGE_ROUTER_HIDDEN_DIM:-128}"
KNOWLEDGE_ROUTER_DROPOUT="${KNOWLEDGE_ROUTER_DROPOUT:-0.0}"
KNOWLEDGE_ROUTER_TEMPERATURE="${KNOWLEDGE_ROUTER_TEMPERATURE:-1.0}"
KNOWLEDGE_ROUTER_TARGET_KEEP_RATIO="${KNOWLEDGE_ROUTER_TARGET_KEEP_RATIO:-0.6}"
KNOWLEDGE_ROUTER_MIN_KEEP_TOKENS="${KNOWLEDGE_ROUTER_MIN_KEEP_TOKENS:-8}"
KNOWLEDGE_ROUTER_HARD_ROUTING="${KNOWLEDGE_ROUTER_HARD_ROUTING:-False}"
KNOWLEDGE_ROUTER_LOSS_COEFF="${KNOWLEDGE_ROUTER_LOSS_COEFF:-0.5}"
KNOWLEDGE_ROUTER_BUDGET_LOSS_COEFF="${KNOWLEDGE_ROUTER_BUDGET_LOSS_COEFF:-0.05}"
KNOWLEDGE_ROUTER_ENTROPY_LOSS_COEFF="${KNOWLEDGE_ROUTER_ENTROPY_LOSS_COEFF:-0.001}"
KNOWLEDGE_ROUTER_WARMUP_STEPS="${KNOWLEDGE_ROUTER_WARMUP_STEPS:-500}"
KNOWLEDGE_ROUTER_FOCAL_GAMMA="${KNOWLEDGE_ROUTER_FOCAL_GAMMA:-2.0}"
KNOWLEDGE_ROUTER_EFFECTIVE_NUM_BETA="${KNOWLEDGE_ROUTER_EFFECTIVE_NUM_BETA:-0.999}"
KNOWLEDGE_ROUTER_TOKEN_FUSION_MODE="${KNOWLEDGE_ROUTER_TOKEN_FUSION_MODE:-no_fusion}"

USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"

BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
SCHEDULER="${SCHEDULER:-CosineAnnealingLR}"         # "MultiStepLR" or "CosineAnnealingLR"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"             # Optional linear LR warmup steps before scheduler stepping
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
MAX_STEPS="${MAX_STEPS:-150005}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SCALE_STEPS_BY_NPROC="${SCALE_STEPS_BY_NPROC:-False}" # If True, divide MAX_STEPS/SAVE_FREQ by NPROC_PER_NODE
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
IMAGE_AUG="${IMAGE_AUG:-True}"
LORA_RANK="${LORA_RANK:-32}"
VISION_LORA="${VISION_LORA:-True}"
RESTRICT_LORA_TO_VISION="${RESTRICT_LORA_TO_VISION:-False}"
FREEZE_VISION_LORA="${FREEZE_VISION_LORA:-False}"
USE_VAL_SET="${USE_VAL_SET:-False}"
VAL_FREQ="${VAL_FREQ:-5000}"
VAL_TIME_LIMIT="${VAL_TIME_LIMIT:-180}"
USE_ROLLOUT_VAL="${USE_ROLLOUT_VAL:-False}"
ROLLOUT_VAL_FREQ="${ROLLOUT_VAL_FREQ:-5000}"
ROLLOUT_TASK_SUITE_NAME="${ROLLOUT_TASK_SUITE_NAME:-}"
if [[ -z "${ROLLOUT_TASK_SUITE_NAME}" ]]; then
  case "${DATASET_NAME}" in
    libero_spatial*) ROLLOUT_TASK_SUITE_NAME="libero_spatial" ;;
    libero_object*) ROLLOUT_TASK_SUITE_NAME="libero_object" ;;
    libero_goal*) ROLLOUT_TASK_SUITE_NAME="libero_goal" ;;
    libero_10*) ROLLOUT_TASK_SUITE_NAME="libero_10" ;;
    *)
      ROLLOUT_TASK_SUITE_NAME="libero_spatial"
      echo "[Warning] Could not infer ROLLOUT_TASK_SUITE_NAME from DATASET_NAME=${DATASET_NAME}; defaulting to libero_spatial."
      ;;
  esac
fi
ROLLOUT_NUM_TRIALS_PER_TASK="${ROLLOUT_NUM_TRIALS_PER_TASK:-50}"
ROLLOUT_MAX_TASKS="${ROLLOUT_MAX_TASKS:-1}"
ROLLOUT_SAVE_VIDEOS="${ROLLOUT_SAVE_VIDEOS:-False}"
ROLLOUT_CENTER_CROP="${ROLLOUT_CENTER_CROP:-True}"
ROLLOUT_SEED="${ROLLOUT_SEED:-7}"
ROLLOUT_LOAD_IN_8BIT="${ROLLOUT_LOAD_IN_8BIT:-False}"
ROLLOUT_LOAD_IN_4BIT="${ROLLOUT_LOAD_IN_4BIT:-False}"
ROLLOUT_LOCAL_LOG_DIR="${ROLLOUT_LOCAL_LOG_DIR:-${ROOT_DIR}/log}"
ROLLOUT_LIBERO_PYTHONPATH="${ROLLOUT_LIBERO_PYTHONPATH:-${DEFAULT_ROLLOUT_LIBERO_PYTHONPATH}}"

WANDB_ENTITY="${WANDB_ENTITY:-YOUR_WANDB_ENTITY}"
WANDB_PROJECT="${WANDB_PROJECT:-YOUR_WANDB_PROJECT}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
CMD_FILE="${LOG_DIR}/${RUN_ID}.cmd"

RESUME_NORMALIZED="$(echo "${RESUME}" | tr '[:upper:]' '[:lower:]')"
RESUME_ENABLED="False"
if [[ "${RESUME_NORMALIZED}" == "1" || "${RESUME_NORMALIZED}" == "true" || "${RESUME_NORMALIZED}" == "yes" ]]; then
  RESUME_ENABLED="True"
fi

if [[ "${RESUME_ENABLED}" == "True" ]]; then
  if [[ -n "${RESUME_CHECKPOINT_DIR}" ]]; then
    RESUME_SOURCE_DIR="${RESUME_CHECKPOINT_DIR}"
  else
    RESUME_SOURCE_DIR="${VLA_PATH}"
  fi
else
  RESUME_SOURCE_DIR=""
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "A run with RUN_ID=${RUN_ID} is already active (pid=${OLD_PID})."
    echo "Use: kill ${OLD_PID}"
    exit 1
  fi
fi

# ===== Preflight checks =====
if [[ -d "${VLA_PATH}" ]]; then
  : # local checkpoint dir
elif [[ "${VLA_PATH}" =~ ^[^/]+/[^/]+$ ]]; then
  : # huggingface repo id
else
  echo "Invalid VLA_PATH: ${VLA_PATH}"
  echo "Expected either:"
  echo "  1) local directory (e.g. /abs/path/openvla-7b)"
  echo "  2) HF repo id (e.g. openvla/openvla-7b)"
  exit 1
fi

if [[ "${RESUME_ENABLED}" == "True" ]] && [[ ! -d "${RESUME_SOURCE_DIR}" ]]; then
  echo "Resume mode requires a local checkpoint directory."
  echo "Set RESUME_CHECKPOINT_DIR to your saved trainable-state folder."
  echo "Current resume source: ${RESUME_SOURCE_DIR}"
  exit 1
fi

if [[ "${RESUME_ENABLED}" == "True" ]] && [[ -z "${RESUME_STEP}" ]]; then
  LATEST_STEP_FILE="${RESUME_SOURCE_DIR}/latest_step.txt"
  if [[ -f "${LATEST_STEP_FILE}" ]]; then
    AUTO_STEP="$(tr -d '[:space:]' < "${LATEST_STEP_FILE}")"
    if [[ "${AUTO_STEP}" =~ ^[0-9]+$ ]]; then
      RESUME_STEP="${AUTO_STEP}"
      echo "Auto-detected RESUME_STEP=${RESUME_STEP} from ${LATEST_STEP_FILE}"
    else
      echo "Found ${LATEST_STEP_FILE}, but it does not contain a valid integer step."
      exit 1
    fi
  else
    AUTO_STEP="$(find "${RESUME_SOURCE_DIR}" -maxdepth 1 -type f \( -name 'fusion_projector--*_checkpoint.pt' -o -name 'single_path_projector--*_checkpoint.pt' -o -name 'action_head--*_checkpoint.pt' -o -name 'vla_projector--*_checkpoint.pt' -o -name 'knowledge_router--*_checkpoint.pt' \) -printf '%f\n' | sed -n 's/.*--\([0-9]\+\)_checkpoint\.pt$/\1/p' | sort -n | tail -n 1)"
    if [[ -n "${AUTO_STEP}" ]]; then
      RESUME_STEP="${AUTO_STEP}"
      echo "Auto-detected RESUME_STEP=${RESUME_STEP} from step-named checkpoints in ${RESUME_SOURCE_DIR}"
    fi
  fi
fi

if [[ "${RESUME_ENABLED}" == "True" ]] && [[ -z "${RESUME_STEP}" ]]; then
  echo "Resume mode requires RESUME_STEP (e.g., RESUME_STEP=5000)."
  echo "Could not auto-detect from ${RESUME_SOURCE_DIR}/latest_step.txt or step-named checkpoints."
  exit 1
fi

if [[ "${RESUME_ENABLED}" == "True" ]]; then
  FOUND_COMPONENT_FOR_STEP="False"
  for PREFIX in fusion_projector single_path_projector action_head proprio_projector attention_aligner noisy_action_projector vla_projector knowledge_router; do
    if [[ -f "${RESUME_SOURCE_DIR}/${PREFIX}--${RESUME_STEP}_checkpoint.pt" ]]; then
      FOUND_COMPONENT_FOR_STEP="True"
      break
    fi
  done
  if [[ "${FOUND_COMPONENT_FOR_STEP}" != "True" ]]; then
    AVAILABLE_STEPS="$(find "${RESUME_SOURCE_DIR}" -maxdepth 1 -type f -name '*--*_checkpoint.pt' -printf '%f\n' | sed -n 's/.*--\([0-9]\+\)_checkpoint\.pt$/\1/p' | sort -n | tr '\n' ' ')"
    if [[ -n "${AVAILABLE_STEPS}" ]]; then
      echo "Resume checkpoint for step ${RESUME_STEP} was not found in ${RESUME_SOURCE_DIR}."
      echo "Available step checkpoints: ${AVAILABLE_STEPS}"
      echo "Set RESUME_STEP to one of the available steps, or leave RESUME_STEP empty for auto-detection."
      exit 1
    elif compgen -G "${RESUME_SOURCE_DIR}/*--latest_checkpoint.pt" > /dev/null; then
      echo "Resume checkpoint for step ${RESUME_STEP} was not found in ${RESUME_SOURCE_DIR}."
      echo "Found latest-only checkpoints; training code will load *--latest_checkpoint.pt files for resume."
      echo "Proceeding with RESUME_STEP=${RESUME_STEP} for logging/scheduling continuity."
    elif [[ -d "${RESUME_SOURCE_DIR}/lora_adapter" ]]; then
      echo "No step-named component checkpoint found in ${RESUME_SOURCE_DIR}; proceeding with adapter-only resume."
    else
      echo "Could not find resume checkpoint files in ${RESUME_SOURCE_DIR}."
      exit 1
    fi
  fi
  if [[ ! -d "${RESUME_SOURCE_DIR}/lora_adapter" ]]; then
    echo "Resume requires LoRA adapter dir at: ${RESUME_SOURCE_DIR}/lora_adapter"
    exit 1
  fi
fi

# Catch a common misconfiguration: passing a training run dir that only contains LoRA adapter files.
if [[ "${RESUME_ENABLED}" != "True" ]] && [[ -d "${VLA_PATH}" ]] && [[ ! -f "${VLA_PATH}/config.json" ]]; then
  if [[ -f "${VLA_PATH}/lora_adapter/adapter_config.json" ]]; then
    BASE_FROM_ADAPTER="$(python - <<'PY'
import json, os
vla_path = os.environ.get("VLA_PATH", "")
cfg_path = os.path.join(vla_path, "lora_adapter", "adapter_config.json")
try:
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    print(cfg.get("base_model_name_or_path", ""))
except Exception:
    print("")
PY
)"
    echo "Invalid VLA_PATH for base model loading: ${VLA_PATH}"
    echo "This directory looks like a training output with only LoRA adapter files (no config.json)."
    if [[ -n "${BASE_FROM_ADAPTER}" ]]; then
      echo "Use the base model path instead, e.g.:"
      echo "  VLA_PATH=\"${BASE_FROM_ADAPTER}\""
    fi
    echo "If you want to continue from this adapter, first merge/export a full checkpoint or implement adapter-resume loading."
    exit 1
  else
    echo "Invalid VLA_PATH: ${VLA_PATH}"
    echo "Local directory is missing required file: config.json"
    exit 1
  fi
fi

if [[ ! -f "${VGGT_PATH}" ]] && [[ ! "${VGGT_PATH}" =~ ^https?:// ]]; then
  if [[ "${AUTO_DOWNLOAD_VGGT}" == "True" ]]; then
    mkdir -p "$(dirname "${VGGT_PATH}")"
    python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/VGGT-1B",
    filename="model.pt",
    local_dir="$(dirname "${VGGT_PATH}")",
    local_dir_use_symlinks=False,
)
PY
  fi
  if [[ ! -f "${VGGT_PATH}" ]] && [[ ! "${VGGT_PATH}" =~ ^https?:// ]]; then
    echo "VGGT checkpoint not found: ${VGGT_PATH}"
    echo "Set VGGT_PATH to either:"
    echo "  1) local .pt file (e.g. /abs/path/model.pt)"
    echo "  2) URL (e.g. https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt)"
    echo "Or auto-download once:"
    echo "  AUTO_DOWNLOAD_VGGT=True ./scripts/run_finetune_align_detached.sh"
    exit 1
  fi
fi

# Optional: keep roughly constant sample budget when changing GPU count
if [[ "${SCALE_STEPS_BY_NPROC}" == "True" ]]; then
  if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "Invalid NPROC_PER_NODE for scaling: ${NPROC_PER_NODE}"
    exit 1
  fi
  ORIG_MAX_STEPS="${MAX_STEPS}"
  ORIG_SAVE_FREQ="${SAVE_FREQ}"
  MAX_STEPS=$(( (MAX_STEPS + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
  SAVE_FREQ=$(( (SAVE_FREQ + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
  if [[ "${SAVE_FREQ}" -lt 1 ]]; then
    SAVE_FREQ=1
  fi
  echo "Scaled steps for NPROC_PER_NODE=${NPROC_PER_NODE}: MAX_STEPS ${ORIG_MAX_STEPS} -> ${MAX_STEPS}, SAVE_FREQ ${ORIG_SAVE_FREQ} -> ${SAVE_FREQ}"
fi

TORCHRUN_ARGS=(
  --standalone
  --nnodes 1
  --nproc-per-node "${NPROC_PER_NODE}"
)

# `--local-ranks-filter` is not available in older torchrun builds.
if torchrun --help 2>&1 | grep -q -- "--local-ranks-filter"; then
  TORCHRUN_ARGS+=(--local-ranks-filter "${LOCAL_RANKS_FILTER}")
else
  echo "torchrun without --local-ranks-filter detected; continuing without rank filter flag."
fi

CMD=(
  torchrun
  "${TORCHRUN_ARGS[@]}"
  vla-scripts/finetune_align.py
  --vla_path "${VLA_PATH}"
  --vggt_path "${VGGT_PATH}"
  --data_root_dir "${DATA_ROOT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --vla_layers_align "${VLA_LAYERS_ALIGN}"
  --vggt_layers_align "${VGGT_LAYERS_ALIGN}"
  --openvla_baseline "${OPENVLA_BASELINE}"
  --use_vggt_alignment "${USE_VGGT_ALIGNMENT}"
  --align_all_layers "${ALIGN_ALL_LAYERS}"
  --align_loss_type "${ALIGN_LOSS_TYPE}"
  --align_loss_coeff "${ALIGN_LOSS_COEFF}"
  --attn_align_hidden_dim "${ATTN_ALIGN_HIDDEN_DIM}"
  --attn_align_temperature "${ATTN_ALIGN_TEMPERATURE}"
  --use_siglip_only_vision "${USE_SIGLIP_ONLY_VISION}"
  --use_dino_only_vision "${USE_DINO_ONLY_VISION}"
  --vla_alignment_branch "${VLA_ALIGNMENT_BRANCH}"
  --train_vla_projector "${TRAIN_VLA_PROJECTOR}"
  --freeze_base_visual_path "${FREEZE_BASE_VISUAL_PATH}"
  --visual_path_mode "${VISUAL_PATH_MODE}"
  --use_knowledge_router "${USE_KNOWLEDGE_ROUTER}"
  --knowledge_router_num_heads "${KNOWLEDGE_ROUTER_NUM_HEADS}"
  --knowledge_router_hidden_dim "${KNOWLEDGE_ROUTER_HIDDEN_DIM}"
  --knowledge_router_dropout "${KNOWLEDGE_ROUTER_DROPOUT}"
  --knowledge_router_temperature "${KNOWLEDGE_ROUTER_TEMPERATURE}"
  --knowledge_router_target_keep_ratio "${KNOWLEDGE_ROUTER_TARGET_KEEP_RATIO}"
  --knowledge_router_min_keep_tokens "${KNOWLEDGE_ROUTER_MIN_KEEP_TOKENS}"
  --knowledge_router_hard_routing "${KNOWLEDGE_ROUTER_HARD_ROUTING}"
  --knowledge_router_loss_coeff "${KNOWLEDGE_ROUTER_LOSS_COEFF}"
  --knowledge_router_budget_loss_coeff "${KNOWLEDGE_ROUTER_BUDGET_LOSS_COEFF}"
  --knowledge_router_entropy_loss_coeff "${KNOWLEDGE_ROUTER_ENTROPY_LOSS_COEFF}"
  --knowledge_router_warmup_steps "${KNOWLEDGE_ROUTER_WARMUP_STEPS}"
  --knowledge_router_focal_gamma "${KNOWLEDGE_ROUTER_FOCAL_GAMMA}"
  --knowledge_router_effective_num_beta "${KNOWLEDGE_ROUTER_EFFECTIVE_NUM_BETA}"
  --knowledge_router_token_fusion_mode "${KNOWLEDGE_ROUTER_TOKEN_FUSION_MODE}"
  --use_l1_regression "${USE_L1_REGRESSION}"
  --use_diffusion "${USE_DIFFUSION}"
  --use_film "${USE_FILM}"
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
  --use_proprio "${USE_PROPRIO}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --scheduler "${SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
  --max_steps "${MAX_STEPS}"
  --save_freq "${SAVE_FREQ}"
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
  --image_aug "${IMAGE_AUG}"
  --lora_rank "${LORA_RANK}"
  --vision_lora "${VISION_LORA}"
  --restrict_lora_to_vision "${RESTRICT_LORA_TO_VISION}"
  --freeze_vision_lora "${FREEZE_VISION_LORA}"
  --use_val_set "${USE_VAL_SET}"
  --val_freq "${VAL_FREQ}"
  --val_time_limit "${VAL_TIME_LIMIT}"
  --use_rollout_val "${USE_ROLLOUT_VAL}"
  --rollout_val_freq "${ROLLOUT_VAL_FREQ}"
  --rollout_task_suite_name "${ROLLOUT_TASK_SUITE_NAME}"
  --rollout_num_trials_per_task "${ROLLOUT_NUM_TRIALS_PER_TASK}"
  --rollout_max_tasks "${ROLLOUT_MAX_TASKS}"
  --rollout_save_videos "${ROLLOUT_SAVE_VIDEOS}"
  --rollout_center_crop "${ROLLOUT_CENTER_CROP}"
  --rollout_seed "${ROLLOUT_SEED}"
  --rollout_load_in_8bit "${ROLLOUT_LOAD_IN_8BIT}"
  --rollout_load_in_4bit "${ROLLOUT_LOAD_IN_4BIT}"
  --rollout_local_log_dir "${ROLLOUT_LOCAL_LOG_DIR}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --run_id_override "${RUN_ID}"
  --resume "${RESUME_ENABLED}"
)

if [[ -n "${ROLLOUT_LIBERO_PYTHONPATH}" ]]; then
  CMD+=(--rollout_libero_pythonpath "${ROLLOUT_LIBERO_PYTHONPATH}")
fi

if [[ -n "${RESUME_STEP}" ]]; then
  CMD+=(--resume_step "${RESUME_STEP}")
fi

if [[ -n "${RESUME_CHECKPOINT_DIR}" ]]; then
  CMD+=(--resume_checkpoint_dir "${RESUME_CHECKPOINT_DIR}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CMD=(env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" QUIET_NON_MAIN_RANKS="${QUIET_NON_MAIN_RANKS}" "${CMD[@]}")
else
  CMD=(env QUIET_NON_MAIN_RANKS="${QUIET_NON_MAIN_RANKS}" "${CMD[@]}")
fi

printf '%q ' "${CMD[@]}" > "${CMD_FILE}"
echo >> "${CMD_FILE}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started detached training."
echo "  RUN_ID : ${RUN_ID}"
echo "  PID    : ${PID}"
echo "  NPROC  : ${NPROC_PER_NODE}"
echo "  RANKS  : ${LOCAL_RANKS_FILTER}"
echo "  QUIET  : ${QUIET_NON_MAIN_RANKS}"
echo "  VISION : siglip_only=${USE_SIGLIP_ONLY_VISION}, dino_only=${USE_DINO_ONLY_VISION}, align_branch=${VLA_ALIGNMENT_BRANCH}"
echo "  PROJ   : train_vla_projector=${TRAIN_VLA_PROJECTOR}"
echo "  ALIGN  : openvla_baseline=${OPENVLA_BASELINE}, use_vggt_alignment=${USE_VGGT_ALIGNMENT}, visual_path_mode=${VISUAL_PATH_MODE}"
echo "  ROUTER : enabled=${USE_KNOWLEDGE_ROUTER}, keep_ratio=${KNOWLEDGE_ROUTER_TARGET_KEEP_RATIO}, warmup=${KNOWLEDGE_ROUTER_WARMUP_STEPS}, hard=${KNOWLEDGE_ROUTER_HARD_ROUTING}, fusion=concat_projector (token_fusion_mode ignored=${KNOWLEDGE_ROUTER_TOKEN_FUSION_MODE})"
echo "  DATA   : dataset=${DATASET_NAME}, shuffle_buffer_size=${SHUFFLE_BUFFER_SIZE}"
echo "  STEPS  : ${MAX_STEPS}"
echo "  TRAIN  : lr=${LEARNING_RATE}, scheduler=${SCHEDULER}, warmup_steps=${LR_WARMUP_STEPS}, step_before_decay=${NUM_STEPS_BEFORE_DECAY}"
echo "  LORA   : rank=${LORA_RANK}, vision_lora=${VISION_LORA}, restrict_to_vision=${RESTRICT_LORA_TO_VISION}, freeze_vision_lora=${FREEZE_VISION_LORA}"
echo "  RESUME : enabled=${RESUME_ENABLED}, step=${RESUME_STEP:-<none>}, ckpt_dir=${RESUME_SOURCE_DIR:-<none>}"
echo "  VAL    : use_val_set=${USE_VAL_SET}, val_freq=${VAL_FREQ}, val_time_limit=${VAL_TIME_LIMIT}"
echo "  RVAL   : use_rollout_val=${USE_ROLLOUT_VAL}, freq=${ROLLOUT_VAL_FREQ}, suite=${ROLLOUT_TASK_SUITE_NAME}, trials/task=${ROLLOUT_NUM_TRIALS_PER_TASK}, max_tasks=${ROLLOUT_MAX_TASKS}, eval_8bit=${ROLLOUT_LOAD_IN_8BIT}, eval_4bit=${ROLLOUT_LOAD_IN_4BIT}"
echo "  LIBERO : rollout_libero_pythonpath=${ROLLOUT_LIBERO_PYTHONPATH:-<auto-detect-at-runtime>}"
echo "  LOG    : ${LOG_FILE}"
echo "  CMD    : ${CMD_FILE}"
echo
# echo "Monitor:"
# echo "  tail -f ${LOG_FILE}"
# echo "  kill -0 ${PID} && echo running || echo stopped"
# echo
# echo "Stop:"
# echo "  kill ${PID}"
