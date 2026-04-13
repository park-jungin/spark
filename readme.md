# SPARK (Spatial Representation Distillation and Knowledge Routing)

This repository contains SPARK modified from Spatial-Forcing code used for LIBERO experiments, including:
- attention-response alignment (VLA \u2194 VGGT),
- knowledge router with token pruning / overlay visualization,
- support for larger Prismatic backbones (including Qwen2.5-7B).

Main entry scripts:
- `./scripts/train.sh`
- `./scripts/resume.sh`
- `./scripts/evaluation.sh`

## 1) Setup

```bash
cd to_github
conda env create -f environment.yml
conda activate vla
pip install -e .
```

Install LIBERO runtime dependencies (if not already installed in this env):
```bash
pip install -r experiments/robot/libero/libero_requirements.txt
```

## 2) Data / Checkpoints

You need:
- RLDS LIBERO dataset root (e.g., `.../data/libero`)
- base Prismatic checkpoint (`VLA_PATH`)
- VGGT checkpoint (`VGGT_PATH`, local `.pt` or URL)

## 3) Important Path Note (Before Running)

`scripts/train.sh`, `scripts/resume.sh`, and `scripts/evaluation.sh` currently contain absolute paths from the original machine (e.g., `/mnt/hdd2/vla/ours/...`).

Before running, edit those paths to your local paths:
- `VLA_PATH`
- `DATA_ROOT_DIR`
- `LOG_DIR` / `ROLLOUT_LOCAL_LOG_DIR`
- `RESUME_CHECKPOINT_DIR` (resume only)
- `PYTHONPATH`, `LIBERO_CONFIG_PATH`, output log path (evaluation only)

## 4) Run

### 4.1 Start Training
```bash
bash scripts/train.sh
```

Edit in `scripts/train.sh`:
- `TASK` (`spatial | object | goal | 10`)
- `BATCH_SIZE`, `NPROC_PER_NODE`
- path values listed above

### 4.2 Resume Training
```bash
bash scripts/resume.sh
```

Edit in `scripts/resume.sh`:
- `TASK`
- `STEP` (checkpoint step to resume from)
- `RESUME_CHECKPOINT_DIR`

### 4.3 Evaluate
```bash
bash scripts/evaluation.sh
```

Edit in `scripts/evaluation.sh`:
- `TASK`
- `STEP`
- `--pretrained_checkpoint`, `--base_model_path`
- `PYTHONPATH`, `LIBERO_CONFIG_PATH`, and output log paths

## 5) Qwen2.5-7B Backbone

This code path supports larger Prismatic backbones.

To use Qwen2.5-7B:
- set `VLA_PATH` to your 7B Prismatic checkpoint (local dir or HF repo),
- keep the rest of the pipeline unchanged.

Practical note: 7B significantly increases memory usage; tune `BATCH_SIZE`, `NPROC_PER_NODE`, and optional quantization flags for rollout eval (`ROLLOUT_LOAD_IN_8BIT`, `ROLLOUT_LOAD_IN_4BIT`).

## 6) Attention Alignment Hyperparameters

These are controlled in `scripts/run_finetune_align_detached.sh` (env vars passed into `vla-scripts/finetune_align.py`):

- `USE_VGGT_ALIGNMENT`:
  enable/disable alignment objective.
- `ALIGN_ALL_LAYERS`:
  align all corresponding layers (`True`) vs selected pair (`False`).
- `VLA_LAYERS_ALIGN`, `VGGT_LAYERS_ALIGN`:
  layer indices used when `ALIGN_ALL_LAYERS=False`.
- `ALIGN_LOSS_TYPE`:
  `l1 | mse | kl`.
- `ALIGN_LOSS_COEFF`:
  weight of alignment loss in total loss.
- `VLA_ALIGNMENT_BRANCH`:
  which VLA visual branch for QKV alignment (`auto | dino | siglip`).
- `ATTN_ALIGN_TEMPERATURE`:
  temperature for KL-style alignment.
- `ATTN_ALIGN_HIDDEN_DIM`:
  compatibility arg (kept for checkpoint/CLI compatibility).

## 7) Knowledge Router Hyperparameters

- `USE_KNOWLEDGE_ROUTER`:
  enable router.
- `KNOWLEDGE_ROUTER_TARGET_KEEP_RATIO`:
  target ratio of kept candidate tokens.
- `KNOWLEDGE_ROUTER_MIN_KEEP_TOKENS`:
  lower bound for kept tokens.
- `KNOWLEDGE_ROUTER_HARD_ROUTING`:
  hard (STE) routing vs soft gating.
- `KNOWLEDGE_ROUTER_WARMUP_STEPS`:
  steps before router gating/loss activates.
- `KNOWLEDGE_ROUTER_LOSS_COEFF`:
  classification/supervision loss weight.
- `KNOWLEDGE_ROUTER_BUDGET_LOSS_COEFF`:
  keep-ratio budget regularization weight.
- `KNOWLEDGE_ROUTER_ENTROPY_LOSS_COEFF`:
  entropy regularizer weight.
- `KNOWLEDGE_ROUTER_NUM_HEADS`, `KNOWLEDGE_ROUTER_HIDDEN_DIM`, `KNOWLEDGE_ROUTER_DROPOUT`, `KNOWLEDGE_ROUTER_TEMPERATURE`:
  router architecture / gating dynamics.
- `KNOWLEDGE_ROUTER_FOCAL_GAMMA`, `KNOWLEDGE_ROUTER_EFFECTIVE_NUM_BETA`:
  focal + effective-number weighting parameters for pseudo-mask supervision.

Additional router detail in code:
- `knowledge_router_importance_ema_momentum` is defined in `vla-scripts/finetune_align.py` (default `0.9`) for smoothing gradient-based token-importance pseudo-labels.

## 8) Evaluation Overlay Semantics

Current overlay logic:
- selected tokens: unmasked (original image),
- not-selected tokens: white mask overlay (`router_overlay_alpha` controls strength).

So white regions indicate tokens pruned by the router.

## 9) Directory Notes

- Third-party LIBERO source is vendored under `third_party/LIBERO`.
- Large artifacts (logs/checkpoints/datasets/wandb runs) are intentionally excluded from this package.
