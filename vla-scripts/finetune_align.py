"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""

import os
import re
import json
import shutil
import subprocess
import sys
import time
import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import numpy as np
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
os.environ["WANDB_MODE"]="offline"

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    AttentionResponseAligner,
    DualPathFusionProjector,
    KnowledgeRouter,
    NoisyActionProjector,
    ProprioProjector,
    SinglePathProjector,
)
import prismatic.extern.hf.modeling_prismatic as hf_modeling_prismatic_module
import prismatic.training.train_utils as train_utils_module
import prismatic.vla.constants as vla_constants_module
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_normed_images

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    vggt_path: str = 'official_ckpts/vggt_model.pt'  # Path to VGGT model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 10000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    openvla_baseline: bool = False                  # If True, run OpenVLA baseline visual path (no VGGT alignment / dual-path projectors)
    use_vggt_alignment: bool = True                  # If True, use VGGT teacher and attention-response alignment loss
    align_loss_type: str = "l1"                      # Loss for attention-response alignment: "mse", "l1", "kl"
    align_loss_coeff: float = 0.5                    # Coefficient for alignment loss (multiplied by align_loss)
    align_all_layers: bool = True                    # If True, align all encoder layers; otherwise align chosen layer pair
    attn_align_hidden_dim: int = 512                 # Deprecated for aligner internals (kept for CLI/checkpoint compatibility)
    attn_align_temperature: float = 1.0              # Temperature used when align_loss_type="kl"
    use_siglip_only_vision: bool = False             # If True, use SigLIP branch only (ignore DINO features) for dual-path vision
    use_dino_only_vision: bool = True                # If True, use DINO branch only (ignore SigLIP features) for dual-path vision
    vla_alignment_branch: str = "dino"               # VLA branch for QKV alignment: "auto" | "siglip" | "dino"
    train_vla_projector: bool = False                # If True, train OpenVLA built-in vision projector on concatenated DINO+SigLIP tokens
    freeze_base_visual_path: bool = True             # If True, base visual path runs in no_grad for preserving general features
    visual_path_mode: str = "dual"                   # Visual mode: dual | base_only | expert_only | base_only_separate | expert_only_separate
    use_knowledge_router: bool = False               # If True, route visual/expert tokens before feeding them into the LLM
    knowledge_router_num_heads: int = 8              # Router cross-attention heads
    knowledge_router_hidden_dim: int = 128           # Router MLP hidden width (Dr.LLM-style default)
    knowledge_router_dropout: float = 0.0            # Router dropout
    knowledge_router_temperature: float = 1.0        # Router sigmoid temperature
    knowledge_router_target_keep_ratio: float = 0.7  # Target ratio of tokens to keep
    knowledge_router_min_keep_tokens: int = 8        # Minimum number of tokens to keep
    knowledge_router_hard_routing: bool = False      # If True, use STE hard routing masks; else soft routing
    knowledge_router_loss_coeff: float = 1.0         # Weight on router focal supervision loss
    knowledge_router_budget_loss_coeff: float = 0.1  # Weight on keep-ratio budget loss
    knowledge_router_entropy_loss_coeff: float = 0.1 # Weight on entropy regularizer (positive -> avoid collapse)
    knowledge_router_warmup_steps: int = 500         # Warmup steps before enabling router gating/loss
    knowledge_router_importance_ema_momentum: float = 0.0  # EMA momentum for gradient-based token-importance pseudo labels
    knowledge_router_focal_gamma: float = 2.0        # Focal loss gamma (Dr.LLM-style)
    knowledge_router_effective_num_beta: float = 0.999  # Effective-number reweighting beta (Dr.LLM-style)
    knowledge_router_token_fusion_mode: str = "no_fusion"  # Deprecated (ignored): kept for CLI/checkpoint compatibility
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    vla_layers_align: int = -1                       # Selected OpenVLA vision layer index for alignment (if not align_all_layers)
    vggt_layers_align: int = -1                      # Selected VGGT vision layer index for alignment (if not align_all_layers)

    # Training configuration
    batch_size: int = 4                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 5000                            # Checkpoint saving frequency in steps
    use_rollout_val: bool = False                    # If True, periodically run LIBERO rollout validation (success rate)
    rollout_val_freq: int = 10_000                   # (When `use_rollout_val==True`) Rollout validation frequency in steps
    rollout_task_suite_name: str = "libero_spatial"  # LIBERO suite for rollout validation
    rollout_num_trials_per_task: int = 2             # Number of rollouts per task during rollout validation
    rollout_max_tasks: int = 2                       # Max number of tasks to evaluate during rollout validation (-1: all)
    rollout_save_videos: bool = False                # If True, save rollout videos during rollout validation
    rollout_center_crop: bool = True                 # Whether to center-crop observations during rollout validation
    rollout_local_log_dir: str = "experiments/logs"  # Base local directory for rollout validation logs
    rollout_seed: int = 7                            # Random seed for rollout validation
    rollout_load_in_8bit: bool = False                # If True, run rollout eval model loading in 8-bit for lower memory
    rollout_load_in_4bit: bool = False               # If True, run rollout eval model loading in 4-bit for lower memory
    rollout_libero_pythonpath: Optional[str] = None  # Optional PYTHONPATH entry pointing to local LIBERO package root
    save_latest_checkpoint_only: bool = True        # If True, saves only 1 checkpoint, overwriting latest checkpoint
    scheduler: str = 'MultiStepLR'                   # "MultiStepLR" or "CosineAnnealingLR"
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    resume_checkpoint_dir: Optional[str] = None      # Optional dir with saved trainable checkpoints (defaults to vla_path)
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    vision_lora: bool = True                         # If True, apply vision-side LoRAs in forward pass
    restrict_lora_to_vision: bool = False           # If True, train only vision-side LoRAs; if False, train all LoRAs (vision + LLM)
    freeze_vision_lora: bool = False                # If True, freeze vision-side LoRAs (train only non-vision LoRAs)
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def is_main_process() -> bool:
    """Return True on global rank 0 (or when distributed is not initialized)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def maybe_smooth_router_importance_with_ema(
    router_module: nn.Module,
    token_importance: torch.Tensor,
    momentum: float,
) -> torch.Tensor:
    """
    EMA-smooth per-token-position importance scores across steps on each process.
    """
    if momentum <= 0.0:
        return torch.nan_to_num(token_importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    token_importance = torch.nan_to_num(token_importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    batch_position_mean = token_importance.mean(dim=0, keepdim=True).detach()
    ema_state = getattr(router_module, "_action_importance_ema", None)
    if (
        ema_state is None
        or (not torch.is_tensor(ema_state))
        or ema_state.shape != batch_position_mean.shape
        or ema_state.device != token_importance.device
    ):
        ema_state = batch_position_mean
    else:
        ema_state = ema_state.to(device=token_importance.device, dtype=token_importance.dtype)
        ema_state = ema_state * float(momentum) + batch_position_mean * (1.0 - float(momentum))

    ema_state = torch.nan_to_num(ema_state, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    setattr(router_module, "_action_importance_ema", ema_state.detach())
    return (ema_state.expand_as(token_importance) * float(momentum) + token_importance * (1.0 - float(momentum))).clamp_min(
        0.0
    )


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = Path(get_resume_checkpoint_dir(cfg)).name
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def get_resume_checkpoint_dir(cfg) -> str:
    """
    Return checkpoint directory used for loading resumed trainable states.

    If `resume_checkpoint_dir` is not provided, falls back to `vla_path` for backward compatibility.
    """
    if cfg.resume_checkpoint_dir is not None and str(cfg.resume_checkpoint_dir).strip() != "":
        return str(cfg.resume_checkpoint_dir).rstrip("/")
    return cfg.vla_path


def resolve_checkpoint_path(module_name: str, path: str, step: int) -> str:
    """
    Resolve a checkpoint path for `module_name` at `step`, with fallback to latest.
    """
    step_checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    latest_checkpoint_path = os.path.join(path, f"{module_name}--latest_checkpoint.pt")

    if os.path.isfile(step_checkpoint_path):
        if is_main_process():
            print(f"Loading checkpoint: {step_checkpoint_path}")
        return step_checkpoint_path

    if os.path.isfile(latest_checkpoint_path):
        if is_main_process():
            print(
                f"Loading checkpoint: {step_checkpoint_path} (missing) -> "
                f"falling back to latest checkpoint `{latest_checkpoint_path}`"
            )
        return latest_checkpoint_path

    raise FileNotFoundError(
        f"Could not find checkpoint for module `{module_name}` at either "
        f"`{step_checkpoint_path}` or `{latest_checkpoint_path}`"
    )


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = resolve_checkpoint_path(module_name, path, step)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def load_raw_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> Any:
    """
    Load a non-module checkpoint object (e.g., optimizer/scheduler state dict).
    """
    checkpoint_path = resolve_checkpoint_path(module_name, path, step)
    return torch.load(checkpoint_path, map_location=device)


def _move_nested_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensor leaves in a nested container to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _move_nested_to_device(value, device)
        return obj
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_nested_to_device(value, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_nested_to_device(value, device) for value in obj)
    return obj


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move all tensor states in an optimizer to the target device."""
    for state in optimizer.state.values():
        _move_nested_to_device(state, device)


def sync_action_token_begin_idx(action_tokenizer: ActionTokenizer) -> int:
    """Synchronize action-token range across modules to support different tokenizers."""
    action_token_begin_idx = int(action_tokenizer.action_token_begin_idx)
    vla_constants_module.ACTION_TOKEN_BEGIN_IDX = action_token_begin_idx
    train_utils_module.ACTION_TOKEN_BEGIN_IDX = action_token_begin_idx
    hf_modeling_prismatic_module.ACTION_TOKEN_BEGIN_IDX = action_token_begin_idx
    if is_main_process():
        print(f"Set ACTION_TOKEN_BEGIN_IDX={action_token_begin_idx}")
    return action_token_begin_idx


def load_vggt_state_dict(vggt_path: str) -> dict:
    """
    Load VGGT checkpoint from a local path or URL.
    """
    if vggt_path.startswith("http://") or vggt_path.startswith("https://"):
        if is_main_process():
            print(f"Loading VGGT checkpoint from URL: {vggt_path}")
        checkpoint = torch.hub.load_state_dict_from_url(vggt_path, map_location="cpu", progress=True)
    else:
        ckpt_path = Path(vggt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"VGGT checkpoint not found at `{vggt_path}`. "
                f"Pass a valid local path or a URL like "
                f"`https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`."
            )
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    return remove_ddp_in_checkpoint(checkpoint)


def _has_hf_pretrained_weights(model_dir: Path) -> bool:
    """Return True if `model_dir` looks like an HF `from_pretrained` checkpoint directory."""
    if not model_dir.is_dir():
        return False
    if (model_dir / "model.safetensors").exists():
        return True
    if (model_dir / "pytorch_model.bin").exists():
        return True
    if (model_dir / "model.safetensors.index.json").exists():
        return True
    if any(model_dir.glob("model-*.safetensors")):
        return True
    return False


def _native_prismatic_checkpoint_path(model_dir: Path) -> Optional[Path]:
    """Pick a native Prismatic checkpoint file from `<model_dir>/checkpoints/`."""
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    latest = checkpoints_dir / "latest-checkpoint.pt"
    if latest.exists():
        return latest

    step_ckpts = list(checkpoints_dir.glob("step-*.pt"))
    if not step_ckpts:
        return None

    def _step_num(path: Path) -> int:
        match = re.search(r"step-(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(step_ckpts, key=_step_num)


def _is_native_prismatic_dir(model_dir: Path) -> bool:
    """Return True for native Prismatic run folders (`config.json` + `checkpoints/*.pt`)."""
    config_path = model_dir / "config.json"
    checkpoint_path = _native_prismatic_checkpoint_path(model_dir)
    if not config_path.exists() or checkpoint_path is None:
        return False

    try:
        with open(config_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return False

    return isinstance(payload, dict) and isinstance(payload.get("model"), dict)


def _convert_native_prismatic_dir_to_hf(
    native_dir: Path,
    output_dir: Optional[Path] = None,
    return_artifacts_only: bool = False,
):
    """
    Convert/load a native Prismatic checkpoint directory into HF-compatible artifacts.

    This mirrors the logic from `scripts/extern/convert_prismatic_weights_to_hf.py`,
    but runs inline without CLI dependencies.
    """
    if return_artifacts_only and output_dir is not None:
        raise ValueError("`output_dir` must be None when `return_artifacts_only=True`.")
    if not return_artifacts_only and output_dir is None:
        raise ValueError("`output_dir` is required when `return_artifacts_only=False`.")
    import timm
    from timm.models.vision_transformer import LayerScale
    from transformers import AutoConfig as HFAutoConfig
    from transformers import AutoTokenizer
    from prismatic.extern.hf.configuration_prismatic import LLM_BACKBONE_TO_HF_PATH, PrismaticConfig
    from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    config_path = native_dir / "config.json"
    checkpoint_pt = _native_prismatic_checkpoint_path(native_dir)
    if checkpoint_pt is None:
        raise FileNotFoundError(f"No native Prismatic checkpoint found under `{native_dir / 'checkpoints'}`")

    with open(config_path, "r") as f:
        prismatic_config = json.load(f)["model"]

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        llm_backbone_id = prismatic_config["llm_backbone_id"]
        llm_hf_id = LLM_BACKBONE_TO_HF_PATH[llm_backbone_id]
        tokenizer = AutoTokenizer.from_pretrained(
            llm_hf_id,
            model_max_length=prismatic_config["llm_max_length"],
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.init_kwargs.pop("add_prefix_space", None)

        llm_text_config = HFAutoConfig.from_pretrained(llm_hf_id).to_dict()

        hf_config = PrismaticConfig(
            vision_backbone_id=prismatic_config["vision_backbone_id"],
            llm_backbone_id=llm_backbone_id,
            arch_specifier=prismatic_config["arch_specifier"],
            image_resize_strategy=prismatic_config["image_resize_strategy"],
            text_config=llm_text_config,
            llm_max_length=prismatic_config["llm_max_length"],
            pad_token_id=int(tokenizer.pad_token_id),
            torch_dtype=torch.bfloat16,
        )
        hf_config.text_config.pad_token_id = hf_config.pad_token_id
        hf_config.text_config.torch_dtype = torch.bfloat16

        # HF transformers rewrites LayerScale.gamma; patch to a stable parameter name before state export.
        def _ls_forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor

        def _patch_layerscale(module: LayerScale) -> None:
            module.scale_factor = nn.Parameter(module.gamma.clone())
            module.forward = _ls_forward.__get__(module, LayerScale)
            del module.gamma

        input_sizes, interpolations, means, stds = [], [], [], []
        for idx, timm_model_id in enumerate(hf_config.timm_model_ids):
            timm_vision_backbone = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                img_size=hf_config.image_sizes[idx],
                act_layer=hf_config.timm_override_act_layers[idx],
            )

            data_cfg = timm.data.resolve_model_data_config(timm_vision_backbone)
            input_sizes.append((3, hf_config.image_sizes[idx], hf_config.image_sizes[idx]))
            interpolations.append(data_cfg["interpolation"])
            means.append(data_cfg["mean"])
            stds.append(data_cfg["std"])

            for module in timm_vision_backbone.modules():
                if isinstance(module, LayerScale):
                    _patch_layerscale(module)

            if idx == 0:
                first_vision_backbone = timm_vision_backbone
            else:
                second_vision_backbone = timm_vision_backbone

        hf_image_processor = PrismaticImageProcessor(
            use_fused_vision_backbone=hf_config.use_fused_vision_backbone,
            image_resize_strategy=hf_config.image_resize_strategy,
            input_sizes=input_sizes,
            interpolations=interpolations,
            means=means,
            stds=stds,
        )
        hf_processor = PrismaticProcessor(image_processor=hf_image_processor, tokenizer=tokenizer)

        model_state_dict = torch.load(checkpoint_pt, map_location="cpu")
        model_state_dict = (
            model_state_dict["model"] if isinstance(model_state_dict, dict) and "model" in model_state_dict else model_state_dict
        )
        if not isinstance(model_state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in `{checkpoint_pt}`")
        if "projector" not in model_state_dict or "llm_backbone" not in model_state_dict:
            raise ValueError(f"Missing expected keys (`projector`, `llm_backbone`) in `{checkpoint_pt}`")

        hf_model = PrismaticForConditionalGeneration(hf_config)

        projector_key_mapping = {
            "projector.0.weight": "fc1.weight",
            "projector.0.bias": "fc1.bias",
            "projector.2.weight": "fc2.weight",
            "projector.2.bias": "fc2.bias",
            "projector.4.weight": "fc3.weight",
            "projector.4.bias": "fc3.bias",
        }
        projector_state_dict = {}
        for key, value in model_state_dict["projector"].items():
            if key not in projector_key_mapping:
                raise KeyError(f"Unexpected projector key `{key}` in `{checkpoint_pt}`")
            projector_state_dict[projector_key_mapping[key]] = value
        hf_model.projector.load_state_dict(projector_state_dict, strict=True)
        del projector_state_dict

        language_model_state_dict = {}
        for key, value in model_state_dict["llm_backbone"].items():
            if not key.startswith("llm."):
                raise KeyError(f"Unexpected LLM key `{key}` in `{checkpoint_pt}`")
            language_model_state_dict[key[len("llm.") :]] = value
        hf_model.language_model.load_state_dict(language_model_state_dict, strict=True)
        del language_model_state_dict
        del model_state_dict

        hf_model.vision_backbone.featurizer.load_state_dict(first_vision_backbone.state_dict(), strict=True)
        del first_vision_backbone
        if hf_config.use_fused_vision_backbone:
            hf_model.vision_backbone.fused_featurizer.load_state_dict(second_vision_backbone.state_dict(), strict=True)
            del second_vision_backbone

        hf_model.to(torch.bfloat16)

        # Register auto classes so code files are saved alongside the checkpoint for trust_remote_code.
        PrismaticConfig.register_for_auto_class()
        PrismaticImageProcessor.register_for_auto_class("AutoImageProcessor")
        PrismaticProcessor.register_for_auto_class("AutoProcessor")
        PrismaticForConditionalGeneration.register_for_auto_class("AutoModelForVision2Seq")

        if return_artifacts_only:
            return hf_processor, hf_model

        tmp_output_dir = output_dir.parent / f".{output_dir.name}.tmp"
        if tmp_output_dir.exists():
            shutil.rmtree(tmp_output_dir)
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        hf_model.save_pretrained(tmp_output_dir, max_shard_size="7GB")
        hf_image_processor.save_pretrained(tmp_output_dir)
        hf_processor.save_pretrained(tmp_output_dir)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_output_dir.rename(output_dir)
    finally:
        torch.set_default_dtype(default_dtype)


def maybe_convert_native_prismatic_checkpoint(
    resolved_vla_path: str,
    source_vla_path: str,
    run_root_dir: Path,
    distributed_state: PartialState,
) -> str:
    """
    If `resolved_vla_path` is a native Prismatic checkpoint, convert it to HF format and return converted path.
    Otherwise, return the original path unchanged.
    """
    resolved_dir = Path(resolved_vla_path)
    if _has_hf_pretrained_weights(resolved_dir):
        return resolved_vla_path
    if not _is_native_prismatic_dir(resolved_dir):
        return resolved_vla_path

    source_name = re.sub(r"[^A-Za-z0-9._-]+", "__", source_vla_path.strip("/"))
    converted_root = (run_root_dir.parent if run_root_dir.parent != Path(".") else Path("ckpts")) / "hf_converted"
    converted_dir = converted_root / source_name

    if distributed_state.is_main_process:
        if _has_hf_pretrained_weights(converted_dir):
            print(f"[Prismatic Convert] Reusing cached converted checkpoint: {converted_dir}")
        else:
            print(
                "[Prismatic Convert] Native Prismatic checkpoint detected. "
                f"Converting `{resolved_dir}` -> `{converted_dir}` ..."
            )
            converted_root.mkdir(parents=True, exist_ok=True)
            _convert_native_prismatic_dir_to_hf(resolved_dir, converted_dir)
            print(f"[Prismatic Convert] Conversion complete: {converted_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return str(converted_dir)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> int:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        int: Number of trainable parameters.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if is_main_process():
        print(f"# trainable params in {name}: {num_params}")
    return num_params


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> nn.Module:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        nn.Module: Initialized module, wrapped with DDP when it has trainable parameters.
    """
    module = module_class(**module_args)
    num_params = count_parameters(module, module_name)

    if cfg.resume:
        resume_ckpt_dir = get_resume_checkpoint_dir(cfg)
        state_dict = load_checkpoint(module_name, resume_ckpt_dir, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    if num_params == 0:
        if is_main_process():
            print(f"[Init] `{module_name}` has no trainable parameters; skipping DDP wrapper.")
        return module

    return wrap_ddp(module, device_id, find_unused_params)


class AttentionQKVCollector:
    """Collects per-layer Q/K/V tensors from ViT attention blocks through forward hooks."""

    def __init__(self, attention_modules: List[nn.Module]) -> None:
        self.attention_modules = attention_modules
        self.enabled = True
        self._handles = []
        self._qkv_cache: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(len(attention_modules))
        ]

        for layer_idx, attn_module in enumerate(attention_modules):
            handle = attn_module.qkv.register_forward_hook(self._build_hook(layer_idx, attn_module))
            self._handles.append(handle)

    def _build_hook(self, layer_idx: int, attn_module: nn.Module):
        def hook(module: nn.Module, args, output: torch.Tensor) -> None:  # noqa: ANN001
            if not self.enabled:
                return
            if isinstance(output, tuple):
                output = output[0]
            if output is None:
                return

            bsz, num_tokens, _ = output.shape
            num_heads = attn_module.num_heads
            head_dim = output.shape[-1] // (3 * num_heads)

            qkv = output.reshape(bsz, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if hasattr(attn_module, "q_norm"):
                q = attn_module.q_norm(q)
            if hasattr(attn_module, "k_norm"):
                k = attn_module.k_norm(k)

            self._qkv_cache[layer_idx].append((q, k, v))

        return hook

    def clear(self) -> None:
        self._qkv_cache = [[] for _ in range(len(self.attention_modules))]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_layerwise_qkv(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        layerwise_qkv = []
        for entries in self._qkv_cache:
            if len(entries) == 0:
                continue
            if len(entries) == 1:
                layerwise_qkv.append(entries[0])
                continue

            # Multi-image OpenVLA forward calls each image branch sequentially; concatenate on token dimension.
            q = torch.cat([item[0] for item in entries], dim=2)
            k = torch.cat([item[1] for item in entries], dim=2)
            v = torch.cat([item[2] for item in entries], dim=2)
            layerwise_qkv.append((q, k, v))
        return layerwise_qkv


def get_vla_core(vla) -> OpenVLAForActionPrediction:
    """Return underlying OpenVLA HF model from DDP / PEFT / raw model wrappers."""
    if isinstance(vla, DDP):
        vla = vla.module
    if hasattr(vla, "base_model") and hasattr(vla.base_model, "model"):
        return vla.base_model.model
    if hasattr(vla, "model"):
        return vla.model
    return vla


def get_vla_peft(vla: DDP):
    """Returns the PEFT wrapper (used for toggling adapters)."""
    return vla.module


def get_attention_modules_from_featurizer(featurizer: nn.Module) -> List[nn.Module]:
    return [block.attn for block in featurizer.blocks if hasattr(block, "attn") and hasattr(block.attn, "qkv")]


def get_vla_vision_backbone(vla_core: OpenVLAForActionPrediction) -> nn.Module:
    """Unwrap FiLM wrapper (if present) and return the underlying Prismatic vision backbone."""
    vision_backbone = vla_core.vision_backbone
    if hasattr(vision_backbone, "vision_backbone"):
        vision_backbone = vision_backbone.vision_backbone
    return vision_backbone


def select_vla_alignment_featurizer(
    vla_core: OpenVLAForActionPrediction,
    vggt_dim: Optional[int],
    branch: str = "auto",
):
    """
    Pick the VLA branch used for attention-response alignment.
    If fused, select the branch whose embed dim is closest to VGGT encoder dim.
    """
    vision_backbone = get_vla_vision_backbone(vla_core)
    candidates = [("featurizer", vision_backbone.featurizer)]
    if hasattr(vision_backbone, "fused_featurizer"):
        candidates.append(("fused_featurizer", vision_backbone.fused_featurizer))
    if hasattr(vision_backbone, "siglip_featurizer"):
        candidates.append(("siglip_featurizer", vision_backbone.siglip_featurizer))
    if hasattr(vision_backbone, "dino_featurizer"):
        candidates.append(("dino_featurizer", vision_backbone.dino_featurizer))

    if branch == "siglip":
        for name, module in candidates:
            if name in {"fused_featurizer", "siglip_featurizer"}:
                selected_name, selected_module = name, module
                break
        else:
            selected_name, selected_module = candidates[0]
            if is_main_process():
                print(
                    f"[Attention Align] Requested branch=siglip but no explicit SigLIP branch found; "
                    f"falling back to `{selected_name}`."
                )
    elif branch == "dino":
        for name, module in candidates:
            if name in {"featurizer", "dino_featurizer"}:
                selected_name, selected_module = name, module
                break
        else:
            selected_name, selected_module = candidates[0]
    elif branch == "auto":
        if vggt_dim is None:
            selected_name, selected_module = candidates[0]
        else:
            selected_name, selected_module = min(candidates, key=lambda x: abs(x[1].embed_dim - vggt_dim))
    else:
        raise ValueError(f"Unsupported vla_alignment_branch={branch}. Use one of: auto, siglip, dino.")

    if is_main_process():
        print(
            f"[Attention Align] Selected VLA vision branch `{selected_name}` "
            f"(embed_dim={selected_module.embed_dim}, target_vggt_dim={vggt_dim}, branch={branch})."
        )
    return selected_module


def set_non_vision_lora_frozen(
    vla_peft,
    siglip_only_vision: bool = False,
    dino_only_vision: bool = False,
) -> None:
    """Keep LoRA trainable only under vision encoder modules (optionally one fused branch only)."""
    if siglip_only_vision and dino_only_vision:
        raise ValueError("Cannot enable both siglip_only_vision and dino_only_vision.")

    has_fused_siglip_lora = any(
        ("lora_" in name) and (("fused_featurizer" in name) or ("siglip_featurizer" in name))
        for name, _ in vla_peft.named_parameters()
    )
    has_named_dino_lora = any(("lora_" in name) and ("dino_featurizer" in name) for name, _ in vla_peft.named_parameters())
    for name, param in vla_peft.named_parameters():
        if "lora_" not in name:
            continue

        keep_trainable = "vision_backbone" in name
        if keep_trainable and siglip_only_vision:
            if has_fused_siglip_lora:
                # Fused backbone case (HF Prismatic): keep only SigLIP branch adapters.
                keep_trainable = ("fused_featurizer" in name) or ("siglip_featurizer" in name)
            else:
                # Single-backbone case: featurizer itself is the only branch.
                keep_trainable = "featurizer" in name
        if keep_trainable and dino_only_vision:
            if has_named_dino_lora:
                keep_trainable = "dino_featurizer" in name
            elif has_fused_siglip_lora:
                # HF fused backbone case: `featurizer` is the first (DINO) branch.
                keep_trainable = ("featurizer" in name) and ("fused_featurizer" not in name)
            else:
                keep_trainable = "featurizer" in name
        param.requires_grad = keep_trainable


def set_vision_lora_frozen(vla_peft, disable_vision_lora_forward: bool = True) -> int:
    """
    Freeze LoRA adapters under vision backbone while keeping non-vision LoRAs trainable.

    If `disable_vision_lora_forward=True`, vision-side LoRA params are also zeroed so the
    adapter branch has no forward contribution (identity behavior on vision modules).
    Returns number of vision LoRA parameters touched.
    """
    touched_params = 0
    with torch.no_grad():
        for name, param in vla_peft.named_parameters():
            if "lora_" not in name:
                continue
            if "vision_backbone" not in name:
                continue
            param.requires_grad = False
            touched_params += param.numel()
            if disable_vision_lora_forward:
                param.zero_()
    return touched_params


def extract_dino_pixels(pixel_values: torch.Tensor, num_images_in_input: int) -> torch.Tensor:
    """
    Extract DINO channel stacks from fused Prismatic inputs.
    Fused format per image is [DINO(3ch), SigLIP(3ch)] => keep DINO channels only.
    """
    channels = pixel_values.shape[1]
    fused_channels = 6 * num_images_in_input
    dino_channels = 3 * num_images_in_input

    # Already single-branch (3 channels per image)
    if channels == dino_channels:
        return pixel_values

    # Fused branch (6 channels per image): keep channels 0:3 in each image chunk
    if channels == fused_channels:
        image_chunks = torch.split(pixel_values, [6] * num_images_in_input, dim=1)
        dino_chunks = [chunk[:, 0:3] for chunk in image_chunks]
        return torch.cat(dino_chunks, dim=1)

    raise ValueError(
        f"Unexpected pixel channel count={channels} for num_images_in_input={num_images_in_input}. "
        f"Expected either {dino_channels} (DINO-only) or {fused_channels} (DINO+SigLIP)."
    )


def run_dino_encoder_only(
    vla_core: OpenVLAForActionPrediction,
    pixel_values: torch.Tensor,
    num_images_in_input: int,
) -> torch.Tensor:
    """Run DINO branch only (no SigLIP) and concatenate patch tokens across images."""
    vision_backbone = get_vla_vision_backbone(vla_core)
    if hasattr(vision_backbone, "dino_featurizer"):
        dino_featurizer = vision_backbone.dino_featurizer
    elif hasattr(vision_backbone, "featurizer"):
        dino_featurizer = vision_backbone.featurizer
    elif hasattr(vision_backbone, "fused_featurizer"):
        dino_featurizer = vision_backbone.fused_featurizer
    else:
        raise ValueError("Unable to locate DINO featurizer in VLA vision backbone.")

    dino_pixels = extract_dino_pixels(pixel_values, num_images_in_input)
    if num_images_in_input == 1:
        return dino_featurizer(dino_pixels)

    per_image_pixels = torch.split(dino_pixels, [3] * num_images_in_input, dim=1)
    per_image_patches = [dino_featurizer(img) for img in per_image_pixels]
    return torch.cat(per_image_patches, dim=1)


def extract_siglip_pixels(pixel_values: torch.Tensor, num_images_in_input: int) -> torch.Tensor:
    """
    Extract SigLIP channel stacks from fused Prismatic inputs.
    Fused format per image is [DINO(3ch), SigLIP(3ch)] => keep SigLIP channels only.
    """
    channels = pixel_values.shape[1]
    fused_channels = 6 * num_images_in_input
    siglip_channels = 3 * num_images_in_input

    # Already single-branch (3 channels per image)
    if channels == siglip_channels:
        return pixel_values

    # Fused branch (6 channels per image): keep channels 3:6 in each image chunk
    if channels == fused_channels:
        image_chunks = torch.split(pixel_values, [6] * num_images_in_input, dim=1)
        siglip_chunks = [chunk[:, 3:6] for chunk in image_chunks]
        return torch.cat(siglip_chunks, dim=1)

    raise ValueError(
        f"Unexpected pixel channel count={channels} for num_images_in_input={num_images_in_input}. "
        f"Expected either {siglip_channels} (SigLIP-only) or {fused_channels} (DINO+SigLIP)."
    )


def run_siglip_encoder_only(
    vla_core: OpenVLAForActionPrediction,
    pixel_values: torch.Tensor,
    num_images_in_input: int,
) -> torch.Tensor:
    """Run SigLIP branch only (no DINO) and concatenate patch tokens across images."""
    vision_backbone = get_vla_vision_backbone(vla_core)
    if hasattr(vision_backbone, "fused_featurizer"):
        siglip_featurizer = vision_backbone.fused_featurizer
    elif hasattr(vision_backbone, "siglip_featurizer"):
        siglip_featurizer = vision_backbone.siglip_featurizer
    elif hasattr(vision_backbone, "featurizer"):
        siglip_featurizer = vision_backbone.featurizer
    else:
        raise ValueError("Unable to locate SigLIP featurizer in VLA vision backbone.")

    siglip_pixels = extract_siglip_pixels(pixel_values, num_images_in_input)
    if num_images_in_input == 1:
        return siglip_featurizer(siglip_pixels)

    per_image_pixels = torch.split(siglip_pixels, [3] * num_images_in_input, dim=1)
    per_image_patches = [siglip_featurizer(img) for img in per_image_pixels]
    return torch.cat(per_image_patches, dim=1)


def qkv_heads_to_token_embeddings(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert [B, Heads, Tokens, HeadDim] into [B, Tokens, Hidden]."""
    q_tokens = q.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
    k_tokens = k.permute(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1)
    v_tokens = v.permute(0, 2, 1, 3).reshape(v.shape[0], v.shape[2], -1)
    return q_tokens, k_tokens, v_tokens


def merge_vggt_batch_images_qkv(
    layerwise_qkv: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_images_in_input: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert VGGT patch-encoder QKV from [B*S, H, N, Dh] to [B, H, S*N, Dh].
    """
    merged = []
    for q, k, v in layerwise_qkv:
        if q.shape[0] == batch_size * num_images_in_input:
            q = q.view(batch_size, num_images_in_input, q.shape[1], q.shape[2], q.shape[3])
            k = k.view(batch_size, num_images_in_input, k.shape[1], k.shape[2], k.shape[3])
            v = v.view(batch_size, num_images_in_input, v.shape[1], v.shape[2], v.shape[3])
            q = q.permute(0, 2, 1, 3, 4).reshape(batch_size, q.shape[2], num_images_in_input * q.shape[3], q.shape[4])
            k = k.permute(0, 2, 1, 3, 4).reshape(batch_size, k.shape[2], num_images_in_input * k.shape[3], k.shape[4])
            v = v.permute(0, 2, 1, 3, 4).reshape(batch_size, v.shape[2], num_images_in_input * v.shape[3], v.shape[4])
        merged.append((q, k, v))
    return merged


def resolve_layer_pairs(
    num_vla_layers: int,
    num_vggt_layers: int,
    align_all_layers: bool,
    vla_layer_idx: int,
    vggt_layer_idx: int,
) -> List[Tuple[int, int]]:
    def _resolve_single_idx(layer_idx: int, num_layers: int, name: str) -> int:
        if layer_idx < 0:
            resolved = num_layers + layer_idx
        elif layer_idx == num_layers:
            # Accept a common 1-based "last layer" input (e.g., 24 for 24 layers).
            resolved = num_layers - 1
            if is_main_process():
                print(f"[Attention Align] {name}={layer_idx} mapped to last valid index {resolved}.")
        else:
            resolved = layer_idx
        return resolved

    if align_all_layers:
        num_pairs = min(num_vla_layers, num_vggt_layers)
        vla_start = num_vla_layers - num_pairs
        vggt_start = num_vggt_layers - num_pairs
        return [(vla_start + i, vggt_start + i) for i in range(num_pairs)]

    resolved_vla = _resolve_single_idx(vla_layer_idx, num_vla_layers, "vla_layers_align")
    resolved_vggt = _resolve_single_idx(vggt_layer_idx, num_vggt_layers, "vggt_layers_align")
    if resolved_vla < 0 or resolved_vla >= num_vla_layers:
        raise ValueError(f"Invalid vla_layers_align={vla_layer_idx} for {num_vla_layers} available layers.")
    if resolved_vggt < 0 or resolved_vggt >= num_vggt_layers:
        raise ValueError(f"Invalid vggt_layers_align={vggt_layer_idx} for {num_vggt_layers} available layers.")
    return [(resolved_vla, resolved_vggt)]


def extract_action_hidden_states_from_tail(
    text_hidden_states: torch.Tensor,
    num_actions_chunk: int = NUM_ACTIONS_CHUNK,
    action_dim: int = ACTION_DIM,
) -> torch.Tensor:
    """
    Extract action token hidden states from the tail of text tokens.

    This avoids tokenizer-specific action-id thresholding and works across
    Llama/Qwen token spaces as long as actions are appended at the end.
    """
    expected_action_tokens = num_actions_chunk * action_dim
    if text_hidden_states.shape[1] < expected_action_tokens:
        raise ValueError(
            f"Not enough text tokens to extract actions: got {text_hidden_states.shape[1]}, "
            f"need at least {expected_action_tokens}."
        )
    return text_hidden_states[:, -expected_action_tokens:, :]


def run_vggt_patch_encoder_forward(
    vggt: VGGT,
    batch: dict,
    processor,
    device_id: int,
    num_images_in_input: int,
) -> Tuple[int, int]:
    """
    Run only VGGT's patch encoder (vision backbone) to trigger Q/K/V hooks.
    Returns batch size and number of images.
    """
    target_device = torch.device("cuda", device_id) if torch.cuda.is_available() else torch.device("cpu")
    unnorm_imgs = preprocess_normed_images(
        batch["pixel_values"], processor.image_processor, num_images_in_input
    ).to(device=target_device, dtype=torch.float32)
    batch_size, num_images = unnorm_imgs.shape[:2]
    normalized_imgs = (unnorm_imgs - vggt.aggregator._resnet_mean) / vggt.aggregator._resnet_std
    flat_imgs = normalized_imgs.view(batch_size * num_images, *normalized_imgs.shape[2:])
    _ = vggt.aggregator.patch_embed(flat_imgs)
    return batch_size, num_images


def forward_dual_path_vla(
    vla: DDP,
    fusion_projector,
    single_path_projector,
    knowledge_router,
    batch: dict,
    device_id: int,
    use_film: bool,
    use_proprio: bool,
    proprio_projector,
    freeze_base_visual_path: bool,
    vla_qkv_collector: Optional[AttentionQKVCollector],
    use_siglip_only_vision: bool,
    use_dino_only_vision: bool,
    visual_path_mode: str,
    num_images_in_input: int,
    noisy_actions: Optional[torch.Tensor] = None,
    noisy_action_projector=None,
    diffusion_timestep_embeddings: Optional[torch.Tensor] = None,
    knowledge_router_enabled: bool = False,
    knowledge_router_target_keep_ratio: float = 0.5,
    knowledge_router_min_keep_tokens: int = 8,
    knowledge_router_hard_routing: bool = False,
    knowledge_router_token_fusion_mode: str = "no_fusion",
    compute_router_loss: bool = True,
) -> Tuple[CausalLMOutputWithPast, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Custom OpenVLA forward with:
      1) base visual path (LoRA disabled),
      2) expert visual path (LoRA enabled),
      3) concat-and-project fusion before language model.
    """
    valid_visual_path_modes = {
        "dual",
        "base_only",
        "expert_only",
        "base_only_separate",
        "expert_only_separate",
    }
    if visual_path_mode not in valid_visual_path_modes:
        raise ValueError(
            f"Unsupported visual_path_mode={visual_path_mode}. "
            f"Use one of: {sorted(valid_visual_path_modes)}."
        )

    vla_core = get_vla_core(vla)
    vla_peft = get_vla_peft(vla)

    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    labels = batch["labels"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)

    router_aux: Optional[Dict[str, torch.Tensor]] = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        input_embeddings = vla_core.get_input_embeddings()(input_ids)
        all_actions_mask = vla_core._process_action_masks(labels)
        language_embeddings = None
        if use_film:
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )

        def _run_vision_encoder():
            if use_siglip_only_vision:
                if use_film:
                    raise NotImplementedError(
                        "use_siglip_only_vision=True is currently unsupported with use_film=True."
                    )
                return run_siglip_encoder_only(
                    vla_core=vla_core,
                    pixel_values=pixel_values,
                    num_images_in_input=num_images_in_input,
                )
            if use_dino_only_vision:
                if use_film:
                    raise NotImplementedError(
                        "use_dino_only_vision=True is currently unsupported with use_film=True."
                    )
                return run_dino_encoder_only(
                    vla_core=vla_core,
                    pixel_values=pixel_values,
                    num_images_in_input=num_images_in_input,
                )
            if use_film:
                return vla_core.vision_backbone(pixel_values, language_embeddings)
            return vla_core.vision_backbone(pixel_values)

        single_branch_vision = use_siglip_only_vision or use_dino_only_vision

        if vla_qkv_collector is not None:
            vla_qkv_collector.clear()
            vla_qkv_collector.enabled = False

        base_projected = None
        expert_projected = None

        if visual_path_mode in {"dual", "base_only", "base_only_separate"}:
            if vla_qkv_collector is not None and visual_path_mode in {"base_only", "base_only_separate"}:
                vla_qkv_collector.enabled = True
            if freeze_base_visual_path:
                with torch.no_grad():
                    with vla_peft.disable_adapter():
                        base_patch_features = _run_vision_encoder()
                        if single_branch_vision:
                            base_projected = base_patch_features
                        else:
                            base_projected = vla_core.projector(base_patch_features)
                base_projected = base_projected.detach()
            else:
                with vla_peft.disable_adapter():
                    base_patch_features = _run_vision_encoder()
                    if single_branch_vision:
                        base_projected = base_patch_features
                    else:
                        base_projected = vla_core.projector(base_patch_features)
            if vla_qkv_collector is not None and visual_path_mode in {"base_only", "base_only_separate"}:
                vla_qkv_collector.enabled = False

        if visual_path_mode in {"dual", "expert_only", "expert_only_separate"}:
            if vla_qkv_collector is not None:
                vla_qkv_collector.enabled = True
            expert_patch_features = _run_vision_encoder()
            if single_branch_vision:
                expert_projected = expert_patch_features
            else:
                expert_projected = vla_core.projector(expert_patch_features)
            if vla_qkv_collector is not None:
                vla_qkv_collector.enabled = False

        def _compose_projected_tokens(
            base_tokens: Optional[torch.Tensor],
            expert_tokens: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if visual_path_mode == "dual":
                if fusion_projector is None:
                    raise RuntimeError("dual visual mode requires fusion_projector.")
                return fusion_projector(base_tokens, expert_tokens)
            if visual_path_mode == "base_only":
                if base_tokens is None:
                    raise RuntimeError("base_only visual mode expected base_projected to be populated.")
                return fusion_projector(base_tokens, torch.zeros_like(base_tokens))
            if visual_path_mode == "expert_only":
                if expert_tokens is None:
                    raise RuntimeError("expert_only visual mode expected expert_projected to be populated.")
                return fusion_projector(torch.zeros_like(expert_tokens), expert_tokens)
            if visual_path_mode == "base_only_separate":
                if base_tokens is None:
                    raise RuntimeError("base_only_separate visual mode expected base_projected to be populated.")
                if single_path_projector is None:
                    raise RuntimeError(
                        "base_only_separate visual mode requires single_path_projector, but got None."
                    )
                return single_path_projector(base_tokens)

            # expert_only_separate
            if expert_tokens is None:
                raise RuntimeError("expert_only_separate visual mode expected expert_projected to be populated.")
            if single_path_projector is None:
                raise RuntimeError(
                    "expert_only_separate visual mode requires single_path_projector, but got None."
                )
            return single_path_projector(expert_tokens)

        def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            expanded = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            return torch.gather(tokens, dim=1, index=expanded)

        def _make_candidate_positions(num_positions: int, batch_size: int, device: torch.device) -> torch.Tensor:
            positions = torch.arange(num_positions, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            return torch.cat((positions, positions), dim=1)

        def _fuse_dual_tokens_with_position_drop(
            base_tokens: torch.Tensor,
            expert_tokens: torch.Tensor,
            gate_probs_dual: torch.Tensor,
            target_keep_ratio: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            batch_size, num_positions, _ = base_tokens.shape
            if fusion_projector is None:
                raise RuntimeError(
                    "dual visual mode with router expects fusion_projector for concat-and-project token fusion."
                )
            if gate_probs_dual.shape[1] != 2 * num_positions:
                raise RuntimeError(
                    "Dual-path per-branch routing expects gate_probs shape (B, 2N). "
                    f"Got gate_probs={tuple(gate_probs_dual.shape)}, N={num_positions}."
                )

            gate_probs_dual = torch.nan_to_num(gate_probs_dual, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            base_gate_probs = gate_probs_dual[:, :num_positions]
            expert_gate_probs = gate_probs_dual[:, num_positions:]

            # Dual-path selection enforces exact per-path keep ratio over N positions.
            # (knowledge_router_min_keep_tokens is still used in router losses, but not for this dual-path top-k.)
            keep_tokens_per_path = int(round(float(target_keep_ratio) * float(num_positions)))
            keep_tokens_per_path = max(1, min(num_positions, keep_tokens_per_path))

            _, base_topk_idx = torch.topk(
                base_gate_probs, k=keep_tokens_per_path, dim=1, largest=True, sorted=False
            )
            _, expert_topk_idx = torch.topk(
                expert_gate_probs, k=keep_tokens_per_path, dim=1, largest=True, sorted=False
            )
            base_selected_mask_full = torch.zeros_like(base_gate_probs, dtype=torch.bool)
            expert_selected_mask_full = torch.zeros_like(expert_gate_probs, dtype=torch.bool)
            base_selected_mask_full.scatter_(1, base_topk_idx, True)
            expert_selected_mask_full.scatter_(1, expert_topk_idx, True)

            # Keep the union of per-path top-k positions.
            # Result length is dynamic per sample (in [K, 2K]), then padded to batch max.
            union_selected_mask_full = base_selected_mask_full | expert_selected_mask_full
            selected_position_counts = union_selected_mask_full.sum(dim=1, dtype=torch.long)
            max_selected_positions = int(selected_position_counts.max().item())
            if max_selected_positions < 1:
                max_selected_positions = 1

            position_grid = torch.arange(
                num_positions, device=base_tokens.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
            sentinel = torch.full_like(position_grid, fill_value=num_positions)
            masked_positions = torch.where(union_selected_mask_full, position_grid, sentinel)
            selected_positions_sorted, _ = torch.sort(masked_positions, dim=1)
            selected_positions = selected_positions_sorted[:, :max_selected_positions]
            valid_positions_mask = selected_positions < num_positions
            selected_positions_safe = torch.where(
                valid_positions_mask, selected_positions, torch.zeros_like(selected_positions)
            )

            gathered_base = _gather_tokens(base_tokens, selected_positions_safe)
            gathered_expert = _gather_tokens(expert_tokens, selected_positions_safe)
            gathered_base_selected_mask = torch.gather(base_selected_mask_full, dim=1, index=selected_positions_safe)
            gathered_expert_selected_mask = torch.gather(expert_selected_mask_full, dim=1, index=selected_positions_safe)
            gathered_base_selected_mask = gathered_base_selected_mask & valid_positions_mask
            gathered_expert_selected_mask = gathered_expert_selected_mask & valid_positions_mask
            updated_base = gathered_base * gathered_base_selected_mask.to(dtype=gathered_base.dtype).unsqueeze(-1)
            updated_expert = gathered_expert * gathered_expert_selected_mask.to(dtype=gathered_expert.dtype).unsqueeze(-1)
            fused_tokens = fusion_projector(updated_base, updated_expert)
            selected_base_counts = base_selected_mask_full.sum(dim=1, dtype=torch.long)
            selected_expert_counts = expert_selected_mask_full.sum(dim=1, dtype=torch.long)
            return (
                fused_tokens,
                valid_positions_mask,
                selected_position_counts,
                selected_base_counts,
                selected_expert_counts,
            )

        projected_patch_embeddings = _compose_projected_tokens(base_projected, expert_projected)
        patch_attention_mask = None

        if knowledge_router_enabled and knowledge_router is not None:
            if base_projected is not None and expert_projected is not None:
                if base_projected.shape != expert_projected.shape:
                    raise RuntimeError(
                        "Knowledge routing requires matched base/expert token shapes for real token pruning, "
                        f"but got base={tuple(base_projected.shape)} and expert={tuple(expert_projected.shape)}."
                    )
                candidate_tokens = torch.cat((base_projected, expert_projected), dim=1)
                candidate_positions = _make_candidate_positions(
                    num_positions=base_projected.shape[1],
                    batch_size=base_projected.shape[0],
                    device=base_projected.device,
                )
            elif base_projected is not None:
                candidate_tokens = base_projected
                candidate_positions = torch.arange(
                    base_projected.shape[1], device=base_projected.device, dtype=torch.long
                ).unsqueeze(0).expand(base_projected.shape[0], -1)
            elif expert_projected is not None:
                candidate_tokens = expert_projected
                candidate_positions = torch.arange(
                    expert_projected.shape[1], device=expert_projected.device, dtype=torch.long
                ).unsqueeze(0).expand(expert_projected.shape[0], -1)
            else:
                candidate_tokens = None
                candidate_positions = None

            if candidate_tokens is not None:
                router_target_keep_ratio = float(knowledge_router_target_keep_ratio)

                text_mask = (~all_actions_mask) & attention_mask.bool()
                no_text_tokens = text_mask.sum(dim=1) == 0
                if no_text_tokens.any():
                    fallback_text_mask = attention_mask.bool()
                    text_mask = torch.where(no_text_tokens.unsqueeze(1), fallback_text_mask, text_mask)

                router_aux = knowledge_router(
                    text_tokens=input_embeddings,
                    candidate_tokens=candidate_tokens,
                    text_mask=text_mask,
                    candidate_positions=candidate_positions,
                    target_keep_ratio=router_target_keep_ratio,
                    min_keep_tokens=knowledge_router_min_keep_tokens,
                    hard_routing=knowledge_router_hard_routing,
                    compute_loss=compute_router_loss,
                )
                router_aux["target_keep_ratio_effective"] = float(router_target_keep_ratio)
                gate_probs = router_aux.get("gate_probs", None)
                if gate_probs is None or (not torch.isfinite(gate_probs).all()):
                    router_aux = None
                elif base_projected is not None and expert_projected is not None:
                    gate_probs = gate_probs.to(dtype=candidate_tokens.dtype)
                    (
                        projected_patch_embeddings,
                        patch_attention_mask,
                        selected_position_counts,
                        selected_base_counts,
                        selected_expert_counts,
                    ) = (
                        _fuse_dual_tokens_with_position_drop(
                            base_tokens=base_projected,
                            expert_tokens=expert_projected,
                            gate_probs_dual=gate_probs,
                            target_keep_ratio=router_target_keep_ratio,
                        )
                    )
                    router_aux["selected_token_count"] = selected_position_counts.to(dtype=gate_probs.dtype).mean()
                    router_aux["selected_base_token_count"] = selected_base_counts.to(dtype=gate_probs.dtype).mean()
                    router_aux["selected_expert_token_count"] = selected_expert_counts.to(dtype=gate_probs.dtype).mean()
                else:
                    selected_indices = router_aux.get("selected_indices", None)
                    selected_gate_probs = router_aux.get("selected_gate_probs", None)
                    if (
                        selected_indices is None
                        or selected_gate_probs is None
                        or (not torch.isfinite(selected_gate_probs).all())
                    ):
                        router_aux = None
                    else:
                        selected_indices = selected_indices.long()
                        selected_gate_probs = selected_gate_probs.to(dtype=candidate_tokens.dtype)
                        selected_gate_probs = torch.nan_to_num(selected_gate_probs, nan=0.0, posinf=1.0, neginf=0.0)
                        selected_gate_probs = selected_gate_probs.clamp(min=0.0, max=1.0)
                        if base_projected is not None:
                            base_projected = _gather_tokens(base_projected, selected_indices)
                            base_projected = base_projected * selected_gate_probs.unsqueeze(-1)
                        if expert_projected is not None:
                            expert_projected = _gather_tokens(expert_projected, selected_indices)
                            expert_projected = expert_projected * selected_gate_probs.unsqueeze(-1)
                        projected_patch_embeddings = _compose_projected_tokens(base_projected, expert_projected)
                        patch_attention_mask = torch.ones(
                            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                            dtype=torch.bool,
                            device=projected_patch_embeddings.device,
                        )
                        selected_token_count = torch.tensor(
                            float(selected_indices.shape[1]),
                            device=selected_indices.device,
                            dtype=selected_gate_probs.dtype,
                        )
                        zero_token_count = torch.tensor(
                            0.0,
                            device=selected_indices.device,
                            dtype=selected_gate_probs.dtype,
                        )
                        router_aux["selected_token_count"] = selected_token_count
                        router_aux["selected_base_token_count"] = (
                            selected_token_count if base_projected is not None else zero_token_count
                        )
                        router_aux["selected_expert_token_count"] = (
                            selected_token_count if expert_projected is not None else zero_token_count
                        )

        projected_patch_embeddings = vla_core._process_proprio_features(
            projected_patch_embeddings,
            batch["proprio"] if use_proprio else None,
            proprio_projector if use_proprio else None,
        )

        if diffusion_timestep_embeddings is not None:
            projected_patch_embeddings = torch.cat((projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

        if noisy_actions is not None:
            bsz = noisy_actions.shape[0]
            noisy_action_tokens = noisy_actions.reshape(bsz, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(noisy_action_tokens)
            input_embeddings = vla_core._replace_input_embeddings(input_embeddings, all_actions_mask, noisy_action_features)
        else:
            input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

        # Keep routed patch mask aligned with any extra tokens appended after routing
        # (e.g., proprio token and/or diffusion timestep token).
        if patch_attention_mask is not None:
            current_patch_len = patch_attention_mask.shape[1]
            final_patch_len = projected_patch_embeddings.shape[1]
            if final_patch_len > current_patch_len:
                extra_valid = torch.ones(
                    (patch_attention_mask.shape[0], final_patch_len - current_patch_len),
                    dtype=patch_attention_mask.dtype,
                    device=patch_attention_mask.device,
                )
                patch_attention_mask = torch.cat((patch_attention_mask, extra_valid), dim=1)
            elif final_patch_len < current_patch_len:
                patch_attention_mask = patch_attention_mask[:, :final_patch_len]

        projected_patch_attention_mask = None
        if attention_mask is not None:
            if patch_attention_mask is None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            else:
                projected_patch_attention_mask = patch_attention_mask.to(
                    dtype=attention_mask.dtype, device=attention_mask.device
                )

        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )
        multimodal_labels = vla_core._build_multimodal_labels(labels, projected_patch_embeddings)

        language_model_output = vla_core.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    return language_model_output, projected_patch_embeddings, router_aux


def forward_openvla_baseline(
    vla: DDP,
    batch: dict,
    device_id: int,
    use_film: bool = False,
    use_proprio: bool = False,
    proprio_projector=None,
) -> Tuple[CausalLMOutputWithPast, torch.Tensor]:
    """
    Strict OpenVLA forward path (no dual-path/projector overrides).
    Uses the model's native multimodal forward with token-level CE loss.
    """
    if use_film:
        raise NotImplementedError("openvla_baseline mode currently requires use_film=False.")

    vla_core = get_vla_core(vla)
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    labels = batch["labels"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        input_ids, attention_mask = vla_core._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = vla_core._prepare_labels_for_action_prediction(labels, input_ids)

        input_embeddings = vla_core.get_input_embeddings()(input_ids)
        all_actions_mask = vla_core._process_action_masks(labels)
        patch_features = vla_core.vision_backbone(pixel_values)
        projected_patch_embeddings = vla_core.projector(patch_features)
        projected_patch_embeddings = vla_core._process_proprio_features(
            projected_patch_embeddings,
            batch["proprio"] if use_proprio else None,
            proprio_projector if use_proprio else None,
        )
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

        multimodal_embeddings, multimodal_attention_mask = vla_core._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        multimodal_labels = vla_core._build_multimodal_labels(labels, projected_patch_embeddings)

        output = vla_core.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    return output, projected_patch_embeddings


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    knowledge_router,
    attention_aligner,
    fusion_projector,
    single_path_projector,
    vla_qkv_collector,
    vggt_qkv_collector,
    vggt,
    layers_align,
    processor,
    batch,
    action_tokenizer,
    device_id,
    use_vggt_alignment,
    align_loss_coeff,
    align_all_layers,
    freeze_base_visual_path,
    visual_path_mode,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    use_siglip_only_vision,
    use_dino_only_vision,
    num_patches,
    num_images_in_input=1,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    openvla_baseline: bool = False,
    use_knowledge_router: bool = False,
    knowledge_router_loss_coeff: float = 1.0,
    knowledge_router_budget_loss_coeff: float = 0.1,
    knowledge_router_entropy_loss_coeff: float = 0.0,
    knowledge_router_target_keep_ratio: float = 0.5,
    knowledge_router_min_keep_tokens: int = 8,
    knowledge_router_hard_routing: bool = False,
    knowledge_router_token_fusion_mode: str = "no_fusion",
    knowledge_router_warmup_steps: int = 500,
    knowledge_router_importance_ema_momentum: float = 0.9,
    current_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    effective_num_patches = num_patches
    router_cls_loss = ground_truth_actions.new_zeros(())
    router_budget_loss = ground_truth_actions.new_zeros(())
    router_entropy_loss = ground_truth_actions.new_zeros(())
    router_keep_ratio = ground_truth_actions.new_zeros(())
    router_action_sup_loss = ground_truth_actions.new_zeros(())
    router_selected_tokens = ground_truth_actions.new_zeros(())
    router_selected_tokens_base = ground_truth_actions.new_zeros(())
    router_selected_tokens_expert = ground_truth_actions.new_zeros(())
    router_total_loss = ground_truth_actions.new_zeros(())
    router_gate_std = ground_truth_actions.new_zeros(())
    router_gate_row_std = ground_truth_actions.new_zeros(())
    router_gate_min = ground_truth_actions.new_zeros(())
    router_gate_max = ground_truth_actions.new_zeros(())
    router_gate_q10 = ground_truth_actions.new_zeros(())
    router_gate_q50 = ground_truth_actions.new_zeros(())
    router_gate_q90 = ground_truth_actions.new_zeros(())
    router_gate_entropy = ground_truth_actions.new_zeros(())
    router_aux: Optional[Dict[str, torch.Tensor]] = None
    router_enabled_now = (
        use_knowledge_router
        and (knowledge_router is not None)
        and (not openvla_baseline)
        and (current_step >= knowledge_router_warmup_steps)
    )
    if openvla_baseline:
        output, projected_patch_embeddings = forward_openvla_baseline(
            vla=vla,
            batch=batch,
            device_id=device_id,
            use_film=use_film,
            use_proprio=use_proprio,
            proprio_projector=proprio_projector,
        )
        effective_num_patches = projected_patch_embeddings.shape[1]
    else:
        output, projected_patch_embeddings, router_aux = forward_dual_path_vla(
            vla=vla,
            fusion_projector=fusion_projector,
            single_path_projector=single_path_projector,
            knowledge_router=knowledge_router,
            batch=batch,
            device_id=device_id,
            use_film=use_film,
            use_proprio=use_proprio,
            proprio_projector=proprio_projector,
            freeze_base_visual_path=freeze_base_visual_path,
            vla_qkv_collector=vla_qkv_collector,
            use_siglip_only_vision=use_siglip_only_vision,
            use_dino_only_vision=use_dino_only_vision,
            visual_path_mode=visual_path_mode,
            num_images_in_input=num_images_in_input,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            knowledge_router_enabled=router_enabled_now,
            knowledge_router_target_keep_ratio=knowledge_router_target_keep_ratio,
            knowledge_router_min_keep_tokens=knowledge_router_min_keep_tokens,
            knowledge_router_hard_routing=knowledge_router_hard_routing,
            knowledge_router_token_fusion_mode=knowledge_router_token_fusion_mode,
            compute_router_loss=False,
        )
        effective_num_patches = projected_patch_embeddings.shape[1]

    llm_visual_tokens = ground_truth_actions.new_tensor(float(effective_num_patches))

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    align_loss = output.logits.new_zeros(())
    if use_vggt_alignment and not openvla_baseline:
        if any(item is None for item in [attention_aligner, vla_qkv_collector, vggt_qkv_collector, vggt, processor]):
            raise RuntimeError(
                "use_vggt_alignment=True requires attention_aligner, vla_qkv_collector, "
                "vggt_qkv_collector, vggt, and processor."
            )

        # Collect VGGT encoder Q/K/V using no_grad (VGGT is frozen teacher).
        vggt_qkv_collector.clear()
        vggt_qkv_collector.enabled = True
        with torch.no_grad():
            batch_size, num_images = run_vggt_patch_encoder_forward(
                vggt=vggt,
                batch=batch,
                processor=processor,
                device_id=device_id,
                num_images_in_input=num_images_in_input,
            )
        vggt_qkv_collector.enabled = False

        vla_layerwise_qkv = vla_qkv_collector.get_layerwise_qkv()
        vggt_layerwise_qkv = merge_vggt_batch_images_qkv(
            vggt_qkv_collector.get_layerwise_qkv(),
            batch_size=batch_size,
            num_images_in_input=num_images,
        )
        layer_pairs = resolve_layer_pairs(
            num_vla_layers=len(vla_layerwise_qkv),
            num_vggt_layers=len(vggt_layerwise_qkv),
            align_all_layers=align_all_layers,
            vla_layer_idx=layers_align[0],
            vggt_layer_idx=layers_align[1],
        )

        layer_losses = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for vla_idx, vggt_idx in layer_pairs:
                q_vla, k_vla, v_vla = qkv_heads_to_token_embeddings(*vla_layerwise_qkv[vla_idx])
                _, k_vggt, v_vggt = qkv_heads_to_token_embeddings(*vggt_layerwise_qkv[vggt_idx])
                layer_losses.append(attention_aligner(q_vla, k_vla, v_vla, k_vggt, v_vggt))
            align_loss = torch.stack(layer_losses).mean()
    
    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        target_token_len = ground_truth_token_ids.shape[1]
        # Align predictions to the label tail directly to avoid dependence on visual-token count.
        predicted_token_ids = output.logits[:, -target_token_len - 1 : -1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                "attn_align_loss": align_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, effective_num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = extract_action_hidden_states_from_tail(text_hidden_states).to(
            torch.bfloat16
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        fusion_projector=fusion_projector,
                        single_path_projector=single_path_projector,
                        knowledge_router=knowledge_router,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=effective_num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                        freeze_base_visual_path=freeze_base_visual_path,
                        use_siglip_only_vision=use_siglip_only_vision,
                        use_dino_only_vision=use_dino_only_vision,
                        visual_path_mode=visual_path_mode,
                        num_images_in_input=num_images_in_input,
                        use_knowledge_router=use_knowledge_router,
                        knowledge_router_target_keep_ratio=knowledge_router_target_keep_ratio,
                        knowledge_router_min_keep_tokens=knowledge_router_min_keep_tokens,
                        knowledge_router_hard_routing=knowledge_router_hard_routing,
                        knowledge_router_token_fusion_mode=knowledge_router_token_fusion_mode,
                        knowledge_router_warmup_steps=knowledge_router_warmup_steps,
                        current_step=current_step,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "attn_align_loss": align_loss.item(),
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    if router_aux is not None and router_enabled_now:
        gate_probs = router_aux.get("gate_probs", None)
        if gate_probs is not None and torch.isfinite(gate_probs).all():
            gate_probs_stats = torch.nan_to_num(gate_probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            router_keep_ratio = torch.nan_to_num(gate_probs_stats.mean(), nan=0.0, posinf=0.0, neginf=0.0)
            router_gate_std = torch.nan_to_num(gate_probs_stats.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0)
            router_gate_min = torch.nan_to_num(gate_probs_stats.min(), nan=0.0, posinf=0.0, neginf=0.0)
            router_gate_max = torch.nan_to_num(gate_probs_stats.max(), nan=0.0, posinf=0.0, neginf=0.0)
            if gate_probs_stats.shape[1] > 1:
                router_gate_row_std = torch.nan_to_num(
                    gate_probs_stats.std(dim=1, unbiased=False).mean(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            else:
                router_gate_row_std = gate_probs_stats.new_zeros(())

            if gate_probs_stats.numel() > 0:
                flat_probs = gate_probs_stats.reshape(-1)
                gate_q = torch.quantile(
                    flat_probs,
                    torch.tensor([0.1, 0.5, 0.9], device=flat_probs.device, dtype=flat_probs.dtype),
                )
                router_gate_q10, router_gate_q50, router_gate_q90 = gate_q[0], gate_q[1], gate_q[2]

            safe_stats_probs = gate_probs_stats.clamp(min=1e-6, max=1.0 - 1e-6)
            router_gate_entropy = torch.nan_to_num(
                -(safe_stats_probs * torch.log(safe_stats_probs) + (1.0 - safe_stats_probs) * torch.log(1.0 - safe_stats_probs)).mean(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            selected_token_count = router_aux.get("selected_token_count", None)
            if selected_token_count is not None:
                router_selected_tokens = torch.nan_to_num(
                    selected_token_count,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            else:
                router_selected_tokens = gate_probs.new_tensor(float(gate_probs.shape[1]))
            selected_base_token_count = router_aux.get("selected_base_token_count", None)
            if selected_base_token_count is not None:
                router_selected_tokens_base = torch.nan_to_num(
                    selected_base_token_count,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            selected_expert_token_count = router_aux.get("selected_expert_token_count", None)
            if selected_expert_token_count is not None:
                router_selected_tokens_expert = torch.nan_to_num(
                    selected_expert_token_count,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            if loss.requires_grad:
                action_gate_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=gate_probs,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )[0]
                if action_gate_grads is None:
                    action_token_importance = torch.zeros_like(gate_probs)
                else:
                    # Keep tokens that reduce action loss when their gate increases.
                    action_token_importance = torch.nan_to_num(
                        torch.relu(-action_gate_grads.detach()),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                fallback_scores = torch.nan_to_num(
                    gate_probs.detach(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp_min(0.0)
                invalid_rows = (~torch.isfinite(action_token_importance).all(dim=1)) | (
                    action_token_importance.sum(dim=1) <= 0.0
                )
                if bool(invalid_rows.any().item()):
                    action_token_importance = action_token_importance.clone()
                    action_token_importance[invalid_rows] = fallback_scores[invalid_rows]

                router_module = knowledge_router.module if isinstance(knowledge_router, DDP) else knowledge_router
                action_token_importance = maybe_smooth_router_importance_with_ema(
                    router_module=router_module,
                    token_importance=action_token_importance,
                    momentum=knowledge_router_importance_ema_momentum,
                )

                invalid_rows_after_ema = (~torch.isfinite(action_token_importance).all(dim=1)) | (
                    action_token_importance.sum(dim=1) <= 0.0
                )
                if bool(invalid_rows_after_ema.any().item()):
                    action_token_importance = action_token_importance.clone()
                    action_token_importance[invalid_rows_after_ema] = fallback_scores[invalid_rows_after_ema]

                target_keep_ratio_for_loss = float(
                    router_aux.get("target_keep_ratio_effective", knowledge_router_target_keep_ratio)
                )
                action_pseudo_targets = router_module._build_online_pseudo_targets(
                    token_scores=action_token_importance,
                    target_keep_ratio=target_keep_ratio_for_loss,
                    min_keep_tokens=knowledge_router_min_keep_tokens,
                )

                router_action_sup_loss = router_module._focal_binary_loss(gate_probs, action_pseudo_targets)
                router_cls_loss = torch.nan_to_num(router_action_sup_loss, nan=0.0, posinf=0.0, neginf=0.0)

                router_budget_loss = torch.abs(gate_probs.mean(dim=1) - target_keep_ratio_for_loss).mean()
                safe_probs = gate_probs.clamp(min=1e-6, max=1.0 - 1e-6)
                entropy = -(safe_probs * torch.log(safe_probs) + (1.0 - safe_probs) * torch.log(1.0 - safe_probs))
                router_entropy_loss = -entropy.mean()

                router_budget_loss = torch.nan_to_num(router_budget_loss, nan=0.0, posinf=0.0, neginf=0.0)
                router_entropy_loss = torch.nan_to_num(router_entropy_loss, nan=0.0, posinf=0.0, neginf=0.0)
                router_total_loss = (
                    router_cls_loss * knowledge_router_loss_coeff
                    + router_budget_loss * knowledge_router_budget_loss_coeff
                    + router_entropy_loss * knowledge_router_entropy_loss_coeff
                )
                router_total_loss = torch.nan_to_num(router_total_loss, nan=0.0, posinf=0.0, neginf=0.0)

    metrics.update(
        {
            "router_loss": router_total_loss.item(),
            "router_cls_loss": router_cls_loss.item(),
            "router_action_sup_loss": router_action_sup_loss.item(),
            "router_budget_loss": router_budget_loss.item(),
            "router_entropy_loss": router_entropy_loss.item(),
            "router_keep_ratio": router_keep_ratio.item(),
            "router_selected_tokens": router_selected_tokens.item(),
            "router_selected_tokens_base": router_selected_tokens_base.item(),
            "router_selected_tokens_expert": router_selected_tokens_expert.item(),
            "llm_visual_tokens": llm_visual_tokens.item(),
            "router_gate_std": router_gate_std.item(),
            "router_gate_row_std": router_gate_row_std.item(),
            "router_gate_min": router_gate_min.item(),
            "router_gate_max": router_gate_max.item(),
            "router_gate_q10": router_gate_q10.item(),
            "router_gate_q50": router_gate_q50.item(),
            "router_gate_q90": router_gate_q90.item(),
            "router_gate_entropy": router_gate_entropy.item(),
        }
    )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss + align_loss * align_loss_coeff + router_total_loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    fusion_projector,
    single_path_projector,
    knowledge_router,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
    freeze_base_visual_path,
    use_siglip_only_vision,
    use_dino_only_vision,
    visual_path_mode,
    num_images_in_input,
    use_knowledge_router=False,
    knowledge_router_target_keep_ratio: float = 0.5,
    knowledge_router_min_keep_tokens: int = 8,
    knowledge_router_hard_routing: bool = False,
    knowledge_router_token_fusion_mode: str = "no_fusion",
    knowledge_router_warmup_steps: int = 500,
    current_step: int = 0,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            router_enabled_now = (
                use_knowledge_router
                and (knowledge_router is not None)
                and (current_step >= knowledge_router_warmup_steps)
            )
            output, _, _ = forward_dual_path_vla(
                vla=vla,
                fusion_projector=fusion_projector,
                single_path_projector=single_path_projector,
                knowledge_router=knowledge_router,
                batch=batch,
                device_id=device_id,
                use_film=use_film,
                use_proprio=use_proprio,
                proprio_projector=proprio_projector,
                freeze_base_visual_path=freeze_base_visual_path,
                vla_qkv_collector=None,
                use_siglip_only_vision=use_siglip_only_vision,
                use_dino_only_vision=use_dino_only_vision,
                visual_path_mode=visual_path_mode,
                num_images_in_input=num_images_in_input,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                knowledge_router_enabled=router_enabled_now,
                knowledge_router_target_keep_ratio=knowledge_router_target_keep_ratio,
                knowledge_router_min_keep_tokens=knowledge_router_min_keep_tokens,
                knowledge_router_hard_routing=knowledge_router_hard_routing,
                knowledge_router_token_fusion_mode=knowledge_router_token_fusion_mode,
                compute_router_loss=False,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = extract_action_hidden_states_from_tail(text_hidden_states)  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def format_metric(metrics: Dict[str, float], key: str, precision: int = 6) -> str:
    """Format a metric value for console logging."""
    value = metrics.get(key, None)
    if value is None:
        return "na"
    return f"{value:.{precision}f}"


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    attention_aligner,
    knowledge_router,
    fusion_projector,
    single_path_projector,
    action_head,
    optimizer,
    scheduler,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        attention_aligner (nn.Module): Attention-response alignment module.
        fusion_projector (nn.Module): Dual-path visual fusion projector.
        action_head (nn.Module): Action head module.
        optimizer (Optimizer): Optimizer instance.
        scheduler (_LRScheduler): Learning-rate scheduler instance.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        base_model_sidecar_path = checkpoint_dir / "base_model_path.txt"
        base_model_sidecar_path.write_text(str(cfg.vla_path).strip() + "\n")

        # Some PEFT save paths leave `base_model_name_or_path` empty; patch it for robust eval loading.
        adapter_cfg_path = adapter_dir / "adapter_config.json"
        if adapter_cfg_path.is_file():
            try:
                adapter_cfg = json.loads(adapter_cfg_path.read_text())
            except (OSError, json.JSONDecodeError):
                adapter_cfg = None
            if isinstance(adapter_cfg, dict):
                if str(adapter_cfg.get("base_model_name_or_path", "") or "").strip() == "":
                    adapter_cfg["base_model_name_or_path"] = str(cfg.vla_path).strip()
                    adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2) + "\n")
                    print(
                        "Patched empty LoRA `base_model_name_or_path` in adapter_config.json "
                        f"-> {adapter_cfg['base_model_name_or_path']}"
                    )

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )
        if attention_aligner is not None:
            torch.save(attention_aligner.state_dict(), checkpoint_dir / f"attention_aligner--{checkpoint_name_suffix}")
        if knowledge_router is not None:
            torch.save(knowledge_router.state_dict(), checkpoint_dir / f"knowledge_router--{checkpoint_name_suffix}")
        if fusion_projector is not None:
            torch.save(fusion_projector.state_dict(), checkpoint_dir / f"fusion_projector--{checkpoint_name_suffix}")
        if single_path_projector is not None:
            torch.save(
                single_path_projector.state_dict(),
                checkpoint_dir / f"single_path_projector--{checkpoint_name_suffix}",
            )
        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        if cfg.train_vla_projector:
            vla_core = get_vla_core(vla)
            torch.save(vla_core.projector.state_dict(), checkpoint_dir / f"vla_projector--{checkpoint_name_suffix}")
        torch.save(optimizer.state_dict(), checkpoint_dir / f"optimizer--{checkpoint_name_suffix}")
        torch.save(scheduler.state_dict(), checkpoint_dir / f"scheduler--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

        # Write an explicit marker for "latest saved step" to make resume unambiguous.
        latest_step_file = run_dir / "latest_step.txt"
        latest_step_file.write_text(f"{int(log_step)}\n")

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    knowledge_router,
    attention_aligner,
    fusion_projector,
    single_path_projector,
    vla_qkv_collector,
    vggt_qkv_collector,
    vggt,
    processor,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    if distributed_state.is_main_process:
        print(f"[Validation @ Step {log_step}] starting (time_limit={val_time_limit}s)")
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                knowledge_router=knowledge_router,
                attention_aligner=attention_aligner,
                fusion_projector=fusion_projector,
                single_path_projector=single_path_projector,
                vla_qkv_collector=vla_qkv_collector,
                vggt_qkv_collector=vggt_qkv_collector,
                vggt=vggt,
                layers_align=(cfg.vla_layers_align, cfg.vggt_layers_align),
                processor=processor,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_vggt_alignment=cfg.use_vggt_alignment,
                align_loss_coeff=cfg.align_loss_coeff,
                align_all_layers=cfg.align_all_layers,
                freeze_base_visual_path=cfg.freeze_base_visual_path,
                visual_path_mode=cfg.visual_path_mode,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                use_siglip_only_vision=cfg.use_siglip_only_vision,
                use_dino_only_vision=cfg.use_dino_only_vision,
                num_patches=num_patches,
                num_images_in_input=cfg.num_images_in_input,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                openvla_baseline=cfg.openvla_baseline,
                use_knowledge_router=cfg.use_knowledge_router,
                knowledge_router_loss_coeff=cfg.knowledge_router_loss_coeff,
                knowledge_router_budget_loss_coeff=cfg.knowledge_router_budget_loss_coeff,
                knowledge_router_entropy_loss_coeff=cfg.knowledge_router_entropy_loss_coeff,
                knowledge_router_target_keep_ratio=cfg.knowledge_router_target_keep_ratio,
                knowledge_router_min_keep_tokens=cfg.knowledge_router_min_keep_tokens,
                knowledge_router_hard_routing=cfg.knowledge_router_hard_routing,
                knowledge_router_token_fusion_mode=cfg.knowledge_router_token_fusion_mode,
                knowledge_router_warmup_steps=cfg.knowledge_router_warmup_steps,
                knowledge_router_importance_ema_momentum=cfg.knowledge_router_importance_ema_momentum,
                current_step=log_step,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut validation short only when a positive time limit is configured.
            # Non-positive values mean "no time limit" (full pass over val loader).
            if val_time_limit > 0 and (time.time() - val_start_time > val_time_limit):
                break

    if not all_val_metrics:
        if distributed_state.is_main_process:
            print(f"[Validation @ Step {log_step}] no validation batches were produced.")
        return

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
        elapsed = time.time() - val_start_time
        print(
            (
                f"[Validation @ Step {log_step}] "
                f"loss={format_metric(avg_val_metrics, 'loss_value')} "
                f"align={format_metric(avg_val_metrics, 'attn_align_loss')} "
                f"router={format_metric(avg_val_metrics, 'router_loss')} "
                f"keep={format_metric(avg_val_metrics, 'router_keep_ratio')} "
                f"kept_tokens={format_metric(avg_val_metrics, 'router_selected_tokens')} "
                f"kept_base={format_metric(avg_val_metrics, 'router_selected_tokens_base')} "
                f"kept_expert={format_metric(avg_val_metrics, 'router_selected_tokens_expert')} "
                f"llm_vis={format_metric(avg_val_metrics, 'llm_visual_tokens')} "
                f"gstd={format_metric(avg_val_metrics, 'router_gate_std')} "
                f"growstd={format_metric(avg_val_metrics, 'router_gate_row_std')} "
                f"gq10={format_metric(avg_val_metrics, 'router_gate_q10')} "
                f"gq50={format_metric(avg_val_metrics, 'router_gate_q50')} "
                f"gq90={format_metric(avg_val_metrics, 'router_gate_q90')} "
                f"gent={format_metric(avg_val_metrics, 'router_gate_entropy')} "
                f"curr_l1={format_metric(avg_val_metrics, 'curr_action_l1_loss')} "
                f"next_l1={format_metric(avg_val_metrics, 'next_actions_l1_loss')} "
                f"batches={val_batches_count} "
                f"time={elapsed:.1f}s"
            )
        )


def get_checkpoint_dir_for_step(cfg: FinetuneConfig, run_dir: Path, log_step: int) -> Path:
    """Return checkpoint directory that corresponds to `log_step`."""
    if cfg.save_latest_checkpoint_only:
        return run_dir
    return Path(str(run_dir) + f"--{log_step}_chkpt")


def parse_rollout_success_rate(eval_output: str) -> Optional[float]:
    """Extract overall success rate from LIBERO eval stdout/stderr."""
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    text = ansi_escape.sub("", eval_output)

    patterns = [
        r"Overall success rate:\s*([0-9]*\.?[0-9]+)",
        r"Current total success rate:\s*([0-9]*\.?[0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue

    # Fallback: infer success rate from totals if present.
    total_successes = re.search(r"Total successes:\s*([0-9]+)", text, flags=re.IGNORECASE)
    total_episodes = re.search(r"Total episodes:\s*([0-9]+)", text, flags=re.IGNORECASE)
    if total_successes is not None and total_episodes is not None:
        try:
            successes = float(total_successes.group(1))
            episodes = float(total_episodes.group(1))
        except ValueError:
            return None
        if episodes > 0:
            return successes / episodes
    return None


def find_rollout_eval_log_file(rollout_log_dir: Path, log_step: int) -> Optional[Path]:
    """Return latest rollout eval txt log for a given train step."""
    candidates = sorted(
        rollout_log_dir.glob(f"*--train-step-{log_step}.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(candidates) == 0:
        return None
    return candidates[0]


def resolve_libero_pythonpath(project_root: Path, cfg: FinetuneConfig) -> Optional[str]:
    """
    Resolve a LIBERO package root to prepend to PYTHONPATH.

    Expected package layout at `<candidate>/libero/__init__.py`.
    """
    candidate_strings: List[Optional[str]] = [
        cfg.rollout_libero_pythonpath,
        os.environ.get("ROLLOUT_LIBERO_PYTHONPATH"),
        os.environ.get("LIBERO_PYTHONPATH"),
    ]

    candidate_paths: List[Path] = []
    for parent in [project_root, project_root.parent]:
        candidate_paths.extend(
            [
                parent / "VLA-Adapter" / "LIBERO",
                parent / "VLA-Adapter" / "LIBERO" / "libero",
                parent / "LIBERO",
                parent / "LIBERO" / "libero",
            ]
        )

    for candidate_str in candidate_strings:
        if candidate_str:
            candidate_paths.insert(0, Path(candidate_str).expanduser())

    for candidate in candidate_paths:
        try:
            # Standard package layout
            if (candidate / "libero" / "__init__.py").is_file():
                return str(candidate.resolve())
            # LIBERO repo layout used in this workspace (namespace package + nested module)
            if (candidate / "libero" / "libero" / "__init__.py").is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    return None


def get_libero_benchmark_root_from_pythonpath(libero_pythonpath: str) -> Optional[Path]:
    """Infer LIBERO benchmark_root (directory with bddl_files/init_files/assets) from pythonpath root."""
    root = Path(libero_pythonpath)
    candidates = [
        root / "libero",             # standard package layout
        root / "libero" / "libero",  # LIBERO repo namespace layout
    ]
    for candidate in candidates:
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def write_libero_config_noninteractive(config_dir: Path, benchmark_root: Path) -> Path:
    """
    Write a minimal LIBERO config.yaml to avoid interactive prompts at first import.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return config_file

    default_paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with open(config_file, "w") as f:
        for key, value in default_paths.items():
            f.write(f"{key}: {value}\n")
    return config_file


def run_rollout_validation(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    distributed_state: PartialState,
    wandb_module,
) -> None:
    """
    Run LIBERO rollout validation from the current checkpoint and log success rate.

    Notes:
    - Runs only on main process; other ranks wait at barriers.
    - This validation is environment rollout based and can take several minutes.
    """
    dist.barrier()
    rollout_sync_dir = run_dir / ".rollout_sync"
    rollout_done_file = rollout_sync_dir / f"step_{log_step}.done"
    if distributed_state.is_main_process:
        rollout_sync_dir.mkdir(parents=True, exist_ok=True)
        rollout_done_file.unlink(missing_ok=True)
    dist.barrier()

    if distributed_state.is_main_process:
        try:
            project_root = Path(__file__).resolve().parents[1]
            checkpoint_dir = get_checkpoint_dir_for_step(cfg, run_dir, log_step)
            rollout_log_dir = str(Path(cfg.rollout_local_log_dir) / "rollout_validation")
            print(
                "[Rollout Validation] "
                f"starting at step={log_step} "
                f"(suite={cfg.rollout_task_suite_name}, "
                f"trials/task={cfg.rollout_num_trials_per_task}, "
                f"max_tasks={cfg.rollout_max_tasks})"
            )

            eval_env = os.environ.copy()
            # Force local repo code to resolve first in rollout subprocesses.
            existing_pythonpath = eval_env.get("PYTHONPATH", "")
            pythonpath_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
            pythonpath_entries = [entry for entry in pythonpath_entries if entry != str(project_root)]
            eval_env["PYTHONPATH"] = os.pathsep.join([str(project_root)] + pythonpath_entries)

            # Ensure rollout subprocess runs as a standalone process, not as part of the current DDP job.
            dist_env_exact = {
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "NODE_RANK",
            }
            dist_env_prefixes = ("TORCHELASTIC_", "ACCELERATE_")
            for env_key in list(eval_env.keys()):
                if env_key in dist_env_exact or any(env_key.startswith(prefix) for prefix in dist_env_prefixes):
                    eval_env.pop(env_key, None)

            rollout_log_dir_path = Path(rollout_log_dir)
            libero_config_dir = rollout_log_dir_path / ".libero"
            eval_env["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
            benchmark_root_for_config: Optional[Path] = None
            if importlib.util.find_spec("libero") is None:
                libero_pythonpath = resolve_libero_pythonpath(project_root, cfg)
                if libero_pythonpath is None:
                    print(
                        f"[Rollout Validation @ Step {log_step}] skipped: `libero` is not importable and "
                        "no local LIBERO package root was found. Set `ROLLOUT_LIBERO_PYTHONPATH` "
                        "(or `LIBERO_PYTHONPATH`) to a directory that contains `libero/__init__.py`."
                    )
                    return
                existing_pythonpath = eval_env.get("PYTHONPATH", "")
                eval_env["PYTHONPATH"] = (
                    f"{libero_pythonpath}:{existing_pythonpath}" if existing_pythonpath else libero_pythonpath
                )
                print(f"[Rollout Validation] using local LIBERO package path: {libero_pythonpath}")
                benchmark_root_for_config = get_libero_benchmark_root_from_pythonpath(libero_pythonpath)
            else:
                spec = importlib.util.find_spec("libero")
                if spec is not None and spec.origin is not None:
                    benchmark_root_for_config = Path(spec.origin).resolve().parent

            if benchmark_root_for_config is not None:
                config_file = write_libero_config_noninteractive(libero_config_dir, benchmark_root_for_config)
                print(f"[Rollout Validation] using LIBERO config: {config_file}")

            probe_cmd = [
                sys.executable,
                "-c",
                (
                    "from experiments.robot.libero.run_libero_eval import get_libero_runtime_modules; "
                    "get_libero_runtime_modules(); "
                    "print('ok')"
                ),
            ]
            probe = subprocess.run(
                probe_cmd,
                cwd=str(project_root),
                env=eval_env,
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                probe_tail = "\n".join((probe.stdout + "\n" + probe.stderr).splitlines()[-20:])
                print(
                    f"[Rollout Validation @ Step {log_step}] skipped: cannot import LIBERO in eval subprocess. "
                    "Training will continue."
                )
                if probe_tail:
                    print("[Rollout Validation] import probe output:")
                    print(probe_tail)
                return

            rollout_load_in_8bit = bool(cfg.rollout_load_in_8bit)
            rollout_load_in_4bit = bool(cfg.rollout_load_in_4bit)
            if (rollout_load_in_8bit or rollout_load_in_4bit) and importlib.util.find_spec("bitsandbytes") is None:
                print(
                    "[Rollout Validation] `bitsandbytes` is not installed in this environment; "
                    "overriding eval quantization flags to load_in_8bit=False, load_in_4bit=False."
                )
                rollout_load_in_8bit = False
                rollout_load_in_4bit = False

            cmd = [
                sys.executable,
                "experiments/robot/libero/run_libero_eval.py",
                "--pretrained_checkpoint",
                str(checkpoint_dir),
                "--base_model_path",
                str(cfg.vla_path),
                "--task_suite_name",
                cfg.rollout_task_suite_name,
                "--num_trials_per_task",
                str(cfg.rollout_num_trials_per_task),
                "--max_tasks",
                str(cfg.rollout_max_tasks),
                "--save_rollout_videos",
                str(cfg.rollout_save_videos),
                "--center_crop",
                str(cfg.rollout_center_crop),
                "--local_log_dir",
                rollout_log_dir,
                "--seed",
                str(cfg.rollout_seed),
                "--use_wandb",
                "False",
                "--use_l1_regression",
                str(cfg.use_l1_regression),
                "--use_diffusion",
                str(cfg.use_diffusion),
                "--openvla_baseline",
                str(cfg.openvla_baseline),
                "--num_diffusion_steps_train",
                str(cfg.num_diffusion_steps_train),
                "--num_diffusion_steps_inference",
                str(cfg.num_diffusion_steps_train),
                "--use_film",
                str(cfg.use_film),
                "--use_siglip_only_vision",
                str(cfg.use_siglip_only_vision),
                "--use_dino_only_vision",
                str(cfg.use_dino_only_vision),
                "--train_vla_projector",
                str(cfg.train_vla_projector),
                "--vision_lora",
                str(cfg.vision_lora),
                "--use_knowledge_router",
                str(cfg.use_knowledge_router),
                "--knowledge_router_num_heads",
                str(cfg.knowledge_router_num_heads),
                "--knowledge_router_hidden_dim",
                str(cfg.knowledge_router_hidden_dim),
                "--knowledge_router_dropout",
                str(cfg.knowledge_router_dropout),
                "--knowledge_router_temperature",
                str(cfg.knowledge_router_temperature),
                "--knowledge_router_target_keep_ratio",
                str(cfg.knowledge_router_target_keep_ratio),
                "--knowledge_router_min_keep_tokens",
                str(cfg.knowledge_router_min_keep_tokens),
                "--knowledge_router_hard_routing",
                str(cfg.knowledge_router_hard_routing),
                "--knowledge_router_focal_gamma",
                str(cfg.knowledge_router_focal_gamma),
                "--knowledge_router_effective_num_beta",
                str(cfg.knowledge_router_effective_num_beta),
                "--knowledge_router_token_fusion_mode",
                str(cfg.knowledge_router_token_fusion_mode),
                "--visual_path_mode",
                str(cfg.visual_path_mode),
                "--num_images_in_input",
                str(cfg.num_images_in_input),
                "--use_proprio",
                str(cfg.use_proprio),
                "--lora_rank",
                str(cfg.lora_rank),
                "--load_in_8bit",
                str(rollout_load_in_8bit),
                "--load_in_4bit",
                str(rollout_load_in_4bit),
                "--run_id_note",
                f"train-step-{log_step}",
            ]

            start = time.time()
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                env=eval_env,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start
            eval_output = (proc.stdout or "") + "\n" + (proc.stderr or "")

            if proc.returncode != 0:
                print(
                    f"[Rollout Validation @ Step {log_step}] failed with return code {proc.returncode} "
                    f"after {elapsed:.1f}s. Training will continue."
                )
                tail_lines = "\n".join(eval_output.splitlines()[-30:])
                if tail_lines:
                    print("[Rollout Validation] last output lines:")
                    print(tail_lines)
            else:
                success_rate = parse_rollout_success_rate(eval_output)
                if success_rate is None:
                    eval_log_file = find_rollout_eval_log_file(rollout_log_dir_path, log_step)
                    if eval_log_file is not None:
                        try:
                            success_rate = parse_rollout_success_rate(eval_log_file.read_text())
                        except OSError:
                            success_rate = None
                if success_rate is None:
                    print(
                        f"[Rollout Validation @ Step {log_step}] completed in {elapsed:.1f}s "
                        "but could not parse `Overall success rate` from eval output."
                    )
                else:
                    print(
                        f"[Rollout Validation @ Step {log_step}] "
                        f"success_rate={success_rate:.4f} ({success_rate * 100:.1f}%) "
                        f"time={elapsed:.1f}s"
                    )
                    wandb_module.log(
                        {
                            "VLA Rollout/Success Rate": success_rate,
                            "VLA Rollout/Elapsed Sec": elapsed,
                            "VLA Rollout/Num Trials Per Task": cfg.rollout_num_trials_per_task,
                            "VLA Rollout/Max Tasks": cfg.rollout_max_tasks,
                        },
                        step=log_step,
                    )
        finally:
            try:
                rollout_done_file.write_text(f"done@{time.time()}\n")
            except OSError:
                pass
    else:
        while not rollout_done_file.exists():
            time.sleep(1.0)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"

    # OpenVLA baseline mode: disable SF-specific alignment/visual projectors.
    if cfg.openvla_baseline:
        cfg.use_vggt_alignment = False
        cfg.align_loss_coeff = 0.0
        cfg.use_l1_regression = True
        cfg.use_diffusion = False
        cfg.use_film = False
        cfg.use_siglip_only_vision = False
        cfg.use_dino_only_vision = False
        cfg.visual_path_mode = "dual"
        cfg.use_knowledge_router = False

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )
    if cfg.use_siglip_only_vision and cfg.use_dino_only_vision:
        raise ValueError("Cannot enable both use_siglip_only_vision and use_dino_only_vision.")
    if cfg.use_siglip_only_vision and cfg.use_film:
        raise ValueError("use_siglip_only_vision=True is currently unsupported with use_film=True.")
    if cfg.use_dino_only_vision and cfg.use_film:
        raise ValueError("use_dino_only_vision=True is currently unsupported with use_film=True.")
    if cfg.visual_path_mode not in {
        "dual",
        "base_only",
        "expert_only",
        "base_only_separate",
        "expert_only_separate",
    }:
        raise ValueError(
            f"Unsupported visual_path_mode={cfg.visual_path_mode}. "
            "Use one of: dual, base_only, expert_only, base_only_separate, expert_only_separate."
        )
    if not (0.0 < cfg.knowledge_router_target_keep_ratio <= 1.0):
        raise ValueError("knowledge_router_target_keep_ratio must be in (0, 1].")
    if cfg.knowledge_router_min_keep_tokens < 1:
        raise ValueError("knowledge_router_min_keep_tokens must be >= 1.")
    if cfg.knowledge_router_warmup_steps < 0:
        raise ValueError("knowledge_router_warmup_steps must be >= 0.")
    if not (0.0 <= cfg.knowledge_router_effective_num_beta < 1.0):
        raise ValueError("knowledge_router_effective_num_beta must be in [0, 1).")
    if cfg.knowledge_router_num_heads < 1:
        raise ValueError("knowledge_router_num_heads must be >= 1.")
    if cfg.knowledge_router_hidden_dim < 1:
        raise ValueError("knowledge_router_hidden_dim must be >= 1.")
    if not (0.0 <= cfg.knowledge_router_importance_ema_momentum < 1.0):
        raise ValueError("knowledge_router_importance_ema_momentum must be in [0, 1).")
    if cfg.restrict_lora_to_vision and (cfg.freeze_vision_lora or not cfg.vision_lora):
        raise ValueError(
            "restrict_lora_to_vision=True with freeze_vision_lora=True or vision_lora=False would disable all trainable "
            "LoRA adapters. Choose only one strategy."
        )
    if cfg.resume and cfg.resume_step is None:
        raise ValueError("resume=True requires a valid `resume_step`.")

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    if cfg.resume_checkpoint_dir is not None:
        cfg.resume_checkpoint_dir = cfg.resume_checkpoint_dir.rstrip("/")
    if cfg.resume:
        resume_ckpt_dir = Path(get_resume_checkpoint_dir(cfg))
        if not resume_ckpt_dir.is_dir():
            raise FileNotFoundError(
                f"Resume requested but checkpoint directory does not exist: `{resume_ckpt_dir}`"
            )
        if is_main_process():
            print(
                f"[Resume] base_vla_path=`{cfg.vla_path}` | "
                f"checkpoint_dir=`{resume_ckpt_dir}` | step={cfg.resume_step}"
            )

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    if os.environ.get("QUIET_NON_MAIN_RANKS", "False").lower() in {"1", "true", "yes"} and not distributed_state.is_main_process:
        # Keep detached log readable by suppressing non-main rank stdout/stderr.
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
        if cfg.openvla_baseline:
            print(
                "[OpenVLA Baseline] Enabled: no VGGT alignment / no dual-path projectors; "
                "using L1 action objective."
            )
        if not cfg.use_vggt_alignment:
            print("[Attention Align] Disabled (`use_vggt_alignment=False`); training without VGGT alignment loss.")
            if cfg.align_loss_coeff != 0:
                print(
                    f"[Attention Align] Ignoring align_loss_coeff={cfg.align_loss_coeff} "
                    "because VGGT alignment is disabled."
                )
        if cfg.visual_path_mode != "dual":
            print(f"[Visual Ablation] visual_path_mode=`{cfg.visual_path_mode}`")
        if cfg.train_vla_projector:
            print("[Projector] Training OpenVLA built-in vision projector (concat DINO+SigLIP -> LLM dim).")
        if not cfg.vision_lora:
            print("[LoRA] `vision_lora=False`: vision LoRAs will be disabled in forward pass.")
        if cfg.use_knowledge_router:
            print(
                "[Knowledge Router] Enabled: "
                f"keep_ratio={cfg.knowledge_router_target_keep_ratio}, "
                f"warmup={cfg.knowledge_router_warmup_steps}, "
                f"hard={cfg.knowledge_router_hard_routing}, "
                "fusion=concat_projector"
            )

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Print detected constants
    if distributed_state.is_main_process:
        print(
            "Detected constants:\n"
            f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
            f"\tACTION_DIM: {ACTION_DIM}\n"
            f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
            f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
        )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    source_vla_path = cfg.vla_path
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    resolved_vla_dir = Path(cfg.vla_path)
    if _is_native_prismatic_dir(resolved_vla_dir) and not _has_hf_pretrained_weights(resolved_vla_dir):
        if distributed_state.is_main_process:
            print(
                "[Prismatic Load] Detected native checkpoint format "
                "(config.json + checkpoints/*.pt); loading directly (MiniVLA-style)."
            )
        processor, vla = _convert_native_prismatic_dir_to_hf(
            resolved_vla_dir,
            return_artifacts_only=True,
        )
        vla = vla.to(device_id)

        # LoRA merge path expects `cfg.vla_path` to be a valid HF `from_pretrained` directory.
        if cfg.merge_lora_during_training:
            if distributed_state.is_main_process:
                print(
                    "[LoRA Merge] Disabled during this run because base model is loaded from "
                    "native Prismatic checkpoint format."
                )
            cfg.merge_lora_during_training = False
    else:
        # Support native Prismatic checkpoints by converting once to HF `from_pretrained` layout and caching locally.
        cfg.vla_path = maybe_convert_native_prismatic_checkpoint(
            resolved_vla_path=cfg.vla_path,
            source_vla_path=source_vla_path,
            run_root_dir=cfg.run_root_dir,
            distributed_state=distributed_state,
        )

        # Update config.json and sync model files
        if distributed_state.is_main_process:
            update_auto_map(cfg.vla_path)
            check_model_logic_mismatch(cfg.vla_path)

        # Wait for model files to be synced
        dist.barrier()

        # Load processor and VLA
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        if cfg.resume:
            resume_ckpt_dir = Path(get_resume_checkpoint_dir(cfg))
            adapter_dir = resume_ckpt_dir / "lora_adapter"
            if not adapter_dir.is_dir():
                raise FileNotFoundError(
                    f"Resume requested but LoRA adapter directory not found: `{adapter_dir}`"
                )
            try:
                vla = PeftModel.from_pretrained(vla, str(adapter_dir), is_trainable=True)
            except TypeError:
                # Older PEFT versions may not support `is_trainable`; manually re-enable LoRA grads.
                vla = PeftModel.from_pretrained(vla, str(adapter_dir))
                for name, param in vla.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
            if distributed_state.is_main_process:
                print(f"[LoRA Resume] Loaded adapter from `{adapter_dir}`")
        else:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
        if cfg.restrict_lora_to_vision:
            set_non_vision_lora_frozen(
                vla,
                siglip_only_vision=cfg.use_siglip_only_vision,
                dino_only_vision=cfg.use_dino_only_vision,
            )
            if distributed_state.is_main_process:
                print("[LoRA] Restricting trainable adapters to vision branch only.")
        elif not cfg.vision_lora:
            num_vision_lora_params = set_vision_lora_frozen(vla, disable_vision_lora_forward=True)
            if distributed_state.is_main_process:
                print(
                    "[LoRA] `vision_lora=False`: vision adapters disabled in forward; "
                    f"training non-vision adapters only. (vision_lora_params={num_vision_lora_params})"
                )
        elif cfg.freeze_vision_lora:
            num_vision_lora_params = set_vision_lora_frozen(vla, disable_vision_lora_forward=True)
            if distributed_state.is_main_process:
                print(
                    "[LoRA] Vision adapters frozen and zeroed (disabled in forward); "
                    f"training non-vision adapters only. (vision_lora_params={num_vision_lora_params})"
                )
        elif distributed_state.is_main_process:
            print("[LoRA] Training all LoRA adapters (vision + LLM).")

        if cfg.train_vla_projector:
            vla_core_projector_owner = get_vla_core(vla)
            for param in vla_core_projector_owner.projector.parameters():
                param.requires_grad = True
            if cfg.resume:
                resume_ckpt_dir = get_resume_checkpoint_dir(cfg)
                projector_state_dict = load_checkpoint("vla_projector", resume_ckpt_dir, cfg.resume_step)
                vla_core_projector_owner.projector.load_state_dict(projector_state_dict)
            if distributed_state.is_main_process:
                count_parameters(vla_core_projector_owner.projector, "vla_projector")

        if distributed_state.is_main_process:
            vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            resume_ckpt_dir = get_resume_checkpoint_dir(cfg)
            state_dict = load_checkpoint("vision_backbone", resume_ckpt_dir, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # Select VLA feature branch for alignment/fusion.
    alignment_branch = cfg.vla_alignment_branch
    if cfg.use_dino_only_vision and alignment_branch == "siglip":
        raise ValueError("use_dino_only_vision=True is incompatible with vla_alignment_branch='siglip'.")
    if cfg.use_siglip_only_vision and alignment_branch == "dino":
        raise ValueError("use_siglip_only_vision=True is incompatible with vla_alignment_branch='dino'.")
    if cfg.use_siglip_only_vision and alignment_branch == "auto":
        alignment_branch = "siglip"
    if cfg.use_dino_only_vision and alignment_branch == "auto":
        alignment_branch = "dino"

    vggt_model = None
    attention_aligner = None
    knowledge_router = None
    vla_qkv_collector = None
    vggt_qkv_collector = None
    vla_alignment_target_dim = None
    vla_alignment_featurizer = None
    fusion_projector = None
    single_path_projector = None
    action_head = None
    noisy_action_projector = None
    proprio_projector = None

    if cfg.use_vggt_alignment:
        # Load VGGT as a frozen vision encoder teacher.
        vggt_model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        vggt_model.load_state_dict(load_vggt_state_dict(cfg.vggt_path), strict=False)
        vggt_model = vggt_model.to(device_id)
        vggt_model.requires_grad_(False)
        vggt_model.eval()
        vla_alignment_target_dim = vggt_model.aggregator.patch_embed.embed_dim

    if not cfg.openvla_baseline:
        vla_alignment_featurizer = select_vla_alignment_featurizer(
            vla_core=get_vla_core(vla),
            vggt_dim=vla_alignment_target_dim,
            branch=alignment_branch,
        )

        # Build visual projector(s) and attention-response aligner.
        use_separate_single_path_projector = cfg.visual_path_mode in {"base_only_separate", "expert_only_separate"}
        if use_separate_single_path_projector:
            single_path_projector_args = {"llm_dim": vla.module.llm_dim}
            if cfg.use_siglip_only_vision or cfg.use_dino_only_vision:
                single_path_projector_args["input_dim"] = vla_alignment_featurizer.embed_dim
            single_path_projector = init_module(
                SinglePathProjector,
                "single_path_projector",
                cfg,
                device_id,
                single_path_projector_args,
            )
        if cfg.visual_path_mode in {"dual", "base_only", "expert_only"}:
            fusion_projector_args = {"llm_dim": vla.module.llm_dim}
            if cfg.use_siglip_only_vision or cfg.use_dino_only_vision:
                fusion_projector_args["input_dim"] = vla_alignment_featurizer.embed_dim
            fusion_projector = init_module(
                DualPathFusionProjector,
                "fusion_projector",
                cfg,
                device_id,
                fusion_projector_args,
            )
        if cfg.use_knowledge_router:
            router_token_dim = (
                vla_alignment_featurizer.embed_dim
                if (cfg.use_siglip_only_vision or cfg.use_dino_only_vision)
                else vla.module.llm_dim
            )
            knowledge_router = init_module(
                KnowledgeRouter,
                "knowledge_router",
                cfg,
                device_id,
                {
                    "text_dim": vla.module.llm_dim,
                    "token_dim": router_token_dim,
                    "num_heads": cfg.knowledge_router_num_heads,
                    "hidden_dim": cfg.knowledge_router_hidden_dim,
                    "dropout": cfg.knowledge_router_dropout,
                    "temperature": cfg.knowledge_router_temperature,
                    "focal_gamma": cfg.knowledge_router_focal_gamma,
                    "effective_num_beta": cfg.knowledge_router_effective_num_beta,
                },
            )
        if cfg.use_vggt_alignment:
            attention_aligner = init_module(
                AttentionResponseAligner,
                "attention_aligner",
                cfg,
                device_id,
                {
                    "vla_dim": vla_alignment_featurizer.embed_dim,
                    "vggt_dim": vggt_model.aggregator.patch_embed.embed_dim,
                    "hidden_dim": cfg.attn_align_hidden_dim,
                    "loss_type": cfg.align_loss_type,
                    "temperature": cfg.attn_align_temperature,
                },
            )

            vla_qkv_collector = AttentionQKVCollector(get_attention_modules_from_featurizer(vla_alignment_featurizer))
            vggt_qkv_collector = AttentionQKVCollector(
                get_attention_modules_from_featurizer(vggt_model.aggregator.patch_embed)
            )

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if fusion_projector is not None:
        trainable_params += [param for param in fusion_projector.parameters() if param.requires_grad]
    if single_path_projector is not None:
        trainable_params += [param for param in single_path_projector.parameters() if param.requires_grad]
    if attention_aligner is not None:
        trainable_params += [param for param in attention_aligner.parameters() if param.requires_grad]
    if knowledge_router is not None:
        trainable_params += [param for param in knowledge_router.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if distributed_state.is_main_process:
        print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    if cfg.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(
            optimizer,
            milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
            gamma=0.1,  # Multiplicative factor of learning rate decay
        )
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_steps,  # Total number of steps for the cosine annealing
            eta_min=cfg.learning_rate * 1e-3,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.scheduler}")

    # Resume optimizer/scheduler states for genuine continuation.
    if cfg.resume:
        resume_ckpt_dir = get_resume_checkpoint_dir(cfg)
        optimizer_loaded = False
        scheduler_loaded = False

        try:
            optimizer_state = load_raw_checkpoint("optimizer", resume_ckpt_dir, cfg.resume_step, device="cpu")
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, torch.device(f"cuda:{device_id}"))
            optimizer_loaded = True
        except FileNotFoundError:
            if distributed_state.is_main_process:
                print(
                    "[Resume] Optimizer checkpoint not found; continuing with a fresh optimizer state. "
                    "(This can cause a temporary loss jump.)"
                )

        try:
            scheduler_state = load_raw_checkpoint("scheduler", resume_ckpt_dir, cfg.resume_step, device="cpu")
            scheduler.load_state_dict(scheduler_state)
            scheduler_loaded = True
        except FileNotFoundError:
            if distributed_state.is_main_process:
                print(
                    "[Resume] Scheduler checkpoint not found; continuing with a fresh scheduler state. "
                    "(This can change LR progression after resume.)"
                )

        if distributed_state.is_main_process:
            print(
                f"[Resume] optimizer_loaded={optimizer_loaded}, "
                f"scheduler_loaded={scheduler_loaded}"
            )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    sync_action_token_begin_idx(action_tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "attn_align_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_cls_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_action_sup_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_budget_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_entropy_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "router_keep_ratio": deque(maxlen=cfg.grad_accumulation_steps),
        "router_selected_tokens": deque(maxlen=cfg.grad_accumulation_steps),
        "router_selected_tokens_base": deque(maxlen=cfg.grad_accumulation_steps),
        "router_selected_tokens_expert": deque(maxlen=cfg.grad_accumulation_steps),
        "llm_visual_tokens": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_std": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_row_std": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_min": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_max": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_q10": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_q50": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_q90": deque(maxlen=cfg.grad_accumulation_steps),
        "router_gate_entropy": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training (offset tqdm when resuming so visual progress matches `log_step`).
    resume_offset = int(cfg.resume_step) if (cfg.resume and cfg.resume_step is not None) else 0
    resume_offset = max(0, min(resume_offset, int(cfg.max_steps)))
    with tqdm.tqdm(
        total=cfg.max_steps,
        initial=resume_offset,
        leave=False,
        disable=not distributed_state.is_main_process,
    ) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                knowledge_router=knowledge_router,
                attention_aligner=attention_aligner,
                fusion_projector=fusion_projector,
                single_path_projector=single_path_projector,
                vla_qkv_collector=vla_qkv_collector,
                vggt_qkv_collector=vggt_qkv_collector,
                vggt=vggt_model,
                layers_align=(cfg.vla_layers_align, cfg.vggt_layers_align),
                processor=processor,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_vggt_alignment=cfg.use_vggt_alignment,
                align_loss_coeff=cfg.align_loss_coeff,
                align_all_layers=cfg.align_all_layers,
                freeze_base_visual_path=cfg.freeze_base_visual_path,
                visual_path_mode=cfg.visual_path_mode,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                use_siglip_only_vision=cfg.use_siglip_only_vision,
                use_dino_only_vision=cfg.use_dino_only_vision,
                num_patches=NUM_PATCHES,
                num_images_in_input=cfg.num_images_in_input,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                openvla_baseline=cfg.openvla_baseline,
                use_knowledge_router=cfg.use_knowledge_router,
                knowledge_router_loss_coeff=cfg.knowledge_router_loss_coeff,
                knowledge_router_budget_loss_coeff=cfg.knowledge_router_budget_loss_coeff,
                knowledge_router_entropy_loss_coeff=cfg.knowledge_router_entropy_loss_coeff,
                knowledge_router_target_keep_ratio=cfg.knowledge_router_target_keep_ratio,
                knowledge_router_min_keep_tokens=cfg.knowledge_router_min_keep_tokens,
                knowledge_router_hard_routing=cfg.knowledge_router_hard_routing,
                knowledge_router_token_fusion_mode=cfg.knowledge_router_token_fusion_mode,
                knowledge_router_warmup_steps=cfg.knowledge_router_warmup_steps,
                knowledge_router_importance_ema_momentum=cfg.knowledge_router_importance_ema_momentum,
                current_step=(
                    (batch_idx + 1) // cfg.grad_accumulation_steps
                    + (cfg.resume_step if (cfg.resume and cfg.resume_step is not None) else 0)
                ),
            )

            # Non-finite guard: if any rank sees NaN/Inf, skip this micro-batch update everywhere.
            finite_flag = torch.tensor(
                int(torch.isfinite(loss.detach()).all().item()),
                device=device_id,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if int(finite_flag.item()) == 0:
                if distributed_state.is_main_process:
                    candidate_step = (
                        (batch_idx + 1) // cfg.grad_accumulation_steps
                        + (cfg.resume_step if (cfg.resume and cfg.resume_step is not None) else 0)
                    )
                    print(f"[Warning] Non-finite loss detected at/near step {candidate_step}; skipping this batch.")
                optimizer.zero_grad()
                continue

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Only trigger logging/validation/checkpointing on real optimizer steps.
            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue

            # Compute gradient step index (1-based update count)
            gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min(gradient_step_idx / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            # Optimizer and LR scheduler step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update()

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push metrics to W&B and console (every wandb_log_freq gradient steps)
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )
                total_steps = cfg.max_steps
                tqdm.tqdm.write(
                    (
                        f"[Step {log_step} / {total_steps}] "
                        f"loss={format_metric(smoothened_metrics, 'loss_value')} "
                        f"align={format_metric(smoothened_metrics, 'attn_align_loss')} "
                        f"router={format_metric(smoothened_metrics, 'router_loss')} "
                        f"r_cls={format_metric(smoothened_metrics, 'router_cls_loss')} "
                        f"r_budget={format_metric(smoothened_metrics, 'router_budget_loss')} "
                        f"r_ent={format_metric(smoothened_metrics, 'router_entropy_loss')} "
                        f"keep={format_metric(smoothened_metrics, 'router_keep_ratio')} "
                        f"kept_tokens={format_metric(smoothened_metrics, 'router_selected_tokens')} "
                        f"kept_base={format_metric(smoothened_metrics, 'router_selected_tokens_base')} "
                        f"kept_expert={format_metric(smoothened_metrics, 'router_selected_tokens_expert')} "
                        f"llm_vis={format_metric(smoothened_metrics, 'llm_visual_tokens')} "
                        f"gstd={format_metric(smoothened_metrics, 'router_gate_std')} "
                        f"growstd={format_metric(smoothened_metrics, 'router_gate_row_std')} "
                        f"gq10={format_metric(smoothened_metrics, 'router_gate_q10')} "
                        f"gq50={format_metric(smoothened_metrics, 'router_gate_q50')} "
                        f"gq90={format_metric(smoothened_metrics, 'router_gate_q90')} "
                        f"gent={format_metric(smoothened_metrics, 'router_gate_entropy')} "
                        f"curr_l1={format_metric(smoothened_metrics, 'curr_action_l1_loss')} "
                        f"next_l1={format_metric(smoothened_metrics, 'next_actions_l1_loss')} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}"
                    )
                )

            checkpoint_saved_this_step = False

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if log_step % cfg.save_freq == 0:
                torch.cuda.empty_cache()
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    attention_aligner=attention_aligner,
                    knowledge_router=knowledge_router,
                    fusion_projector=fusion_projector,
                    single_path_projector=single_path_projector,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )
                checkpoint_saved_this_step = True

            # Test model on validation set
            if cfg.use_val_set and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    knowledge_router=knowledge_router,
                    attention_aligner=attention_aligner,
                    fusion_projector=fusion_projector,
                    single_path_projector=single_path_projector,
                    vla_qkv_collector=vla_qkv_collector,
                    vggt_qkv_collector=vggt_qkv_collector,
                    vggt=vggt_model,
                    processor=processor,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Rollout-based validation on LIBERO environment (success rate).
            if cfg.use_rollout_val and log_step % cfg.rollout_val_freq == 0:
                if not checkpoint_saved_this_step:
                    torch.cuda.empty_cache()
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=log_step,
                        vla=vla,
                        processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        attention_aligner=attention_aligner,
                        knowledge_router=knowledge_router,
                        fusion_projector=fusion_projector,
                        single_path_projector=single_path_projector,
                        action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_dataset=train_dataset,
                        distributed_state=distributed_state,
                    )
                run_rollout_validation(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    wandb_module=wandb,
                )
                vla.train()

            # Stop training when max_steps is reached
            if log_step >= cfg.max_steps:
                if distributed_state.is_main_process:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
