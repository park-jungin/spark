"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import contextlib
import filecmp
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    DualPathFusionProjector,
    KnowledgeRouter,
    NoisyActionProjector,
    ProprioProjector,
    SinglePathProjector,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Last router visualization payload captured at inference time (single model query).
_LAST_ROUTER_VIZ: Optional[Dict[str, Any]] = None


def clear_last_router_viz() -> None:
    """Clear cached router visualization payload."""
    global _LAST_ROUTER_VIZ
    _LAST_ROUTER_VIZ = None


def set_last_router_viz(payload: Optional[Dict[str, Any]]) -> None:
    """Store router visualization payload for downstream tooling (e.g., eval video overlays)."""
    global _LAST_ROUTER_VIZ
    if payload is None:
        _LAST_ROUTER_VIZ = None
        return
    copied: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    _LAST_ROUTER_VIZ = copied


def get_last_router_viz() -> Optional[Dict[str, Any]]:
    """Return a copy of the latest cached router visualization payload."""
    if _LAST_ROUTER_VIZ is None:
        return None
    copied: Dict[str, Any] = {}
    for key, value in _LAST_ROUTER_VIZ.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Selection policy:
        - Prefer `--latest_checkpoint.pt` if present.
        - Otherwise choose the largest numeric step `--{step}_checkpoint.pt`.

    Raises:
        AssertionError: If no files match the pattern.
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) >= 1, (
        f"Expected at least 1 {file_pattern} checkpoint but found 0 in directory: {pretrained_checkpoint}"
    )

    def _sort_key(path: str) -> Tuple[int, int]:
        filename = os.path.basename(path)
        if filename.endswith("--latest_checkpoint.pt"):
            return (2, 0)
        match = re.search(r"--(\d+)_checkpoint\.pt$", filename)
        if match is not None:
            return (1, int(match.group(1)))
        return (0, 0)

    selected = max(checkpoint_files, key=_sort_key)
    if len(checkpoint_files) > 1:
        print(
            f"WARNING: found {len(checkpoint_files)} files matching `{file_pattern}` in `{pretrained_checkpoint}`; "
            f"using `{os.path.basename(selected)}`"
        )
    return selected


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_vla_core_model(vla: torch.nn.Module) -> torch.nn.Module:
    """Return underlying OpenVLA model from possible PEFT wrapper."""
    if hasattr(vla, "base_model") and hasattr(vla.base_model, "model"):
        return vla.base_model.model
    return vla


def get_vla_adapter_context(vla: torch.nn.Module):
    """Return adapter-disable context if available, otherwise no-op context."""
    if hasattr(vla, "disable_adapter"):
        return vla.disable_adapter()
    return contextlib.nullcontext()


def get_vla_vision_backbone(vla_core: torch.nn.Module) -> torch.nn.Module:
    """Unwrap FiLM wrapper (if present) and return the raw vision backbone."""
    vision_backbone = vla_core.vision_backbone
    if hasattr(vision_backbone, "vision_backbone"):
        vision_backbone = vision_backbone.vision_backbone
    return vision_backbone


def disable_vision_lora_inplace(vla: torch.nn.Module) -> int:
    """
    Disable vision-side LoRA contribution by zeroing LoRA params under vision_backbone.
    Returns number of parameters touched.
    """
    touched_params = 0
    with torch.no_grad():
        for name, param in vla.named_parameters():
            if "lora_" not in name or "vision_backbone" not in name:
                continue
            param.requires_grad = False
            param.zero_()
            touched_params += param.numel()
    return touched_params


def extract_dino_pixels(pixel_values: torch.Tensor, num_images_in_input: int) -> torch.Tensor:
    """Extract DINO channels from fused [DINO(3), SigLIP(3)] stacks per image."""
    channels = pixel_values.shape[1]
    fused_channels = 6 * num_images_in_input
    dino_channels = 3 * num_images_in_input
    if channels == dino_channels:
        return pixel_values
    if channels == fused_channels:
        chunks = torch.split(pixel_values, [6] * num_images_in_input, dim=1)
        dino_chunks = [chunk[:, 0:3] for chunk in chunks]
        return torch.cat(dino_chunks, dim=1)
    raise ValueError(
        f"Unexpected pixel channel count={channels}. Expected {dino_channels} (DINO-only) "
        f"or {fused_channels} (DINO+SigLIP) for num_images_in_input={num_images_in_input}."
    )


def run_dino_encoder_only(
    vla_core: torch.nn.Module,
    pixel_values: torch.Tensor,
    num_images_in_input: int,
) -> torch.Tensor:
    """Run only the DINO branch and return patch embeddings."""
    vision_backbone = get_vla_vision_backbone(vla_core)
    if hasattr(vision_backbone, "dino_featurizer"):
        dino_featurizer = vision_backbone.dino_featurizer
    elif hasattr(vision_backbone, "featurizer"):
        dino_featurizer = vision_backbone.featurizer
    elif hasattr(vision_backbone, "fused_featurizer"):
        dino_featurizer = vision_backbone.fused_featurizer
    else:
        raise ValueError("Unable to locate DINO featurizer in vision backbone.")

    dino_pixels = extract_dino_pixels(pixel_values, num_images_in_input)
    if num_images_in_input == 1:
        return dino_featurizer(dino_pixels)

    per_image_pixels = torch.split(dino_pixels, [3] * num_images_in_input, dim=1)
    per_image_patches = [dino_featurizer(img) for img in per_image_pixels]
    return torch.cat(per_image_patches, dim=1)


def extract_siglip_pixels(pixel_values: torch.Tensor, num_images_in_input: int) -> torch.Tensor:
    """Extract SigLIP channels from fused [DINO(3), SigLIP(3)] stacks per image."""
    channels = pixel_values.shape[1]
    fused_channels = 6 * num_images_in_input
    siglip_channels = 3 * num_images_in_input
    if channels == siglip_channels:
        return pixel_values
    if channels == fused_channels:
        chunks = torch.split(pixel_values, [6] * num_images_in_input, dim=1)
        siglip_chunks = [chunk[:, 3:6] for chunk in chunks]
        return torch.cat(siglip_chunks, dim=1)
    raise ValueError(
        f"Unexpected pixel channel count={channels}. Expected {siglip_channels} (SigLIP-only) "
        f"or {fused_channels} (DINO+SigLIP) for num_images_in_input={num_images_in_input}."
    )


def run_siglip_encoder_only(
    vla_core: torch.nn.Module,
    pixel_values: torch.Tensor,
    num_images_in_input: int,
) -> torch.Tensor:
    """Run only the SigLIP branch and return patch embeddings."""
    vision_backbone = get_vla_vision_backbone(vla_core)
    if hasattr(vision_backbone, "fused_featurizer"):
        siglip_featurizer = vision_backbone.fused_featurizer
    elif hasattr(vision_backbone, "siglip_featurizer"):
        siglip_featurizer = vision_backbone.siglip_featurizer
    elif hasattr(vision_backbone, "featurizer"):
        siglip_featurizer = vision_backbone.featurizer
    else:
        raise ValueError("Unable to locate SigLIP featurizer in vision backbone.")

    siglip_pixels = extract_siglip_pixels(pixel_values, num_images_in_input)
    if num_images_in_input == 1:
        return siglip_featurizer(siglip_pixels)

    per_image_pixels = torch.split(siglip_pixels, [3] * num_images_in_input, dim=1)
    per_image_patches = [siglip_featurizer(img) for img in per_image_pixels]
    return torch.cat(per_image_patches, dim=1)


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    use_siglip_only_vision = getattr(cfg, "use_siglip_only_vision", False)
    use_dino_only_vision = getattr(cfg, "use_dino_only_vision", False)
    openvla_baseline = bool(getattr(cfg, "openvla_baseline", False))
    if use_siglip_only_vision and use_dino_only_vision:
        raise ValueError("Cannot enable both use_siglip_only_vision and use_dino_only_vision.")
    use_single_branch_vision = use_siglip_only_vision or use_dino_only_vision
    adapter_dir = os.path.join(cfg.pretrained_checkpoint, "lora_adapter")

    def _resolve_adapter_base_model_path() -> str:
        """
        Resolve base model path for loading a local LoRA adapter.

        Priority:
        1) `adapter_config.json` (`peft_cfg.base_model_name_or_path`)
        2) explicit eval arg `cfg.base_model_path`
        3) `<checkpoint_dir>/base_model_path.txt` (written during training checkpoint save)
        """
        peft_cfg = PeftConfig.from_pretrained(adapter_dir)
        base_model_path = str(getattr(peft_cfg, "base_model_name_or_path", "") or "").strip()
        if base_model_path:
            return base_model_path

        explicit_base_path = str(getattr(cfg, "base_model_path", "") or "").strip()
        if explicit_base_path:
            print(
                "WARNING: adapter base_model_name_or_path is empty. "
                f"Falling back to --base_model_path={explicit_base_path}"
            )
            return explicit_base_path

        sidecar_path = os.path.join(cfg.pretrained_checkpoint, "base_model_path.txt")
        if os.path.isfile(sidecar_path):
            try:
                sidecar_base_path = Path(sidecar_path).read_text().strip()
            except OSError:
                sidecar_base_path = ""
            if sidecar_base_path:
                print(
                    "WARNING: adapter base_model_name_or_path is empty. "
                    f"Falling back to checkpoint sidecar file: {sidecar_path}"
                )
                return sidecar_base_path

        raise ValueError(
            "Cannot resolve base model path for local LoRA adapter evaluation. "
            "Adapter config has empty `base_model_name_or_path`. "
            "Pass `--base_model_path <HF repo id or local model dir>` to run_libero_eval.py."
        )

    quantized_load = bool(getattr(cfg, "load_in_8bit", False) or getattr(cfg, "load_in_4bit", False))
    quantized_device_map = "auto" if quantized_load else None

    # Load PEFT adapter if requested and available.
    # - single-branch visual inference requires adapter context
    # - OpenVLA baseline mode may still use local adapter checkpoints
    should_load_local_adapter = (use_single_branch_vision or openvla_baseline)
    if should_load_local_adapter and (not model_is_on_hf_hub(cfg.pretrained_checkpoint)) and os.path.isdir(adapter_dir):
        base_model_path = _resolve_adapter_base_model_path()
        base_vla = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            device_map=quantized_device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    else:
        if use_single_branch_vision or openvla_baseline:
            print(
                "WARNING: adapter-style loading was requested but no local `lora_adapter/` was found. "
                "Loading checkpoint directly via AutoModelForVision2Seq.from_pretrained(...)."
            )
        # Load the model
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            device_map=quantized_device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if not bool(getattr(cfg, "vision_lora", True)):
        num_vision_lora_params = disable_vision_lora_inplace(vla)
        print(
            "[LoRA Eval] `vision_lora=False`: vision adapters disabled in forward "
            f"(vision_lora_params={num_vision_lora_params})."
        )

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    _load_vla_projector_if_requested(cfg, vla)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla_core = get_vla_core_model(vla)
        vla_core.norm_stats = norm_stats
        # Keep compatibility for wrappers that proxy attributes.
        if hasattr(vla, "norm_stats"):
            vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def _load_vla_projector_if_requested(cfg: Any, vla: torch.nn.Module) -> None:
    """Load fine-tuned OpenVLA projector weights when requested."""
    if not bool(getattr(cfg, "train_vla_projector", False)):
        return
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        raise ValueError(
            "train_vla_projector=True expects a local checkpoint directory with "
            "`vla_projector--*_checkpoint.pt`."
        )
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vla_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    vla_core = get_vla_core_model(vla)
    vla_core.projector.load_state_dict(state_dict, strict=True)
    print(f"Loaded fine-tuned VLA projector from: {checkpoint_path}")


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_fusion_projector(cfg: Any, llm_dim: int) -> DualPathFusionProjector:
    """
    Get dual-path fusion projector for SigLIP-only dual-path inference.
    """
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        raise ValueError("SigLIP-only dual-path inference currently expects a local checkpoint directory.")

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "fusion_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    if "fc2.weight" not in state_dict:
        raise ValueError(f"Invalid fusion projector checkpoint at {checkpoint_path}: missing `fc2.weight`.")

    fusion_input_dim = state_dict["fc2.weight"].shape[1] // 2
    fusion_projector = DualPathFusionProjector(llm_dim=llm_dim, input_dim=fusion_input_dim).to(DEVICE)
    fusion_projector = fusion_projector.to(torch.bfloat16).to(DEVICE)
    fusion_projector.eval()
    fusion_projector.load_state_dict(state_dict)
    return fusion_projector


def get_single_path_projector(cfg: Any, llm_dim: int) -> SinglePathProjector:
    """Get single-path projector for separate single-path visual ablations."""
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        raise ValueError("Single-path projector loading currently expects a local checkpoint directory.")

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "single_path_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    if "fc2.weight" not in state_dict:
        raise ValueError(f"Invalid single-path projector checkpoint at {checkpoint_path}: missing `fc2.weight`.")

    input_dim = state_dict["fc2.weight"].shape[1]
    single_path_projector = SinglePathProjector(llm_dim=llm_dim, input_dim=input_dim).to(DEVICE)
    single_path_projector = single_path_projector.to(torch.bfloat16).to(DEVICE)
    single_path_projector.eval()
    single_path_projector.load_state_dict(state_dict)
    return single_path_projector


def get_knowledge_router(cfg: Any, llm_dim: int) -> KnowledgeRouter:
    """Load knowledge router from local checkpoint."""
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        raise ValueError("Knowledge router loading currently expects a local checkpoint directory.")

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "knowledge_router")
    state_dict = load_component_state_dict(checkpoint_path)
    required_keys = {"query_proj.weight", "fc1.weight"}
    if not required_keys.issubset(state_dict.keys()):
        raise ValueError(
            f"Invalid knowledge router checkpoint at {checkpoint_path}: "
            f"missing one of {sorted(required_keys)}."
        )

    if "token_proj.weight" in state_dict:
        # New router format: token_proj: [text_dim, token_dim], query_proj: [text_dim, text_dim]
        text_dim = state_dict["token_proj.weight"].shape[0]
        token_dim = state_dict["token_proj.weight"].shape[1]
    else:
        # Backward compatibility for old checkpoints (no token_proj).
        token_dim = state_dict["query_proj.weight"].shape[0]
        text_dim = state_dict["query_proj.weight"].shape[1]

    hidden_dim = state_dict["fc1.weight"].shape[0]
    if text_dim != llm_dim:
        print(
            f"WARNING: router text_dim={text_dim} but model llm_dim={llm_dim}. "
            "Using checkpoint text_dim for router init."
        )
    knowledge_router = KnowledgeRouter(
        text_dim=text_dim,
        token_dim=token_dim,
        num_heads=getattr(cfg, "knowledge_router_num_heads", 8),
        hidden_dim=hidden_dim,
        dropout=getattr(cfg, "knowledge_router_dropout", 0.0),
        temperature=getattr(cfg, "knowledge_router_temperature", 1.0),
        focal_gamma=getattr(cfg, "knowledge_router_focal_gamma", 2.0),
        effective_num_beta=getattr(cfg, "knowledge_router_effective_num_beta", 0.999),
    ).to(DEVICE)
    knowledge_router = knowledge_router.to(torch.bfloat16).to(DEVICE)
    knowledge_router.eval()
    knowledge_router.load_state_dict(state_dict, strict=True)
    return knowledge_router


def apply_knowledge_routing(
    cfg: Any,
    knowledge_router: Optional[KnowledgeRouter],
    input_embeddings: torch.Tensor,
    all_actions_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    base_patch_features: Optional[torch.Tensor],
    expert_patch_features: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply optional router token pruning to base/expert patch features."""
    if knowledge_router is None:
        set_last_router_viz(None)
        return base_patch_features, expert_patch_features

    candidate_positions = None
    if base_patch_features is not None and expert_patch_features is not None:
        if base_patch_features.shape != expert_patch_features.shape:
            raise RuntimeError(
                "Knowledge routing requires matched base/expert token shapes for pruning, "
                f"but got base={tuple(base_patch_features.shape)} and expert={tuple(expert_patch_features.shape)}."
            )
        candidate_tokens = torch.cat((base_patch_features, expert_patch_features), dim=1)
        num_positions = base_patch_features.shape[1]
        base_positions = torch.arange(num_positions, device=candidate_tokens.device, dtype=torch.long).unsqueeze(0)
        base_positions = base_positions.expand(candidate_tokens.shape[0], -1)
        candidate_positions = torch.cat((base_positions, base_positions), dim=1)
    elif base_patch_features is not None:
        candidate_tokens = base_patch_features
        candidate_positions = torch.arange(
            candidate_tokens.shape[1], device=candidate_tokens.device, dtype=torch.long
        ).unsqueeze(0).expand(candidate_tokens.shape[0], -1)
    elif expert_patch_features is not None:
        candidate_tokens = expert_patch_features
        candidate_positions = torch.arange(
            candidate_tokens.shape[1], device=candidate_tokens.device, dtype=torch.long
        ).unsqueeze(0).expand(candidate_tokens.shape[0], -1)
    else:
        return base_patch_features, expert_patch_features

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
        target_keep_ratio=float(getattr(cfg, "knowledge_router_target_keep_ratio", 0.5)),
        min_keep_tokens=int(getattr(cfg, "knowledge_router_min_keep_tokens", 8)),
        hard_routing=bool(getattr(cfg, "knowledge_router_hard_routing", False)),
        compute_loss=False,
    )
    selected_indices = router_aux.get("selected_indices", None)
    selected_gate_probs = router_aux.get("selected_gate_probs", None)
    if selected_indices is None or selected_gate_probs is None:
        set_last_router_viz(None)
        return base_patch_features, expert_patch_features

    selected_indices = selected_indices.long()
    selected_gate_probs = selected_gate_probs.to(candidate_tokens.dtype)
    selected_gate_probs = torch.nan_to_num(selected_gate_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

    # Inference-time routing diagnostics: report selected token count and gate confidence.
    selected_tokens = int(selected_indices.shape[1])
    total_tokens = int(candidate_tokens.shape[1])
    keep_ratio = (float(selected_tokens) / float(total_tokens)) if total_tokens > 0 else 0.0
    effective_gate = float(selected_gate_probs.mean().item()) if selected_gate_probs.numel() > 0 else 0.0
    if base_patch_features is not None and expert_patch_features is not None:
        batch_size, num_positions, _ = base_patch_features.shape
        zeros_scores = torch.zeros((batch_size, num_positions), device=selected_indices.device, dtype=selected_gate_probs.dtype)
        base_scores = zeros_scores.clone()
        expert_scores = zeros_scores.clone()
        base_selected_binary = zeros_scores.clone()
        expert_selected_binary = zeros_scores.clone()

        base_selected_mask = selected_indices < num_positions
        expert_selected_mask = ~base_selected_mask

        if base_selected_mask.any():
            base_indices = torch.where(base_selected_mask, selected_indices, torch.zeros_like(selected_indices))
            base_values = torch.where(base_selected_mask, selected_gate_probs, torch.zeros_like(selected_gate_probs))
            base_scores.scatter_add_(1, base_indices, base_values)
            base_binary_values = torch.where(
                base_selected_mask,
                torch.ones_like(selected_gate_probs),
                torch.zeros_like(selected_gate_probs),
            )
            base_selected_binary.scatter_add_(1, base_indices, base_binary_values)
        if expert_selected_mask.any():
            expert_indices = torch.where(
                expert_selected_mask,
                selected_indices - num_positions,
                torch.zeros_like(selected_indices),
            )
            expert_values = torch.where(
                expert_selected_mask,
                selected_gate_probs,
                torch.zeros_like(selected_gate_probs),
            )
            expert_scores.scatter_add_(1, expert_indices, expert_values)
            expert_binary_values = torch.where(
                expert_selected_mask,
                torch.ones_like(selected_gate_probs),
                torch.zeros_like(selected_gate_probs),
            )
            expert_selected_binary.scatter_add_(1, expert_indices, expert_binary_values)

        base_scores = torch.nan_to_num(base_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        expert_scores = torch.nan_to_num(expert_scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        base_selected_binary = torch.nan_to_num(base_selected_binary, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
        expert_selected_binary = torch.nan_to_num(expert_selected_binary, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        # Cache router scores for optional rollout visualization (batch index 0).
        set_last_router_viz(
            {
                "base_scores": base_scores[0].detach().float().cpu().numpy(),
                "expert_scores": expert_scores[0].detach().float().cpu().numpy(),
                "base_selected_binary": base_selected_binary[0].detach().float().cpu().numpy(),
                "expert_selected_binary": expert_selected_binary[0].detach().float().cpu().numpy(),
                "num_positions": int(num_positions),
                "num_images_in_input": int(getattr(cfg, "num_images_in_input", 1)),
                "selected_tokens": int(selected_tokens),
                "total_tokens": int(total_tokens),
                "keep_ratio": float(keep_ratio),
                "gate_mean": float(effective_gate),
            }
        )

        active_position_mask = (base_scores > 0) | (expert_scores > 0)
        selected_position_count = int(active_position_mask.sum(dim=1).max().item()) if active_position_mask.numel() > 0 else 0
        selected_position_count = max(1, selected_position_count)

        position_ids = torch.arange(num_positions, device=selected_indices.device, dtype=torch.long).unsqueeze(0).expand(
            batch_size, -1
        )
        pad_id = num_positions
        padded_position_ids = torch.where(
            active_position_mask,
            position_ids,
            torch.full_like(position_ids, fill_value=pad_id),
        )
        sorted_positions, _ = torch.sort(padded_position_ids, dim=1)
        selected_positions = sorted_positions[:, :selected_position_count]
        valid_positions_mask = selected_positions < pad_id
        selected_positions_safe = selected_positions.clamp(max=max(0, num_positions - 1))

        selected_pos_expanded = selected_positions_safe.unsqueeze(-1).expand(-1, -1, base_patch_features.shape[-1])
        updated_base = torch.gather(base_patch_features, dim=1, index=selected_pos_expanded)
        updated_expert = torch.gather(expert_patch_features, dim=1, index=selected_pos_expanded)
        selected_base_scores = torch.gather(base_scores, dim=1, index=selected_positions_safe)
        selected_expert_scores = torch.gather(expert_scores, dim=1, index=selected_positions_safe)
        valid_scale = valid_positions_mask.to(dtype=updated_base.dtype).unsqueeze(-1)
        updated_base = updated_base * selected_base_scores.unsqueeze(-1) * valid_scale
        updated_expert = updated_expert * selected_expert_scores.unsqueeze(-1) * valid_scale

        position_keep_ratio = float(active_position_mask.float().mean().item()) if active_position_mask.numel() > 0 else 0.0
        print(
            f"[Router][Inference] selected_tokens={selected_tokens}/{total_tokens} "
            f"keep_ratio={keep_ratio:.4f} kept_positions={selected_position_count}/{num_positions} "
            f"pos_keep_ratio={position_keep_ratio:.4f} gate_mean={effective_gate:.4f}"
        )
        return updated_base, updated_expert

    print(
        f"[Router][Inference] selected_tokens={selected_tokens}/{total_tokens} "
        f"keep_ratio={keep_ratio:.4f} gate_mean={effective_gate:.4f}"
    )
    set_last_router_viz(None)
    selected_idx_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, candidate_tokens.shape[-1])
    updated_base = base_patch_features
    updated_expert = expert_patch_features
    if base_patch_features is not None:
        updated_base = torch.gather(base_patch_features, dim=1, index=selected_idx_expanded)
        updated_base = updated_base * selected_gate_probs.unsqueeze(-1)
    if expert_patch_features is not None:
        updated_expert = torch.gather(expert_patch_features, dim=1, index=selected_idx_expanded)
        updated_expert = updated_expert * selected_gate_probs.unsqueeze(-1)
    return updated_base, updated_expert


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector


def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead, DiffusionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        # Set number of diffusion steps for inference
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def predict_action_siglip_dual_path(
    cfg: Any,
    vla: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    unnorm_key: str,
    fusion_projector: Optional[DualPathFusionProjector],
    single_path_projector: Optional[SinglePathProjector] = None,
    knowledge_router: Optional[KnowledgeRouter] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    visual_path_mode: str = "dual",
):
    """
    Dual-path SigLIP-only action prediction:
      1) base path with LoRA disabled,
      2) expert path with LoRA enabled,
      3) concat-and-project via fusion_projector.
    """
    if use_film:
        raise NotImplementedError("SigLIP-only dual-path inference currently does not support FiLM.")

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
    if visual_path_mode in {"dual", "base_only", "expert_only"} and fusion_projector is None:
        raise ValueError(
            f"visual_path_mode={visual_path_mode} requires `fusion_projector`, but got None."
        )
    if visual_path_mode in {"base_only_separate", "expert_only_separate"} and single_path_projector is None:
        raise ValueError(
            f"visual_path_mode={visual_path_mode} requires `single_path_projector`, but got None."
        )

    vla_core = get_vla_core_model(vla)

    # Match training-time prompt formatting.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871], dtype=torch.long, device=input_ids.device), dim=0)), dim=1
        )

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    num_prompt_tokens = input_ids.shape[-1] - 1

    input_ids, attention_mask = vla_core._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla_core._prepare_labels_for_action_prediction(labels, input_ids)
    input_embeddings = vla_core.get_input_embeddings()(input_ids)
    all_actions_mask = vla_core._process_action_masks(labels)

    num_images_in_input = vla_core.vision_backbone.get_num_images_in_input()
    base_patch_features = None
    expert_patch_features = None
    if visual_path_mode in {"dual", "base_only", "base_only_separate"}:
        with torch.no_grad():
            with get_vla_adapter_context(vla):
                base_patch_features = run_siglip_encoder_only(vla_core, pixel_values, num_images_in_input)
        base_patch_features = base_patch_features.detach()
    if visual_path_mode in {"dual", "expert_only", "expert_only_separate"}:
        expert_patch_features = run_siglip_encoder_only(vla_core, pixel_values, num_images_in_input)

    base_patch_features, expert_patch_features = apply_knowledge_routing(
        cfg=cfg,
        knowledge_router=knowledge_router,
        input_embeddings=input_embeddings,
        all_actions_mask=all_actions_mask,
        attention_mask=attention_mask,
        base_patch_features=base_patch_features,
        expert_patch_features=expert_patch_features,
    )

    if visual_path_mode == "dual":
        projected_patch_embeddings = fusion_projector(base_patch_features, expert_patch_features)
    elif visual_path_mode == "base_only":
        projected_patch_embeddings = fusion_projector(base_patch_features, torch.zeros_like(base_patch_features))
    elif visual_path_mode == "expert_only":
        projected_patch_embeddings = fusion_projector(torch.zeros_like(expert_patch_features), expert_patch_features)
    elif visual_path_mode == "base_only_separate":
        projected_patch_embeddings = single_path_projector(base_patch_features)
    else:  # expert_only_separate
        projected_patch_embeddings = single_path_projector(expert_patch_features)

    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = vla_core._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

    use_diffusion = noisy_action_projector is not None and action_head is not None and hasattr(action_head, "noise_scheduler")

    num_patches = projected_patch_embeddings.shape[1]
    if use_diffusion:
        num_patches += 1

    if use_diffusion:
        noise = torch.randn(
            size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        normalized_actions, actions_hidden_states = vla_core._run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            num_patches,
            num_prompt_tokens,
            noisy_action_projector,
        )
    else:
        normalized_actions, actions_hidden_states = vla_core._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            num_patches,
            num_prompt_tokens,
            action_head,
        )

    actions = vla_core._unnormalize_actions(normalized_actions, unnorm_key)
    return actions, actions_hidden_states


def predict_action_dino_dual_path(
    cfg: Any,
    vla: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    unnorm_key: str,
    fusion_projector: Optional[DualPathFusionProjector],
    single_path_projector: Optional[SinglePathProjector] = None,
    knowledge_router: Optional[KnowledgeRouter] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    visual_path_mode: str = "dual",
):
    """
    Dual-path DINO-only action prediction:
      1) base path with LoRA disabled,
      2) expert path with LoRA enabled,
      3) concat-and-project via fusion_projector.
    """
    if use_film:
        raise NotImplementedError("DINO-only dual-path inference currently does not support FiLM.")

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
    if visual_path_mode in {"dual", "base_only", "expert_only"} and fusion_projector is None:
        raise ValueError(
            f"visual_path_mode={visual_path_mode} requires `fusion_projector`, but got None."
        )
    if visual_path_mode in {"base_only_separate", "expert_only_separate"} and single_path_projector is None:
        raise ValueError(
            f"visual_path_mode={visual_path_mode} requires `single_path_projector`, but got None."
        )

    vla_core = get_vla_core_model(vla)

    # Match training-time prompt formatting.
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871], dtype=torch.long, device=input_ids.device), dim=0)), dim=1
        )

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    num_prompt_tokens = input_ids.shape[-1] - 1

    input_ids, attention_mask = vla_core._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla_core._prepare_labels_for_action_prediction(labels, input_ids)
    input_embeddings = vla_core.get_input_embeddings()(input_ids)
    all_actions_mask = vla_core._process_action_masks(labels)

    num_images_in_input = vla_core.vision_backbone.get_num_images_in_input()
    base_patch_features = None
    expert_patch_features = None
    if visual_path_mode in {"dual", "base_only", "base_only_separate"}:
        with torch.no_grad():
            with get_vla_adapter_context(vla):
                base_patch_features = run_dino_encoder_only(vla_core, pixel_values, num_images_in_input)
        base_patch_features = base_patch_features.detach()
    if visual_path_mode in {"dual", "expert_only", "expert_only_separate"}:
        expert_patch_features = run_dino_encoder_only(vla_core, pixel_values, num_images_in_input)

    base_patch_features, expert_patch_features = apply_knowledge_routing(
        cfg=cfg,
        knowledge_router=knowledge_router,
        input_embeddings=input_embeddings,
        all_actions_mask=all_actions_mask,
        attention_mask=attention_mask,
        base_patch_features=base_patch_features,
        expert_patch_features=expert_patch_features,
    )

    if visual_path_mode == "dual":
        projected_patch_embeddings = fusion_projector(base_patch_features, expert_patch_features)
    elif visual_path_mode == "base_only":
        projected_patch_embeddings = fusion_projector(base_patch_features, torch.zeros_like(base_patch_features))
    elif visual_path_mode == "expert_only":
        projected_patch_embeddings = fusion_projector(torch.zeros_like(expert_patch_features), expert_patch_features)
    elif visual_path_mode == "base_only_separate":
        projected_patch_embeddings = single_path_projector(base_patch_features)
    else:  # expert_only_separate
        projected_patch_embeddings = single_path_projector(expert_patch_features)

    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = vla_core._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

    use_diffusion = noisy_action_projector is not None and action_head is not None and hasattr(action_head, "noise_scheduler")

    num_patches = projected_patch_embeddings.shape[1]
    if use_diffusion:
        num_patches += 1

    if use_diffusion:
        noise = torch.randn(
            size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        normalized_actions, actions_hidden_states = vla_core._run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            num_patches,
            num_prompt_tokens,
            noisy_action_projector,
        )
    else:
        normalized_actions, actions_hidden_states = vla_core._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            num_patches,
            num_prompt_tokens,
            action_head,
        )

    actions = vla_core._unnormalize_actions(normalized_actions, unnorm_key)
    return actions, actions_hidden_states


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    fusion_projector: Optional[torch.nn.Module] = None,
    single_path_projector: Optional[torch.nn.Module] = None,
    knowledge_router: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        fusion_projector: Optional dual-path fusion projector (used when single-branch dual-path inference is enabled)
        single_path_projector: Optional single-path projector (used in *_separate visual ablation modes)
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():
        # Reset visualization cache for this query so callers only read fresh router outputs.
        clear_last_router_viz()

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # Generate action
        if action_head is None:
            # Standard VLA output (single-image inputs, discrete actions)
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            use_siglip_only_vision = getattr(cfg, "use_siglip_only_vision", False)
            use_dino_only_vision = getattr(cfg, "use_dino_only_vision", False)
            if use_siglip_only_vision and use_dino_only_vision:
                raise ValueError("Cannot enable both use_siglip_only_vision and use_dino_only_vision.")

            if use_siglip_only_vision:
                action, _ = predict_action_siglip_dual_path(
                    cfg=cfg,
                    vla=vla,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    unnorm_key=cfg.unnorm_key,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector,
                    fusion_projector=fusion_projector,
                    single_path_projector=single_path_projector,
                    knowledge_router=knowledge_router,
                    use_film=use_film,
                    visual_path_mode=getattr(cfg, "visual_path_mode", "dual"),
                )
            elif use_dino_only_vision:
                action, _ = predict_action_dino_dual_path(
                    cfg=cfg,
                    vla=vla,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    unnorm_key=cfg.unnorm_key,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector,
                    fusion_projector=fusion_projector,
                    single_path_projector=single_path_projector,
                    knowledge_router=knowledge_router,
                    use_film=use_film,
                    visual_path_mode=getattr(cfg, "visual_path_mode", "dual"),
                )
            else:
                # Custom action head for continuous actions
                action, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                )

    # Return action chunk as list of actions
    return [action[i] for i in range(len(action))]


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
