"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import importlib.util
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from PIL import Image

import draccus
import numpy as np
import tqdm

import wandb

# Ensure project root is importable regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.openvla_utils import (
    get_action_head,
    get_fusion_projector,
    get_knowledge_router,
    get_last_router_viz,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    get_single_path_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_LIBERO_BENCHMARK = None
_LIBERO_UTILS = None


def _is_libero_benchmark_root(path: Path) -> bool:
    """Return True when path contains expected LIBERO benchmark assets."""
    required = ("bddl_files", "init_files", "assets")
    return all((path / key).exists() for key in required)


def _resolve_libero_benchmark_root() -> Optional[Path]:
    """
    Find LIBERO benchmark root to write config non-interactively.

    Root must contain `bddl_files/`, `init_files/`, and `assets/`.
    """
    raw_candidates = []
    for env_key in ("LIBERO_BENCHMARK_ROOT", "ROLLOUT_LIBERO_PYTHONPATH", "LIBERO_PYTHONPATH", "PYTHONPATH"):
        env_val = os.environ.get(env_key)
        if not env_val:
            continue
        if env_key == "PYTHONPATH":
            raw_candidates.extend([entry for entry in env_val.split(os.pathsep) if entry])
        else:
            raw_candidates.append(env_val)

    spec = importlib.util.find_spec("libero")
    if spec is not None:
        if spec.submodule_search_locations:
            raw_candidates.extend(spec.submodule_search_locations)
        if spec.origin:
            raw_candidates.append(str(Path(spec.origin).resolve().parent))

    visited = set()
    for candidate in raw_candidates:
        base = Path(candidate).expanduser()
        for maybe_root in (base, base / "libero", base / "libero" / "libero"):
            try:
                resolved = maybe_root.resolve()
            except OSError:
                continue
            if str(resolved) in visited:
                continue
            visited.add(str(resolved))
            if _is_libero_benchmark_root(resolved):
                return resolved
    return None


def ensure_libero_config_noninteractive() -> Optional[Path]:
    """
    Write LIBERO config file if missing to avoid interactive prompts in subprocess eval.
    """
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return config_file

    benchmark_root = _resolve_libero_benchmark_root()
    if benchmark_root is None:
        return None

    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        f.write(f"benchmark_root: {benchmark_root}\n")
        f.write(f"bddl_files: {benchmark_root / 'bddl_files'}\n")
        f.write(f"init_states: {benchmark_root / 'init_files'}\n")
        f.write(f"datasets: {benchmark_root.parent / 'datasets'}\n")
        f.write(f"assets: {benchmark_root / 'assets'}\n")
    logger.info("Created LIBERO config at %s", config_file)
    return config_file


def get_libero_runtime_modules():
    """
    Lazy import LIBERO runtime modules after non-interactive config bootstrap.
    """
    global _LIBERO_BENCHMARK, _LIBERO_UTILS
    if _LIBERO_BENCHMARK is not None and _LIBERO_UTILS is not None:
        return _LIBERO_BENCHMARK, _LIBERO_UTILS

    ensure_libero_config_noninteractive()
    try:
        from libero.libero import benchmark as benchmark_module
        from experiments.robot.libero import libero_utils as libero_utils_module
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise ModuleNotFoundError(
            f"Missing dependency `{missing}` required for LIBERO rollout evaluation. "
            "Install requirements with: `pip install -r experiments/robot/libero/libero_requirements.txt`."
        ) from exc
    except EOFError as exc:
        raise RuntimeError(
            "LIBERO attempted interactive configuration in a non-interactive run. "
            "Set LIBERO_CONFIG_PATH to a writable directory and ensure LIBERO benchmark files are discoverable."
        ) from exc

    _LIBERO_BENCHMARK = benchmark_module
    _LIBERO_UTILS = libero_utils_module
    return _LIBERO_BENCHMARK, _LIBERO_UTILS


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    base_model_path: Optional[str] = None            # Optional base model path (used if adapter config lacks it)
    openvla_baseline: bool = False                   # If True, run OpenVLA baseline inference path (no VGGT/dual-path visual projectors)

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    use_siglip_only_vision: bool = False             # If True, use SigLIP-only dual-path inference with fusion projector
    use_dino_only_vision: bool = True                # If True, use DINO-only dual-path inference with fusion projector
    train_vla_projector: bool = False                # If True, load and use fine-tuned OpenVLA vision projector checkpoint
    use_knowledge_router: bool = False               # If True, apply token routing before projector/LLM input
    knowledge_router_num_heads: int = 8              # Router cross-attention heads
    knowledge_router_hidden_dim: int = 128           # Router MLP hidden width
    knowledge_router_dropout: float = 0.0            # Router dropout
    knowledge_router_temperature: float = 1.0        # Router sigmoid temperature
    knowledge_router_target_keep_ratio: float = 0.5  # Router target keep ratio
    knowledge_router_min_keep_tokens: int = 8        # Router min tokens to keep
    knowledge_router_hard_routing: bool = False      # Hard/STE gating in router
    knowledge_router_focal_gamma: float = 2.0        # Focal gamma (for checkpoint compatibility)
    knowledge_router_effective_num_beta: float = 0.999  # Effective-number beta (for checkpoint compatibility)
    visual_path_mode: str = "dual"                   # Visual mode: dual | base_only | expert_only | base_only_separate | expert_only_separate
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    vision_lora: bool = True                         # If False, disable vision-side LoRA contribution in forward pass

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    max_tasks: int = -1                              # Maximum number of tasks to evaluate (-1: all tasks)
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    save_rollout_videos: bool = True                 # Whether to save rollout videos during evaluation
    save_router_overlay_videos: bool = False         # Whether to save exo/wrist router-overlay videos
    router_overlay_alpha: float = 0.7                # Overlay alpha for router masks (white color)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if cfg.openvla_baseline:
        if (not cfg.use_l1_regression) or cfg.use_diffusion:
            logger.warning(
                "openvla_baseline=True: forcing `use_l1_regression=True` and `use_diffusion=False`."
            )
            cfg.use_l1_regression = True
            cfg.use_diffusion = False
        if cfg.use_film:
            logger.warning("openvla_baseline=True: forcing `use_film=False` for native OpenVLA eval.")
            cfg.use_film = False
        if cfg.use_siglip_only_vision or cfg.use_dino_only_vision:
            logger.warning(
                "openvla_baseline=True: forcing `use_siglip_only_vision=False` and `use_dino_only_vision=False`."
            )
            cfg.use_siglip_only_vision = False
            cfg.use_dino_only_vision = False
        if cfg.visual_path_mode != "dual":
            logger.warning("openvla_baseline=True: forcing `visual_path_mode=dual`.")
            cfg.visual_path_mode = "dual"
        if cfg.use_knowledge_router:
            logger.warning("openvla_baseline=True: forcing `use_knowledge_router=False`.")
            cfg.use_knowledge_router = False

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert not (cfg.use_siglip_only_vision and cfg.use_dino_only_vision), (
        "Cannot enable both use_siglip_only_vision and use_dino_only_vision!"
    )
    assert cfg.visual_path_mode in {
        "dual",
        "base_only",
        "expert_only",
        "base_only_separate",
        "expert_only_separate",
    }, (
        f"Invalid visual_path_mode: {cfg.visual_path_mode}"
    )
    if cfg.use_knowledge_router and not (cfg.use_siglip_only_vision or cfg.use_dino_only_vision):
        logger.warning(
            "use_knowledge_router=True but neither single-branch vision mode is enabled; "
            "forcing `use_knowledge_router=False`."
        )
        cfg.use_knowledge_router = False

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assert 0.0 <= cfg.router_overlay_alpha <= 1.0, "router_overlay_alpha must be in [0, 1]."


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    fusion_projector = None
    single_path_projector = None
    knowledge_router = None
    if cfg.use_siglip_only_vision or cfg.use_dino_only_vision:
        if cfg.visual_path_mode in {"base_only_separate", "expert_only_separate"}:
            single_path_projector = get_single_path_projector(cfg, model.llm_dim)
        else:
            fusion_projector = get_fusion_projector(cfg, model.llm_dim)
        if cfg.use_knowledge_router:
            knowledge_router = get_knowledge_router(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        fusion_projector,
        single_path_projector,
        knowledge_router,
        processor,
    )


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    _, libero_utils = get_libero_runtime_modules()

    # Get preprocessed images
    img = libero_utils.get_libero_image(obs)
    wrist_img = libero_utils.get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                libero_utils.quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }

    # Return processed observation + original images for replay/visualization.
    return observation, img, wrist_img


def _clone_router_masks(router_masks: Optional[dict]) -> Optional[dict]:
    """Clone router mask payload to avoid accidental in-place mutation across timesteps."""
    if router_masks is None:
        return None
    cloned = {}
    for key, value in router_masks.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def _extract_router_overlay_masks(router_viz: Optional[dict], num_images_in_input: int) -> Optional[dict]:
    """
    Convert cached router payload into 4 overlay masks:
      - exo_visual, exo_3d, wrist_visual, wrist_3d
    Each mask is a flat per-patch array where 1 means "masked (not selected)".
    """
    if router_viz is None:
        return None

    base_selected = router_viz.get("base_selected_binary", None)
    expert_selected = router_viz.get("expert_selected_binary", None)

    # Backward compatibility for older checkpoints / cached payloads:
    # infer selected map from nonzero scores when explicit binary maps are absent.
    if base_selected is None or expert_selected is None:
        base_scores = router_viz.get("base_scores", None)
        expert_scores = router_viz.get("expert_scores", None)
        if base_scores is None or expert_scores is None:
            return None
        base_selected = np.asarray(base_scores, dtype=np.float32).reshape(-1) > 0
        expert_selected = np.asarray(expert_scores, dtype=np.float32).reshape(-1) > 0

    base_selected = np.asarray(base_selected, dtype=np.float32).reshape(-1)
    expert_selected = np.asarray(expert_selected, dtype=np.float32).reshape(-1)
    if base_selected.shape != expert_selected.shape:
        return None

    # White overlay should indicate pruned (not selected) tokens.
    base_pruned = 1.0 - np.clip(base_selected, 0.0, 1.0)
    expert_pruned = 1.0 - np.clip(expert_selected, 0.0, 1.0)

    if num_images_in_input < 2:
        return None
    if base_pruned.size % num_images_in_input != 0:
        return None

    patches_per_image = base_pruned.size // num_images_in_input
    if patches_per_image <= 0:
        return None
    if 2 * patches_per_image > base_pruned.size:
        return None

    return {
        "exo_visual": base_pruned[0:patches_per_image].copy(),
        "wrist_visual": base_pruned[patches_per_image : 2 * patches_per_image].copy(),
        "exo_3d": expert_pruned[0:patches_per_image].copy(),
        "wrist_3d": expert_pruned[patches_per_image : 2 * patches_per_image].copy(),
    }


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    fusion_projector=None,
    single_path_projector=None,
    knowledge_router=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    _, libero_utils = get_libero_runtime_modules()

    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_exo_images = []
    replay_wrist_images = []
    replay_router_masks = []
    current_router_masks = None
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(libero_utils.get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, exo_img, wrist_img = prepare_observation(obs, resize_size)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    fusion_projector=fusion_projector,
                    single_path_projector=single_path_projector,
                    knowledge_router=knowledge_router,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
                current_router_masks = _extract_router_overlay_masks(
                    get_last_router_viz(), cfg.num_images_in_input
                )

            replay_exo_images.append(exo_img)
            replay_wrist_images.append(wrist_img)
            replay_router_masks.append(_clone_router_masks(current_router_masks))

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_exo_images, replay_wrist_images, replay_router_masks


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    fusion_projector=None,
    single_path_projector=None,
    knowledge_router=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    _, libero_utils = get_libero_runtime_modules()

    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = libero_utils.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    try:
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            log_message(f"\nTask: {task_description}", log_file)

            # Handle initial state
            if cfg.initial_states_path == "DEFAULT":
                # Use default initial state
                initial_state = initial_states[episode_idx]
            else:
                # Get keys for fetching initial episode state from JSON
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"

                # Skip episode if expert demonstration failed to complete the task
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                    continue

                # Get initial state
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            log_message(f"Starting episode {task_episodes + 1}...", log_file)

            # Run episode
            success, replay_exo_images, replay_wrist_images, replay_router_masks = run_episode(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                fusion_projector,
                single_path_projector,
                knowledge_router,
                initial_state,
                log_file,
            )

            # Update counters
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save replay video
            if cfg.save_rollout_videos:
                libero_utils.save_rollout_video(
                    replay_exo_images, total_episodes, success=success, task_description=task_description, log_file=log_file
                )
            if cfg.save_router_overlay_videos:
                libero_utils.save_router_overlay_videos(
                    exo_rollout_images=replay_exo_images,
                    wrist_rollout_images=replay_wrist_images,
                    router_overlay_masks=replay_router_masks,
                    idx=total_episodes,
                    success=success,
                    task_description=task_description,
                    overlay_alpha=cfg.router_overlay_alpha,
                    log_file=log_file,
                )

            # Log results
            log_message(f"Success: {success}", log_file)
            log_message(f"# episodes completed so far: {total_episodes}", log_file)
            log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    finally:
        # Explicit env teardown helps avoid EGL context warnings at interpreter shutdown.
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:
                log_message(f"Environment close warning: {e}", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Import LIBERO modules early so dependency/config issues fail fast.
    benchmark_module, _ = get_libero_runtime_modules()

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        fusion_projector,
        single_path_projector,
        knowledge_router,
        processor,
    ) = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark_module.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks if cfg.max_tasks <= 0 else min(task_suite.n_tasks, cfg.max_tasks)

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Evaluating {num_tasks}/{task_suite.n_tasks} tasks", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            fusion_projector,
            single_path_projector,
            knowledge_router,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
