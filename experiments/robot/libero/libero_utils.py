"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import Dict, List, Optional

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def _patch_scores_to_mask_image(
    patch_scores: Optional[np.ndarray], image_height: int, image_width: int
) -> np.ndarray:
    """
    Convert flat per-patch scores to full-resolution mask image.
    Returns float mask in [0, 1] with shape (H, W).
    """
    if patch_scores is None:
        return np.zeros((image_height, image_width), dtype=np.float32)

    scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros((image_height, image_width), dtype=np.float32)

    grid_side = int(round(math.sqrt(float(scores.size))))
    if grid_side <= 0 or grid_side * grid_side != scores.size:
        # Fallback for unexpected token counts.
        grid_side = int(math.sqrt(float(scores.size)))
        if grid_side <= 0:
            return np.zeros((image_height, image_width), dtype=np.float32)
        scores = scores[: grid_side * grid_side]

    patch_grid = np.clip(scores.reshape(grid_side, grid_side), 0.0, 1.0)
    patch_grid_img = Image.fromarray((patch_grid * 255.0).astype(np.uint8), mode="L")
    patch_grid_img = patch_grid_img.resize((image_width, image_height), Image.NEAREST)
    return np.asarray(patch_grid_img, dtype=np.float32) / 255.0


def _overlay_white_mask(frame: np.ndarray, mask: np.ndarray, overlay_alpha: float = 0.7) -> np.ndarray:
    """Overlay white mask on RGB frame with configurable alpha."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    alpha_map = np.clip(mask, 0.0, 1.0).astype(np.float32) * float(overlay_alpha)
    alpha_map = alpha_map[..., None]

    frame_f = frame.astype(np.float32)
    white = np.full_like(frame_f, 255.0)
    blended = frame_f * (1.0 - alpha_map) + white * alpha_map
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def save_router_overlay_videos(
    exo_rollout_images: List[np.ndarray],
    wrist_rollout_images: List[np.ndarray],
    router_overlay_masks: List[Optional[Dict[str, np.ndarray]]],
    idx: int,
    success: bool,
    task_description: str,
    overlay_alpha: float = 0.7,
    log_file=None,
) -> Dict[str, str]:
    """
    Save four router-overlay rollout videos:
      1) exoview + visual mask
      2) exoview + 3D mask
      3) wrist view + visual mask
      4) wrist view + 3D mask
    """
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    base_name = f"{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}"

    output_paths = {
        "exo_visual": f"{rollout_dir}/{base_name}--exo-visual-mask.mp4",
        "exo_3d": f"{rollout_dir}/{base_name}--exo-3d-mask.mp4",
        "wrist_visual": f"{rollout_dir}/{base_name}--wrist-visual-mask.mp4",
        "wrist_3d": f"{rollout_dir}/{base_name}--wrist-3d-mask.mp4",
    }

    if not exo_rollout_images or not wrist_rollout_images:
        if log_file is not None:
            log_file.write("Skipping router overlay video save: missing exo/wrist rollout images.\n")
        return output_paths

    num_frames = min(len(exo_rollout_images), len(wrist_rollout_images), len(router_overlay_masks))
    if num_frames == 0:
        if log_file is not None:
            log_file.write("Skipping router overlay video save: no frames available.\n")
        return output_paths

    writers = {key: imageio.get_writer(path, fps=30) for key, path in output_paths.items()}
    try:
        for frame_idx in range(num_frames):
            exo_frame = exo_rollout_images[frame_idx]
            wrist_frame = wrist_rollout_images[frame_idx]
            masks = router_overlay_masks[frame_idx] if frame_idx < len(router_overlay_masks) else None

            exo_h, exo_w = exo_frame.shape[0], exo_frame.shape[1]
            wrist_h, wrist_w = wrist_frame.shape[0], wrist_frame.shape[1]

            exo_visual_mask = _patch_scores_to_mask_image(
                masks.get("exo_visual") if masks is not None else None, exo_h, exo_w
            )
            exo_3d_mask = _patch_scores_to_mask_image(
                masks.get("exo_3d") if masks is not None else None, exo_h, exo_w
            )
            wrist_visual_mask = _patch_scores_to_mask_image(
                masks.get("wrist_visual") if masks is not None else None, wrist_h, wrist_w
            )
            wrist_3d_mask = _patch_scores_to_mask_image(
                masks.get("wrist_3d") if masks is not None else None, wrist_h, wrist_w
            )

            writers["exo_visual"].append_data(_overlay_white_mask(exo_frame, exo_visual_mask, overlay_alpha))
            writers["exo_3d"].append_data(_overlay_white_mask(exo_frame, exo_3d_mask, overlay_alpha))
            writers["wrist_visual"].append_data(_overlay_white_mask(wrist_frame, wrist_visual_mask, overlay_alpha))
            writers["wrist_3d"].append_data(_overlay_white_mask(wrist_frame, wrist_3d_mask, overlay_alpha))
    finally:
        for writer in writers.values():
            writer.close()

    for key, path in output_paths.items():
        print(f"Saved router overlay MP4 ({key}) at path {path}")
        if log_file is not None:
            log_file.write(f"Saved router overlay MP4 ({key}) at path {path}\n")
    return output_paths


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
