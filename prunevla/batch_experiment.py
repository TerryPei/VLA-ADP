"""
PruneVLA 小批量实验脚本：
- 加载原始 VLA 模型
- 原样评测一轮（original）
- 注入 KV-Pruning 注意力（一次选列 + 持久裁剪）
- 再评测一轮（prune）
- 保存视频与简单日志，便于快速验证可行性
"""

import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import imageio

# 复用现有评测管线
sys.path.append("experiments/robot")

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    get_image_resize_size,
    normalize_gripper_action,
    invert_gripper_action,
)
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from libero.libero import benchmark

try:
    from .configs.prune_config import (
        PruneVLAConfig,
        PruneExperimentConfig,
        get_suite_info,
    )
    from .prune_kv_attention import replace_attention_with_prune
except Exception:
    # 兼容脚本直跑：python prunevla/batch_experiment.py 或由 experiment_controller 脚本导入
    import sys as _sys, os as _os
    _CUR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR not in _sys.path:
        _sys.path.append(_CUR)
    from configs.prune_config import PruneVLAConfig, PruneExperimentConfig, get_suite_info
    from prune_kv_attention import replace_attention_with_prune


DEFAULT_MAX_STEPS = 220
NUM_STEPS_WAIT = 10
ENV_IMG_RES = 256
SAVE_VIDEOS = True


def _create_video_dir() -> str:
    d = "videos"
    os.makedirs(d, exist_ok=True)
    return d


def _save_video(frames, task_description: str, tag: str) -> Optional[str]:
    if not frames:
        return None
    video_dir = _create_video_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = task_description.lower().replace(" ", "_")[:50]
    path = os.path.join(video_dir, f"{ts}--{tag}--{name}.mp4")
    try:
        writer = imageio.get_writer(path, fps=30)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return path
    except Exception:
        return None


def _prepare_observation(obs):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    return {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])  # type: ignore
        ),
    }, img


def run_one_episode(
    cfg: GenerateConfig,
    model,
    processor,
    action_head,
    proprio_projector,
    task_description: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    tag: str = "original",
):
    success = False
    total_actions_generated = 0
    frames = []
    t = 0

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(0)
    env, env_task_description = get_libero_env(task, cfg.model_family, resolution=ENV_IMG_RES)

    resize_size = get_image_resize_size(cfg)
    env.reset()
    initial_states = task_suite.get_task_init_states(0)
    obs = env.set_init_state(initial_states[0])

    q = deque(maxlen=cfg.num_open_loop_steps)
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        observation, img = _prepare_observation(obs)
        if SAVE_VIDEOS:
            frames.append(img)
        if len(q) == 0:
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                use_film=cfg.use_film,
            )
            q.extend(actions)
            total_actions_generated += len(actions)
        action = q.popleft()
        action = normalize_gripper_action(action, binarize=True)
        if cfg.model_family == "openvla":
            action = invert_gripper_action(action)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    video_path = _save_video(frames, task_description, tag) if SAVE_VIDEOS else None
    return success, total_actions_generated, max(0, t - NUM_STEPS_WAIT), video_path


def run_prune_experiment(prune_config: PruneVLAConfig, task_suite_name: str = "libero_spatial", model_path: Optional[str] = None):
    # 1) 构造生成配置（与 sparsevla 一致的基本参数）
    cfg = GenerateConfig(
        pretrained_checkpoint=model_path or "moojink/openvla-7b-oft-finetuned-libero-spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
        model_family="openvla",
        task_suite_name=task_suite_name,
        num_steps_wait=NUM_STEPS_WAIT,
        env_img_res=ENV_IMG_RES,
    )

    # 2) 加载模型与组件
    model = get_model(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)

    # 3) 获取任务描述
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_description = task_suite.get_task(0).language

    # 4) baseline(original)
    print("[PruneVLA] Running original...")
    orig_success, orig_actions, orig_steps, orig_video = run_one_episode(
        cfg, model, processor, action_head, proprio_projector, task_description, tag="original"
    )

    # 5) 注入 KV-Pruning 并运行
    print("[PruneVLA] Injecting KV-Pruning...")
    # 动态覆盖关键区间，避免写死 257/固定视觉长度（失败即报错）
    per_image_patches = int(model.vision_backbone.get_num_patches())
    num_images = int(model.vision_backbone.get_num_images_in_input())
    if per_image_patches <= 0 or num_images <= 0:
        raise RuntimeError("Invalid vision backbone patch/images settings for prune attention")
    num_patches = per_image_patches * num_images

    # 额外前缀伪 token（本体/扩散）不应计入 instruction 段
    extra = (1 if cfg.use_proprio else 0) + (1 if getattr(cfg, "use_diffusion", False) else 0)
    prune_config.instruction_start = 1 + num_patches + extra
    # 同步视觉范围为真实 patch 数，避免默认值导致范围错误
    prune_config.vision_range = (1, 1 + num_patches)

    unnorm_key = getattr(cfg, "unnorm_key", None)
    action_dim = int(model.get_action_dim(unnorm_key))
    if action_dim <= 0:
        raise RuntimeError("Failed to resolve action_dim for prune attention; check unnorm_key and model stats")
    prune_config.expected_action_tokens = int(action_dim * NUM_ACTIONS_CHUNK)
    replace_attention_with_prune(model, prune_config)

    print("[PruneVLA] Running pruned...")
    pr_success, pr_actions, pr_steps, pr_video = run_one_episode(
        cfg, model, processor, action_head, proprio_projector, task_description, tag="pruned"
    )

    # 6) 打印结果
    print("\n===== PruneVLA Result =====")
    print(f"Original  -> success={orig_success}, steps={orig_steps}, actions={orig_actions}, video={orig_video}")
    print(f"Pruned    -> success={pr_success}, steps={pr_steps}, actions={pr_actions}, video={pr_video}")

    # 7) 记录日志
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{ts}--result.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("===== PruneVLA Result =====\n")
            f.write(f"task_suite={task_suite_name}\n")
            f.write(f"prune_schedule={getattr(prune_config, 'prune_schedule', None)}\n")
            # min_keep_vision 已移除，保持日志兼容留空
            f.write(f"original_success={orig_success}, steps={orig_steps}, actions={orig_actions}, video={orig_video}\n")
            f.write(f"pruned_success={pr_success}, steps={pr_steps}, actions={pr_actions}, video={pr_video}\n")
    except Exception:
        pass

    return {
        "original": {
            "success": orig_success,
            "steps": orig_steps,
            "actions": orig_actions,
            "video": orig_video,
        },
        "pruned": {
            "success": pr_success,
            "steps": pr_steps,
            "actions": pr_actions,
            "video": pr_video,
        },
    }


def run_one_episode_with_task_index(
    cfg: GenerateConfig,
    model,
    processor,
    action_head,
    proprio_projector,
    task_suite_name: str,
    task_index: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    tag: str = "original",
    save_videos: bool = True,
):
    global SAVE_VIDEOS
    prev_save = SAVE_VIDEOS
    SAVE_VIDEOS = save_videos
    try:
        success = False
        total_actions_generated = 0
        frames = []

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_index)
        env, env_task_description = get_libero_env(task, cfg.model_family, resolution=ENV_IMG_RES)

        env.reset()
        initial_states = task_suite.get_task_init_states(task_index)
        obs = env.set_init_state(initial_states[0])

        q = deque(maxlen=cfg.num_open_loop_steps)
        t = 0
        while t < max_steps + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            observation, img = _prepare_observation(obs)
            if SAVE_VIDEOS:
                frames.append(img)
            if len(q) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task.language,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=cfg.use_film,
                )
                q.extend(actions)
                total_actions_generated += len(actions)
            action = q.popleft()
            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

        video_path = _save_video(frames, task.language, tag) if SAVE_VIDEOS else None
        return success, total_actions_generated, max(0, t - NUM_STEPS_WAIT), video_path
    finally:
        SAVE_VIDEOS = prev_save


def run_prune_experiments(exp_cfg) -> Dict[str, Any]:
    """
    多任务/多次/对比实验控制：
    - test_mode: "original" / "pruned" / "both"
    - 循环 task_indices 与 num_repeats
    - 日志落盘
    """
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # 根据套件选择生成配置（task_suite_name / unnorm_key）
    suite_info = get_suite_info(exp_cfg.suite)
    task_suite_name = suite_info["task_suite_name"]
    unnorm_key = suite_info["unnorm_key"]

    # 若用户未自定义，与套件默认权重对齐
    model_path = exp_cfg.model_path
    default_model = suite_info.get("default_model")
    if model_path is None or ("libero-" in default_model and "libero-" in model_path and exp_cfg.suite not in model_path):
        # 简单启发式：当路径里不包含当前 suite 关键词时，用默认权重
        model_path = default_model

    gen_cfg = GenerateConfig(
        pretrained_checkpoint=model_path,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        model_family="openvla",
        task_suite_name=task_suite_name,
        num_steps_wait=NUM_STEPS_WAIT,
        env_img_res=ENV_IMG_RES,
    )

    # 加载一次模型与组件
    model = get_model(gen_cfg)
    processor = get_processor(gen_cfg)
    action_head = get_action_head(gen_cfg, llm_dim=model.llm_dim)
    proprio_projector = get_proprio_projector(gen_cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)

    results: Dict[str, Any] = {"original": [], "pruned": []}

    # 先跑 original（如果需要）
    if exp_cfg.test_mode in ("original", "both"):
        for task_idx in exp_cfg.task_indices:
            for r in range(exp_cfg.num_repeats):
                ok, acts, steps, vid = run_one_episode_with_task_index(
                    gen_cfg, model, processor, action_head, proprio_projector,
                    task_suite_name=gen_cfg.task_suite_name,
                    task_index=task_idx,
                    max_steps=exp_cfg.max_steps,
                    tag=f"original_t{task_idx}_r{r}",
                    save_videos=exp_cfg.save_videos,
                )
                results["original"].append({
                    "task": task_idx, "repeat": r, "success": ok, "steps": steps, "actions": acts, "video": vid,
                })

    # 注入裁剪再跑 pruned（如果需要）
    if exp_cfg.test_mode in ("pruned", "both"):
        print("[PruneVLA] Injecting KV-Pruning...")
        # 动态覆盖关键区间（失败即报错）
        per_image_patches = int(model.vision_backbone.get_num_patches())
        num_images = int(model.vision_backbone.get_num_images_in_input())
        if per_image_patches <= 0 or num_images <= 0:
            raise RuntimeError("Invalid vision backbone patch/images settings for prune attention")
        num_patches = per_image_patches * num_images

        extra = (1 if gen_cfg.use_proprio else 0) + (1 if getattr(gen_cfg, "use_diffusion", False) else 0)
        exp_cfg.prune_config.instruction_start = 1 + num_patches + extra
        # 同步视觉范围为真实 patch 数
        exp_cfg.prune_config.vision_range = (1, 1 + num_patches)

        action_dim = int(model.get_action_dim(getattr(gen_cfg, "unnorm_key", None)))
        if action_dim <= 0:
            raise RuntimeError("Failed to resolve action_dim for prune attention; check unnorm_key and model stats")
        exp_cfg.prune_config.expected_action_tokens = int(action_dim * NUM_ACTIONS_CHUNK)

        replace_attention_with_prune(model, exp_cfg.prune_config)
        for task_idx in exp_cfg.task_indices:
            for r in range(exp_cfg.num_repeats):
                # 跨任务 / 跨 episode 时重置一次索引缓存，避免序列长度变化导致越界
                exp_cfg.prune_config.kept_vision_key_indices = None
                ok, acts, steps, vid = run_one_episode_with_task_index(
                    gen_cfg, model, processor, action_head, proprio_projector,
                    task_suite_name=gen_cfg.task_suite_name,
                    task_index=task_idx,
                    max_steps=exp_cfg.max_steps,
                    tag=f"pruned_t{task_idx}_r{r}",
                    save_videos=exp_cfg.save_videos,
                )
                results["pruned"].append({
                    "task": task_idx, "repeat": r, "success": ok, "steps": steps, "actions": acts, "video": vid,
                })

    # 保存汇总日志
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sum_path = os.path.join("logs", f"{ts}--summary.json")
    try:
        import json
        # 统计成功率
        def _succ_rate(items):
            return (sum(1 for it in items if it.get("success")) / max(1, len(items))) if items else None
        original_rate = _succ_rate(results.get("original", []))
        pruned_rate = _succ_rate(results.get("pruned", []))
        if original_rate is not None:
            print(f"[PruneVLA] Original success rate: {original_rate*100:.2f}% ({len(results['original'])} episodes)")
        if pruned_rate is not None:
            print(f"[PruneVLA] Pruned   success rate: {pruned_rate*100:.2f}% ({len(results['pruned'])} episodes)")
        results["original_success_rate"] = original_rate
        results["pruned_success_rate"] = pruned_rate
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[PruneVLA] Saved summary to {sum_path}")
        # 额外提示当前套件与权重
        print(f"[PruneVLA] suite={exp_cfg.suite}, task_suite_name={task_suite_name}, unnorm_key={unnorm_key}")
        print(f"[PruneVLA] model_path={model_path}")
    except Exception:
        pass

    return results


