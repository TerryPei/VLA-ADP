from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class PruneVLAConfig:
    """
    KV 裁剪配置（支持多层渐进裁剪）。

    - prune_schedule: 渐进裁剪日程，按 layer 触发累计保留比例（相对原始视觉长度 V0）。
      例如：[{"layer":2, "cumulative_keep":0.75}, {"layer":8, "cumulative_keep":0.5}]
    - use_dynamic_ranges/instruction_start/expected_action_tokens: 动态定位各段 token 范围。
    - debug: 调试信息开关。
    """

    prune_schedule: List[Dict[str, float]] = None

    use_dynamic_ranges: bool = True
    instruction_start: int = 513
    expected_action_tokens: int = 56

    # 与 sparsevla 的 VLASparseAttentionConfig 对齐的默认固定范围（保持向后兼容）
    bos_range: Tuple[int, int] = (0, 1)
    vision_range: Tuple[int, int] = (1, 513)
    instruction_range: Tuple[int, int] = (515, 550)
    action_range: Tuple[int, int] = (551, 607)

    debug: bool = False

    # 运行时缓存（跨层共享）
    kept_vision_key_indices: Optional[object] = None  # 全序列 key 维的保留索引（Tensor[int]）

    # ---- 重要性打分消融开关 ----
    # importance_use_causal_mask: 在计算视觉重要性时，是否对 attn_scores 施加因果/模型提供的 attention_mask
    # importance_use_softmax: 在打分前是否对 key 维做 softmax（概率意义的打分）。
    # 说明：仅开 causal 而不开 softmax，对于“instruction->vision”的条目基本与原实现数值等价；
    #      两者同时开启才是有辨识度的对照（使用带因果的注意力概率质量作为重要性）。
    importance_use_causal_mask: bool = False
    importance_use_softmax: bool = False

    # ---- 消融：不做重要性与裁剪，但仍走包装层完整注意力路径 ----
    use_original: bool = False

    # ----- 注意力可视化/落盘（可选）-----
    dump_attn: bool = False                      # 是否落盘注意力图
    dump_attn_layers: Optional[List[int]] = None # 指定层（0-based）；None 按 scope 判断
    dump_attn_scope: str = "trigger_only"        # "trigger_only" | "all_after"
    dump_attn_what: str = "probs"                # "probs" | "scores"
    dump_attn_dtype: str = "float16"             # "float16" | "float32"
    dump_pre_prune_at_trigger: bool = False      # 触发层额外保存未裁剪完整 attn map
    dump_pre_prune_layers: Optional[List[int]] = None  # 额外指定若干层也保存未裁剪 attn（0-based）
    dump_attn_dir_root: str = "logs/attn"        # 根目录
    dump_suite_name: Optional[str] = None        # 任务类型（如 libero_spatial）
    dump_task_index: Optional[int] = None        # 任务序号
    _dump_session_dir: Optional[str] = None      # 运行时缓存：本次会话目录
    # 仅在第几个“整模型前向/动作chunk”进行保存（基于 layer_idx==0 计数）。
    # None=每个前向都保存；[0]=仅第一个动作chunk；[0,2]=第1与第3个动作chunk。
    dump_forward_indices: Optional[List[int]] = None
    _dump_forward_count: int = 0                 # 运行时累计的前向计数（从0开始）
    _dump_current_forward_allowed: bool = True   # 本次前向是否允许落盘（由第0层决定）

    def __post_init__(self):
        if self.prune_schedule is None:
            # 默认不裁剪（可等价于 cumulative_keep=1.0 的无效步骤）
            self.prune_schedule = []

    # 注意：动态范围计算已迁移至 prune_kv_attention.compute_dynamic_ranges_for_config，
    # 此处不再保留方法以避免重复定义。


@dataclass
class PruneExperimentConfig:
    """
    实验配置（对齐 sparsevla 的 experiment_config 结构）：
    - test_mode: "original" / "pruned" / "both"
    - task_indices: 任务索引列表，None=全部10个
    - num_repeats: 每个任务重复次数
    - model_path: 预训练权重
    - max_steps: 每个 episode 最大步数
    - save_videos / save_logs: 输出控制
    - prune_config: 上面的 KV 裁剪配置
    """

    test_mode: str = "both"
    task_indices: Optional[List[int]] = None
    num_repeats: int = 1

    # 任务套件选择：spatial / object / goal / 10
    suite: str = "spatial"

    # 默认权重（可被覆盖）。若与 suite 不匹配，运行时将自动提示并尝试切换到对应默认权重
    model_path: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    max_steps: int = 220

    save_videos: bool = True
    save_logs: bool = True

    prune_config: PruneVLAConfig = PruneVLAConfig()

    def __post_init__(self):
        if self.task_indices is None:
            self.task_indices = list(range(10))


# =============================
# 套件元信息与帮助函数
# =============================

SUITE_META: Dict[str, Dict[str, str]] = {
    "spatial": {
        "task_suite_name": "libero_spatial",
        "unnorm_key": "libero_spatial_no_noops",
        "default_model": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    },
    "object": {
        "task_suite_name": "libero_object",
        "unnorm_key": "libero_object_no_noops",
        "default_model": "moojink/openvla-7b-oft-finetuned-libero-object",
    },
    "goal": {
        "task_suite_name": "libero_goal",
        "unnorm_key": "libero_goal_no_noops",
        "default_model": "moojink/openvla-7b-oft-finetuned-libero-goal",
    },
    "10": {
        "task_suite_name": "libero_10",
        "unnorm_key": "libero_10_no_noops",
        "default_model": "moojink/openvla-7b-oft-finetuned-libero-10",
    },
}


def get_suite_info(suite: str) -> Dict[str, str]:
    if suite not in SUITE_META:
        raise ValueError(f"Unknown suite: {suite}. Choose from {list(SUITE_META.keys())}")
    return SUITE_META[suite]


# =============================
# 预定义配置
# =============================

def get_quick_test_config() -> PruneExperimentConfig:
    return PruneExperimentConfig(
        test_mode="pruned",
        task_indices=[0],
        num_repeats=1,
        save_videos=True,
        save_logs=True,
        suite="goal",
        prune_config=PruneVLAConfig(
            prune_schedule=[{"layer": 16, "cumulative_keep": 0.3}],
            debug=True,
        ),
    )


def get_comprehensive_test_config() -> PruneExperimentConfig:
    return PruneExperimentConfig(
        test_mode="both",
        task_indices=[0, 1, 2, 3, 4],
        num_repeats=2,
        save_videos=True,
        save_logs=True,
        suite="spatial",
        prune_config=PruneVLAConfig(
            prune_schedule=[{"layer": 16, "cumulative_keep": 0.4}],
            debug=False,
        ),
    )


def get_sparse_ratio_comparison_config() -> list[PruneExperimentConfig]:
    cfgs = []
    for ratio in [0.3, 0.5, 0.7, 0.9]:
        keep = 1.0 - ratio
        cfgs.append(
            PruneExperimentConfig(
                test_mode="pruned",
                task_indices=[0, 1, 2],
                num_repeats=2,
                suite="spatial",
                prune_config=PruneVLAConfig(
                    prune_schedule=[{"layer": 16, "cumulative_keep": keep}],
                    debug=False,
                ),
            )
        )
    return cfgs


def get_debug_config() -> PruneExperimentConfig:
    return PruneExperimentConfig(
        test_mode="pruned",
        task_indices=[0],
        num_repeats=1,
        save_videos=False,
        save_logs=True,
        prune_config=PruneVLAConfig(
            prune_schedule=[{"layer": 12, "cumulative_keep": 0.4}],
            debug=True,
        ),
    )


# 保存注意力图的配置：
# - 保存触发层及其后的所有层的 attn map（dump_attn_scope="all_after"）
# - 触发层额外保存未裁剪完整 attn（dump_pre_prune_at_trigger=True）
# - 使用 softmax 之后的概率（dump_attn_what="probs"）
def get_attn_dump_config() -> PruneExperimentConfig:
    return PruneExperimentConfig(
        test_mode="pruned",
        task_indices=[0],
        num_repeats=1,
        save_videos=False,
        save_logs=True,
        suite="spatial",
        prune_config=PruneVLAConfig(
            prune_schedule=[
                {"layer": 15, "cumulative_keep": 0.75},
                {"layer": 24, "cumulative_keep": 0.50},
                {"layer": 31, "cumulative_keep": 0.25},
            ],
            debug=True,
            dump_attn=True,
            dump_attn_layers=list(range(32)),
            dump_attn_scope="all_after",
            dump_attn_what="probs",
            dump_attn_dtype="float16",
            dump_pre_prune_at_trigger=True,
            dump_attn_dir_root="logs/attn",
            dump_forward_indices=[0],
        ),
    )

# 渐进裁剪：spatial 套件 Task 4，仅跑 pruned
def get_progressive_spatial_task4_config() -> PruneExperimentConfig:
    return PruneExperimentConfig(
        test_mode="pruned",
        task_indices=[0],
        num_repeats=1,
        save_videos=True,
        save_logs=True,
        suite="spatial",
        prune_config=PruneVLAConfig(
            prune_schedule=[
                # {"layer": 2, "cumulative_keep": 0.75},
                # {"layer": 8, "cumulative_keep": 0.50},
                # {"layer": 16, "cumulative_keep": 0.25},
                {"layer": 15, "cumulative_keep": 0.75},
                {"layer": 24, "cumulative_keep": 0.50},
                {"layer": 31, "cumulative_keep": 0.25},
            ],
            debug=True,
        ),
    )


# 活跃配置（如需）
ACTIVE_CONFIG = get_quick_test_config()


def get_config() -> PruneExperimentConfig:
    return ACTIVE_CONFIG
