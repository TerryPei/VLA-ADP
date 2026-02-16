#!/usr/bin/env python3
"""
PruneVLA 实验控制器：quick / comprehensive / debug

独立于 openvla 主体与 sparsevla，便于快速验证 KV-Pruning 的可行性。
"""

import argparse
import sys
import os
import time
from datetime import datetime

try:
    from .configs.prune_config import (
        PruneVLAConfig,
        PruneExperimentConfig,
        get_quick_test_config,
        get_comprehensive_test_config,
        get_debug_config,
        get_progressive_spatial_task4_config,
        get_attn_dump_config,
    )
    from .batch_experiment import run_prune_experiment, run_prune_experiments
except Exception:
    # 兼容脚本直跑：python prunevla/experiment_controller.py
    import sys, os
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from configs.prune_config import (
        PruneVLAConfig,
        PruneExperimentConfig,
        get_quick_test_config,
        get_comprehensive_test_config,
        get_debug_config,
        get_progressive_spatial_task4_config,
        get_attn_dump_config,
    )
    from batch_experiment import run_prune_experiment, run_prune_experiments


def run_quick(mode_override: str | None = None, suite_override: str | None = None):
    exp_cfg = get_quick_test_config()
    if mode_override:
        exp_cfg.test_mode = mode_override
    if suite_override:
        exp_cfg.suite = suite_override
    return run_prune_experiments(exp_cfg)


def run_debug(mode_override: str | None = None, suite_override: str | None = None):
    exp_cfg = get_debug_config()
    if mode_override:
        exp_cfg.test_mode = mode_override
    if suite_override:
        exp_cfg.suite = suite_override
    return run_prune_experiments(exp_cfg)


def run_comprehensive(mode_override: str | None = None, suite_override: str | None = None):
    exp_cfg = get_comprehensive_test_config()
    if mode_override:
        exp_cfg.test_mode = mode_override
    if suite_override:
        exp_cfg.suite = suite_override
    return run_prune_experiments(exp_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="PruneVLA 实验控制器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        choices=["quick", "comprehensive", "debug", "progressive", "attn_dump"],
        default="quick",
        help="选择实验配置类型",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "pruned", "both"],
        default=None,
        help="覆盖配置中的 test_mode",
    )
    parser.add_argument(
        "--suite",
        choices=["spatial", "object", "goal", "10"],
        default=None,
        help="选择 LIBERO 任务套件",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    print("🎉 欢迎使用 PruneVLA 实验控制器!", flush=True)
    print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    t0 = time.time()
    print(f"[PruneVLA] argv={sys.argv}", flush=True)
    print(f"[PruneVLA] CWD={os.getcwd()}", flush=True)
    print(f"[PruneVLA] __package__={__package__}", flush=True)
    print(f"[PruneVLA] __name__={__name__}", flush=True)

    if args.config == "quick":
        exp_results = run_quick(args.mode, args.suite)
    elif args.config == "comprehensive":
        exp_results = run_comprehensive(args.mode, args.suite)
    elif args.config == "progressive":
        exp_cfg = get_progressive_spatial_task4_config()
        if args.mode:
            exp_cfg.test_mode = args.mode
        if args.suite:
            exp_cfg.suite = args.suite
        exp_results = run_prune_experiments(exp_cfg)
    elif args.config == "attn_dump":
        exp_cfg = get_attn_dump_config()
        if args.mode:
            exp_cfg.test_mode = args.mode
        if args.suite:
            exp_cfg.suite = args.suite
        # 注入 suite 信息到落盘目录命名
        exp_cfg.prune_config.dump_suite_name = exp_cfg.suite
        exp_results = run_prune_experiments(exp_cfg)
    else:
        exp_results = run_debug(args.mode, args.suite)

    # 若指定了套件，覆盖结果中的 suite（实际执行在 batch_experiment 内读取配置时已生效）
    if args.suite:
        # 直接提示：在 configs.get_quick_test_config 里改 suite 更合适
        pass
    t1 = time.time()

    print("\n===== 实验完成 =====", flush=True)
    print(f"总耗时: {(t1-t0):.2f}s", flush=True)
    print(exp_results, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")


