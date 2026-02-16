"""
PruneVLA: KV 裁剪版本的 OpenVLA 实验代码（外部注入注意力层）。

目录结构：
- configs/prune_config.py         配置定义
- prune_kv_attention.py          K/V 裁剪注意力实现与注入入口
- batch_experiment.py            小批量实验脚本
- experiment_controller.py       实验控制器（quick / comprehensive / debug）
"""

__all__ = []


