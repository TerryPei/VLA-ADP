#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
view_attn_npz.py

用途：查看并可视化 PruneVLA 落盘的注意力 .npz 文件。

功能：
- 支持读取单个 .npz 文件或整个目录（递归搜索）；
- 打印关键元信息（layer_idx, S, S_kept, num_heads 等）；
- 生成“头均值”的热力图（pruned SxS_kept）与“重构后的 SxS”热力图；
- 若存在同层的 preprune 文件，生成 preprune vs pruned 的对比图与差分图；
- 可选提取 text->vision 子块并画图（需提供 instruction_start 与 expected_action_tokens）。

注意：
- 图像默认保存到 --outdir；默认不弹窗显示（--show 可显示）。
"""

import argparse
import os
import sys
import glob
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _print_meta(npz: Dict[str, np.ndarray], path: str) -> None:
    layer = int(npz.get("layer_idx", -1)) if "layer_idx" in npz else -1
    S = int(npz.get("S", -1)) if "S" in npz else -1
    S_kept = int(npz.get("S_kept", -1)) if "S_kept" in npz else -1
    shape = npz.get("attn").shape if "attn" in npz else None
    print(f"[info] {os.path.basename(path)} | layer={layer} S={S} S_kept={S_kept} shape={shape}")


def _imshow(mat: np.ndarray, title: str, outpath: Optional[str] = None, show: bool = False,
            vmin: Optional[float] = None, vmax: Optional[float] = None,
            percent_clip: Optional[float] = 99.0) -> None:
    plt.figure(figsize=(6, 5))
    # 对比增强：按百分位裁剪上下界（默认 99%），避免极小/极大值“淹没”色彩
    if percent_clip is not None:
        lo = np.percentile(mat, 100 - percent_clip)
        hi = np.percentile(mat, percent_clip)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()


def _plot_pruned_and_reconstructed(npz: Dict[str, np.ndarray], outdir: str, fname_stem: str, show: bool,
                                   percent_clip: Optional[float]) -> None:
    attn = npz["attn"]  # [B,H,S,S_kept]
    layer = int(npz["layer_idx"]) if "layer_idx" in npz else -1
    S = int(npz["S"]) if "S" in npz else attn.shape[2]
    S_kept = int(npz["S_kept"]) if "S_kept" in npz else attn.shape[3]

    attn_mean = attn[0].mean(axis=0)  # [S,S_kept]
    _imshow(attn_mean, f"layer{layer} pruned (SxS_kept)", os.path.join(outdir, f"{fname_stem}.mean.pruned.png"), show, percent_clip=percent_clip)

    kept = npz.get("kept_indices_abs", None)
    if kept is not None:
        kept = kept.astype(np.int64)
        full = np.zeros((S, S), dtype=attn_mean.dtype)
        full[:, kept] = attn_mean
        _imshow(full, f"layer{layer} pruned reconstructed (SxS)", os.path.join(outdir, f"{fname_stem}.mean.pruned_recon.png"), show, percent_clip=percent_clip)


def _plot_preprune(npz: Dict[str, np.ndarray], outdir: str, fname_stem: str, show: bool,
                   percent_clip: Optional[float]) -> None:
    attn = npz["attn"]  # [B,H,S,S]
    layer = int(npz["layer_idx"]) if "layer_idx" in npz else -1
    attn_mean = attn[0].mean(axis=0)  # [S,S]
    _imshow(attn_mean, f"layer{layer} preprune (SxS)", os.path.join(outdir, f"{fname_stem}.mean.preprune.png"), show, percent_clip=percent_clip)


def _plot_diff(pre_npz: Dict[str, np.ndarray], pruned_npz: Dict[str, np.ndarray], outdir: str, fname_stem: str, show: bool,
               percent_clip: Optional[float]) -> None:
    pre = pre_npz["attn"][0].mean(axis=0)  # [S,S]
    S = int(pre_npz.get("S", pre.shape[-1]))
    attn = pruned_npz["attn"][0].mean(axis=0)  # [S,S_kept]
    kept = pruned_npz.get("kept_indices_abs")
    full = np.zeros((S, S), dtype=attn.dtype)
    if kept is not None:
        full[:, kept.astype(np.int64)] = attn
    _imshow(pre - full, "preprune - pruned_recon", os.path.join(outdir, f"{fname_stem}.mean.diff.png"), show, percent_clip=percent_clip)


def _plot_text2vision(npz_any: Dict[str, np.ndarray], outdir: str, fname_stem: str, show: bool,
                      instruction_start: int, expected_action_tokens: int, prepruned: bool = False,
                      percent_clip: Optional[float] = None) -> None:
    attn = npz_any["attn"][0].mean(axis=0)  # [S,S] 或 [S,S_kept]
    S = int(npz_any.get("S", attn.shape[-2]))
    V = instruction_start - 1
    text_rows = np.arange(instruction_start, S - expected_action_tokens)
    if prepruned:
        vis_cols = np.arange(1, V + 1)
        sub = attn[text_rows[:, None], vis_cols[None, :]]  # [T,V]
    else:
        kept = npz_any.get("kept_indices_abs")
        if kept is None:
            return
        kept = kept.astype(np.int64)
        kept_index = {int(k): i for i, k in enumerate(kept)}
        kept_vis_abs = [int(k) for k in kept if 1 <= int(k) <= V]
        kept_vis_rel = np.array([kept_index[k] for k in kept_vis_abs], dtype=np.int64)
        sub = attn[text_rows[:, None], kept_vis_rel[None, :]]  # [T, keep_v]

    # 画热力图与均值曲线
    _imshow(sub, f"text->vision (rows=T, cols=V or kept_v)", os.path.join(outdir, f"{fname_stem}.t2v.heat.png"), show, percent_clip=percent_clip)
    mean_vec = sub.mean(axis=0)
    plt.figure(figsize=(6, 3))
    plt.plot(mean_vec)
    plt.title("text->vision mean over text")
    if outdir:
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{fname_stem}.t2v.mean.png")); plt.close()
    elif show:
        plt.show()
    else:
        plt.close()


def _pair_files_by_layer(files: List[str]) -> Dict[int, Dict[str, str]]:
    pairs: Dict[int, Dict[str, str]] = {}
    for p in files:
        base = os.path.basename(p)
        # 解析 layer 编号
        try:
            # 名称包含 layer{L}
            if "layer" in base:
                seg = base.split("layer", 1)[1]
                layer_str = "".join([ch for ch in seg if ch.isdigit()])
                if layer_str:
                    L = int(layer_str)
                else:
                    continue
            else:
                continue
        except Exception:
            continue
        entry = pairs.setdefault(L, {})
        if base.startswith("attn.preprune"):
            entry["pre"] = p
        else:
            entry.setdefault("pruned", p)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="npz 文件或包含 npz 的目录（建议传 logs/attn/MMDD_HHMMSS_suite/）")
    parser.add_argument("--outdir", type=str, default="attn_plots", help="输出图片根目录")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示（默认保存图片不显示）")
    parser.add_argument("--instruction_start", type=int, default=257, help="文本起始（默认 257）")
    parser.add_argument("--expected_action_tokens", type=int, default=56, help="动作段长度（默认 56）")
    parser.add_argument("--clip", type=float, default=99.0, help="热力图百分位裁剪（默认99，None=不裁剪）")
    args = parser.parse_args()

    # 若 path 是形如 logs/attn/MMDD_HHMMSS_suite/，则将该目录名拼到 outdir 下
    base_dirname = os.path.basename(os.path.normpath(args.path))
    out_root = os.path.join(args.outdir, base_dirname)
    _ensure_outdir(out_root)

    targets: List[str] = []
    if os.path.isdir(args.path):
        targets = glob.glob(os.path.join(args.path, "**", "*.npz"), recursive=True)
    else:
        targets = [args.path]

    # 按层配对 preprune/pruned
    pairs = _pair_files_by_layer(sorted(targets))
    if not pairs:
        print("[warn] 未找到任何 npz 文件")
        return

    for layer, files in sorted(pairs.items()):
        pruned_p = files.get("pruned")
        pre_p = files.get("pre")
        print(f"\n=== Layer {layer} ===")
        if pruned_p:
            pruned = _load_npz(pruned_p)
            _print_meta(pruned, pruned_p)
            stem = os.path.splitext(os.path.basename(pruned_p))[0]
            outdir = os.path.join(out_root, f"layer{layer}")
            _ensure_outdir(outdir)
            _plot_pruned_and_reconstructed(pruned, outdir, stem, args.show, args.clip)
            _plot_text2vision(pruned, outdir, stem, args.show, args.instruction_start, args.expected_action_tokens, prepruned=False, percent_clip=args.clip)
        if pre_p:
            pre = _load_npz(pre_p)
            _print_meta(pre, pre_p)
            stem_pre = os.path.splitext(os.path.basename(pre_p))[0]
            outdir = os.path.join(out_root, f"layer{layer}")
            _ensure_outdir(outdir)
            _plot_preprune(pre, outdir, stem_pre, args.show, args.clip)
            if pruned_p:
                _plot_diff(pre, pruned, outdir, f"layer{layer}.mean", args.show, args.clip)
                _plot_text2vision(pre, outdir, stem_pre, args.show, args.instruction_start, args.expected_action_tokens, prepruned=True, percent_clip=args.clip)


if __name__ == "__main__":
    main()


