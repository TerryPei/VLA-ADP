"""
VLA KV-Pruning Attention: 一次计算视觉重要性 -> 跨层持久裁剪 K/V 列，降低注意力主干 FLOPs。

实现要点：
- 不改变序列长度与输出形状（B, S, D）以兼容 OpenVLA 解码与动作切片。
- 仅在注意力的 key/value 维度进行列裁剪（gather），缩短 QK^T 的列维与 probs@V 的列维。
- 重要性仅在指定层计算一次（或后续扩展分段刷新），后续层持久复用保留索引。
"""

from typing import Optional, Tuple, Any, List
import json
import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .configs.prune_config import PruneVLAConfig
except Exception:
    # 兼容脚本直跑：python prunevla/experiment_controller.py
    import sys as _sys, os as _os
    _CUR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR not in _sys.path:
        _sys.path.append(_CUR)
    from configs.prune_config import PruneVLAConfig

def compute_dynamic_ranges_for_config(config: PruneVLAConfig, seq_len: int) -> dict:
    """
    基于 config 与实际序列长度动态计算 token 范围。
    返回包含：'bos_range'、'vision_range'、'instruction_range'、'action_range'、
    'has_instruction'、'has_action'，可选 'stop_position'。
    """
    if not getattr(config, "use_dynamic_ranges", True):
        return {
            'bos_range': getattr(config, 'bos_range', (0, 1)),
            'vision_range': getattr(config, 'vision_range', (1, 513)),
            'instruction_range': getattr(config, 'instruction_range', (515, 550)),
            'action_range': getattr(config, 'action_range', (551, 607)),
            'has_instruction': True,
            'has_action': seq_len >= getattr(config, 'action_range', (551, 607))[0],
        }

    bos_range = getattr(config, 'bos_range', (0, 1))
    vision_range = getattr(config, 'vision_range', (1, 513))
    instruction_start = int(getattr(config, 'instruction_start', 513))
    expected_action_tokens = int(getattr(config, 'expected_action_tokens', 56))

    ranges = {
        'bos_range': bos_range,
        'vision_range': vision_range,
    }

    if seq_len <= instruction_start:
        ranges['instruction_range'] = None
        ranges['action_range'] = None
        ranges['has_instruction'] = False
        ranges['has_action'] = False
    else:
        # 最小包含完整 action 的序列长度：instruction_start + 1(至少一个文本) + action_tokens + 1(STOP)
        min_seq_for_actions = instruction_start + 1 + expected_action_tokens + 1
        if seq_len >= min_seq_for_actions:
            action_start = seq_len - expected_action_tokens - 1
            action_end = seq_len - 1
            ranges['instruction_range'] = (instruction_start, action_start)
            ranges['action_range'] = (action_start, action_end)
            ranges['has_instruction'] = True
            ranges['has_action'] = True
            ranges['stop_position'] = seq_len - 1
        else:
            ranges['instruction_range'] = (instruction_start, seq_len)
            ranges['action_range'] = None
            ranges['has_instruction'] = True
            ranges['has_action'] = False
            ranges['stop_position'] = None

    if getattr(config, 'debug', False):
        instr = ranges.get('instruction_range')
        act = ranges.get('action_range')
        if instr:
            s, e = instr
            print(f"Text tokens: {s}-{e-1}")
        if act:
            s, e = act
            print(f"Action tokens: {s}-{e-1}")

    return ranges


class VLAKVPruningAttention(nn.Module):
    def __init__(self, original_attention: nn.Module, config: PruneVLAConfig, layer_idx: int) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.config = config
        self.layer_idx = layer_idx
        self._last_seq_len: Optional[int] = None

        # 尝试复制原注意力的一些基础属性（便于一致行为与调试）
        for name in (
            "hidden_size",
            "num_heads",
            "head_dim",
            "num_key_value_heads",
            "num_key_value_groups",
            "max_position_embeddings",
            "rope_theta",
        ):
            if hasattr(original_attention, name):
                setattr(self, name, getattr(original_attention, name))

        # RoPE 支持（与 LLaMA 行为对齐）
        self.has_rope = hasattr(self.original_attention, "rotary_emb")

    def _compute_dynamic_ranges(self, seq_len: int) -> dict:
        # 使用本文件的动态范围推断
        ranges = compute_dynamic_ranges_for_config(self.config, seq_len)

        # 统一规范化：若期望存在 action_tokens，则强制将 instruction_range 截断到 [instruction_start, seq_len - expected_action_tokens)
        exp_actions = int(getattr(self.config, "expected_action_tokens", 0) or 0)
        if exp_actions > 0 and ranges.get("instruction_range") is not None:
            instr_start, instr_end = ranges["instruction_range"]
            # 仅当截断后仍有文本长度且不与 vision 段重叠时，才应用截断
            if (seq_len - exp_actions) > instr_start:
                new_instr_end = max(instr_start, seq_len - exp_actions)
                ranges["instruction_range"] = (instr_start, new_instr_end)
                ranges["has_instruction"] = True
                # 标注 action 范围
                ranges["action_range"] = (new_instr_end, seq_len)
                ranges["has_action"] = True
        return ranges

    def _extract_qkv(self, hidden_states: torch.Tensor, position_ids: Optional[torch.LongTensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.original_attention.q_proj(hidden_states)
        k = self.original_attention.k_proj(hidden_states)
        v = self.original_attention.v_proj(hidden_states)

        bsz, seqlen, hidden = hidden_states.shape
        num_heads = getattr(self.original_attention, "num_heads", None)
        if num_heads is None:
            # 回退：从 q 形状推断
            num_heads = q.shape[-1] // (hidden // (q.shape[-1] // seqlen))
        head_dim = q.shape[-1] // num_heads

        q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)

        if self.has_rope:
            try:
                # rotary_emb(value_states, position_ids)
                if position_ids is None:
                    position_ids = torch.arange(seqlen, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                cos, sin = self.original_attention.rotary_emb(v, position_ids)
                # HF 的 apply_rotary_pos_emb 接口
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except Exception:
                pass

        return q, k, v

    @torch.no_grad()
    def _maybe_compute_kept_indices_once(self, hidden_states: torch.Tensor,
                                         attention_mask: Optional[torch.Tensor],
                                         position_ids: Optional[torch.LongTensor],
                                         *,
                                         force_recompute: bool = False) -> None:
        bsz, seqlen, _ = hidden_states.shape
        # 若已有缓存且序列长度未变，且未强制重算，则直接返回
        if (not force_recompute) and self.config.kept_vision_key_indices is not None and self._last_seq_len == seqlen:
            return

        ranges = self._compute_dynamic_ranges(seqlen)
        vis_start, vis_end = ranges["vision_range"]
        instr_range = ranges.get("instruction_range")
        has_instr = ranges.get("has_instruction", False)
        if not has_instr or not instr_range:
            # 没有文本区域，不允许继续
            raise RuntimeError(
                f"[PruneVLA] No instruction tokens detected at layer={self.layer_idx}, seq_len={seqlen}."
            )

        instr_start, instr_end = instr_range

        # 先确定候选视觉列索引（相对 vis_start 的相对索引），以便后续只取子矩阵
        # 渐进裁剪：基于 schedule 的累计保留比例。
        schedule = getattr(self.config, "prune_schedule", []) or []
        target_keep_ratio: Optional[float] = None
        if schedule:
            for item in sorted(schedule, key=lambda x: x.get("layer", 0)):
                if int(item.get("layer", -1)) == self.layer_idx:
                    target_keep_ratio = float(item.get("cumulative_keep", 1.0))
        if target_keep_ratio is None:
            # 该层不是触发层，不更新
            self._last_seq_len = seqlen
            return

        # 原始视觉长度（按输入布局固定）：
        V0 = vis_end - vis_start
        target_keep_v = max(1, min(V0, int(round(V0 * target_keep_ratio))))

        # 候选视觉索引：若已有裁剪，则只在“当前剩余的视觉列”范围内继续筛选；否则使用全视觉范围
        if self.config.kept_vision_key_indices is not None:
            prev_kept = self.config.kept_vision_key_indices.to(device=hidden_states.device)
            # 取交集：上一阶段保留列 ∩ 当前视觉列范围 [vis_start, vis_end)
            mask = (prev_kept >= vis_start) & (prev_kept < vis_end)
            candidate_vis_abs = prev_kept[mask]
            # 按绝对索引升序
            candidate_vis_abs = candidate_vis_abs.sort().values
            # 转为相对视觉子区间的索引（便于从 k_vis_full 中 gather）
            candidate_vis_rel = candidate_vis_abs - vis_start
        else:
            candidate_vis_abs = torch.arange(vis_start, vis_end, device=hidden_states.device)
            candidate_vis_rel = torch.arange(0, V0, device=hidden_states.device)

        candidate_count = int(candidate_vis_rel.numel())
        if candidate_count == 0:
            raise RuntimeError("[PruneVLA] No candidate visual tokens to prune from; check earlier steps.")

        # 提取 q/k/v，改为仅计算 text->vision 子矩阵分数，避免全量 [S,S]
        q, k, v = self._extract_qkv(hidden_states, position_ids)
        num_heads = q.shape[1]
        head_dim = q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)

        # 子矩阵：q_text @ k_vis_cand^T → [B,H,T,V_cand]
        q_text = q[:, :, instr_start:instr_end, :]                                   # [B,H,T,d]
        k_vis_full = k[:, :, vis_start:vis_end, :]                                   # [B,H,V,d]
        k_vis_cand = k_vis_full.index_select(dim=2, index=candidate_vis_rel)         # [B,H,V_cand,d]
        scoring_tensor = torch.matmul(q_text, k_vis_cand.transpose(-1, -2)) * scale  # [B,H,T,V_cand]

        # 仅当显式要求时才在触发层对子矩阵做 softmax；因果对 text->vision 无实际作用，忽略
        use_softmax = bool(getattr(self.config, "importance_use_softmax", False))
        if use_softmax:
            scoring_tensor = F.softmax(scoring_tensor, dim=-1, dtype=torch.float32).to(q.dtype)

        # 因果掩码检查 abs
        # if use_causal or use_softmax:
        #     try:
        #         msg_parts = [f"[PruneVLA][check] layer={self.layer_idx} causal={use_causal} softmax={use_softmax}"]
        #         if use_causal:
        #             tri_bool = torch.triu(
        #                 torch.ones((1, 1, seqlen, seqlen), device=attn_scores.device, dtype=torch.bool), diagonal=1
        #             )
        #             upper = tri_bool.expand_as(attn_scores)
        #             neg_inf = torch.finfo(attn_scores.dtype).min
        #             delta = scoring_tensor - attn_scores
        #             masked_frac = (delta == neg_inf)[upper].float().mean().item() if upper.any() else 1.0
        #             msg_parts.append(f"upper_masked_frac={masked_frac:.3f}")
        #         if use_softmax:
        #             row_sums = scoring_tensor[:, :, instr_start:instr_end, :].sum(dim=-1)
        #             row_sums_mean = float(row_sums.mean().item()) if row_sums.numel() > 0 else float('nan')
        #             msg_parts.append(f"row_sum_mean~{row_sums_mean:.3f}")
        #         print(" ".join(msg_parts))
        #     except Exception:
        #         pass

        # （后续逻辑保持不变）

        # 计算仅在候选视觉列上的重要性（按头均值、再跨文本均值）
        importance = scoring_tensor.mean(dim=2).mean(dim=1)  # [B, cand]

        # 需要从候选中保留的数量：不能多于候选，也不能少于目标（若候选已≤目标，则不再变小）
        keep_now = min(candidate_count, target_keep_v)

        _, top_idx = torch.topk(importance, k=keep_now, dim=-1)  # [B, keep_now]
        top_idx = top_idx.sort(dim=-1).values  # 升序

        # 映射到全序列 key 维绝对索引，并与“非视觉列”合并
        kept_vis_absolute = candidate_vis_abs.index_select(dim=0, index=top_idx[0])  # [keep_now]
        # 非视觉列：0..vis_start-1 与 vis_end..seqlen-1 全部保留
        non_vis_left = torch.arange(0, vis_start, device=hidden_states.device)
        non_vis_right = torch.arange(vis_end, seqlen, device=hidden_states.device)
        base_non_vis = torch.cat([non_vis_left, non_vis_right], dim=0)  # [S - V]

        # 评测默认 B=1，这里先实现 B=1（后续可扩展到 per-sample gather）
        if bsz != 1:
            # 暂不支持 batch>1，避免静默回退
            raise NotImplementedError("[PruneVLA] KV-Pruning currently supports batch size == 1 only.")

        kept_all = torch.cat([base_non_vis, kept_vis_absolute], dim=0).sort().values  # [S - V + keep_now]
        self.config.kept_vision_key_indices = kept_all  # Tensor[int], key维保留索引
        self._last_seq_len = seqlen

        if self.config.debug:
            kept_ratio = float(kept_all.numel()) / float(seqlen)
            # 计算对照统计：text->vision 与 vision->vision 的均值（vv 可采样）
            try:
                tv_mean = float(scoring_tensor.mean().item())
            except Exception:
                tv_mean = None
            vv_mean = None
            # 子矩阵路径无完整 vision->vision 分数，跳过 vv_mean 计算

            # 打印 top-10 保留视觉列及其分数
            top_show = min(10, keep_now)
            kept_scores = importance[0].index_select(dim=0, index=top_idx[0])
            kept_pairs = list(zip(kept_vis_absolute[:top_show].tolist(), kept_scores[:top_show].tolist()))
            print(
                f"[PruneVLA] layer={self.layer_idx} vis=({vis_start},{vis_end}) instr=({instr_start},{instr_end}) "
                f"candidates={candidate_count} target_keep={target_keep_v} keep_now={keep_now} "
                f"keys_kept={kept_all.numel()}/{seqlen} ({kept_ratio:.2%}) tv_mean={tv_mean:.4f}"
                + (f" vv_mean={vv_mean:.4f}" if vv_mean is not None else "")
            )
            print(f"[PruneVLA] kept_top{top_show} (abs_idx, score) sample: {kept_pairs}")

            # 落盘 JSON（每层一个文件）
            try:
                os.makedirs("logs", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dump = {
                    "layer": self.layer_idx,
                    "vision_range": [int(vis_start), int(vis_end)],
                    "instruction_range": [int(instr_start), int(instr_end)],
                    "candidate_count": int(candidate_count),
                    "target_keep_v": int(target_keep_v),
                    "keep_now": int(keep_now),
                    "keys_kept_count": int(kept_all.numel()),
                    "seq_len": int(seqlen),
                    "tv_mean": tv_mean,
                    "vv_mean": vv_mean,
                    "kept_indices_abs": [int(x) for x in kept_vis_absolute.tolist()],
                    "top_pairs_sample": [(int(i), float(s)) for i, s in kept_pairs],
                }
                with open(os.path.join("logs", f"prune_debug.{ts}.layer{self.layer_idx}.json"), "w", encoding="utf-8") as f:
                    json.dump(dump, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 消融：不计算重要性、不做裁剪，但仍走包装层完整注意力路径
        if bool(getattr(self.config, "use_original", False)):
            bsz, seqlen, _ = hidden_states.shape
            # 计算 Q/K/V
            q = self.original_attention.q_proj(hidden_states)
            k = self.original_attention.k_proj(hidden_states)
            v = self.original_attention.v_proj(hidden_states)

            num_heads = getattr(self.original_attention, "num_heads", 32)
            head_dim = q.shape[-1] // num_heads
            q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)

            # RoPE（如可用）
            if self.has_rope:
                try:
                    if position_ids is None:
                        position_ids = torch.arange(seqlen, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                    cos, sin = self.original_attention.rotary_emb(v, position_ids)
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                except Exception:
                    pass

            scale = 1.0 / (head_dim ** 0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # 加上 attention_mask（不做任何 key 维裁剪）
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1)  # [B,1,S,S]
                attn_scores = attn_scores + attention_mask

            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, num_heads * head_dim)
            out = self.original_attention.o_proj(attn_output)
            return out, None if not output_attentions else attn_probs, None
        # 在第 0 层，判定本次整模型前向是否允许落盘（基于 dump_forward_indices）
        if getattr(self.config, "dump_attn", False) and int(self.layer_idx) == 0:
            indices = getattr(self.config, "dump_forward_indices", None)
            if indices is None:
                self.config._dump_current_forward_allowed = True
            else:
                self.config._dump_current_forward_allowed = (self.config._dump_forward_count in set(indices))
            # 增加计数供下一次整模型前向使用
            self.config._dump_forward_count += 1

        # 未到任何 schedule 触发层：走原层计算（保持完全一致）
        schedule_layers = sorted([int(it.get("layer", -1)) for it in (getattr(self.config, "prune_schedule", []) or [])])
        next_trigger = min([l for l in schedule_layers if l >= 0], default=None)
        if next_trigger is None or self.layer_idx < next_trigger:
            # 若需要在触发层之前的层也保存“未裁剪的完整 attn map”，则在此落盘
            need_dump = False
            if getattr(self.config, "dump_attn", False) and getattr(self.config, "_dump_current_forward_allowed", True):
                layers_sel = getattr(self.config, "dump_attn_layers", None)
                if layers_sel is not None:
                    need_dump = int(self.layer_idx) in set(layers_sel)
                else:
                    # 未指定列表时，默认仅根据 scope；但对于触发前层 scope=all_after 不包含此层
                    # 因此只有显式 dump_attn_layers 才会保存此层
                    need_dump = False
            if need_dump:
                try:
                    q_full, k_full, _ = self._extract_qkv(hidden_states, position_ids)
                    head_dim_full = q_full.shape[-1]
                    scale_full = 1.0 / (head_dim_full ** 0.5)
                    full_scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale_full
                    full_probs = F.softmax(full_scores, dim=-1, dtype=torch.float32).to(q_full.dtype)

                    # 组织目录：logs/attn/{mmdd_HHMMSS}_{suite}/task_{idx}/
                    root_dir = getattr(self.config, "dump_attn_dir_root", "logs/attn")
                    os.makedirs(root_dir, exist_ok=True)
                    session_dir = getattr(self.config, "_dump_session_dir", None)
                    if session_dir is None:
                        ts = datetime.now().strftime("%m%d_%H%M%S")
                        suite = getattr(self.config, "dump_suite_name", None) or "unknown"
                        dir_name = f"{ts}_{suite}"
                        session_dir = os.path.join(root_dir, dir_name)
                        task_idx = getattr(self.config, "dump_task_index", None)
                        if task_idx is not None:
                            session_dir = os.path.join(session_dir, f"task_{int(task_idx)}")
                        os.makedirs(session_dir, exist_ok=True)
                        self.config._dump_session_dir = session_dir

                    dtype = torch.float16 if getattr(self.config, "dump_attn_dtype", "float16") == "float16" else torch.float32
                    what = getattr(self.config, "dump_attn_what", "probs")
                    full_dump = full_probs if what == "probs" else full_scores.to(full_probs.dtype)
                    full_np = full_dump.to(dtype).detach().cpu().numpy()
                    s_np = int(full_probs.shape[-1])
                    fname = os.path.join(session_dir, f"attn.preprune.layer{self.layer_idx}.S{s_np}.Skept{s_np}.{ 'fp16' if dtype==torch.float16 else 'fp32' }.npz")
                    np.savez_compressed(
                        fname,
                        attn=full_np,
                        layer_idx=int(self.layer_idx),
                        S=int(s_np),
                        S_kept=int(s_np),
                        num_heads=int(full_probs.shape[1]),
                    )
                except Exception as _e:
                    if getattr(self.config, "debug", False):
                        print(f"[PruneVLA][warn] failed saving pre-trigger attn at layer={self.layer_idx}: {_e}")
            return self.original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # 在 schedule 触发层：计算 / 更新 kept indices
        schedule_here = any(int(it.get("layer", -1)) == self.layer_idx for it in (getattr(self.config, "prune_schedule", []) or []))
        if schedule_here:
            # 若需要在触发层保存“未裁剪的完整 attn map”（[B,H,S,S]）
            need_pre_at_trigger = getattr(self.config, "dump_attn", False) and getattr(self.config, "dump_pre_prune_at_trigger", False)
            # 或者：配置了 dump_pre_prune_layers 包含本层
            pre_layers = getattr(self.config, "dump_pre_prune_layers", None)
            need_pre_layers = bool(pre_layers) and (int(self.layer_idx) in set(pre_layers))
            if need_pre_at_trigger or need_pre_layers:
                try:
                    q_full, k_full, _ = self._extract_qkv(hidden_states, position_ids)
                    head_dim_full = q_full.shape[-1]
                    scale_full = 1.0 / (head_dim_full ** 0.5)
                    full_scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale_full
                    full_probs = F.softmax(full_scores, dim=-1, dtype=torch.float32).to(q_full.dtype)

                    # 组织目录：logs/attn/{mmdd_HHMMSS}_{suite}/task_{idx}/
                    root_dir = getattr(self.config, "dump_attn_dir_root", "logs/attn")
                    os.makedirs(root_dir, exist_ok=True)
                    session_dir = getattr(self.config, "_dump_session_dir", None)
                    if session_dir is None:
                        ts = datetime.now().strftime("%m%d_%H%M%S")
                        suite = getattr(self.config, "dump_suite_name", None) or "unknown"
                        dir_name = f"{ts}_{suite}"
                        parts = [root_dir, dir_name]
                        task_idx = getattr(self.config, "dump_task_index", None)
                        if task_idx is not None:
                            parts.append(f"task_{int(task_idx)}")
                        session_dir = os.path.join(*parts)
                        os.makedirs(session_dir, exist_ok=True)
                        self.config._dump_session_dir = session_dir

                    dtype = torch.float16 if getattr(self.config, "dump_attn_dtype", "float16") == "float16" else torch.float32
                    what = getattr(self.config, "dump_attn_what", "probs")
                    full_dump = full_probs if what == "probs" else full_scores.to(full_probs.dtype)
                    full_np = full_dump.to(dtype).detach().cpu().numpy()
                    s_np = int(full_probs.shape[-1])
                    fname = os.path.join(session_dir, f"attn.preprune.layer{self.layer_idx}.S{s_np}.Skept{s_np}.{ 'fp16' if dtype==torch.float16 else 'fp32' }.npz")
                    np.savez_compressed(
                        fname,
                        attn=full_np,
                        layer_idx=int(self.layer_idx),
                        S=int(s_np),
                        S_kept=int(s_np),
                        num_heads=int(full_probs.shape[1]),
                    )
                except Exception as _e:
                    if getattr(self.config, "debug", False):
                        print(f"[PruneVLA][warn] failed saving pre-prune attn at layer={self.layer_idx}: {_e}")

            self._maybe_compute_kept_indices_once(hidden_states, attention_mask, position_ids)

        # 之后层：对 K/V 的 key 维做真裁剪（列裁剪），输出形状保持不变
        kept = self.config.kept_vision_key_indices
        if kept is None:
            # 不允许静默回退
            raise RuntimeError("[PruneVLA] KV-Pruning indices are missing before pruned layers.")
        # 防止跨 episode 序列长度变化导致的越界
        if kept.numel() == 0 or int(kept.max().item()) >= hidden_states.shape[1]:
            # 尝试强制按最近的触发层重算一次
            self._maybe_compute_kept_indices_once(hidden_states, attention_mask, position_ids, force_recompute=True)
            kept = self.config.kept_vision_key_indices
            if kept is None or kept.numel() == 0 or int(kept.max().item()) >= hidden_states.shape[1]:
                raise RuntimeError("[PruneVLA] KV-Pruning indices invalid after recompute.")

        bsz, seqlen, _ = hidden_states.shape

        # 1) 计算 Q/K/V
        q = self.original_attention.q_proj(hidden_states)
        k = self.original_attention.k_proj(hidden_states)
        v = self.original_attention.v_proj(hidden_states)

        num_heads = getattr(self.original_attention, "num_heads", None)
        if num_heads is None:
            # 尝试从权重推断（保底）
            num_heads = 32
        head_dim = q.shape[-1] // num_heads

        q = q.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)

        # RoPE（如可用）
        if self.has_rope:
            try:
                if position_ids is None:
                    position_ids = torch.arange(seqlen, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
                cos, sin = self.original_attention.rotary_emb(v, position_ids)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except Exception:
                pass

        # 在第 0 层，判定本次整模型前向是否允许落盘（基于 dump_forward_indices）
        if getattr(self.config, "dump_attn", False) and int(self.layer_idx) == 0:
            indices = getattr(self.config, "dump_forward_indices", None)
            if indices is None:
                self.config._dump_current_forward_allowed = True
            else:
                allow = (self.config._dump_forward_count in set(indices))
                self.config._dump_current_forward_allowed = allow
            # 增加计数供下一次整模型前向使用
            self.config._dump_forward_count += 1

        # 2) 对 key 维度做列裁剪（gather）
        # kept: [S_kept], 对应全序列 key 维的索引
        kept_exp = kept.view(1, 1, -1).to(device=k.device)
        k_pruned = k.index_select(dim=2, index=kept)
        v_pruned = v.index_select(dim=2, index=kept)

        # 3) 注意力计算（列维缩短）
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.matmul(q, k_pruned.transpose(-2, -1)) * scale  # [B, H, S, S_kept]

        # 4) attention_mask 在 key 维同步 slice（若提供）
        if attention_mask is not None:
            # 期望形状 [B, 1, S, S] 或 [B, S, S]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S, S]
            if attention_mask.shape[-1] == seqlen:
                # 只在 key 维做 slice 到 S_kept
                attention_mask = attention_mask.index_select(dim=-1, index=kept)
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v_pruned)  # [B, H, S, head_dim]

        # 可选：注意力落盘（剪裁后 [B,H,S,S_kept]）
        if getattr(self.config, "dump_attn", False) and getattr(self.config, "_dump_current_forward_allowed", True):
            # 选择层：dump_attn_layers 指定或按 scope（trigger_only/all_after）
            layers_sel = getattr(self.config, "dump_attn_layers", None)
            if layers_sel is not None:
                should_dump = int(self.layer_idx) in set(layers_sel)
            else:
                scope = getattr(self.config, "dump_attn_scope", "trigger_only")
                trigger_layers = {int(it.get("layer", -1)) for it in (getattr(self.config, "prune_schedule", []) or [])}
                should_dump = (self.layer_idx in trigger_layers) if scope == "trigger_only" else any(self.layer_idx >= tl for tl in trigger_layers)
            if should_dump:
                try:
                    root_dir = getattr(self.config, "dump_attn_dir_root", "logs/attn")
                    os.makedirs(root_dir, exist_ok=True)
                    session_dir = getattr(self.config, "_dump_session_dir", None)
                    if session_dir is None:
                        ts = datetime.now().strftime("%m%d_%H%M%S")
                        suite = getattr(self.config, "dump_suite_name", None) or "unknown"
                        dir_name = f"{ts}_{suite}"
                        session_dir = os.path.join(root_dir, dir_name)
                        task_idx = getattr(self.config, "dump_task_index", None)
                        if task_idx is not None:
                            session_dir = os.path.join(session_dir, f"task_{int(task_idx)}")
                        os.makedirs(session_dir, exist_ok=True)
                        self.config._dump_session_dir = session_dir

                    dtype = torch.float16 if getattr(self.config, "dump_attn_dtype", "float16") == "float16" else torch.float32
                    what = getattr(self.config, "dump_attn_what", "probs")
                    dump_tensor = attn_probs if what == "probs" else attn_scores.to(attn_probs.dtype)
                    dump_np = dump_tensor.to(dtype).detach().cpu().numpy()

                    kept_np = self.config.kept_vision_key_indices.detach().cpu().numpy()
                    bsz_np, num_heads_np, s_np, ske_np = int(attn_probs.shape[0]), int(attn_probs.shape[1]), int(attn_probs.shape[2]), int(attn_probs.shape[3])
                    fname = os.path.join(session_dir, f"attn.layer{self.layer_idx}.S{s_np}.Skept{ske_np}.{what}.{ 'fp16' if dtype==torch.float16 else 'fp32' }.npz")
                    np.savez_compressed(
                        fname,
                        attn=dump_np,
                        layer_idx=int(self.layer_idx),
                        S=int(s_np),
                        S_kept=int(ske_np),
                        num_heads=int(num_heads_np),
                        kept_indices_abs=kept_np,
                    )
                except Exception as _e:
                    if getattr(self.config, "debug", False):
                        print(f"[PruneVLA][warn] failed saving attn npz at layer={self.layer_idx}: {_e}")

        # 5) 还原形状并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, num_heads * head_dim)
        out = self.original_attention.o_proj(attn_output)

        # 去除任何输出回退/门控；保持输出形状与数值路径与未裁剪一致（仅注意力列侧裁剪）

        return out, None if not output_attentions else attn_probs, None


def _find_transformer_layers_root(model: nn.Module) -> Optional[List[nn.Module]]:
    """参照 sparse_openvla.py 的路径搜索，找到 LLaMA 层列表。"""
    paths = [
        "llm_backbone.llm.model.layers",  # Prismatic backbone
        "language_model.model.layers",    # HF standard path
        "llm.model.layers",              # Alternative path
    ]
    for path in paths:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def replace_attention_with_prune(model: nn.Module, prune_config: PruneVLAConfig) -> None:
    """在现有 VLA 模型上，就地替换注意力层为 KV-Pruning 版本。"""
    layers = _find_transformer_layers_root(model)
    if layers is None:
        raise RuntimeError("[PruneVLA] Could not find transformer layers for replacement")
    replaced_count = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            orig = layer.self_attn
            layer.self_attn = VLAKVPruningAttention(orig, prune_config, layer_idx=idx)
            replaced_count += 1
    if replaced_count == 0:
        raise RuntimeError("[PruneVLA] No attention layers were replaced; model structure unexpected")
    if prune_config.debug:
        print(f"[PruneVLA] Replaced attention on {replaced_count} layers")


