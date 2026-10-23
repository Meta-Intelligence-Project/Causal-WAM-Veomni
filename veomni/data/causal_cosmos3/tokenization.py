# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""对齐原 causal Cosmos 的文本编码和生成边界 token。"""

from ...models import build_tokenizer


def add_special_tokens(tokenizer):
    """沿用原实现的补词方式，保留已有词表和特殊 token 的定义。"""
    existing = []
    for value in tokenizer.special_tokens_map.values():
        if isinstance(value, str):
            existing.append(value)
        elif isinstance(value, list):
            existing.extend(value)
    missing = [token for token in ("<|vision_start|>", "<|vision_end|>") if token not in existing]
    if missing:
        # 原实现使用 add_tokens；不改变已有 token 的特殊词属性。
        tokenizer.add_tokens(missing)
    return tokenizer, {
        "start_of_generation": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_generation": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }


def build_prompt_tokenizer(path):
    """通过 VeOmni 加载 tokenizer，并补齐原 Cosmos 使用的视觉边界 token。"""
    tokenizer = build_tokenizer(str(path))
    tokenizer, _ = add_special_tokens(tokenizer)
    return tokenizer


def tokenize_prompt(prompt, *, tokenizer):
    """生成可直接参与 packing 的完整文本 IDs，包含 EOS 和视觉生成起点。"""
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=False,
    )
    # 原 augmentor 先截取 4096 个 chat token，packing 再追加两个边界 token。
    return tokens[:4096] + [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|vision_start|>")]
