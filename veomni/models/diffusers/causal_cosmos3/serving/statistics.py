# SPDX-License-Identifier: OpenMDW-1.1
"""Measured quantiles for online native action encoding and robot adapters."""

import json
from dataclasses import dataclass, field

import torch


@dataclass
class Quantiles:
    low: torch.Tensor
    high: torch.Tensor
    valid: torch.Tensor
    # 可选实测统计；与训练读取的 low/high 分开保存，旧文件仍可加载。
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            self.low.ndim != 1
            or not self.low.numel()
            or any(x.shape != self.low.shape for x in (self.high, self.valid))
        ):
            raise ValueError("Quantiles and validity must have the same nonempty [D] shape")
        if not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Quantile bounds must be finite and ordered")
        if self.valid.dtype != torch.bool:
            raise ValueError("Statistics validity must be boolean")

    @classmethod
    def fit(cls, values, masks):
        values, masks = torch.cat(values).float(), torch.cat(masks).bool()
        lo, hi = values.new_zeros(values.shape[-1]), values.new_zeros(values.shape[-1])
        valid = masks.any(0)
        for d in torch.where(valid)[0].tolist():
            lo[d], hi[d] = torch.quantile(values[masks[:, d], d], values.new_tensor([0.01, 0.99]))
        return cls(lo, hi, valid)

    def normalize(self, values, mask):
        lo, hi = self.low.to(values), self.high.to(values)
        span = hi - lo
        scaled = 2 * (values - lo) / span.clamp_min(1e-12) - 1
        # low/high 仍映射到 -1/+1；额外保留尾部幅度，超过 ±1.5 才计为裁剪。
        clip_bound = 1.5
        clipped = mask & (span > 0) & (scaled.abs() > clip_bound)
        out = scaled.clamp(-clip_bound, clip_bound).masked_fill(~mask | (span == 0), 0)
        return out, clipped.sum(0) / mask.sum(0).clamp_min(1)

    def denormalize(self, values, mask):
        lo, hi = self.low.to(values), self.high.to(values)
        return ((values + 1) * 0.5 * (hi - lo) + lo).masked_fill(~mask, 0)

    def as_dict(self):
        return dict(self.metrics, **{k: getattr(self, k).tolist() for k in ("low", "high", "valid")})


@dataclass
class BlockStatistics:
    state: Quantiles
    action: Quantiles
    provenance: dict = field(default_factory=dict)

    def validate(self, width, state_mask, action_mask, *, allow_mock=False):
        """训练适配层首次加载时检查维度、mask 覆盖和模拟统计开关。"""
        if self.state.low.numel() != width or self.action.low.numel() != width:
            raise ValueError("Statistics width differs from the action template")
        for mask, quantiles in ((state_mask, self.state), (action_mask, self.action)):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=quantiles.valid.device)
            if (mask & ~quantiles.valid).any():
                raise ValueError("Statistics do not cover valid source channels")
        if self.provenance.get("kind") == "mock" and not allow_mock:
            raise ValueError("Mock statistics require allow_mock_statistics=True")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)

        def read(name):
            q = data[name]
            return Quantiles(
                torch.tensor(q["low"]),
                torch.tensor(q["high"]),
                torch.tensor(q["valid"]).bool(),
                metrics={k: v for k, v in q.items() if k not in ("low", "high", "valid")},
            )

        return cls(read("state"), read("action"), data.get("provenance", {}))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                dict(state=self.state.as_dict(), action=self.action.as_dict(), provenance=self.provenance),
                f,
                indent=2,
            )
