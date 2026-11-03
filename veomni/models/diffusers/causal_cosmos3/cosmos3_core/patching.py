# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Video token patching shared by input packing and loss reconstruction."""

from torch.nn import functional as F


def patchify_video(latents, patch_size):
    """[C,T,H,W] -> [T*ceil(H/p)*ceil(W/p),p*p*C], with channels last."""
    c, t, h, w = latents.shape
    p = patch_size
    padded = F.pad(latents, (0, -w % p, 0, -h % p))
    return (
        padded.reshape(c, t, (h + p - 1) // p, p, (w + p - 1) // p, p).permute(1, 2, 4, 3, 5, 0).reshape(-1, p * p * c)
    )


def unpatchify_video(tokens, shape, patch_size):
    """Invert patchify_video and crop spatial padding to the original [C,T,H,W]."""
    c, t, h, w = shape
    p = patch_size
    hp, wp = (h + p - 1) // p, (w + p - 1) // p
    return tokens.reshape(t, hp, wp, p, p, c).permute(5, 0, 1, 3, 2, 4).reshape(c, t, hp * p, wp * p)[:, :, :h, :w]
