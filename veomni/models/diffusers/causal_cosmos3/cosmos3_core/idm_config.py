# SPDX-License-Identifier: OpenMDW-1.1
"""Canonical IDM video mode and compatibility with the original field name."""


def resolve_idm_video_mode(idm_video_mode=None, idm_denoising_mode=None):
    if idm_video_mode is not None and idm_denoising_mode is not None and idm_video_mode != idm_denoising_mode:
        raise ValueError("idm_video_mode and legacy idm_denoising_mode must match")
    mode = idm_video_mode if idm_video_mode is not None else idm_denoising_mode
    if mode is None:
        mode = "partial_denoised"
    if mode not in ("partial_denoised", "full_denoised"):
        raise ValueError("idm_video_mode must be partial_denoised or full_denoised")
    return mode


def normalize_idm_video_mode(options):
    """Consume the legacy key while preserving whether a mode was specified."""
    if "idm_denoising_mode" in options:
        options["idm_video_mode"] = resolve_idm_video_mode(
            options.get("idm_video_mode"), options.pop("idm_denoising_mode")
        )
    return options
