# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LingBot/OpenPI-compatible msgpack NumPy wire format; no Torch dependency."""

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        if isinstance(value, np.generic):
            return {b"__npgeneric__": True, b"dtype": value.dtype.str, b"data": value.item()}
        return {b"__ndarray__": True, b"dtype": value.dtype.str, b"shape": value.shape, b"data": value.tobytes()}
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def packb(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_decode, raw=False)
