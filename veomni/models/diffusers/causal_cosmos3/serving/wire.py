# SPDX-License-Identifier: OpenMDW-1.1
"""NumPy/msgpack framing shared with the RoboTwin client-server branch."""

import msgpack
import numpy as np


PROTOCOL = "cosmos-causal-action-rtc-v1"
HISTORY_TRANSPORT = "raw-history-refs-v1"
# A cold 32-block request can contain 257 prepared 640x640 RGB frames.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


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
