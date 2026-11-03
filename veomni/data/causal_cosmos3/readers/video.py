"""Timestamp-based PyAV decoding directly into a caller-owned uint8 CTHW buffer."""

import numpy as np


def decode_into(path, timestamps, output, *, tolerance_s=1e-4, threads=0):
    """Match nearest PTS (earlier frame on ties), resize selected frames only.

    The caller owns output, including a non-contiguous camera region in a larger
    canvas. No decoded frame list, decoder cache, or persistent file descriptor.
    Repeated timestamps (tail padding) reuse the last converted RGB frame.
    threads=0 lets the decoder choose its thread count automatically.
    """
    import av
    from av.video.reformatter import Interpolation

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError("Expected nonempty finite video timestamps")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Video timestamps must be nondecreasing")
    if output.dtype != np.uint8 or output.shape[:2] != (3, len(timestamps)) or min(output.shape[2:]) <= 0:
        raise ValueError("Expected uint8 output [3,T,H,W]")
    cursor, converted_frame, rgb = 0, None, None

    def write(frame, pts, index):
        nonlocal converted_frame, rgb
        if abs(pts - timestamps[index]) >= tolerance_s:
            raise ValueError(f"{path}: nearest PTS {pts} exceeds tolerance for {timestamps[index]}")
        if frame is not converted_frame:
            resized = frame.reformat(
                width=output.shape[3], height=output.shape[2], format="rgb24", interpolation=Interpolation.BICUBIC
            )
            rgb = resized.to_ndarray()
            converted_frame = frame
        output[:, index] = rgb.transpose(2, 0, 1)

    with av.open(str(path), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        container.seek(
            int(round(max(0, timestamps[0]) / stream.time_base)), backward=True, any_frame=False, stream=stream
        )
        previous, previous_pts = None, None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * frame.time_base)
            if previous_pts is not None and pts == previous_pts:
                continue
            while cursor < len(timestamps) and timestamps[cursor] <= pts:
                if previous is not None and abs(previous_pts - timestamps[cursor]) <= abs(pts - timestamps[cursor]):
                    write(previous, previous_pts, cursor)
                else:
                    write(frame, pts, cursor)
                cursor += 1
            previous, previous_pts = frame, pts
            if cursor == len(timestamps):
                break
        if previous is None:
            raise ValueError(f"{path}: no video frames decoded")
        while cursor < len(timestamps):
            write(previous, previous_pts, cursor)
            cursor += 1
