"""Local reader semantics and resource bounds, independent of accelerator kernels."""

import json
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from veomni.data.causal_cosmos3 import ActionColumns, Geometry, plan_window, read_metadata, read_rows, read_window
from veomni.data.causal_cosmos3.readers.video import decode_into


def plan(root, episode, *args, **kwargs):
    info, episodes = read_metadata(root)
    i = int(np.flatnonzero(episodes["episode_index"] == episode)[0])
    ep = {k: v[i].item() for k, v in episodes.items()}
    return plan_window(root, episode, *args, info=info, ep=ep, **kwargs)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    info = dict(
        codebase_version="v3.0",
        fps=30,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        features={
            key: dict(dtype="float32", shape=[2])
            for key in ["state_unified", "action_unified", "mask_state", "mask_action"]
        },
    )
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    ep = pa.table(
        dict(
            episode_index=[42, 99],
            dataset_from_index=[100, 109],
            dataset_to_index=[109, 118],
            **{"data/chunk_index": [0, 0], "data/file_index": [0, 0]},
        )
    )
    pq.write_table(ep, tmp_path / "meta/episodes/chunk-000/file-000.parquet", row_group_size=1)
    state = np.arange(36, dtype=np.float32).reshape(18, 2)

    def vec(x):
        return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), 2)

    table = pa.table(
        dict(
            state_unified=vec(state),
            action_unified=vec(state + 1000),
            mask_state=vec(np.ones_like(state)),
            mask_action=vec(np.tile([1.0, 0.0], (18, 1)).astype(np.float32)),
            timestamp=np.tile(np.arange(9) / 30, 2),
            task_index=np.repeat([0, 1], 9),
        )
    )
    pq.write_table(table, tmp_path / "data/chunk-000/file-000.parquet", row_group_size=3)
    return tmp_path


@pytest.mark.parametrize("offset,start,real", [(0, 2, 6), (1, 2, 6), (-2, 2, 6), (2, 2, 5)])
def test_shift_padding_and_noncontiguous_episode_ids(root, offset, start, real):
    window = plan(root, 99, start, 8, block_size=4, video_stride=1, columns=ActionColumns(target_offset=offset))
    result = read_window(window, video=False)
    assert window.real_actions == real
    expected = torch.arange(36, dtype=torch.float32).reshape(18, 2)
    assert torch.equal(result["state"][: real + 1], expected[9 + start : 9 + start + real + 1])
    assert torch.equal(result["action"][:real], expected[9 + start + offset : 9 + start + offset + real] + 1000)
    assert torch.equal(result["action"][real:], result["action"][real - 1 : real].expand(window.actions - real, 2))
    assert result["action_mask"].tolist() == [True, False]
    assert result["timestamps"].dtype == torch.float64


def test_state_target_offset_once_and_independent_outputs(root):
    window = plan(
        root, 42, 0, 4, block_size=4, video_stride=1, columns=ActionColumns(target="state_unified", target_offset=1)
    )
    first = read_window(window, video=False)
    assert torch.equal(first["action"], first["state"][1:])
    first["state"].fill_(-999)
    second = read_window(window, video=False)
    assert second["state"][0, 0] == 0
    assert first["action"][0, 0] == 2


def test_row_group_pruning_crossing_and_repeated_reads(root):
    path = root / "data/chunk-000/file-000.parquet"
    expected = np.arange(36, dtype=np.float32).reshape(18, 2)
    for start, stop in [(1, 3), (2, 5), (8, 17), (0, 18)]:
        out = read_rows(path, start, stop, ["state_unified"])["state_unified"]
        np.testing.assert_array_equal(out, expected[start:stop])
        assert out.flags.owndata


def test_array_list_column_slice(root):
    p = root / "variable.parquet"
    pq.write_table(pa.table(dict(x=pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))), p, row_group_size=2)
    result = read_rows(p, 1, 3, ["x"])["x"]
    np.testing.assert_array_equal(result, [[3.0, 4.0], [5.0, 6.0]])


def test_video_nearest_pts_padding_and_canvas(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "sample.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for i in range(12):
            pixels = np.full((24, 32, 3), i * 15, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    queries = np.array([0, 2 / 30, 7 / 30, 11 / 30, 11 / 30], dtype=np.float64)
    canvas = np.zeros((3, len(queries), 30, 40), dtype=np.uint8)
    region = canvas[:, :, 3:15, 4:20]
    decode_into(path, queries, region)
    from av.video.reformatter import Interpolation

    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        pts = np.array([float(f.pts * f.time_base) for f in frames])
        expected = np.stack(
            [
                frames[int(np.argmin(abs(pts - q)))]
                .reformat(width=16, height=12, format="rgb24", interpolation=Interpolation.BICUBIC)
                .to_ndarray()
                for q in queries
            ]
        ).transpose(3, 0, 1, 2)
    np.testing.assert_array_equal(region, expected)
    assert not canvas[:, :, :3].any()
    np.testing.assert_array_equal(region[:, -1], region[:, -2])
    with pytest.raises(ValueError, match="tolerance"):
        decode_into(path, [1.0], np.empty((3, 1, 12, 16), dtype=np.uint8))


def test_token_budget_uses_padded_lengths(root):
    w = plan(root, 42, 0, 32)
    w = replace(w, height=720, width=640)
    assert w.actions == 32 and w.real_actions == 8
    assert sum(Geometry().token_layout(w.actions, w.height, w.width, 10).values()) == 10 + 2 * 3 * 23 * 20 + 32 + 1


def test_explicit_file_offsets_and_window_limit(root):
    meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(meta_path).append_column("data/file_from_offset", pa.array([0, 9]))
    pq.write_table(table, meta_path, row_group_size=1)
    window = plan(root, 99, 2, 4, block_size=4, video_stride=1)
    assert window.row_start == 11
    assert read_window(window, video=False)["state"][0, 0] == 22


def test_segment_preserves_unaligned_length_when_padding_disabled(root):
    window = plan(root, 42, 1, 5, pad_to_block=False)
    sample = read_window(window, video=False)
    assert window.actions == window.real_actions == 5
    assert sample["state"].shape == (6, 2)
    assert sample["action"].shape == (5, 2)
    assert window.video_frames == 2


def test_optional_null_metadata_falls_back(root):
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.append_column("source_fps", pa.array([25.0, None]))
    table = table.append_column("data/file_from_offset", pa.array([0, None], type=pa.int64()))
    pq.write_table(table, path)
    _, episodes = read_metadata(root)
    assert episodes["source_fps"][0] == 25 and np.isnan(episodes["source_fps"][1])
    assert episodes["data/file_from_offset"].tolist() == [0, 9]
