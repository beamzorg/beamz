import numpy as np
import pytest

from beamz.optimization.adjoint_memmap import (
    ForwardFieldChunkWriter,
    MemmapConfig,
    ReverseFieldChunkReader,
    compute_overlap_gradient_memmap,
)


def test_memmap_chunk_roundtrip_and_overlap_gradient(tmp_path):
    cfg = MemmapConfig(directory=str(tmp_path), chunk_size=3, dtype="float32")
    writer = ForwardFieldChunkWriter(
        config=cfg,
        field_names=("Ez",),
        field_shape=(2, 2),
    )

    n_steps = 10
    forward = []
    for i in range(n_steps):
        arr = np.full((2, 2), float(i + 1), dtype=np.float32)
        forward.append(arr)
        writer.append({"Ez": arr})
    writer.finalize()

    reader = ReverseFieldChunkReader(str(tmp_path), delete_consumed=False)
    assert reader.total_steps == n_steps

    adjoint = [np.full((2, 2), 2.0, dtype=np.float32) for _ in range(n_steps)]
    grad = compute_overlap_gradient_memmap(reader, adjoint, field_key="Ez")

    expected = np.zeros((2, 2), dtype=np.float32)
    for t in range(n_steps):
        expected += forward[t] * adjoint[n_steps - 1 - t]

    assert np.allclose(grad, expected)


def test_memmap_reverse_reader_delete_consumed(tmp_path):
    cfg = MemmapConfig(directory=str(tmp_path), chunk_size=2, dtype="float32")
    writer = ForwardFieldChunkWriter(
        config=cfg,
        field_names=("Ez",),
        field_shape=(1, 1),
    )
    for i in range(5):
        writer.append({"Ez": np.array([[float(i)]], dtype=np.float32)})
    writer.finalize()

    reader = ReverseFieldChunkReader(str(tmp_path), delete_consumed=True)
    # Exhaust iterator
    for _chunk_id, _chunk in reader.iter_chunks_reverse():
        pass

    # All chunk files should be removed, metadata stays.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("chunk_")]
    assert leftovers == []
    assert (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("n_steps", [0, 1, 2, 3, 5])
def test_memmap_flush_and_finalize_preserve_chunk_metadata(tmp_path, n_steps):
    writer = ForwardFieldChunkWriter(
        MemmapConfig(directory=str(tmp_path), chunk_size=2), ("Ez", "Hy"), (1,)
    )
    for step in range(n_steps):
        writer.append({"Ez": np.array([step]), "Hy": np.array([-step])})
        # Explicit flushes create short chunks; empty flushes must add no chunks.
        writer.flush()
        writer.flush()
    writer.finalize()
    writer.finalize()

    reader = ReverseFieldChunkReader(str(tmp_path))
    assert reader.total_steps == n_steps
    assert reader.chunk_lengths == [1] * n_steps
    chunks = list(reader.iter_chunks_reverse())
    assert [index for index, _ in chunks] == list(reversed(range(n_steps)))
    for index, chunk in chunks:
        np.testing.assert_array_equal(chunk["Ez"], [[index]])
        np.testing.assert_array_equal(chunk["Hy"], [[-index]])
