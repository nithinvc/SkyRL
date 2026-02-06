import pytest
import torch
from skyrl_train.training_batch import TensorBatch
import pickle
import ray
import numpy as np


def test_train_batch_initialization():
    # Test basic initialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": response_mask,
        }
    )
    assert isinstance(data, TensorBatch)
    assert data.batch_size == batch_size
    assert torch.equal(data["sequences"], sequences)
    assert torch.equal(data["attention_mask"], attention_mask)


def test_train_batch_validation():
    # Test validation of batch sizes
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size - 1, seq_len)  # Different size

    with pytest.raises(ValueError, match="Batch size mismatch"):
        batch = TensorBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
            }
        )
        TensorBatch(batch=batch, metadata={})


def test_train_batch_chunk():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    assert chunks[0]["sequences"].shape == (2, seq_len)
    assert chunks[1]["sequences"].shape == (2, seq_len)


def test_train_batch_slice():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert sliced["sequences"].shape == (2, seq_len)
    assert sliced["attention_mask"].shape == (2, seq_len)


def test_train_batch_to_dtype():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = None
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    assert data["attention_mask"] is None


def test_train_batch_select():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    metadata = {"info": "test", "extra": "data"}

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
    )
    data.metadata = metadata

    selected = data.select(["sequences", "attention_mask"], ["info"])
    assert "sequences" in selected
    assert "attention_mask" in selected
    assert "loss_mask" not in selected
    assert "info" in selected.metadata
    assert "extra" not in selected.metadata


def test_train_batch_cat():
    batch_size = 3
    seq_len = 4
    sequences1 = torch.randn(batch_size, seq_len)
    attention_mask1 = torch.ones(batch_size, seq_len)
    data1 = TensorBatch(
        {
            "sequences": sequences1,
            "attention_mask": attention_mask1,
        }
    )
    sequences2 = torch.randn(batch_size, seq_len)
    attention_mask2 = torch.ones(batch_size, seq_len)
    data2 = TensorBatch(
        {
            "sequences": sequences2,
            "attention_mask": attention_mask2,
        }
    )

    concatenated = data1.cat([data1, data2])
    assert len(concatenated) == 2 * batch_size
    assert concatenated["sequences"].shape == (2 * batch_size, seq_len)
    assert concatenated["attention_mask"].shape == (2 * batch_size, seq_len)


def test_train_batch_pickle():
    # Test pickle serialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )
    metadata = {"info": "test"}
    data.metadata = metadata

    # Serialize
    pickled = pickle.dumps(data)

    # Deserialize
    unpickled = pickle.loads(pickled)

    # Verify all components are preserved
    assert len(unpickled) == len(data)
    assert all(torch.equal(unpickled[k], data[k]) for k in data.keys())
    assert unpickled.metadata == data.metadata


def test_train_batch_setitem():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    # Test setting tensor
    new_sequences = torch.randn(batch_size, seq_len)
    data["sequences"] = new_sequences
    assert torch.equal(data["sequences"], new_sequences)

    # Test invalid tensor shape
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["sequences"] = torch.randn(batch_size + 1, seq_len)

    # Test invalid types
    # 1. string
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = "invalid"
    # 2. numpy array
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = np.zeros((batch_size, seq_len))


def test_train_batch_ray_serialization():
    data = TensorBatch(
        **{"a": torch.tensor([1.2, 2.4, 3.6, 4.8]), "b": torch.tensor([4, 5, 6, 7])},
    )
    data.metadata = {"hello": "world"}

    def _task(inp: TensorBatch):
        assert inp == data

    _inp_ray = ray.put(data)
    ray.remote(_task).remote(_inp_ray)


def test_train_batch_repeat():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"d": 1, "e": "test"}
    repeated = data.repeat(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 2, 3, 1, 2, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 5, 6, 4, 5, 6]))
    assert repeated.metadata == {"d": 1, "e": "test"}


def test_train_batch_repeat_interleave():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    repeated = data.repeat_interleave(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 1, 2, 2, 3, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 4, 5, 5, 6, 6]))
    assert repeated.metadata == {"c": "test"}


def test_train_batch_get_item():
    batch = {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([4, 5, 6, 7])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    new_data = data[:2]
    assert torch.equal(new_data["a"], torch.tensor([1, 2]))
    assert torch.equal(new_data["b"], torch.tensor([4, 5]))


# ---------------------------------------------------------------------------
# Helper: create a TensorBatch that mimics multi_modal_inputs
# ---------------------------------------------------------------------------
def _make_mm_inputs(batch_size):
    """Return a nested TensorBatch with list-of-tensor values."""
    pixel_values = [torch.randn(3, 4) for _ in range(batch_size)]
    image_grid_thw = [torch.tensor([1, 2, 3]) for _ in range(batch_size)]
    return TensorBatch({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})


# ---------------------------------------------------------------------------
# Tests for list values
# ---------------------------------------------------------------------------
def test_list_value_initialization():
    batch_size = 3
    pixel_values = [torch.randn(3, 4) for _ in range(batch_size)]
    data = TensorBatch({"pixel_values": pixel_values})
    assert data.batch_size == batch_size
    assert len(data["pixel_values"]) == batch_size
    # device is not set for list-only batches
    assert data.device is None


def test_list_value_consistency_mismatch():
    with pytest.raises(ValueError, match="Batch size mismatch"):
        TensorBatch({
            "a": [torch.randn(2) for _ in range(3)],
            "b": [torch.randn(2) for _ in range(4)],  # wrong length
        })


def test_list_value_setitem():
    batch_size = 3
    data = TensorBatch({"a": [torch.randn(2) for _ in range(batch_size)]})
    new_list = [torch.randn(2) for _ in range(batch_size)]
    data["a"] = new_list
    assert data["a"] is new_list

    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["a"] = [torch.randn(2) for _ in range(batch_size + 1)]


def test_list_value_operations():
    """Combined test for list-only batch: slice, chunk, repeat, repeat_interleave, cat."""
    batch_size = 4
    items = [torch.tensor([float(i)]) for i in range(batch_size)]
    data = TensorBatch({"x": items})

    # slice
    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert torch.equal(sliced["x"][0], items[1])
    assert torch.equal(sliced["x"][1], items[2])

    # chunk
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert torch.equal(chunks[0]["x"][0], items[0])
    assert torch.equal(chunks[1]["x"][0], items[2])

    # repeat (tile semantics: [a,b,c,d]*2 = [a,b,c,d,a,b,c,d])
    repeated = data.repeat(2)
    assert len(repeated) == 8
    assert torch.equal(repeated["x"][0], items[0])
    assert torch.equal(repeated["x"][4], items[0])

    # repeat_interleave ([a,b]*2 -> [a,a,b,b])
    ri = data.repeat_interleave(2)
    assert len(ri) == 8
    assert torch.equal(ri["x"][0], items[0])
    assert torch.equal(ri["x"][1], items[0])
    assert torch.equal(ri["x"][2], items[1])

    # cat
    data2 = TensorBatch({"x": [torch.tensor([float(i + 10)]) for i in range(batch_size)]})
    cat_result = TensorBatch.cat([data, data2])
    assert len(cat_result) == 2 * batch_size
    assert torch.equal(cat_result["x"][0], items[0])
    assert torch.equal(cat_result["x"][batch_size], data2["x"][0])


# ---------------------------------------------------------------------------
# Tests for nested TensorBatch
# ---------------------------------------------------------------------------
def test_nested_tensorbatch_initialization():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})
    assert data.batch_size == batch_size
    assert isinstance(dict.__getitem__(data, "multi_modal_inputs"), TensorBatch)
    assert len(dict.__getitem__(data, "multi_modal_inputs")) == batch_size


def test_nested_tensorbatch_consistency_mismatch():
    mm = _make_mm_inputs(5)  # batch_size=5
    with pytest.raises(ValueError, match="Batch size mismatch"):
        TensorBatch({"sequences": torch.randn(3, 4), "multi_modal_inputs": mm})


def test_nested_tensorbatch_setitem():
    batch_size = 3
    data = TensorBatch({"sequences": torch.randn(batch_size, 4)})
    mm = _make_mm_inputs(batch_size)
    data["multi_modal_inputs"] = mm
    assert isinstance(dict.__getitem__(data, "multi_modal_inputs"), TensorBatch)

    mm_wrong = _make_mm_inputs(batch_size + 1)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["multi_modal_inputs"] = mm_wrong


def test_nested_tensorbatch_slice():
    batch_size = 4
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert sliced["sequences"].shape == (2, 4)
    nested = dict.__getitem__(sliced, "multi_modal_inputs")
    assert isinstance(nested, TensorBatch)
    assert len(nested) == 2
    assert len(nested["pixel_values"]) == 2
    assert torch.equal(nested["pixel_values"][0], mm["pixel_values"][1])


def test_nested_tensorbatch_chunk():
    batch_size = 4
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})

    chunks = data.chunk(2)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk) == 2
        nested = dict.__getitem__(chunk, "multi_modal_inputs")
        assert isinstance(nested, TensorBatch)
        assert len(nested) == 2
        assert len(nested["pixel_values"]) == 2

    # Verify the split content
    assert torch.equal(
        dict.__getitem__(chunks[0], "multi_modal_inputs")["pixel_values"][0],
        mm["pixel_values"][0],
    )
    assert torch.equal(
        dict.__getitem__(chunks[1], "multi_modal_inputs")["pixel_values"][0],
        mm["pixel_values"][2],
    )


def test_nested_tensorbatch_repeat():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.tensor([1, 2, 3]), "multi_modal_inputs": mm})

    repeated = data.repeat(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["sequences"], torch.tensor([1, 2, 3, 1, 2, 3]))
    nested = dict.__getitem__(repeated, "multi_modal_inputs")
    assert len(nested) == 6
    # tile semantics: [a,b,c,a,b,c]
    assert torch.equal(nested["pixel_values"][0], mm["pixel_values"][0])
    assert torch.equal(nested["pixel_values"][3], mm["pixel_values"][0])


def test_nested_tensorbatch_repeat_interleave():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.tensor([1, 2, 3]), "multi_modal_inputs": mm})

    ri = data.repeat_interleave(2)
    assert len(ri) == 6
    assert torch.equal(ri["sequences"], torch.tensor([1, 1, 2, 2, 3, 3]))
    nested = dict.__getitem__(ri, "multi_modal_inputs")
    assert len(nested) == 6
    # interleave semantics: [a,a,b,b,c,c]
    assert torch.equal(nested["pixel_values"][0], mm["pixel_values"][0])
    assert torch.equal(nested["pixel_values"][1], mm["pixel_values"][0])
    assert torch.equal(nested["pixel_values"][2], mm["pixel_values"][1])


def test_nested_tensorbatch_cat():
    batch_size = 3
    mm1 = _make_mm_inputs(batch_size)
    mm2 = _make_mm_inputs(batch_size)
    data1 = TensorBatch({"sequences": torch.tensor([1, 2, 3]), "multi_modal_inputs": mm1})
    data2 = TensorBatch({"sequences": torch.tensor([4, 5, 6]), "multi_modal_inputs": mm2})

    cat_result = TensorBatch.cat([data1, data2])
    assert len(cat_result) == 2 * batch_size
    assert torch.equal(cat_result["sequences"], torch.tensor([1, 2, 3, 4, 5, 6]))
    nested = dict.__getitem__(cat_result, "multi_modal_inputs")
    assert len(nested) == 2 * batch_size
    assert torch.equal(nested["pixel_values"][0], mm1["pixel_values"][0])
    assert torch.equal(nested["pixel_values"][batch_size], mm2["pixel_values"][0])


def test_nested_tensorbatch_to_dtype():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})

    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    nested = dict.__getitem__(data, "multi_modal_inputs")
    for t in nested["pixel_values"]:
        assert t.dtype == torch.float16
    for t in nested["image_grid_thw"]:
        assert t.dtype == torch.float16


def test_nested_tensorbatch_contiguous():
    batch_size = 4
    # Create non-contiguous tensors via transposition
    pixel_values = [torch.randn(4, 3).T for _ in range(batch_size)]  # (3,4) non-contiguous
    mm = TensorBatch({"pixel_values": pixel_values})
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})
    assert not dict.__getitem__(data, "multi_modal_inputs")["pixel_values"][0].is_contiguous()

    data.contiguous()
    for t in dict.__getitem__(data, "multi_modal_inputs")["pixel_values"]:
        assert t.is_contiguous()


def test_nested_tensorbatch_select():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({
        "sequences": torch.randn(batch_size, 4),
        "attention_mask": torch.ones(batch_size, 4),
        "multi_modal_inputs": mm,
    })
    data.metadata = {"info": "test", "extra": "data"}

    selected = data.select(["sequences", "multi_modal_inputs"], ["info"])
    assert "sequences" in selected
    assert "multi_modal_inputs" in selected
    assert "attention_mask" not in selected
    assert "info" in selected.metadata
    assert "extra" not in selected.metadata
    nested = dict.__getitem__(selected, "multi_modal_inputs")
    assert isinstance(nested, TensorBatch)
    assert len(nested) == batch_size


def test_nested_tensorbatch_pickle():
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.randn(batch_size, 4), "multi_modal_inputs": mm})
    data.metadata = {"info": "test"}

    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == len(data)
    assert torch.equal(unpickled["sequences"], data["sequences"])
    nested_orig = dict.__getitem__(data, "multi_modal_inputs")
    nested_loaded = dict.__getitem__(unpickled, "multi_modal_inputs")
    assert isinstance(nested_loaded, TensorBatch)
    assert len(nested_loaded) == len(nested_orig)
    for a, b in zip(nested_loaded["pixel_values"], nested_orig["pixel_values"]):
        assert torch.equal(a, b)
    for a, b in zip(nested_loaded["image_grid_thw"], nested_orig["image_grid_thw"]):
        assert torch.equal(a, b)
    assert unpickled.metadata == data.metadata


def test_nested_tensorbatch_eq():
    batch_size = 3
    pixel_values = [torch.randn(3, 4) for _ in range(batch_size)]
    grid_thw = [torch.tensor([1, 2, 3]) for _ in range(batch_size)]

    mm1 = TensorBatch({"pixel_values": list(pixel_values), "image_grid_thw": list(grid_thw)})
    mm2 = TensorBatch({"pixel_values": list(pixel_values), "image_grid_thw": list(grid_thw)})
    data1 = TensorBatch({"multi_modal_inputs": mm1})
    data2 = TensorBatch({"multi_modal_inputs": mm2})
    assert data1 == data2

    # Different content
    mm3 = _make_mm_inputs(batch_size)
    data3 = TensorBatch({"multi_modal_inputs": mm3})
    assert data1 != data3

    # Different batch size
    mm4 = _make_mm_inputs(batch_size + 1)
    data4 = TensorBatch({"multi_modal_inputs": mm4})
    assert data1 != data4


def test_nested_tensorbatch_get_item():
    """Test integer and slice indexing with nested TensorBatch."""
    batch_size = 4
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({"sequences": torch.tensor([10, 20, 30, 40]), "multi_modal_inputs": mm})

    # Slice indexing
    sliced = data[:2]
    assert len(sliced) == 2
    assert torch.equal(sliced["sequences"], torch.tensor([10, 20]))
    nested = dict.__getitem__(sliced, "multi_modal_inputs")
    assert len(nested) == 2

    # Integer indexing
    single = data[1]
    assert len(single) == 1
    assert torch.equal(single["sequences"], torch.tensor([20]))
    nested_single = dict.__getitem__(single, "multi_modal_inputs")
    assert len(nested_single) == 1
    assert torch.equal(nested_single["pixel_values"][0], mm["pixel_values"][1])


def test_mixed_tensor_and_nested():
    """TensorBatch with both tensor and nested TensorBatch values -- test all ops."""
    batch_size = 4
    mm = _make_mm_inputs(batch_size)
    seq = torch.randn(batch_size, 5)
    mask = torch.ones(batch_size, 5)
    data = TensorBatch({"sequences": seq, "attention_mask": mask, "multi_modal_inputs": mm})
    data.metadata = {"response_length": 10}

    # slice
    s = data.slice(0, 2)
    assert len(s) == 2
    assert s["sequences"].shape == (2, 5)
    assert len(dict.__getitem__(s, "multi_modal_inputs")) == 2

    # chunk
    chunks = data.chunk(2)
    assert len(chunks) == 2
    for c in chunks:
        assert len(c) == 2
        assert isinstance(dict.__getitem__(c, "multi_modal_inputs"), TensorBatch)

    # cat
    cat_result = TensorBatch.cat(chunks)
    assert len(cat_result) == batch_size
    assert torch.equal(cat_result["sequences"], seq)
    nested_cat = dict.__getitem__(cat_result, "multi_modal_inputs")
    for i in range(batch_size):
        assert torch.equal(nested_cat["pixel_values"][i], mm["pixel_values"][i])

    # repeat / repeat_interleave with 1D tensors + nested (repeat only works with 1D tensors)
    mm_1d = _make_mm_inputs(batch_size)
    data_1d = TensorBatch({"ids": torch.tensor([1, 2, 3, 4]), "multi_modal_inputs": mm_1d})
    rep = data_1d.repeat(2)
    assert len(rep) == 2 * batch_size
    assert torch.equal(rep["ids"], torch.tensor([1, 2, 3, 4, 1, 2, 3, 4]))

    ri = data_1d.repeat_interleave(2)
    assert len(ri) == 2 * batch_size
    nested_ri = dict.__getitem__(ri, "multi_modal_inputs")
    assert torch.equal(nested_ri["pixel_values"][0], mm_1d["pixel_values"][0])
    assert torch.equal(nested_ri["pixel_values"][1], mm_1d["pixel_values"][0])

    # select
    sel = data.select(["sequences", "multi_modal_inputs"])
    assert "sequences" in sel
    assert "multi_modal_inputs" in sel
    assert "attention_mask" not in sel

    # to dtype
    data_copy = TensorBatch({"sequences": seq.clone(), "multi_modal_inputs": _make_mm_inputs(batch_size)})
    data_copy.to(dtype=torch.float16)
    assert data_copy["sequences"].dtype == torch.float16
    for t in dict.__getitem__(data_copy, "multi_modal_inputs")["pixel_values"]:
        assert t.dtype == torch.float16

    # pickle roundtrip
    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)
    assert len(unpickled) == len(data)
    assert torch.equal(unpickled["sequences"], data["sequences"])


def test_nested_with_none_values():
    """TensorBatch with both nested TensorBatch and None values."""
    batch_size = 3
    mm = _make_mm_inputs(batch_size)
    data = TensorBatch({
        "sequences": torch.randn(batch_size, 4),
        "multi_modal_inputs": mm,
        "optional_field": None,
    })
    assert data.batch_size == batch_size

    # slice should propagate None
    sliced = data.slice(0, 2)
    assert sliced["optional_field"] is None
    assert len(dict.__getitem__(sliced, "multi_modal_inputs")) == 2

    # chunk
    chunks = data.chunk(1)
    assert len(chunks) == 3
    assert chunks[0]["optional_field"] is None

    # cat
    cat_result = TensorBatch.cat(chunks)
    assert cat_result["optional_field"] is None
    assert len(cat_result) == batch_size
