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


# ==================== Tests for Nested TensorBatch and List Support ====================


def test_list_initialization():
    """Test TensorBatch with list values"""
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    # List of tensors (like pixel_values for multimodal)
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]

    data = TensorBatch(
        {
            "sequences": sequences,
            "pixel_values": pixel_values,
        }
    )
    assert isinstance(data, TensorBatch)
    assert data.batch_size == batch_size
    assert len(data["pixel_values"]) == batch_size
    assert torch.equal(data["sequences"], sequences)


def test_list_batch_size_mismatch():
    """Test that list length must match batch_size"""
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size + 1)]  # Wrong size

    with pytest.raises(ValueError, match="Batch size mismatch"):
        TensorBatch(
            {
                "sequences": sequences,
                "pixel_values": pixel_values,
            }
        )


def test_nested_tensorbatch_initialization():
    """Test TensorBatch with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)

    # Create nested TensorBatch (like multi_modal_inputs)
    nested_batch = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
            "image_grid_thw": [torch.tensor([1, 224, 224]) for _ in range(batch_size)],
        }
    )

    data = TensorBatch(
        {
            "sequences": sequences,
            "multi_modal_inputs": nested_batch,
        }
    )
    assert isinstance(data, TensorBatch)
    assert data.batch_size == batch_size
    assert isinstance(data["multi_modal_inputs"], TensorBatch)
    assert len(data["multi_modal_inputs"]["pixel_values"]) == batch_size


def test_nested_tensorbatch_batch_size_mismatch():
    """Test that nested TensorBatch batch_size must match"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)

    nested_batch = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size + 1)],  # Wrong size
        }
    )

    with pytest.raises(ValueError, match="Batch size mismatch"):
        TensorBatch(
            {
                "sequences": sequences,
                "multi_modal_inputs": nested_batch,
            }
        )


def test_list_slice():
    """Test slicing with list values"""
    batch_size = 4
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert len(sliced["pixel_values"]) == 2
    assert sliced["sequences"].shape[0] == 2
    # Verify correct items were sliced
    assert torch.equal(sliced["pixel_values"][0], pixel_values[1])
    assert torch.equal(sliced["pixel_values"][1], pixel_values[2])


def test_nested_tensorbatch_slice():
    """Test slicing with nested TensorBatch"""
    batch_size = 4
    sequences = torch.randn(batch_size, 10)
    nested = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
            "grid": [torch.tensor([i]) for i in range(batch_size)],
        }
    )

    data = TensorBatch({"sequences": sequences, "nested": nested})

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert len(sliced["nested"]["pixel_values"]) == 2
    assert len(sliced["nested"]["grid"]) == 2


def test_list_chunk():
    """Test chunking with list values"""
    batch_size = 4
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})

    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0]["pixel_values"]) == 2
    assert len(chunks[1]["pixel_values"]) == 2
    # Verify correct items in each chunk
    assert torch.equal(chunks[0]["pixel_values"][0], pixel_values[0])
    assert torch.equal(chunks[0]["pixel_values"][1], pixel_values[1])
    assert torch.equal(chunks[1]["pixel_values"][0], pixel_values[2])
    assert torch.equal(chunks[1]["pixel_values"][1], pixel_values[3])


def test_nested_tensorbatch_chunk():
    """Test chunking with nested TensorBatch"""
    batch_size = 4
    sequences = torch.randn(batch_size, 10)
    nested = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
        }
    )

    data = TensorBatch({"sequences": sequences, "nested": nested})

    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0]["nested"]["pixel_values"]) == 2
    assert len(chunks[1]["nested"]["pixel_values"]) == 2


def test_list_cat():
    """Test concatenation with list values"""
    batch_size = 2
    pixel_values1 = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    pixel_values2 = [torch.randn(3, 224, 224) for _ in range(batch_size)]

    data1 = TensorBatch(
        {"sequences": torch.randn(batch_size, 10), "pixel_values": pixel_values1}
    )
    data2 = TensorBatch(
        {"sequences": torch.randn(batch_size, 10), "pixel_values": pixel_values2}
    )

    concatenated = TensorBatch.cat([data1, data2])
    assert len(concatenated) == 2 * batch_size
    assert len(concatenated["pixel_values"]) == 2 * batch_size
    # Verify order is preserved
    assert torch.equal(concatenated["pixel_values"][0], pixel_values1[0])
    assert torch.equal(concatenated["pixel_values"][2], pixel_values2[0])


def test_nested_tensorbatch_cat():
    """Test concatenation with nested TensorBatch"""
    batch_size = 2
    sequences1 = torch.randn(batch_size, 10)
    sequences2 = torch.randn(batch_size, 10)

    nested1 = TensorBatch({"pixel_values": [torch.tensor([i]) for i in range(batch_size)]})
    nested2 = TensorBatch(
        {"pixel_values": [torch.tensor([i + batch_size]) for i in range(batch_size)]}
    )

    data1 = TensorBatch({"sequences": sequences1, "nested": nested1})
    data2 = TensorBatch({"sequences": sequences2, "nested": nested2})

    concatenated = TensorBatch.cat([data1, data2])
    assert len(concatenated) == 2 * batch_size
    assert len(concatenated["nested"]["pixel_values"]) == 2 * batch_size


def test_list_repeat():
    """Test repeat with list values"""
    batch_size = 2
    pixel_values = [torch.tensor([i]) for i in range(batch_size)]

    # Use 1D tensor for sequences (like original repeat tests)
    data = TensorBatch(
        {"sequences": torch.tensor([1, 2]), "pixel_values": pixel_values}
    )

    repeated = data.repeat(2)
    assert len(repeated) == 2 * batch_size
    assert len(repeated["pixel_values"]) == 2 * batch_size
    # List repeat: [a, b] * 2 = [a, b, a, b]
    assert torch.equal(repeated["pixel_values"][0], pixel_values[0])
    assert torch.equal(repeated["pixel_values"][1], pixel_values[1])
    assert torch.equal(repeated["pixel_values"][2], pixel_values[0])
    assert torch.equal(repeated["pixel_values"][3], pixel_values[1])


def test_list_repeat_interleave():
    """Test repeat_interleave with list values"""
    batch_size = 2
    pixel_values = [torch.tensor([i]) for i in range(batch_size)]

    # Use 1D tensor for sequences (like original repeat_interleave tests)
    data = TensorBatch(
        {"sequences": torch.tensor([1, 2]), "pixel_values": pixel_values}
    )

    repeated = data.repeat_interleave(2)
    assert len(repeated) == 2 * batch_size
    assert len(repeated["pixel_values"]) == 2 * batch_size
    # Interleave: [a, b] -> [a, a, b, b]
    assert torch.equal(repeated["pixel_values"][0], pixel_values[0])
    assert torch.equal(repeated["pixel_values"][1], pixel_values[0])
    assert torch.equal(repeated["pixel_values"][2], pixel_values[1])
    assert torch.equal(repeated["pixel_values"][3], pixel_values[1])


def test_nested_tensorbatch_repeat():
    """Test repeat with nested TensorBatch"""
    batch_size = 2
    nested = TensorBatch({"values": [torch.tensor([i]) for i in range(batch_size)]})

    # Use 1D tensor for sequences (like original repeat tests)
    data = TensorBatch({"sequences": torch.tensor([1, 2]), "nested": nested})

    repeated = data.repeat(2)
    assert len(repeated) == 2 * batch_size
    assert len(repeated["nested"]["values"]) == 2 * batch_size


def test_nested_tensorbatch_repeat_interleave():
    """Test repeat_interleave with nested TensorBatch"""
    batch_size = 2
    nested = TensorBatch({"values": [torch.tensor([i]) for i in range(batch_size)]})

    # Use 1D tensor for sequences (like original repeat_interleave tests)
    data = TensorBatch({"sequences": torch.tensor([1, 2]), "nested": nested})

    repeated = data.repeat_interleave(2)
    assert len(repeated) == 2 * batch_size
    assert len(repeated["nested"]["values"]) == 2 * batch_size


def test_list_to_device():
    """Test to() with list containing tensors"""
    batch_size = 3
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})

    # Test dtype conversion
    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    for pv in data["pixel_values"]:
        assert pv.dtype == torch.float16


def test_nested_tensorbatch_to_device():
    """Test to() with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    nested = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
        }
    )

    data = TensorBatch({"sequences": sequences, "nested": nested})

    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    for pv in data["nested"]["pixel_values"]:
        assert pv.dtype == torch.float16


def test_list_contiguous():
    """Test contiguous() with list containing tensors"""
    batch_size = 3
    # Create non-contiguous tensors by transposing
    pixel_values = [torch.randn(224, 3, 224).transpose(0, 1) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})

    # Check initial state
    for pv in data["pixel_values"]:
        assert not pv.is_contiguous()

    data.contiguous()

    for pv in data["pixel_values"]:
        assert pv.is_contiguous()


def test_nested_tensorbatch_contiguous():
    """Test contiguous() with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    nested = TensorBatch(
        {
            "pixel_values": [
                torch.randn(224, 3, 224).transpose(0, 1) for _ in range(batch_size)
            ],
        }
    )

    data = TensorBatch({"sequences": sequences, "nested": nested})

    # Check initial state
    for pv in data["nested"]["pixel_values"]:
        assert not pv.is_contiguous()

    data.contiguous()

    for pv in data["nested"]["pixel_values"]:
        assert pv.is_contiguous()


def test_list_pickle():
    """Test pickle serialization with list values"""
    batch_size = 3
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})
    data.metadata = {"info": "test"}

    # Serialize and deserialize
    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == len(data)
    assert torch.equal(unpickled["sequences"], data["sequences"])
    assert len(unpickled["pixel_values"]) == len(data["pixel_values"])
    for i in range(batch_size):
        assert torch.equal(unpickled["pixel_values"][i], data["pixel_values"][i])
    assert unpickled.metadata == data.metadata


def test_nested_tensorbatch_pickle():
    """Test pickle serialization with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    nested = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
            "grid": [torch.tensor([i]) for i in range(batch_size)],
        }
    )

    data = TensorBatch({"sequences": sequences, "nested": nested})
    data.metadata = {"info": "test"}

    # Serialize and deserialize
    pickled = pickle.dumps(data)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == len(data)
    assert torch.equal(unpickled["sequences"], data["sequences"])
    assert isinstance(unpickled["nested"], TensorBatch)
    assert len(unpickled["nested"]["pixel_values"]) == batch_size
    for i in range(batch_size):
        assert torch.equal(
            unpickled["nested"]["pixel_values"][i], data["nested"]["pixel_values"][i]
        )
    assert unpickled.metadata == data.metadata


def test_list_equality():
    """Test equality comparison with list values"""
    batch_size = 3
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    sequences = torch.randn(batch_size, 10)

    data1 = TensorBatch({"sequences": sequences, "pixel_values": pixel_values})
    data2 = TensorBatch({"sequences": sequences.clone(), "pixel_values": [pv.clone() for pv in pixel_values]})

    assert data1 == data2

    # Modify one element
    data2["pixel_values"][0] = torch.randn(3, 224, 224)
    assert data1 != data2


def test_nested_tensorbatch_equality():
    """Test equality comparison with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    nested1 = TensorBatch({"values": [torch.tensor([i]) for i in range(batch_size)]})
    nested2 = TensorBatch({"values": [torch.tensor([i]) for i in range(batch_size)]})

    data1 = TensorBatch({"sequences": sequences, "nested": nested1})
    data2 = TensorBatch({"sequences": sequences.clone(), "nested": nested2})

    assert data1 == data2

    # Modify nested value
    data2["nested"]["values"][0] = torch.tensor([999])
    assert data1 != data2


def test_setitem_list():
    """Test __setitem__ with list values"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    data = TensorBatch({"sequences": sequences})

    # Add list field
    pixel_values = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    data["pixel_values"] = pixel_values
    assert len(data["pixel_values"]) == batch_size

    # Test batch size mismatch
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["wrong_size"] = [torch.randn(3, 224, 224) for _ in range(batch_size + 1)]


def test_setitem_nested_tensorbatch():
    """Test __setitem__ with nested TensorBatch"""
    batch_size = 3
    sequences = torch.randn(batch_size, 10)
    data = TensorBatch({"sequences": sequences})

    # Add nested TensorBatch field
    nested = TensorBatch({"values": [torch.tensor([i]) for i in range(batch_size)]})
    data["nested"] = nested
    assert isinstance(data["nested"], TensorBatch)

    # Test batch size mismatch
    wrong_nested = TensorBatch(
        {"values": [torch.tensor([i]) for i in range(batch_size + 1)]}
    )
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["wrong_nested"] = wrong_nested


def test_multimodal_inputs_use_case():
    """Test the specific multi_modal_inputs use case"""
    batch_size = 4

    # Create multi_modal_inputs TensorBatch with pixel_values and image_grid_thw
    multi_modal_inputs = TensorBatch(
        {
            "pixel_values": [torch.randn(3, 224, 224) for _ in range(batch_size)],
            "image_grid_thw": [torch.tensor([1, 224, 224]) for _ in range(batch_size)],
        }
    )

    # Create main training input batch
    training_input = TensorBatch(
        {
            "sequences": torch.randn(batch_size, 512),
            "attention_mask": torch.ones(batch_size, 512),
            "multi_modal_inputs": multi_modal_inputs,
        }
    )

    # Test slicing (micro-batching)
    micro_batch = training_input.slice(0, 2)
    assert len(micro_batch) == 2
    assert len(micro_batch["multi_modal_inputs"]["pixel_values"]) == 2
    assert len(micro_batch["multi_modal_inputs"]["image_grid_thw"]) == 2

    # Test chunking
    chunks = training_input.chunk(2)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk["multi_modal_inputs"]["pixel_values"]) == 2

    # Test to() for device/dtype conversion
    training_input.to(dtype=torch.float16)
    for pv in training_input["multi_modal_inputs"]["pixel_values"]:
        assert pv.dtype == torch.float16

    # Test contiguous
    training_input.contiguous()

    # Test pickle round-trip
    pickled = pickle.dumps(training_input)
    unpickled = pickle.loads(pickled)
    assert len(unpickled["multi_modal_inputs"]["pixel_values"]) == batch_size
