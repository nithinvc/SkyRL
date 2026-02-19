import pytest
from pydantic import ValidationError

from skyrl.tinker import api


def _make_datum() -> api.Datum:
    return api.Datum(
        model_input=api.ModelInput(chunks=[api.ModelInputChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "weights": api.TensorData(data=[1.0, 1.0, 1.0]),
        },
    )


def test_forward_backward_input_accepts_ppo_threshold_keys():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
    )
    assert req.loss_fn_config == {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}


def test_forward_backward_input_rejects_invalid_ppo_loss_fn_config_keys():
    with pytest.raises(ValidationError, match="Invalid loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="ppo",
            loss_fn_config={"clip_ratio": 0.2},
        )


def test_forward_backward_input_rejects_loss_fn_config_for_cross_entropy():
    with pytest.raises(ValidationError, match="does not accept loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="cross_entropy",
            loss_fn_config={"clip_low_threshold": 0.9},
        )


def test_model_input_chunk_accepts_typed_encoded_text():
    chunk = api.ModelInputChunk(type="encoded_text", tokens=[1, 2, 3])
    converted = chunk.to_types()
    assert isinstance(converted, api.types.EncodedTextChunk)
    assert converted.tokens == [1, 2, 3]


def test_model_input_chunk_accepts_typed_image():
    chunk = api.ModelInputChunk(type="image", data=b"img-bytes", format="jpeg", expected_tokens=8)
    converted = chunk.to_types()
    assert isinstance(converted, api.types.ImageChunk)
    assert converted.expected_tokens == 8
    assert converted.format == "jpeg"
    assert converted.data == b"img-bytes"


def test_model_input_chunk_rejects_unsupported_type():
    with pytest.raises(ValidationError, match="Unsupported chunk type"):
        api.ModelInputChunk(type="audio", data=b"123")
