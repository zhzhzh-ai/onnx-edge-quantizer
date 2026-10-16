import numpy as np
import pytest
from onnx_edge_quantizer.quantization import DTYPES, minmax_params, quantize, dequantize, validate_native, reconstruct_weight


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("symmetric", [False, True])
def test_zero_representable_and_error_bound(dtype, symmetric):
    p = minmax_params(-2.3, 3.7, dtype, symmetric)
    values = np.linspace(-2.3, 3.7, 2001, dtype=np.float32)
    quantized = quantize(values, p)
    assert quantized.min() >= p.qmin and quantized.max() <= p.qmax
    # Affine zero-point rounding can move an endpoint by <= half a step.
    assert np.max(np.abs(dequantize(quantized, p) - values)) <= p.scale * 0.501 + 1e-6
    assert dequantize(quantize(np.array([0.0]), p), p)[0] == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_and_constant_positive_negative(dtype):
    for constant in [0.0, 3.2, -1.7]:
        p = minmax_params(constant, constant, dtype, False)
        assert p.scale > 0
        assert abs(float(dequantize(quantize([constant], p), p)[0]) - constant) <= p.scale + 1e-6


def test_narrow_symmetric_signed_and_offset_unsigned():
    signed = minmax_params(-2, 2, "int8", True)
    unsigned = minmax_params(-2, 2, "uint8", True)
    assert (signed.qmin, signed.qmax, signed.zero_point) == (-127, 127, 0)
    assert (unsigned.qmin, unsigned.qmax, unsigned.zero_point) == (1, 255, 128)


def test_native_rejects_silent_precision_or_scheme_changes():
    validate_native("int8", "int8", True, False)
    validate_native("int4", "int16", True, True)
    for weights, activations, ws, acs in [("uint8", "int8", True, False), ("int16", "int8", True, False),
                                         ("int8", "uint4", True, False), ("int8", "int16", True, False),
                                         ("int8", "int8", False, False)]:
        with pytest.raises(ValueError):
            validate_native(weights, activations, ws, acs)


@pytest.mark.parametrize("lo,hi", [(float("nan"), 1), (-1, float("inf")), (2, 1)])
def test_invalid_range_rejected(lo, hi):
    with pytest.raises(ValueError):
        minmax_params(lo, hi, "int8", False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("symmetric", [True, False])
def test_per_channel_handles_different_ranges_and_zero_channels(dtype, symmetric):
    # Depthwise layout: channel axis is the last dimension, including a zero channel.
    values = np.array([[[[0.01, 10.0, 0.0], [-0.02, -20.0, 0.0]],
                        [[0.03, 30.0, 0.0], [-0.01, -10.0, 0.0]]]], dtype=np.float32)
    restored, report = reconstruct_weight(values, dtype, symmetric, "per-channel", 3)
    assert restored.shape == values.shape and report["axis"] == 3
    assert report["scales"][1] / report["scales"][0] == pytest.approx(1000, rel=1e-6)
    assert report["scales"][2] == 1
    np.testing.assert_array_equal(restored[..., 2], 0)
    for c in range(3):
        assert np.max(np.abs(restored[..., c] - values[..., c])) <= report["scales"][c] * 0.501 + 1e-6


def test_channel_axes_follow_tflite_layout_and_matmul_transposes():
    from types import SimpleNamespace
    from onnx_edge_quantizer.runtime import modules, weight_channel_axis
    _, s = modules()
    assert weight_channel_axis(s.BuiltinOperator.FULLY_CONNECTED, 1, 2) == 0
    assert weight_channel_axis(s.BuiltinOperator.CONV_2D, 1, 4) == 0
    assert weight_channel_axis(s.BuiltinOperator.DEPTHWISE_CONV_2D, 1, 4) == 3
    assert weight_channel_axis(s.BuiltinOperator.TRANSPOSE_CONV, 1, 4) == 0
    for transpose in [False, True]:
        options = SimpleNamespace(adjX=transpose, adjY=transpose)
        assert weight_channel_axis(s.BuiltinOperator.BATCH_MATMUL, 0, 3, options) == (2 if transpose else 1)
        assert weight_channel_axis(s.BuiltinOperator.BATCH_MATMUL, 1, 3, options) == (1 if transpose else 2)
