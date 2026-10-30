"""External TFLite weight buffers must survive inspection and serialization."""

import importlib.util
import numpy as np
import pytest
from onnx_edge_quantizer.runtime import (read_model, write_model, make_interpreter, map_details,
                                         run, calibrate, check_native_precision, modules)
from onnx_edge_quantizer.data import input_specs, load_samples


def externalize(source, destination, buffer_index):
    model = read_model(source)
    buffer = model.buffers[buffer_index]
    payload = bytes(buffer.data)
    buffer.data, buffer.offset, buffer.size = None, 1024 * 1024, len(payload)
    write_model(model, destination)
    header = destination.read_bytes()
    assert len(header) < buffer.offset
    destination.write_bytes(header + bytes(buffer.offset - len(header)) + payload)
    return payload


def test_external_float_weights_round_trip(demo):
    source, external, restored = demo / "float.tflite", demo / "external.tflite", demo / "restored.tflite"
    payload = externalize(source, external, 1)
    raw = external.read_bytes()
    model = read_model(external)
    assert bytes(model.buffers[1].data) == payload
    assert model.buffers[1].offset == model.buffers[1].size == 0
    assert external.read_bytes() == raw
    write_model(model, restored)
    _, specs = input_specs(demo / "demo.onnx", {})
    samples = load_samples(demo / "calibration.npz", specs)
    expected = make_interpreter(source)
    mapping = map_details(specs, expected.get_input_details())
    for path in [external, restored]:
        actual = make_interpreter(path)
        for sample in samples[:2]:
            np.testing.assert_array_equal(run(expected, sample, mapping)["output"],
                                          run(actual, sample, mapping)["output"])


@pytest.mark.parametrize("offset,size", [(1024 * 1024, 48), (1024 * 1024, 0), (2, 2 ** 32)])
def test_external_buffer_bounds_are_checked(demo, offset, size):
    model = read_model(demo / "float.tflite")
    buffer = model.buffers[1]
    buffer.data, buffer.offset, buffer.size = None, offset, size
    path = demo / "invalid.tflite"
    write_model(model, path)
    with pytest.raises(ValueError, match="Invalid external TFLite buffer"):
        read_model(path)


@pytest.mark.skipif(importlib.util.find_spec("ai_edge_quantizer") is None,
                    reason="AI Edge Quantizer optional dependency is absent")
def test_external_int4_weights_are_not_mistaken_for_activations(demo):
    from onnx_edge_quantizer.backends import native_export
    from onnx_edge_quantizer.cli import Config
    _, specs = input_specs(demo / "demo.onnx", {})
    samples = load_samples(demo / "calibration.npz", specs)
    source = demo / "float.tflite"
    _, mapping = calibrate(source, specs, samples)
    quantized = demo / "quantized.tflite"
    native_export(source, quantized, specs, samples, mapping,
                  Config(weight_dtype="int4", weight_granularity="per-channel"))
    model = read_model(quantized)
    _, schema = modules()
    weight = next(int(op.inputs[1]) for op in model.subgraphs[0].operators
                  if len(op.inputs) >= 2
                  and model.subgraphs[0].tensors[op.inputs[1]].type == schema.TensorType.INT4)
    external = demo / "external_int4.tflite"
    externalize(quantized, external, model.subgraphs[0].tensors[weight].buffer)
    result = check_native_precision(external, "int4", "int8", "per-channel")
    assert result["checked_weight_tensors"] == 1
