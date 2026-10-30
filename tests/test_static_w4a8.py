"""Static W4A8 packing and strict deployment structure publishing checks."""
import json

import numpy as np
import pytest

from onnx_edge_quantizer.cli import main
from onnx_edge_quantizer.quantization import minmax_params, quantize
from onnx_edge_quantizer.runtime import modules, read_model, write_model, check_native_precision
from onnx_edge_quantizer.structure import compare_structure


@pytest.mark.parametrize('granularity', ['per-tensor', 'per-channel'])
def test_w4_weights_and_bias_are_derived_from_original_float(demo, granularity):
    output = demo / 'w4.tflite'
    source = read_model(demo / 'float.tflite')
    _, s = modules()
    float_weights = np.frombuffer(bytes(source.buffers[1].data), dtype='<f4').reshape(3, 4)
    float_bias = np.frombuffer(bytes(source.buffers[2].data), dtype='<f4')
    assert main([str(demo / 'demo.onnx'), '-o', str(output), '--float-tflite', str(demo / 'float.tflite'),
                 '--weight-dtype', 'int4', '--weight-granularity', granularity,
                 '--calibration', str(demo / 'calibration.npz')]) == 0
    model = read_model(output)
    graph = model.subgraphs[0]
    dense = next(op for op in graph.operators
                 if model.operatorCodes[op.opcodeIndex].builtinCode == s.BuiltinOperator.FULLY_CONNECTED)
    weight = graph.tensors[int(dense.inputs[1])]
    data = np.asarray(model.buffers[weight.buffer].data)
    unpacked = np.empty(data.size * 2, dtype=np.int8)
    unpacked[::2] = ((data & 15).astype(np.int8) ^ 8) - 8
    unpacked[1::2] = ((data >> 4).astype(np.int8) ^ 8) - 8
    expected = []
    channels = float_weights if granularity == 'per-channel' else [float_weights]
    for values in channels:
        params = minmax_params(float(values.min()), float(values.max()), 'int4', True)
        expected.extend(quantize(values, params).reshape(-1))
    np.testing.assert_array_equal(unpacked[:float_weights.size], expected)
    assert len(data) == 6
    assert weight.type == s.TensorType.INT4
    bias = graph.tensors[int(dense.inputs[2])]
    activation = graph.tensors[int(dense.inputs[0])]
    scales = (np.float64(activation.quantization.scale[0]) * weight.quantization.scale.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(bias.quantization.scale, scales)
    np.testing.assert_array_equal(np.frombuffer(bytes(model.buffers[bias.buffer].data), dtype='<i4'),
                                  np.rint(float_bias.astype(np.float64) / scales).astype(np.int32))
    # A malformed or unpacked INT4 carrier must never pass precision checks.
    model.buffers[weight.buffer].data = np.zeros(12, dtype=np.uint8)
    malformed = demo / 'malformed.tflite'
    write_model(model, malformed)
    with pytest.raises(ValueError, match='packed INT4 buffer'):
        check_native_precision(malformed, 'int4', 'int8', granularity)


def test_structure_ignores_names_but_detects_edges_shapes_and_options(demo):
    source = demo / 'float.tflite'
    model = read_model(source)
    for i, tensor in enumerate(model.subgraphs[0].tensors):
        tensor.name = f'renamed_{i}'.encode()
    other = demo / 'renamed.tflite'
    write_model(model, other)
    assert compare_structure(other, source)['matched']
    model.subgraphs[0].operators[0].builtinOptions.keepNumDims = True
    write_model(model, other)
    result = compare_structure(other, source)
    assert not result['matched']
    assert any('keepNumDims' in diff['path'] for diff in result['differences'])


def test_structure_mismatch_does_not_replace_existing_output(demo):
    source = demo / 'float.tflite'
    reference = demo / 'wrong_reference.tflite'
    model = read_model(source)
    model.subgraphs[0].tensors[1].shape[1] += 1
    write_model(model, reference)
    output = demo / 'existing.tflite'
    output.write_bytes(b'preserve previous model')
    report = output.with_suffix('.quantization.json')
    report.write_text('{}')
    assert main([str(demo / 'demo.onnx'), '-o', str(output), '--float-tflite', str(source),
                 '--weight-dtype', 'int4', '--structure-reference', str(reference), '--overwrite']) == 2
    assert output.read_bytes() == b'preserve previous model'
    assert json.loads(report.read_text()) == {}


def test_static_int4_odd_element_count_and_zero_channel(demo):
    import onnx
    from onnx import numpy_helper
    values = np.array([[-0.7, 0, 0.7], [0, 0, 0], [0.2, -0.4, 0.6]], dtype='<f4')
    source_path = demo / 'demo.onnx'
    source = onnx.load(source_path)
    source.graph.input[0].type.tensor_type.shape.dim[-1].dim_value = 3
    source.graph.initializer[0].CopyFrom(numpy_helper.from_array(values, 'weight'))
    onnx.save(source, source_path)
    float_path = demo / 'float.tflite'
    model = read_model(float_path)
    graph = model.subgraphs[0]
    graph.tensors[0].shape[-1] = 3
    graph.tensors[1].shape[-1] = 3
    model.buffers[1].data = np.frombuffer(values.tobytes(), dtype=np.uint8)
    write_model(model, float_path)
    output = demo / 'odd.tflite'
    assert main([str(source_path), '-o', str(output), '--float-tflite', str(float_path),
                 '--weight-dtype', 'int4', '--weight-granularity', 'per-channel']) == 0
    model = read_model(output)
    tensor = next(t for t in model.subgraphs[0].tensors if t.name == b'weight')
    packed = model.buffers[tensor.buffer].data
    assert len(packed) == 5 and packed[-1] >> 4 == 0
    np.testing.assert_array_equal(tensor.quantization.scale,
                                  np.array([0.1, 1.0, 0.6 / 7], dtype=np.float32))
    # [-7,0,7,0,0,0,2,-5,7], low nibble first, including a full zero row.
    np.testing.assert_array_equal(packed, [0x09, 0x07, 0x00, 0xB2, 0x07])
