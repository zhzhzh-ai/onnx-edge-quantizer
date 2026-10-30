"""Independent CPU checks for packed INT4 transposed convolution exports."""
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from onnx_edge_quantizer.cli import main
from onnx_edge_quantizer.runtime import read_model, write_model, check_native_precision, make_interpreter
from onnx_edge_quantizer.structure import compare_structure
from onnx_edge_quantizer.transpose_conv import fuse_transpose_conv_bias
from scripts.validate_model_zoo import dynamic_input_flags, InputShapeMismatch
from test_deployment_rewrites import Graph, run


def make_transpose_conv(folder, *, separate_bias=False, bias_shape=(1, 1, 1, 3)):
    rng = np.random.default_rng(84)
    weights = rng.uniform(-0.7, 0.7, (3, 2, 2, 2)).astype(np.float32)
    bias = np.array([-0.5, 0.2, 0.7], np.float32)
    b = Graph()
    x = b.tensor('x', [1, 2, 2, 2])
    w = b.tensor('weight', weights.shape, weights)
    shape = b.tensor('output_shape', [4], [1, 4, 4, 3], 'INT32')
    c = b.tensor('bias', bias_shape if separate_bias else [3], bias)
    y = b.tensor('y', [1, 4, 4, 3])
    if separate_bias:
        inner = b.tensor('deconv', [1, 4, 4, 3])
        b.op('TRANSPOSE_CONV', [shape, w, x], [inner], 'TransposeConvOptions',
             padding=b.s.Padding.VALID, strideH=2, strideW=2)
        b.op('ADD', [inner, c], [y], 'AddOptions', fusedActivationFunction=b.s.ActivationFunctionType.RELU)
    else:
        b.op('TRANSPOSE_CONV', [shape, w, x, c], [y], 'TransposeConvOptions',
             padding=b.s.Padding.VALID, strideH=2, strideW=2)
    path = b.save(folder / 'float.tflite', [x], [y])
    signature = b.s.SignatureDefT()
    signature.signatureKey, signature.subgraphIndex = b'serving_default', 0
    signature.inputs, signature.outputs = [], []
    for name, index, entries in [(b'x', x, signature.inputs), (b'y', y, signature.outputs)]:
        entry = b.s.TensorMapT()
        entry.name, entry.tensorIndex = name, index
        entries.append(entry)
    b.model.signatureDefs = [signature]
    write_model(b.model, path)
    graph = helper.make_graph([
        helper.make_node('Transpose', ['x'], ['nchw'], perm=[0, 3, 1, 2]),
        helper.make_node('ConvTranspose', ['nchw', 'weight', 'bias'], ['deconv'], strides=[2, 2]),
        helper.make_node('Transpose', ['deconv'], ['y'], perm=[0, 2, 3, 1]),
    ], 'transpose_conv', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 4, 3])],
        [numpy_helper.from_array(weights.transpose(3, 0, 1, 2).copy(), 'weight'),
         numpy_helper.from_array(bias, 'bias')])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9), folder / 'model.onnx')
    np.savez(folder / 'calibration.npz', x=rng.uniform(-1, 1, (3, 1, 2, 2, 2)).astype(np.float32))
    return path, weights, bias


@pytest.mark.parametrize('granularity', ['per-tensor', 'per-channel'])
def test_transpose_conv_int4_storage_bias_slots_and_structure(tmp_path, granularity):
    source, weights, bias = make_transpose_conv(tmp_path)
    reference = tmp_path / 'w8.tflite'
    common = [str(tmp_path / 'model.onnx'), '--float-tflite', str(source),
              '--calibration', str(tmp_path / 'calibration.npz'), '--weight-granularity', granularity]
    assert main([*common, '-o', str(reference)]) == 0
    output = tmp_path / 'w4.tflite'
    assert main([*common, '-o', str(output), '--weight-dtype', 'int4',
                 '--structure-reference', str(reference), '--validation-mode', 'structure']) == 0
    assert compare_structure(output, reference)['matched']
    report = json.loads(output.with_suffix('.quantization.json').read_text())
    assert report['runtime_validation'] == report['onnx_parity_validation'] == 'not_requested'
    assert report['validation'] == []
    assert report['backend_details']['runtime_requirements']
    assert check_native_precision(output, 'int4', 'int8', granularity)['checked_weight_tensors'] == 1
    model = read_model(output)
    g, s = model.subgraphs[0], Graph().s
    op = next(op for op in g.operators if model.operatorCodes[op.opcodeIndex].builtinCode == s.BuiltinOperator.TRANSPOSE_CONV)
    w, a, c = (g.tensors[int(op.inputs[i])] for i in (1, 2, 3))
    assert g.tensors[int(op.inputs[0])].type == s.TensorType.INT32
    assert w.type == s.TensorType.INT4 and c.type == s.TensorType.INT32
    packed = np.asarray(model.buffers[w.buffer].data)
    unpacked = np.empty(packed.size * 2, np.int8)
    unpacked[::2] = ((packed & 15).astype(np.int8) ^ 8) - 8
    unpacked[1::2] = ((packed >> 4).astype(np.int8) ^ 8) - 8
    scales = np.max(np.abs(weights), axis=(1, 2, 3)) / 7 if granularity == 'per-channel' else np.array([np.max(np.abs(weights)) / 7])
    np.testing.assert_allclose(w.quantization.scale, scales)
    expected = np.clip(np.rint(weights / scales.reshape(-1, 1, 1, 1)), -7, 7).astype(np.int8)
    np.testing.assert_array_equal(unpacked[:weights.size], expected.ravel())
    bias_scales = (float(a.quantization.scale[0]) * scales.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(np.frombuffer(bytes(model.buffers[c.buffer].data), '<i4'),
                                  np.rint(bias.astype(np.float64) / bias_scales).astype(np.int32))
    # Expanding exactly the exported INT4 codes (without changing scales) lets
    # the stock INT8 kernel verify these weight/bias encodings independently.
    w.type = s.TensorType.INT8
    model.buffers[w.buffer].data = unpacked[:weights.size].view(np.uint8)
    expanded = tmp_path / 'expanded_for_cpu_test.tflite'
    write_model(model, expanded)
    instance = make_interpreter(expanded)
    for detail in instance.get_input_details():
        instance.set_tensor(detail['index'], np.zeros(detail['shape'], detail['dtype']))
    instance.invoke()
    # Strict default still runs the real candidate and refuses unsupported
    # stock-runtime execution instead of silently publishing an INT8 carrier.
    strict = tmp_path / 'strict.tflite'
    assert main([*common, '-o', str(strict), '--weight-dtype', 'int4']) == 2
    assert not strict.exists()


@pytest.mark.parametrize('bias_shape', [(3,), (1, 1, 1, 3)])
def test_fuse_transpose_conv_bias_and_relu_preserves_cpu_values(tmp_path, bias_shape):
    path, _, _ = make_transpose_conv(tmp_path, separate_bias=True, bias_shape=bias_shape)
    sample = np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 2, 2, 2)
    expected = run(path, [sample])[0]
    assert fuse_transpose_conv_bias(path)
    model = read_model(path)
    assert len(model.subgraphs[0].operators) == 1
    assert len(model.subgraphs[0].operators[0].inputs) == 4
    np.testing.assert_allclose(run(path, [sample])[0], expected, atol=1e-6)


def test_shape_only_requires_reference_and_static_input_mismatch_is_explicit(demo):
    assert main([str(demo / 'demo.onnx'), '-o', str(demo / 'unchecked.tflite'),
                 '--validation-mode', 'structure']) == 2
    model = read_model(demo / 'float.tflite')
    model.subgraphs[0].tensors[int(model.subgraphs[0].inputs[0])].shape[-1] += 1
    ref = demo / 'shape_mismatch.tflite'
    write_model(model, ref)
    with pytest.raises(InputShapeMismatch, match='Input shape mismatch'):
        dynamic_input_flags(demo / 'demo.onnx', ref, check_fixed=True)


def test_transpose_conv_output_shape_control_is_part_of_structure(tmp_path):
    path, _, _ = make_transpose_conv(tmp_path)
    m = read_model(path)
    op = m.subgraphs[0].operators[0]
    t = m.subgraphs[0].tensors[int(op.inputs[0])]
    m.buffers[t.buffer].data = np.frombuffer(np.array([1, 5, 4, 3], '<i4').tobytes(), np.uint8)
    malformed = tmp_path / 'wrong_output_shape.tflite'
    write_model(m, malformed)
    result = compare_structure(path, malformed)
    assert not result['matched']
    assert any('.value' in d['path'] for d in result['differences'])
