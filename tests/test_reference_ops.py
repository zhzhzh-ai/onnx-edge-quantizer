"""Numerical and negative tests for reference-guided deployment rewrites."""
from copy import deepcopy

import numpy as np
from onnx import TensorProto, helper
import pytest

from onnx_edge_quantizer.onnx_compat import restore_concat_axes
from onnx_edge_quantizer.converter import converted_output_layouts
from onnx_edge_quantizer.quantized_optimizations import restore_integer_reductions
from onnx_edge_quantizer.reference_ops import align_reduction_indices, align_scalar_transposes, align_flattened_concat
from onnx_edge_quantizer.layout_normalization import lower_nearest_resize_to_gathers
from onnx_edge_quantizer.tflite_optimizations import normalize_deployment_graph
from onnx_edge_quantizer.reference_interfaces import align_terminal_output_layout
from onnx_edge_quantizer.runtime import inventory, read_model, write_model
from onnx_edge_quantizer.structure import compare_structure
from test_deployment_rewrites import Graph, run, metadata


@pytest.mark.parametrize('input_layout', [False, True])
def test_restore_concat_axis_requires_explicit_unchanged_input_layouts(tmp_path, input_layout):
    b = Graph()
    a = b.tensor('a', [1, 2, 2, 3])
    c = b.tensor('b', [1, 2, 2, 3])
    z = b.tensor('z', [1, 2, 2, 6])
    b.op('CONCATENATION', [a, c], [z], 'ConcatenationOptions', axis=3)
    path = b.save(tmp_path / 'concat.tflite', [a, c], [z])
    model = helper.make_model(helper.make_graph(
        [helper.make_node('Concat', ['a', 'b'], ['z'], axis=1)], 'concat',
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2, 2, 3]) for n in ('a', 'b')],
        [helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 4, 2, 3])]))
    records = [{'onnx_output_name': name, 'resolved_tflite_tensor_name': name,
                'exists_in_final_model': True, 'assume_channel_last_layout': input_layout if name == 'a' else False}
               for name in ('a', 'b', 'z')]
    metadata(path, records)
    changes = restore_concat_axes(path, model)
    assert bool(changes) is (not input_layout)
    values = [np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3), np.full((1, 2, 2, 3), 20, np.float32)]
    axis = 3 if input_layout else 1
    np.testing.assert_array_equal(run(path, values)[0], np.concatenate(values, axis=axis))


@pytest.mark.parametrize('kind', ['MUL', 'SUB'])
def test_scalar_commutes_through_transpose_with_operand_order_preserved(tmp_path, kind):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    perm = b.tensor('perm', [4], [0, 2, 3, 1], 'INT32')
    t = b.tensor('transpose', [1, 3, 4, 2])
    c = b.tensor('scalar', [], [2.0])
    y = b.tensor('y', [1, 3, 4, 2])
    b.op('TRANSPOSE', [x, perm], [t], 'TransposeOptions')
    b.op(kind, [c, t], [y], kind.title() + 'Options')
    path = b.save(tmp_path / 'pointwise.tflite', [x], [y])
    sample = np.linspace(-4, 5, 24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [sample])[0]
    ref = deepcopy(b.model)
    first, second = ref.subgraphs[0].operators
    second.inputs[1] = x
    second.outputs = np.array([t], np.int32)
    ref.subgraphs[0].tensors[t].shape = np.array([1, 2, 3, 4], np.int32)
    first.inputs[0] = t
    first.outputs = np.array([y], np.int32)
    ref.subgraphs[0].operators = [second, first]
    reference = tmp_path / 'reference.tflite'
    write_model(ref, reference)
    assert align_scalar_transposes(path, reference)
    assert compare_structure(path, reference)['matched']
    np.testing.assert_array_equal(run(path, [sample])[0], expected)


def make_reductions(folder, *, quant_scale=0.25):
    b = Graph()
    x = b.tensor('codes', [2, 3], dtype='INT8', quant=(0.25, -2))
    f = b.tensor('float', [2, 3])
    axis = b.tensor('axis', [1], [1], 'INT32')
    m = b.tensor('max', [2])
    q = b.tensor('quantized_max', [2], dtype='INT8', quant=(quant_scale, -2))
    index = b.tensor('indices', [2], dtype='INT32')
    b.op('DEQUANTIZE', [x], [f], 'DequantizeOptions')
    b.op('REDUCE_MAX', [f, axis], [m], 'ReducerOptions', keepDims=False)
    b.op('QUANTIZE', [m], [q], 'QuantizeOptions', version=2)
    b.op('ARG_MAX', [f, axis], [index], 'ArgMaxOptions', outputType=b.s.TensorType.INT32)
    return b.save(folder / 'reduction.tflite', [x], [q, index])


def test_integer_reductions_preserve_real_cpu_values_and_shared_dq(tmp_path):
    path = make_reductions(tmp_path)
    values = np.array([[-8, 12, 12], [7, -20, 4]], np.int8)
    expected = run(path, [values])
    assert len(restore_integer_reductions(path)) == 2
    result = run(path, [values])
    for a, b in zip(result, expected):
        np.testing.assert_array_equal(a, b)
    assert not inventory(path)['float_compute_ops']
    assert len(read_model(path).subgraphs[0].operators) == 2


def test_integer_reduce_max_does_not_discard_real_requantization(tmp_path):
    path = make_reductions(tmp_path, quant_scale=0.5)
    changes = restore_integer_reductions(path)
    assert len(changes) == 1  # ArgMax can still use the quantized source.
    remaining = inventory(path)['float_compute_ops']
    assert any(op['op'] == 'REDUCE_MAX' for op in remaining)
    # The shared DQ must remain for the unconverted ReduceMax consumer.
    values = np.array([[-8, 12, 12], [7, -20, 4]], np.int8)
    assert len(run(path, [values])) == 2


def test_argmax_axis_and_bounded_integer_cast_are_exact(tmp_path):
    b = Graph()
    x = b.tensor('x', [2, 3])
    axis = b.tensor('axis', [1], [-1], 'INT32')
    index = b.tensor('index', [2], dtype='INT32')
    b.g.tensors[index].type = b.s.TensorType.INT64
    b.op('ARG_MAX', [x, axis], [index], 'ArgMaxOptions', outputType=b.s.TensorType.INT64)
    path = b.save(tmp_path / 'argmax.tflite', [x], [index])
    ref = deepcopy(b.model)
    op = ref.subgraphs[0].operators[0]
    op.builtinOptions.outputType = b.s.TensorType.INT32
    ref.subgraphs[0].tensors[index].type = b.s.TensorType.INT32
    ref.subgraphs[0].tensors[axis].shape = np.array([], np.int32)
    reference = tmp_path / 'argmax_reference.tflite'
    write_model(ref, reference)
    values = np.array([[0.0, 5.0, 5.0], [-2.0, -4.0, -1.0]], np.float32)
    expected = run(path, [values])[0]
    assert align_reduction_indices(path, reference)
    actual = run(path, [values])[0]
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)


def test_absorb_terminal_transpose_records_exact_output_inverse(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    mid = b.tensor('mid', [1, 2, 3, 4])
    perm = b.tensor('perm', [4], [0, 3, 1, 2], 'INT32')
    out = b.tensor('output', [1, 4, 2, 3])
    b.op('RELU', [x], [mid])
    b.op('TRANSPOSE', [mid, perm], [out], 'TransposeOptions')
    path = b.save(tmp_path / 'terminal.tflite', [x], [out])
    ref = deepcopy(b.model)
    ref.subgraphs[0].operators.pop()
    ref.subgraphs[0].outputs = np.array([mid], np.int32)
    reference = tmp_path / 'terminal_reference.tflite'
    write_model(ref, reference)
    original = helper.make_model(helper.make_graph([], 'outputs', [],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 2, 3])]))
    layouts = {}
    values = np.linspace(-4, 5, 24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [values])[0]
    assert align_terminal_output_layout(path, reference, original, layouts)
    actual = run(path, [values])[0]
    np.testing.assert_array_equal(actual.transpose(layouts['output']['tflite_to_onnx_permutation']), expected)
    assert compare_structure(path, reference)['matched']


@pytest.mark.parametrize('insert_transpose', [False, True])
def test_unary_output_layout_requires_uninterrupted_conv_contract(tmp_path, insert_transpose):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 1])
    w = b.tensor('weight', [1, 1, 1, 1], [1.0])
    bias = b.tensor('bias', [1], [0.0])
    conv = b.tensor('conv', [1, 2, 3, 1])
    b.op('CONV_2D', [x, w, bias], [conv], 'Conv2DOptions',
         padding=b.s.Padding.VALID, strideH=1, strideW=1, dilationHFactor=1, dilationWFactor=1)
    source = conv
    if insert_transpose:
        perm = b.tensor('perm', [4], [0, 3, 1, 2], 'INT32')
        source = b.tensor('nchw', [1, 1, 2, 3])
        b.op('TRANSPOSE', [conv, perm], [source], 'TransposeOptions')
    out = b.tensor('output', [1, 1, 2, 3] if insert_transpose else [1, 2, 3, 1])
    b.op('LOGISTIC', [source], [out])
    path = b.save(tmp_path / 'unary.tflite', [x], [out])
    original = helper.make_model(helper.make_graph([
        helper.make_node('Conv', ['x', 'weight', 'bias'], ['conv']),
        helper.make_node('Sigmoid', ['conv'], ['output']),
    ], 'unary', [], [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 3])]))
    metadata(path, [{'onnx_output_name': 'output', 'resolved_tflite_tensor_name': 'output',
                     'onnx_op_type': 'Sigmoid', 'producer_op_type': 'LOGISTIC',
                     'exists_in_final_model': True, 'is_graph_output': True,
                     'assume_channel_last_layout': False}])
    layouts = converted_output_layouts(original, path)
    assert bool(layouts) is (not insert_transpose)
    if layouts:
        assert layouts['output']['tflite_to_onnx_permutation'] == [0, 3, 1, 2]


@pytest.mark.parametrize('index_origin', [False, True])
def test_float_index_output_casts_are_distinguished_from_float_compute(tmp_path, index_origin):
    b = Graph()
    if index_origin:
        x = b.tensor('x', [2, 3], dtype='INT8', quant=(0.25, -2))
        axis = b.tensor('axis', [], [1], 'INT32')
        index = b.tensor('indices', [2], dtype='INT32')
        b.op('ARG_MAX', [x, axis], [index], 'ArgMaxOptions', outputType=b.s.TensorType.INT32)
    else:
        x = index = b.tensor('integer_input', [2], dtype='INT32')
    value = b.tensor('float_output', [2])
    q = b.tensor('output', [2], dtype='INT8', quant=(0.25, -2))
    b.op('CAST', [index], [value], 'CastOptions', inDataType=b.s.TensorType.INT32, outDataType=b.s.TensorType.FLOAT32)
    b.op('QUANTIZE', [value], [q], 'QuantizeOptions', version=2)
    path = b.save(tmp_path / 'index_output.tflite', [x], [q])
    result = inventory(path)
    assert bool(result['index_output_casts']) is index_origin
    assert bool(result['float_compute_ops']) is (not index_origin)


def test_channel_major_detection_branches_preserve_complete_output_array(tmp_path):
    def build(path, channel_major):
        b = Graph()
        inputs, outputs = [], []
        for n, shape in enumerate([[1, 2, 3, 2], [1, 1, 2, 2]]):
            x = b.tensor(f'x{n}', shape)
            inputs.append(x)
            current = x
            if channel_major:
                p = b.tensor(f'perm{n}', [4], [0, 3, 1, 2], 'INT32')
                current = b.tensor(f'transposed{n}', [shape[0], shape[3], shape[1], shape[2]])
                b.op('TRANSPOSE', [x, p], [current], 'TransposeOptions')
            target = [1, 2, shape[1] * shape[2]] if channel_major else [1, shape[1] * shape[2], 2]
            control = b.tensor(f'shape{n}', [3], target, 'INT32')
            y = b.tensor(f'flat{n}', target)
            b.op('RESHAPE', [current, control], [y], 'ReshapeOptions')
            outputs.append(y)
        joined = b.tensor('joined', [1, 2, 8] if channel_major else [1, 8, 2])
        b.op('CONCATENATION', outputs, [joined], 'ConcatenationOptions', axis=2 if channel_major else 1)
        result = joined
        if channel_major:
            control = b.tensor('output_perm', [3], [0, 2, 1], 'INT32')
            result = b.tensor('output', [1, 8, 2])
            b.op('TRANSPOSE', [joined, control], [result], 'TransposeOptions')
        return b.save(path, inputs, [result])
    source = build(tmp_path / 'flatten.tflite', False)
    reference = build(tmp_path / 'reference.tflite', True)
    samples = [np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2),
               np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2) + 20]
    expected = np.concatenate([sample.reshape(1, -1, 2) for sample in samples], axis=1)
    assert align_flattened_concat(source, reference)
    np.testing.assert_array_equal(run(source, samples)[0], expected)
    assert compare_structure(source, reference)['matched']


def test_identity_resize_is_retained_only_for_explicit_reference_sampling(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 1])
    size = b.tensor('size', [2], [2, 3], 'INT32')
    mid = b.tensor('identity_resize', [1, 2, 3, 1])
    y = b.tensor('y', [1, 2, 3, 1])
    b.op('RESIZE_NEAREST_NEIGHBOR', [x, size], [mid], 'ResizeNearestNeighborOptions')
    b.op('RELU', [mid], [y])
    source = b.save(tmp_path / 'resize.tflite', [x], [y])
    default = tmp_path / 'default.tflite'
    default.write_bytes(source.read_bytes())
    reference = tmp_path / 'reference.tflite'
    reference.write_bytes(source.read_bytes())
    lower_nearest_resize_to_gathers(reference)
    normalize_deployment_graph(default)
    assert len(read_model(default).subgraphs[0].operators) == 1
    normalize_deployment_graph(source, reference)
    assert len(read_model(source).subgraphs[0].operators) == 2
    lower_nearest_resize_to_gathers(source)
    assert compare_structure(source, reference)['matched']
    values = np.linspace(-2, 3, 6, dtype=np.float32).reshape(1, 2, 3, 1)
    np.testing.assert_array_equal(run(source, [values])[0], np.maximum(values, 0))
