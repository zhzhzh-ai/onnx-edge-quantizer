"""Independent LiteRT execution of layout, shape and quantization rewrites."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_edge_quantizer.layout_normalization import (
    coalesce_slice_partitions, lower_nearest_resize_to_gathers,
    normalize_channel_layouts, normalize_unit_strided_slices,
    lower_reference_splits,
)
from onnx_edge_quantizer.onnx_compat import lower_custom_gemm, restore_inferred_reshapes
from onnx_edge_quantizer.quantized_optimizations import (
    restore_integer_leaky_relu, satisfy_quantization_constraints,
)
from onnx_edge_quantizer.reference_interfaces import align_terminal_output_layout
from onnx_edge_quantizer.reference_quantization import align_quantize_boundaries
from onnx_edge_quantizer.runtime import make_interpreter, modules, read_model, write_model
from onnx_edge_quantizer.static_folding import fold_static_tensors
from onnx_edge_quantizer.channel_shuffle import align_channel_shuffles
from onnx_edge_quantizer.padding_helpers import simplify_padding_helpers
from onnx_edge_quantizer.structure import compare_structure
from onnx_edge_quantizer.tflite_optimizations import (
    normalize_elementwise_constants, fuse_scalar_gather_reshapes, restore_pointwise_rank,
    normalize_deployment_graph, commute_unary_before_reshape,
)


class Graph:
    def __init__(self):
        _, self.s = modules()
        self.model = self.s.ModelT()
        self.model.version = 3
        self.model.buffers, self.model.operatorCodes = [self.s.BufferT()], []
        self.g = self.s.SubGraphT()
        self.g.tensors, self.g.operators = [], []
        self.model.subgraphs = [self.g]

    def tensor(self, name, shape, data=None, dtype='FLOAT32', quant=None):
        t = self.s.TensorT()
        t.name, t.shape, t.type, t.buffer = name.encode(), np.array(shape, np.int32), getattr(self.s.TensorType, dtype), 0
        if data is not None:
            buffer = self.s.BufferT()
            array = np.array(data, dtype={'FLOAT32': '<f4', 'INT32': '<i4', 'INT8': 'i1'}[dtype])
            buffer.data = np.frombuffer(array.tobytes(), np.uint8)
            self.model.buffers.append(buffer)
            t.buffer = len(self.model.buffers) - 1
        if quant is not None:
            t.quantization = self.s.QuantizationParametersT()
            t.quantization.scale = np.array([quant[0]], np.float32)
            t.quantization.zeroPoint = np.array([quant[1]], np.int64)
        self.g.tensors.append(t)
        return len(self.g.tensors) - 1

    def op(self, name, inputs, outputs, table=None, version=1, **options):
        code = self.s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = getattr(self.s.BuiltinOperator, name)
        code.version = version
        self.model.operatorCodes.append(code)
        op = self.s.OperatorT()
        op.opcodeIndex = len(self.model.operatorCodes) - 1
        op.inputs, op.outputs = np.array(inputs, np.int32), np.array(outputs, np.int32)
        if table:
            op.builtinOptionsType = getattr(self.s.BuiltinOptions, table)
            op.builtinOptions = getattr(self.s, table + 'T')()
            for key, value in options.items():
                setattr(op.builtinOptions, key, value)
        self.g.operators.append(op)
        return op

    def save(self, path, inputs, outputs):
        self.g.inputs, self.g.outputs = np.array(inputs, np.int32), np.array(outputs, np.int32)
        write_model(self.model, path)
        return path


def run(path, inputs):
    instance = make_interpreter(path)
    for detail, value in zip(instance.get_input_details(), inputs):
        instance.set_tensor(detail['index'], value)
    instance.invoke()
    return [instance.get_tensor(d['index']) for d in instance.get_output_details()]


def metadata(path, records=None):
    path.with_name(path.stem.removesuffix('_float32') + '_tensor_correspondence_report.json').write_text(
        json.dumps({'schema_version': 1, 'records': records or []}))


@pytest.mark.parametrize('kind', ['ADD', 'MUL', 'SUB', 'DIV'])
def test_elementwise_constants_preserve_values_and_polarity(tmp_path, kind):
    b = Graph()
    x = b.tensor('x', [2, 3])
    c = b.tensor('c', [1], [2.0])
    y = b.tensor('y', [2, 3])
    b.op(kind, [c, x], [y], kind.title() + 'Options')
    path = b.save(tmp_path / 'scalar.tflite', [x], [y])
    sample = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    expected = run(path, [sample])[0]
    assert normalize_elementwise_constants(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    g = read_model(path).subgraphs[0]
    dynamic_slot = 0 if kind in ('ADD', 'MUL') else 1
    assert g.operators[0].inputs[dynamic_slot] == g.inputs[0]
    assert len(g.tensors[g.operators[0].inputs[1 - dynamic_slot]].shape) == 0


def test_elementwise_scalar_does_not_remove_broadcast_rank(tmp_path):
    b = Graph()
    x = b.tensor('x', [])
    c = b.tensor('c', [1, 1], [[2.0]])
    y = b.tensor('y', [1, 1])
    b.op('SUB', [c, x], [y], 'SubOptions')
    path = b.save(tmp_path / 'rank.tflite', [x], [y])
    assert not normalize_elementwise_constants(path)
    assert run(path, [np.array(3, np.float32)])[0].shape == (1, 1)


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_scalar_gather_preserves_axis_and_values(tmp_path, axis):
    b = Graph()
    shape = [2, 3, 4]
    normalized = axis % len(shape)
    x = b.tensor('x', shape)
    indices = b.tensor('index', [1], [1], 'INT32')
    selected_shape = shape[:normalized] + [1] + shape[normalized + 1:]
    target_shape = shape[:normalized] + shape[normalized + 1:]
    selected = b.tensor('selected', selected_shape)
    y = b.tensor('y', target_shape)
    control = b.tensor('shape', [len(target_shape)], target_shape, 'INT32')
    b.op('GATHER', [x, indices], [selected], 'GatherOptions', axis=axis)
    b.op('RESHAPE', [selected, control], [y], 'ReshapeOptions')
    path = b.save(tmp_path / 'gather.tflite', [x], [y])
    sample = np.arange(24, dtype=np.float32).reshape(shape)
    expected = run(path, [sample])[0]
    assert fuse_scalar_gather_reshapes(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    assert len(read_model(path).subgraphs[0].operators) == 1


def test_pointwise_chain_can_keep_rank_five(tmp_path):
    b = Graph()
    shape = [1, 2, 3, 4, 5]
    x = b.tensor('x', shape)
    flat = b.tensor('flat', [1, 120])
    flat_control = b.tensor('flat_shape', [2], [1, 120], 'INT32')
    intermediate = b.tensor('intermediate', [1, 120])
    result = b.tensor('result', [1, 120])
    c = b.tensor('c', [], [2.0])
    y = b.tensor('y', shape)
    control = b.tensor('shape', [5], shape, 'INT32')
    b.op('RESHAPE', [x, flat_control], [flat], 'ReshapeOptions')
    b.op('MUL', [flat, c], [intermediate], 'MulOptions')
    b.op('SUB', [c, intermediate], [result], 'SubOptions')
    b.op('RESHAPE', [result, control], [y], 'ReshapeOptions')
    path = b.save(tmp_path / 'rank5.tflite', [x], [y])
    sample = np.arange(120, dtype=np.float32).reshape(shape)
    expected = run(path, [sample])[0]
    assert restore_pointwise_rank(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    assert len(read_model(path).subgraphs[0].operators) == 2


def test_aeq_single_encoding_rank_fix_and_restoration():
    from onnx_edge_quantizer.aeq_compat import scalar_quantization_ranks
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uq
    original = uq.fix_quantization_params_rank
    params = qtyping.UniformQuantParams(num_bits=8, quantized_dimension=None,
                                       scale=np.array([[[[0.25]]]]), zero_point=np.array([[[[0]]]]))
    with pytest.raises(RuntimeError):
        with scalar_quantization_ranks() as stats:
            for shape in [(), (3,), (1, 2, 3, 4, 5)]:
                values = np.full(shape, 0.5, np.float32)
                codes = uq.uniform_quantize(values, params)
                assert codes.shape == shape
                np.testing.assert_array_equal(codes, np.full(shape, 2, np.int8))
            assert stats['scalar_parameter_rank_fixes'] == 3
            raise RuntimeError('test restoration on export failure')
    assert uq.fix_quantization_params_rank is original


def test_channel_shuffle_expansion_preserves_permutation(tmp_path):
    def build(path, expanded):
        b = Graph()
        x = b.tensor('x', [1, 6, 2, 3])
        y = b.tensor('y', [1, 6, 2, 3])
        if expanded:
            shape = b.tensor('split_shape', [5], [1, 2, 3, 2, 3], 'INT32')
            split = b.tensor('split', [1, 2, 3, 2, 3])
            perm = b.tensor('perm', [5], [0, 2, 1, 3, 4], 'INT32')
            swapped = b.tensor('swapped', [1, 3, 2, 2, 3])
            target = b.tensor('target', [4], [1, 6, 2, 3], 'INT32')
            b.op('RESHAPE', [x, shape], [split], 'ReshapeOptions')
            b.op('TRANSPOSE', [split, perm], [swapped], 'TransposeOptions')
            b.op('RESHAPE', [swapped, target], [y], 'ReshapeOptions')
        else:
            index = b.tensor('index', [6], [0, 3, 1, 4, 2, 5], 'INT32')
            b.op('GATHER', [x, index], [y], 'GatherOptions', axis=1)
        return b.save(path, [x], [y])
    path, ref = build(tmp_path / 'shuffle.tflite', False), build(tmp_path / 'reference.tflite', True)
    sample = np.arange(36, dtype=np.float32).reshape(1, 6, 2, 3)
    expected = run(path, [sample])[0]
    assert align_channel_shuffles(path, ref)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    assert compare_structure(path, ref)['matched']


def test_padding_scaffolding_and_same_conv_are_exact(tmp_path):
    def build(path, scaffolding):
        b = Graph()
        x = b.tensor('x', [1, 3, 4, 1])
        source = x
        if scaffolding:
            zero = b.tensor('zero', [1, 3, 4, 1], np.zeros((1, 3, 4, 1)))
            joined = b.tensor('joined', [2, 3, 4, 1])
            selected = b.tensor('selected', [1, 3, 4, 1])
            begin = b.tensor('begin', [4], [0, 0, 0, 0], 'INT32')
            size = b.tensor('size', [4], [1, 3, 4, 1], 'INT32')
            b.op('CONCATENATION', [x, zero], [joined], 'ConcatenationOptions', axis=0)
            b.op('SLICE', [joined, begin, size], [selected], 'SliceOptions')
            added = b.tensor('added', [1, 3, 4, 1])
            b.op('ADD', [selected, zero], [added], 'AddOptions')
            pads = b.tensor('pads', [4, 2], [[0, 0], [1, 1], [1, 1], [0, 0]], 'INT32')
            source = b.tensor('padded', [1, 5, 6, 1])
            b.op('PAD', [added, pads], [source], 'PadOptions')
        w = b.tensor('weight', [1, 3, 3, 1], np.arange(9, dtype=np.float32).reshape(1, 3, 3, 1)/10)
        bias = b.tensor('bias', [1], [0])
        y = b.tensor('y', [1, 3, 4, 1])
        b.op('CONV_2D', [source, w, bias], [y], 'Conv2DOptions', strideH=1, strideW=1,
             dilationHFactor=1, dilationWFactor=1, padding=b.s.Padding.VALID if scaffolding else b.s.Padding.SAME)
        return b.save(path, [x], [y])
    path, ref = build(tmp_path / 'padding.tflite', True), build(tmp_path / 'reference.tflite', False)
    sample = np.linspace(-2, 3, 12, dtype=np.float32).reshape(1, 3, 4, 1)
    expected = run(path, [sample])[0]
    assert simplify_padding_helpers(path, ref)
    np.testing.assert_allclose(run(path, [sample])[0], expected, atol=1e-6)
    assert compare_structure(path, ref)['matched']


@pytest.mark.parametrize('reference_shape,accepted', [([1, 2], True), ([2, 1], False)])
def test_reference_broadcast_only_moves_leading_singletons(tmp_path, reference_shape, accepted):
    def build(path, reference):
        b = Graph()
        x = b.tensor('x', [2, 2], dtype='INT8', quant=(0.25, 0))
        source = x
        if reference:
            source = b.tensor('q', [2, 2], dtype='INT8', quant=(0.25, 0))
            b.op('QUANTIZE', [x], [source], version=2)
        c = b.tensor('c', reference_shape if reference else [2], [1, 2], 'INT8', (0.25, 0))
        y = b.tensor('y', [2, 2], dtype='INT8', quant=(0.25, 0))
        b.op('ADD', [source, c], [y], 'AddOptions', version=2)
        return b.save(path, [x], [y])
    path, ref = build(tmp_path / 'broadcast.tflite', False), build(tmp_path / 'reference.tflite', True)
    before = path.read_bytes()
    sample = np.array([[-3, 2], [7, -6]], np.int8)
    expected = run(path, [sample])[0]
    assert bool(align_quantize_boundaries(path, ref)) == accepted
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    if not accepted:
        assert path.read_bytes() == before


def test_inferred_reshape_uses_onnx_control_not_stale_annotation(tmp_path):
    from onnx import helper, numpy_helper, TensorProto
    b = Graph()
    x = b.tensor('x', [2, 3])
    control = b.tensor('stale_shape', [2], [8, 1], 'INT32')
    y = b.tensor('y', [8, 1])
    b.op('RESHAPE', [x, control], [y], 'ReshapeOptions')
    path = b.save(tmp_path / 'case_float32.tflite', [x], [y])
    metadata(path, [{'onnx_op_type': 'Reshape', 'exists_in_final_model': True,
                     'assume_channel_last_layout': False, 'onnx_output_name': 'y',
                     'resolved_tflite_tensor_name': 'y'}])
    graph = helper.make_graph([helper.make_node('Reshape', ['x', 'shape'], ['y'])], 'reshape',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [8, 1])],
                              [numpy_helper.from_array(np.array([-1, 1], np.int64), 'shape')])
    assert restore_inferred_reshapes(path, helper.make_model(graph))
    assert fold_static_tensors(path)
    np.testing.assert_array_equal(run(path, [np.arange(6, dtype=np.float32).reshape(2, 3)])[0],
                                  np.arange(6, dtype=np.float32).reshape(6, 1))


def test_inferred_reshape_does_not_override_converter_layout(tmp_path):
    from onnx import helper, numpy_helper, TensorProto
    b = Graph()
    x = b.tensor('x', [1, 6])
    control = b.tensor('physical_shape', [3], [1, 3, 2], 'INT32')
    y = b.tensor('y', [1, 3, 2])
    b.op('RESHAPE', [x, control], [y], 'ReshapeOptions')
    path = b.save(tmp_path / 'case_float32.tflite', [x], [y])
    metadata(path, [{'onnx_op_type': 'Reshape', 'exists_in_final_model': True,
                     'assume_channel_last_layout': False, 'onnx_output_name': 'y',
                     'resolved_tflite_tensor_name': 'y'}])
    graph = helper.make_graph([helper.make_node('Reshape', ['x', 'shape'], ['y'])], 'reshape',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 6])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 3])],
                              [numpy_helper.from_array(np.array([1, 2, -1], np.int64), 'shape')])
    before = path.read_bytes()
    assert not restore_inferred_reshapes(path, helper.make_model(graph))
    assert path.read_bytes() == before


def test_reference_activation_before_flatten_preserves_public_output(tmp_path):
    def build(path, first):
        b = Graph()
        x = b.tensor('x', [1, 2, 3, 4])
        control = b.tensor('shape', [2], [6, 4], 'INT32')
        y = b.tensor('y', [6, 4])
        if first:
            intermediate = b.tensor('intermediate', [1, 2, 3, 4])
            b.op('LOGISTIC', [x], [intermediate])
            b.op('RESHAPE', [intermediate, control], [y], 'ReshapeOptions')
        else:
            intermediate = b.tensor('intermediate', [6, 4])
            b.op('RESHAPE', [x, control], [intermediate], 'ReshapeOptions')
            b.op('LOGISTIC', [intermediate], [y])
        return b.save(path, [x], [y])
    path, ref = build(tmp_path / 'activation.tflite', False), build(tmp_path / 'reference.tflite', True)
    sample = np.linspace(-4, 3, 24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [sample])[0]
    assert commute_unary_before_reshape(path, ref)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    assert compare_structure(path, ref)['matched']
    assert make_interpreter(path).get_output_details()[0]['name'] == 'y'


@pytest.mark.parametrize('kind', ['SPLIT', 'SPLIT_V'])
def test_split_shapes_derive_from_static_partition(tmp_path, kind):
    b = Graph()
    x = b.tensor('x', [1, 2, 6, 4])
    axis = b.tensor('axis', [], [2], 'INT32')
    first, second = b.tensor('first', [1, 1, 1, 4]), b.tensor('second', [1, 1, 1, 4])
    extent = 3 if kind == 'SPLIT' else 2
    if kind == 'SPLIT':
        b.op(kind, [axis, x], [first, second], 'SplitOptions', numSplits=2)
    else:
        sizes = b.tensor('sizes', [2], [2, -1], 'INT32')
        b.op(kind, [x, sizes, axis], [first, second], 'SplitVOptions', numSplits=2)
    shape = b.tensor('shape', [4], dtype='INT32')
    output = b.tensor('hw', [2], dtype='INT32')
    indices = b.tensor('indices', [2], [1, 2], 'INT32')
    b.op('SHAPE', [first], [shape], 'ShapeOptions', outType=b.s.TensorType.INT32)
    b.op('GATHER', [shape, indices], [output], 'GatherOptions', axis=0)
    path = b.save(tmp_path / 'shape.tflite', [x], [output])
    assert fold_static_tensors(path)
    np.testing.assert_array_equal(run(path, [np.ones((1, 2, 6, 4), np.float32)])[0], [2, extent])


def test_reference_slice_partition_replaces_split(tmp_path):
    def build(path, slices):
        b = Graph()
        x = b.tensor('x', [1, 2, 3, 4])
        outputs = [b.tensor(f'y{i}', [1, 2, 3, 2]) for i in range(2)]
        if slices:
            for i, y in enumerate(outputs):
                begin = b.tensor(f'begin{i}', [4], [0, 0, 0, 2*i], 'INT32')
                size = b.tensor(f'size{i}', [4], [1, 2, 3, 2], 'INT32')
                b.op('SLICE', [x, begin, size], [y], 'SliceOptions')
        else:
            axis = b.tensor('axis', [], [3], 'INT32')
            b.op('SPLIT', [axis, x], outputs, 'SplitOptions', numSplits=2)
        return b.save(path, [x], outputs)
    path, ref = build(tmp_path / 'split.tflite', False), build(tmp_path / 'reference.tflite', True)
    sample = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [sample])
    assert lower_reference_splits(path, ref)
    for actual, wanted in zip(run(path, [sample]), expected):
        np.testing.assert_array_equal(actual, wanted)
    assert compare_structure(path, ref)['matched']


@pytest.mark.parametrize('kind', ['RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR'])
def test_identity_resize_is_removed_without_changing_values(tmp_path, kind):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    size = b.tensor('size', [2], [2, 3], 'INT32')
    resized = b.tensor('resized', [1, 2, 3, 4])
    y = b.tensor('y', [1, 2, 3, 4])
    table = 'ResizeBilinearOptions' if kind == 'RESIZE_BILINEAR' else 'ResizeNearestNeighborOptions'
    b.op(kind, [x, size], [resized], table, alignCorners=True, halfPixelCenters=False)
    b.op('LOGISTIC', [resized], [y])
    path = b.save(tmp_path / 'resize.tflite', [x], [y])
    sample = np.linspace(-4, 3, 24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [sample])[0]
    assert normalize_deployment_graph(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    assert len(read_model(path).subgraphs[0].operators) == 1


def test_channel_views_split_and_concat_preserve_values(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    p = b.tensor('to_nc', [4], [0, 3, 1, 2], 'INT32')
    q = b.tensor('to_nhwc', [4], [0, 2, 3, 1], 'INT32')
    nc = b.tensor('nc', [1, 4, 2, 3])
    sigmoid = b.tensor('sigmoid', [1, 4, 2, 3])
    silu = b.tensor('silu', [1, 4, 2, 3])
    b.op('TRANSPOSE', [x, p], [nc], 'TransposeOptions')
    b.op('LOGISTIC', [nc], [sigmoid])
    b.op('MUL', [nc, sigmoid], [silu], 'MulOptions')
    branches = []
    for i in range(2):
        start = b.tensor(f'begin{i}', [4], [0, i * 2, 0, 0], 'INT32')
        size = b.tensor(f'size{i}', [4], [1, 2, 2, 3], 'INT32')
        part = b.tensor(f'part{i}', [1, 2, 2, 3])
        target = b.tensor(f'target{i}', [1, 2, 3, 2])
        b.op('SLICE', [silu, start, size], [part], 'SliceOptions')
        b.op('TRANSPOSE', [part, q], [target], 'TransposeOptions')
        branches.append(target)
    output = b.tensor('output', [1, 2, 3, 4])
    b.op('CONCATENATION', branches, [output], 'ConcatenationOptions', axis=3)
    path = b.save(tmp_path / 'case_float32.tflite', [x], [output])
    metadata(path)
    sample = np.linspace(-5, 3, 24, dtype=np.float32).reshape(1, 2, 3, 4)
    expected = run(path, [sample])[0]
    assert normalize_channel_layouts(path)
    assert coalesce_slice_partitions(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    model = read_model(path)
    names = [model.operatorCodes[o.opcodeIndex].builtinCode for o in model.subgraphs[0].operators]
    assert b.s.BuiltinOperator.TRANSPOSE not in names
    assert names.count(b.s.BuiltinOperator.SPLIT_V) == 1


def test_static_shapes_are_derived_from_contracts_not_stale_annotations(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 6, 8, 3])
    w = b.tensor('w', [2, 3, 3, 3], np.ones((2, 3, 3, 3)))
    bias = b.tensor('bias', [2], [0, 0])
    conv = b.tensor('stale_conv_shape', [1, 1, 1, 2])
    b.op('CONV_2D', [x, w, bias], [conv], 'Conv2DOptions', padding=b.s.Padding.VALID,
         strideH=1, strideW=1, dilationHFactor=1, dilationWFactor=1)
    dims = b.tensor('dims', [4], dtype='INT32')
    b.op('SHAPE', [conv], [dims], 'ShapeOptions', outType=b.s.TensorType.INT32)
    index = b.tensor('index', [2], [1, 2], 'INT32')
    hw = b.tensor('hw', [2], dtype='INT32')
    b.op('GATHER', [dims, index], [hw], 'GatherOptions', axis=0)
    begin, size = b.tensor('begin', [1], [4], 'INT32'), b.tensor('size', [1], [0], 'INT32')
    empty = b.tensor('empty', [0], dtype='INT32')
    b.op('SLICE', [dims, begin, size], [empty], 'SliceOptions')
    output = b.tensor('output', [2], dtype='INT32')
    b.op('CONCATENATION', [hw, empty], [output], 'ConcatenationOptions', axis=0)
    path = b.save(tmp_path / 'shape.tflite', [x], [conv, output])
    assert fold_static_tensors(path)
    actual = run(path, [np.ones((1, 6, 8, 3), np.float32)])
    assert actual[0].shape == (1, 4, 6, 2)
    np.testing.assert_array_equal(actual[1], [4, 6])
    assert len(read_model(path).subgraphs[0].operators) == 1


@pytest.mark.parametrize('align,half', [(False, False), (True, False), (False, True)])
def test_nearest_resize_gather_indices_match_litert(tmp_path, align, half):
    b = Graph()
    x = b.tensor('x', [1, 3, 4, 2])
    size = b.tensor('size', [2], [5, 7], 'INT32')
    output = b.tensor('output', [1, 5, 7, 2])
    b.op('RESIZE_NEAREST_NEIGHBOR', [x, size], [output], 'ResizeNearestNeighborOptions',
         version=3, alignCorners=align, halfPixelCenters=half)
    path = b.save(tmp_path / 'resize.tflite', [x], [output])
    sample = np.arange(24, dtype=np.float32).reshape(1, 3, 4, 2)
    expected = run(path, [sample])[0]
    assert lower_nearest_resize_to_gathers(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)


def test_unit_stride_slice_clamps_negative_and_unbounded_ends(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 3, 5, 2])
    begin = b.tensor('begin', [4], [0, 1, 0, 0], 'INT32')
    end = b.tensor('end', [4], [2147483647, 2147483647, -1, 2147483647], 'INT32')
    stride = b.tensor('stride', [4], [1, 1, 1, 1], 'INT32')
    output = b.tensor('output', [1, 2, 4, 2])
    b.op('STRIDED_SLICE', [x, begin, end, stride], [output], 'StridedSliceOptions')
    path = b.save(tmp_path / 'slice.tflite', [x], [output])
    sample = np.arange(30, dtype=np.float32).reshape(1, 3, 5, 2)
    expected = run(path, [sample])[0]
    assert normalize_unit_strided_slices(path)
    np.testing.assert_array_equal(run(path, [sample])[0], expected)


def test_shared_float_dequantize_leaky_islands_become_native_int8(tmp_path):
    b = Graph()
    x = b.tensor('x', [4], dtype='INT8', quant=(0.5, -3))
    raw = b.tensor('raw', [4])
    b.op('DEQUANTIZE', [x], [raw], version=2)
    outputs = []
    for i, alpha in enumerate([0.1, 0.2]):
        value = b.tensor(f'float{i}', [4])
        output = b.tensor(f'out{i}', [4], dtype='INT8', quant=(0.25, 2))
        b.op('LEAKY_RELU', [raw], [value], 'LeakyReluOptions', alpha=alpha)
        b.op('QUANTIZE', [value], [output])
        outputs.append(output)
    path = b.save(tmp_path / 'leaky.tflite', [x], outputs)
    sample = np.array([-128, -3, 0, 127], np.int8)
    expected = run(path, [sample])
    assert restore_integer_leaky_relu(path)[0]['count'] == 2
    for actual, reference in zip(run(path, [sample]), expected):
        assert np.max(np.abs(actual.astype(int) - reference.astype(int))) <= 1
    assert len(read_model(path).subgraphs[0].operators) == 2


def test_conflicting_sigmoid_and_tanh_anchors_retain_requantization(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2], dtype='INT8', quant=(0.1, 0))
    sigmoid = b.tensor('sigmoid', [1, 2], dtype='INT8', quant=(1 / 256, -128))
    tanh = b.tensor('tanh', [1, 2], dtype='INT8', quant=(1 / 128, 0))
    q = b.tensor('requant', [1, 2], dtype='INT8', quant=(1 / 128, 0))
    output = b.tensor('output', [1, 4], dtype='INT8', quant=(1 / 128, 0))
    b.op('LOGISTIC', [x], [sigmoid], version=2)
    b.op('TANH', [x], [tanh], version=2)
    b.op('QUANTIZE', [sigmoid], [q], version=2)
    b.op('CONCATENATION', [q, tanh], [output], 'ConcatenationOptions', version=2, axis=1)
    path = b.save(tmp_path / 'anchors.tflite', [x], [output])
    before = path.read_bytes()
    assert satisfy_quantization_constraints(path) == []
    assert path.read_bytes() == before
    run(path, [np.array([[-13, 11]], np.int8)])


def test_reference_quantize_is_inserted_as_identity_only_after_core_match(tmp_path):
    def build(path, boundary):
        b = Graph()
        x = b.tensor('x', [1, 2], dtype='INT8', quant=(0.25, 0))
        data = x
        if boundary:
            data = b.tensor('q', [1, 2], dtype='INT8', quant=(0.5, 3))
            b.op('QUANTIZE', [x], [data], version=2)
        output = b.tensor('output', [1, 2], dtype='INT8', quant=(1 / 256, -128))
        b.op('LOGISTIC', [data], [output], version=2)
        return b.save(path, [x], [output])
    path, reference = build(tmp_path / 'candidate.tflite', False), build(tmp_path / 'reference.tflite', True)
    sample = np.array([[-7, 13]], np.int8)
    expected = run(path, [sample])[0]
    assert align_quantize_boundaries(path, reference)[0]['core_graph_verified']
    np.testing.assert_array_equal(run(path, [sample])[0], expected)
    model = read_model(path)
    op = model.subgraphs[0].operators[0]
    first, second = [model.subgraphs[0].tensors[int(i)].quantization for i in [*op.inputs, *op.outputs]]
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.zeroPoint, second.zeroPoint)


def test_custom_gemm_recovers_original_onnx_contract(demo):
    import onnx
    path = demo / 'float.tflite'
    expected = run(path, [np.ones((1, 4), np.float32)])[0]
    model = read_model(path)
    _, s = modules()
    code = model.operatorCodes[model.subgraphs[0].operators[0].opcodeIndex]
    code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.CUSTOM
    code.customCode = b'ONNX_GEMM'
    model.subgraphs[0].operators[0].builtinOptions = None
    model.subgraphs[0].operators[0].builtinOptionsType = 0
    write_model(model, path)
    assert lower_custom_gemm(path, onnx.load(demo / 'demo.onnx'))
    np.testing.assert_array_equal(run(path, [np.ones((1, 4), np.float32)])[0], expected)


def test_reference_terminal_layout_records_inverse_for_validation(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    output = b.tensor('output', [1, 2, 3, 4])
    b.op('LOGISTIC', [x], [output])
    path = b.save(tmp_path / 'candidate.tflite', [x], [output])
    permutation = b.tensor('perm', [4], [0, 2, 3, 1], 'INT32')
    target = b.tensor('target', [1, 3, 4, 2])
    b.op('TRANSPOSE', [output, permutation], [target], 'TransposeOptions')
    reference = b.save(tmp_path / 'reference.tflite', [x], [target])
    layouts = {}
    onnx = SimpleNamespace(graph=SimpleNamespace(output=[SimpleNamespace(name='output')]))
    sample = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 10
    expected = run(path, [sample])[0]
    assert align_terminal_output_layout(path, reference, onnx, layouts)
    actual = run(path, [sample])[0]
    np.testing.assert_array_equal(actual.transpose(layouts['output']['tflite_to_onnx_permutation']), expected)
