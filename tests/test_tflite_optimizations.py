"""Prove MaxPool padding rewrites and their safety boundaries."""

import numpy as np
import pytest
from onnx_edge_quantizer.runtime import modules, read_model, write_model, make_interpreter
from onnx_edge_quantizer.tflite_optimizations import normalize_maxpool_padding, normalize_deployment_graph


def make_pad_pool(path, producer="relu", before=0, expose_padding=False, extra_consumer=False,
                  fill=-np.finfo(np.float32).max):
    _, s = modules()
    model = s.ModelT()
    model.version, model.buffers, model.operatorCodes = 3, [s.BufferT()], []
    graph = s.SubGraphT()
    graph.tensors, graph.operators = [], []
    model.subgraphs = [graph]

    def tensor(name, shape, values=None, dtype=None):
        t = s.TensorT()
        t.name, t.shape, t.type = name.encode(), np.array(shape, np.int32), s.TensorType.FLOAT32 if dtype is None else dtype
        t.buffer = 0
        if values is not None:
            buffer = s.BufferT()
            buffer.data = np.frombuffer(values.tobytes(), dtype=np.uint8)
            model.buffers.append(buffer)
            t.buffer = len(model.buffers) - 1
        graph.tensors.append(t)
        return len(graph.tensors) - 1

    def operator(code, inputs, output, options_type=0, options=None):
        entry = s.OperatorCodeT()
        entry.builtinCode, entry.deprecatedBuiltinCode = code, min(code, 127)
        entry.version = 1
        model.operatorCodes.append(entry)
        op = s.OperatorT()
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.inputs, op.outputs = np.array(inputs, np.int32), np.array([output], np.int32)
        op.builtinOptionsType, op.builtinOptions = options_type, options
        graph.operators.append(op)

    source = tensor("image", [1, 4, 6, 1])
    graph.inputs = np.array([source], np.int32)
    if producer != "unknown":
        source = tensor("relu", [1, 4, 6, 1])
        operator(s.BuiltinOperator.RELU, [0], source)
    channels = 1
    if producer == "concat":
        merged = tensor("concat", [1, 4, 6, 2])
        options = s.ConcatenationOptionsT()
        options.axis = 3
        operator(s.BuiltinOperator.CONCATENATION, [source, source], merged,
                 s.BuiltinOptions.ConcatenationOptions, options)
        source, channels = merged, 2
    pads = np.array([[0, 0], [before, 2], [before, 2], [0, 0]], np.int32)
    padding = tensor("paddings", [4, 2], pads, s.TensorType.INT32)
    value = tensor("pad_value", [1], np.array([fill], np.float32))
    padded_shape = [1, 4 + before + 2, 6 + before + 2, channels]
    padded = tensor("padded", padded_shape)
    operator(s.BuiltinOperator.PADV2, [source, padding, value], padded, s.BuiltinOptions.PadV2Options, s.PadV2OptionsT())
    output_shape = [1, (padded_shape[1] - 3) // 2 + 1, (padded_shape[2] - 3) // 2 + 1, channels]
    output = tensor("pooled", output_shape)
    options = s.Pool2DOptionsT()
    options.padding = s.Padding.VALID
    options.filterHeight = options.filterWidth = 3
    options.strideH = options.strideW = 2
    operator(s.BuiltinOperator.MAX_POOL_2D, [padded], output, s.BuiltinOptions.Pool2DOptions, options)
    outputs = [output]
    if expose_padding:
        outputs.append(padded)
    if extra_consumer:
        extra = tensor("abs_padding", padded_shape)
        operator(s.BuiltinOperator.ABS, [padded], extra)
        outputs.append(extra)
    graph.outputs = np.array(outputs, np.int32)
    write_model(model, path)


def evaluate(path, values):
    interpreter = make_interpreter(path)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], values)
    interpreter.invoke()
    return [interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()]


@pytest.mark.parametrize("producer", ["relu", "concat"])
@pytest.mark.parametrize("fill", [-np.finfo(np.float32).max, -np.inf])
def test_nonnegative_pool_padding_is_exact(tmp_path, producer, fill):
    path = tmp_path / "pool.tflite"
    make_pad_pool(path, producer=producer, fill=fill)
    values = np.linspace(-2, 3, 24, dtype=np.float32).reshape(1, 4, 6, 1)
    expected = evaluate(path, values)
    changes = normalize_maxpool_padding(path)
    assert len(changes) == 1 and changes[0]["rewrite"] == "PADV2_to_PAD"
    for a, b in zip(expected, evaluate(path, values)):
        np.testing.assert_array_equal(a, b)
    _, s = modules()
    model = read_model(path)
    assert all(max(c.builtinCode, c.deprecatedBuiltinCode) != s.BuiltinOperator.PADV2
               for op in model.subgraphs[0].operators for c in [model.operatorCodes[op.opcodeIndex]])


@pytest.mark.parametrize("settings", [dict(producer="unknown"), dict(before=3),
                                       dict(expose_padding=True), dict(extra_consumer=True), dict(fill=1.0)])
def test_unsafe_padding_is_not_rewritten(tmp_path, settings):
    path = tmp_path / "unchanged.tflite"
    make_pad_pool(path, **settings)
    original = path.read_bytes()
    assert normalize_maxpool_padding(path) == []
    assert path.read_bytes() == original


@pytest.mark.parametrize('producer', ['relu', 'concat'])
def test_trailing_unused_maxpool_pad_folds_into_same(tmp_path, producer):
    path = tmp_path / 'pool.tflite'
    make_pad_pool(path, producer=producer)
    samples = np.linspace(-4, 6, 24, dtype=np.float32).reshape(1, 4, 6, 1)
    expected = evaluate(path, samples)
    normalize_maxpool_padding(path)
    changes = normalize_deployment_graph(path)
    assert {'rewrite': 'maxpool_pad_to_same'} in changes
    for a, b in zip(expected, evaluate(path, samples)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('settings', [dict(producer='unknown', fill=0),
                                      dict(before=1), dict(expose_padding=True),
                                      dict(extra_consumer=True)])
def test_pool_same_fold_preserves_fill_alignment_and_observable_edges(tmp_path, settings):
    path = tmp_path / 'pool.tflite'
    make_pad_pool(path, **settings)
    normalize_maxpool_padding(path)
    before = path.read_bytes()
    samples = np.linspace(-5, 2, 24, dtype=np.float32).reshape(1, 4, 6, 1)
    expected = evaluate(path, samples)
    changes = normalize_deployment_graph(path)
    assert all(c['rewrite'] != 'maxpool_pad_to_same' for c in changes)
    if changes:
        assert settings == dict(producer='unknown', fill=0)
        assert changes == [{'rewrite': 'zero_padv2_to_pad'}]
        for a, b in zip(expected, evaluate(path, samples)):
            np.testing.assert_array_equal(a, b)
    else:
        assert path.read_bytes() == before


def test_fold_channel_affine_into_private_fc_weights(demo):
    from copy import deepcopy
    _, s = modules()
    path = demo / 'float.tflite'
    model = read_model(path)
    graph = model.subgraphs[0]

    def tensor(values=None):
        t = deepcopy(graph.tensors[3])
        t.name = f'new_{len(graph.tensors)}'.encode()
        if values is not None:
            values = np.array(values, dtype='<f4')
            buf = s.BufferT()
            buf.data = np.frombuffer(values.tobytes(), dtype=np.uint8)
            model.buffers.append(buf)
            t.shape, t.buffer = np.array([3], dtype=np.int32), len(model.buffers) - 1
        graph.tensors.append(t)
        return len(graph.tensors) - 1

    def operation(builtin, source, constant, output, option_class, option_type):
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = builtin
        code.version = 1
        model.operatorCodes.append(code)
        op = s.OperatorT()
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.inputs, op.outputs = np.array([source, constant], np.int32), np.array([output], np.int32)
        op.builtinOptions, op.builtinOptionsType = option_class(), option_type
        return op

    raw, shifted = tensor(), tensor()
    mul = operation(s.BuiltinOperator.MUL, 3, tensor([1.5, -2, 0.25]), raw, s.MulOptionsT, s.BuiltinOptions.MulOptions)
    add = operation(s.BuiltinOperator.ADD, raw, tensor([0.2, 1, -0.8]), shifted, s.AddOptionsT, s.BuiltinOptions.AddOptions)
    graph.operators[-1].inputs[0] = shifted
    graph.operators = [graph.operators[0], mul, add, graph.operators[-1]]
    write_model(model, path)
    samples = np.array([[-2, 3, 0.25, -0.75]], dtype=np.float32)
    expected = evaluate(path, samples)
    changes = normalize_deployment_graph(path)
    assert any(c['rewrite'] == 'fold_weighted_channel_affine' for c in changes)
    assert len(read_model(path).subgraphs[0].operators) == 1
    for a, b in zip(expected, evaluate(path, samples)):
        np.testing.assert_allclose(a, b, atol=2e-6)


@pytest.mark.parametrize('constant', [True, False])
def test_constant_scalar_reshape_with_absent_shape_vector(demo, constant):
    _, s = modules()
    path = demo / 'float.tflite'
    model = read_model(path)
    graph = model.subgraphs[0]
    buf = s.BufferT()
    buf.data = np.frombuffer(np.float32(0.5).tobytes(), dtype=np.uint8)
    model.buffers.append(buf)
    start = len(graph.tensors)
    for name, shape, dtype, buffer in [
        (b'constant', [1], s.TensorType.FLOAT32, len(model.buffers) - 1 if constant else 0),
        (b'empty_shape', [0], s.TensorType.INT32, 0),
        (b'scalar', None, s.TensorType.FLOAT32, 0),
    ]:
        tensor = s.TensorT()
        tensor.name, tensor.shape, tensor.type, tensor.buffer = name, (None if shape is None else np.array(shape, np.int32)), dtype, buffer
        graph.tensors.append(tensor)
    code = s.OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.RESHAPE
    code.version = 1
    model.operatorCodes.append(code)
    op = s.OperatorT()
    op.opcodeIndex = len(model.operatorCodes) - 1
    op.inputs, op.outputs = np.array([start, start + 1], np.int32), np.array([start + 2], np.int32)
    op.builtinOptionsType, op.builtinOptions = s.BuiltinOptions.ReshapeOptions, s.ReshapeOptionsT()
    graph.operators.append(op)
    graph.outputs = np.append(graph.outputs, np.int32(start + 2))
    if not constant:
        graph.inputs = np.append(graph.inputs, np.int32(start))
    write_model(model, path)
    changes = normalize_deployment_graph(path)
    assert any(c['rewrite'] == ('fold_constant_reshape' if constant else 'scalar_reshape_options') for c in changes)
    result = read_model(path)
    scalar = next(t for t in result.subgraphs[0].tensors if t.name == b'scalar')
    if constant:
        assert scalar.buffer > 0
        np.testing.assert_array_equal(np.frombuffer(bytes(result.buffers[scalar.buffer].data), '<f4'), [0.5])
    instance = make_interpreter(path)
    for detail in instance.get_input_details():
        instance.set_tensor(detail['index'], np.full(detail['shape'], 0.5, np.float32))
    instance.invoke()
    scalar_output = next(d for d in instance.get_output_details() if d['name'] == 'scalar')
    assert instance.get_tensor(scalar_output['index']).item() == 0.5
    from onnx_edge_quantizer.structure import compare_structure
    assert compare_structure(path, path)['matched']
