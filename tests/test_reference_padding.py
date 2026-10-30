"""Check explicit reference padding semantics and exact layout rewrites on CPU."""
import numpy as np
import pytest

from onnx_edge_quantizer.channel_shuffle import align_channel_shuffle_partitions
from onnx_edge_quantizer.cli import main
from onnx_edge_quantizer.padding_helpers import deduplicate_float_padding
from onnx_edge_quantizer.reference_ops import align_pad_transposes
from onnx_edge_quantizer.reference_integerization import integerize_depthwise_reference
from onnx_edge_quantizer.runtime import inventory, read_model
from onnx_edge_quantizer.reference_padding import adopt_reference_pool_padding
from onnx_edge_quantizer.structure import compare_structure
from test_deployment_rewrites import Graph, run


def pool_model(path, kind, padding, *, fill=None):
    b = Graph()
    x = b.tensor('x', [1, 3, 3, 1])
    source = x
    if padding == 'explicit':
        pads = b.tensor('pads', [4, 2], [[0, 0], [1, 1], [1, 1], [0, 0]], 'INT32')
        source = b.tensor('padded', [1, 5, 5, 1])
        if fill is None:
            b.op('PAD', [x, pads], [source], 'PadOptions')
        else:
            value = b.tensor('value', [], [fill])
            b.op('PADV2', [x, pads, value], [source], 'PadV2Options')
    y = b.tensor('y', [1, 3, 3, 1])
    b.op(kind, [source], [y], 'Pool2DOptions', strideH=1, strideW=1,
         filterHeight=3, filterWidth=3,
         padding=b.s.Padding.SAME if padding == 'same' else b.s.Padding.VALID)
    return b.save(path, [x], [y])


@pytest.mark.parametrize('kind', ['AVERAGE_POOL_2D', 'MAX_POOL_2D'])
def test_reference_padding_is_an_explicit_semantics_change(tmp_path, kind):
    if kind == 'AVERAGE_POOL_2D':
        source = pool_model(tmp_path / 'source.tflite', kind, 'same')
        values = np.ones((1, 3, 3, 1), np.float32)
    else:
        source = pool_model(tmp_path / 'source.tflite', kind, 'explicit', fill=-np.inf)
        values = -np.ones((1, 3, 3, 1), np.float32)
    reference = pool_model(tmp_path / 'reference.tflite', kind, 'explicit')
    original = run(source, [values])[0]
    desired = run(reference, [values])[0]
    assert not np.array_equal(original, desired)
    changes = adopt_reference_pool_padding(source, reference)
    assert len(changes) == 1 and changes[0]['semantics_changed']
    np.testing.assert_allclose(run(source, [values])[0], desired)
    assert compare_structure(source, reference)['matched']


def test_baseline_padding_cannot_disable_default_parity_implicitly(demo):
    output = demo / 'policy.tflite'
    assert main([str(demo / 'demo.onnx'), '-o', str(output), '--structure-reference', str(demo / 'float.tflite'),
                 '--reference-padding-policy', 'baseline']) == 2
    assert not output.exists()


@pytest.mark.parametrize('wrong_indices', [False, True])
def test_partial_channel_gathers_equal_complete_reference_shuffle_slices(tmp_path, wrong_indices):
    b = Graph()
    x = b.tensor('x', [1, 2, 3, 4])
    outputs = []
    for n, indices in enumerate(([0, 3], [3, 0]) if wrong_indices else ([0, 2], [1, 3])):
        index = b.tensor(f'indices{n}', [2], indices, 'INT32')
        out = b.tensor(f'out{n}', [1, 2, 3, 2])
        b.op('GATHER', [x, index], [out], 'GatherOptions', axis=3)
        outputs.append(out)
    source = b.save(tmp_path / 'gathers.tflite', [x], outputs)
    r = Graph()
    rx = r.tensor('x', [1, 2, 3, 4])
    current = rx
    for n, (kind, control, shape) in enumerate([
        ('TRANSPOSE', [0, 3, 1, 2], [1, 4, 2, 3]),
        ('RESHAPE', [1, 2, 2, 2, 3], [1, 2, 2, 2, 3]),
        ('TRANSPOSE', [0, 2, 1, 3, 4], [1, 2, 2, 2, 3]),
        ('RESHAPE', [1, 4, 2, 3], [1, 4, 2, 3]),
        ('TRANSPOSE', [0, 2, 3, 1], [1, 2, 3, 4]),
    ]):
        c = r.tensor(f'control{n}', [len(control)], control, 'INT32')
        y = r.tensor(f'view{n}', shape)
        r.op(kind, [current, c], [y], 'TransposeOptions' if kind == 'TRANSPOSE' else 'ReshapeOptions')
        current = y
    rout = []
    for n in range(2):
        begin = r.tensor(f'begin{n}', [4], [0, 0, 0, n * 2], 'INT32')
        size = r.tensor(f'size{n}', [4], [1, 2, 3, 2], 'INT32')
        out = r.tensor(f'out{n}', [1, 2, 3, 2])
        r.op('SLICE', [current, begin, size], [out])
        rout.append(out)
    reference = r.save(tmp_path / 'reference.tflite', [rx], rout)
    values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    original = run(source, [values])
    changes = align_channel_shuffle_partitions(source, reference)
    assert bool(changes) is (not wrong_indices)
    for actual, expected in zip(run(source, [values]), original):
        np.testing.assert_array_equal(actual, expected)
    assert compare_structure(source, reference)['matched'] is (not wrong_indices)


def test_transpose_can_move_before_padding_without_changing_values(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 2, 3])
    pads = b.tensor('pads', [4, 2], [[0, 0], [0, 0], [1, 2], [2, 1]], 'INT32')
    padded = b.tensor('padded', [1, 2, 5, 6])
    axes = b.tensor('axes', [4], [0, 2, 3, 1], 'INT32')
    out = b.tensor('out', [1, 5, 6, 2])
    b.op('PAD', [x, pads], [padded], 'PadOptions')
    b.op('TRANSPOSE', [padded, axes], [out], 'TransposeOptions')
    source = b.save(tmp_path / 'pad_transpose.tflite', [x], [out])
    r = Graph()
    rx = r.tensor('x', [1, 2, 2, 3])
    rp = r.tensor('axes', [4], [0, 2, 3, 1], 'INT32')
    rt = r.tensor('transposed', [1, 2, 3, 2])
    rc = r.tensor('pads', [4, 2], [[0, 0], [1, 2], [2, 1], [0, 0]], 'INT32')
    ry = r.tensor('out', [1, 5, 6, 2])
    r.op('TRANSPOSE', [rx, rp], [rt], 'TransposeOptions')
    r.op('PAD', [rt, rc], [ry], 'PadOptions')
    reference = r.save(tmp_path / 'ref.tflite', [rx], [ry])
    values = np.linspace(-2, 3, 12, dtype=np.float32).reshape(1, 2, 2, 3)
    expected = run(source, [values])[0]
    assert align_pad_transposes(source, reference)
    np.testing.assert_array_equal(run(source, [values])[0], expected)
    assert compare_structure(source, reference)['matched']


def test_duplicate_padding_shares_data_but_different_padding_does_not(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 2, 2, 1])
    pads = b.tensor('pads', [4, 2], [[0, 0], [1, 1], [1, 1], [0, 0]], 'INT32')
    left, right = b.tensor('left', [1, 4, 4, 1]), b.tensor('right', [1, 4, 4, 1])
    out = b.tensor('out', [1, 4, 4, 1])
    b.op('PAD', [x, pads], [left], 'PadOptions')
    b.op('PAD', [x, pads], [right], 'PadOptions')
    b.op('ADD', [left, right], [out], 'AddOptions')
    source = b.save(tmp_path / 'duplicate.tflite', [x], [out])
    values = np.arange(4, dtype=np.float32).reshape(1, 2, 2, 1)
    expected = run(source, [values])[0]
    assert deduplicate_float_padding(source)[0]['count'] == 1
    np.testing.assert_array_equal(run(source, [values])[0], expected)


@pytest.mark.parametrize('dtype,zero', [('INT8', -3), ('UINT8', 128)])
def test_float_depthwise_reference_becomes_a_separate_runnable_integer_copy(tmp_path, dtype, zero):
    b = Graph()
    x = b.tensor('input', [1, 2, 2, 1], dtype=dtype, quant=(0.25, zero))
    f = b.tensor('float', [1, 2, 2, 1])
    pads = b.tensor('pads', [4, 2], [[0, 0], [1, 1], [1, 1], [0, 0]], 'INT32')
    padded = b.tensor('padded', [1, 4, 4, 1])
    w = b.tensor('weight', [1, 3, 3, 1], np.arange(-4, 5, dtype=np.float32) / 16)
    bias = b.tensor('bias', [1], [0.25])
    y = b.tensor('float_conv', [1, 2, 2, 1])
    q = b.tensor('output', [1, 2, 2, 1], dtype=dtype, quant=(0.125, zero))
    b.op('DEQUANTIZE', [x], [f], 'DequantizeOptions', version=2)
    b.op('PAD', [f, pads], [padded], 'PadOptions')
    b.op('DEPTHWISE_CONV_2D', [padded, w, bias], [y], 'DepthwiseConv2DOptions',
         padding=b.s.Padding.VALID, strideH=1, strideW=1, depthMultiplier=1,
         dilationHFactor=1, dilationWFactor=1)
    b.op('QUANTIZE', [y], [q], 'QuantizeOptions', version=2)
    source = b.save(tmp_path / 'mixed.tflite', [x], [q])
    original = source.read_bytes()
    values = (np.arange(4) + zero - 2).astype(np.int8 if dtype == 'INT8' else np.uint8).reshape(1, 2, 2, 1)
    expected = run(source, [values])[0]
    output = tmp_path / 'integer.tflite'
    report = integerize_depthwise_reference(source, output)
    assert source.read_bytes() == original
    assert report['stock_litert_prepare'] == 'passed'
    assert not inventory(output)['float_compute_ops']
    assert len(read_model(output).subgraphs[0].operators) == 2
    np.testing.assert_allclose(run(output, [values])[0], expected, atol=1)
    with pytest.raises(ValueError, match='already exists'):
        integerize_depthwise_reference(source, output)
