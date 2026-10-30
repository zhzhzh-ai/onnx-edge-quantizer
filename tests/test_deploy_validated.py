"""Fixed-input specialization, strided layouts and reuse-only deployment."""
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import pytest

from onnx_edge_quantizer.cli import main
from onnx_edge_quantizer.input_specialization import specialize_to_reference
from onnx_edge_quantizer.layout_normalization import normalize_channel_layouts
from onnx_edge_quantizer.padding_helpers import deduplicate_static_slices
from onnx_edge_quantizer.reference_interfaces import align_terminal_output_layout
from onnx_edge_quantizer.runtime import read_model
from onnx_edge_quantizer.structure import compare_structure
from scripts import deploy_validated_w4a8
from test_deployment_rewrites import Graph, metadata, run


def test_fixed_shape_specialization_preserves_source_nodes_and_weights(tmp_path):
    graph = helper.make_graph([helper.make_node('Conv', ['x', 'weight'], ['y'])], 'fixed',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])],
        [numpy_helper.from_array(np.array([[[[2.0]]]], np.float32), 'weight')])
    source = tmp_path / 'source.onnx'
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9), source)
    original = source.read_bytes()
    b = Graph()
    x, y = b.tensor('x', [1, 2, 2, 1]), b.tensor('y', [1, 2, 2, 1])
    b.op('RELU', [x], [y])
    reference = b.save(tmp_path / 'reference.tflite', [x], [y])
    destination = tmp_path / 'specialized.onnx'
    report = specialize_to_reference(source, reference, destination)
    assert source.read_bytes() == original
    assert report['original_input_shape'] == [1, 1, 4, 4]
    assert report['specialized_input_shape'] == [1, 1, 2, 2]
    session = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
    values = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    np.testing.assert_array_equal(session.run(None, {'x': values})[0], values * 2)


@pytest.mark.parametrize('negative', [False, True])
def test_strided_slices_propagate_channel_layout_and_axis_masks(tmp_path, negative):
    b = Graph()
    x = b.tensor('x', [1, 4, 6, 2])
    p = b.tensor('perm', [4], [0, 3, 1, 2], 'INT32')
    nchw = b.tensor('nchw', [1, 2, 4, 6])
    b.op('TRANSPOSE', [x, p], [nchw], 'TransposeOptions')
    begin = b.tensor('begin', [4], [0, 0, 0, 5 if negative else 1], 'INT32')
    end = b.tensor('end', [4], [0, 0, 4, 0 if negative else 6], 'INT32')
    stride = b.tensor('strides', [4], [1, 1, 2, -2 if negative else 2], 'INT32')
    cropped = b.tensor('cropped', [1, 2, 2, 3])
    b.op('STRIDED_SLICE', [nchw, begin, end, stride], [cropped], 'StridedSliceOptions',
         beginMask=3, endMask=11 if negative else 3)
    inverse = b.tensor('inverse', [4], [0, 2, 3, 1], 'INT32')
    output = b.tensor('output', [1, 2, 3, 2])
    b.op('TRANSPOSE', [cropped, inverse], [output], 'TransposeOptions')
    path = b.save(tmp_path / 'strided.tflite', [x], [output])
    metadata(path)
    values = np.arange(48, dtype=np.float32).reshape(1, 4, 6, 2)
    expected = run(path, [values])[0]
    assert normalize_channel_layouts(path)
    np.testing.assert_array_equal(run(path, [values])[0], expected)
    assert len(read_model(path).subgraphs[0].operators) == 1


def test_identical_strided_slice_branches_share_their_result(tmp_path):
    b = Graph()
    x = b.tensor('x', [1, 4, 4, 1])
    begin = b.tensor('begin', [4], [0, 0, 0, 0], 'INT32')
    end = b.tensor('end', [4], [1, 4, 4, 1], 'INT32')
    stride = b.tensor('stride', [4], [1, 2, 2, 1], 'INT32')
    a, c, y = [b.tensor(name, [1, 2, 2, 1]) for name in ('a', 'c', 'y')]
    for output in (a, c):
        b.op('STRIDED_SLICE', [x, begin, end, stride], [output], 'StridedSliceOptions')
    b.op('ADD', [a, c], [y], 'AddOptions')
    path = b.save(tmp_path / 'duplicates.tflite', [x], [y])
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    expected = run(path, [values])[0]
    assert deduplicate_static_slices(path)[0]['count'] == 1
    np.testing.assert_array_equal(run(path, [values])[0], expected)


def test_rank_three_terminal_transpose_has_a_three_element_control(tmp_path):
    b = Graph()
    x, y = b.tensor('x', [1, 2, 3]), b.tensor('output', [1, 2, 3])
    b.op('RELU', [x], [y])
    path = b.save(tmp_path / 'source.tflite', [x], [y])
    p = b.tensor('perm', [3], [0, 2, 1], 'INT32')
    z = b.tensor('reference_out', [1, 3, 2])
    b.op('TRANSPOSE', [y, p], [z], 'TransposeOptions')
    reference = b.save(tmp_path / 'reference.tflite', [x], [z])
    model = helper.make_model(helper.make_graph([], 'public', [],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 3])]))
    layouts = {}
    assert align_terminal_output_layout(path, reference, model, layouts)
    assert compare_structure(path, reference)['matched']
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    actual = run(path, [values])[0]
    np.testing.assert_array_equal(actual.transpose(layouts['output']['tflite_to_onnx_permutation']), values)


def test_deployment_reuses_outputs_without_invoking_conversion(demo, monkeypatch):
    refs = demo / 'refs'
    refs.mkdir()
    reference = refs / 'demo_uint8.tflite'
    common = [str(demo / 'demo.onnx'), '--float-tflite', str(demo / 'float.tflite')]
    assert main([*common, '-o', str(reference)]) == 0
    output = demo / 'old.tflite'
    assert main([*common, '-o', str(output), '--weight-dtype', 'int4']) == 0
    old = output.read_bytes()
    summary = demo / 'validated.json'
    summary.write_text(json.dumps({'models': [{'model': 'demo', 'onnx': str(demo / 'demo.onnx'),
        'reference': str(reference), 'output': str(output), 'status': 'structure_passed'}]}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Previously converted models must not be converted again')
    monkeypatch.setattr(deploy_validated_w4a8, 'execute', forbidden)
    args = ['--validated-summary', str(summary), '--reference-root', str(refs),
            '--output-dir', str(demo / 'deployed'), '--work-dir', str(demo / 'work')]
    assert deploy_validated_w4a8.main(args) == 0
    copied = demo / 'deployed/demo_w4a8.tflite'
    assert copied.read_bytes() == old
    before = copied.stat().st_mtime_ns
    assert deploy_validated_w4a8.main(args) == 0
    assert copied.stat().st_mtime_ns == before


def test_canceled_batch_work_root_rejects_queued_launcher(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / '.w4a8-cancel').touch()
    launcher = Path(__file__).resolve().parents[1] / 'run_w4a8.sh'
    result = subprocess.run(['bash', str(launcher), str(tmp_path / 'unused.onnx'),
        '--keep-workdir', str(work / 'queued_child')], capture_output=True, text=True)
    assert result.returncode == 130
    assert not (work / 'queued_child').exists()
