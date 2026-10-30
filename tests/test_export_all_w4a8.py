"""Batch exports preserve source namespaces, launcher use and failure statuses."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from examples.make_demo import make_demo
from scripts.export_all_w4a8 import discover_sources, reference_index, select_reference


def test_source_roots_and_case_insensitive_extensions_do_not_overwrite_names(tmp_path):
    first, second = tmp_path / 'aicv', tmp_path / 'model_zoo' / 'onnx'
    first.mkdir()
    second.mkdir(parents=True)
    (first / 'same.onnx').write_bytes(b'first')
    (second / 'same.ONNX').write_bytes(b'second')
    records = discover_sources([first, second])
    assert {r['relative_output'] for r in records} == {'aicv/same_w4a8.tflite', 'model_zoo/same_w4a8.tflite'}


def test_reference_selection_respects_model_zoo_group_and_local_directory(tmp_path):
    root, refs = tmp_path / 'sources', tmp_path / 'references'
    source = root / 'onnx2' / 'same.onnx'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'onnx')
    for group in ('onnx1', 'onnx2'):
        folder = refs / group
        folder.mkdir(parents=True)
        (folder / 'same_uint8.tflite').write_bytes(b'not loaded by this selection')
    selected, _ = select_reference(source, root, reference_index([refs]))
    assert selected == refs / 'onnx2' / 'same_uint8.tflite'
    local = source.with_name('same_uint8_modify.tflite')
    local.write_bytes(b'local')
    selected, _ = select_reference(source, root, reference_index([refs, root]))
    assert selected == local


@pytest.mark.skipif(importlib.util.find_spec('onnx2tf') is None
                    or importlib.util.find_spec('ai_edge_quantizer') is None,
                    reason='Conversion and quantization backends are required')
def test_shell_batch_launcher_exports_both_roots_and_keeps_existing_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    first, second = tmp_path / 'first', tmp_path / 'second'
    for folder in (first, second):
        folder.mkdir()
        make_demo(folder)
    target = tmp_path / 'target'
    target.mkdir()
    old = target / 'existing.tflite'
    old.write_bytes(b'preserve existing user output')
    env = os.environ.copy()
    env['W4A8_PYTHON'] = sys.executable
    result = subprocess.run([
        'bash', str(root / 'run_w4a8.sh'), '--batch',
        '--onnx-root', str(first), '--onnx-root', str(second),
        '--output-dir', str(target), '--work-dir', str(tmp_path / 'work'),
        '--workers', '1', '--timeout', '60',
    ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((target / 'export_summary.json').read_text())
    assert summary['total'] == summary['completed'] == summary['generated'] == 2
    assert old.read_bytes() == b'preserve existing user output'
    for record in summary['models']:
        assert record['command'][:2] == ['bash', str(root / 'run_w4a8.sh')]
        assert record['status'] == 'generated_full_verified'
        assert record['reference'] is None
        assert record['runtime_validation'] == 'passed'
        assert Path(record['output']).is_file()
        assert len(record['output_sha256']) == len(record['source_sha256']) == 64
