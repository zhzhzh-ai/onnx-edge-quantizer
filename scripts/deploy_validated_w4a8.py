"""Reuse a validated pair manifest and convert only missing W4A8 artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onnx_edge_quantizer.input_specialization import specialize_to_reference
from onnx_edge_quantizer.runtime import check_native_precision, inventory
from onnx_edge_quantizer.structure import compare_structure
from scripts.export_all_w4a8 import execute, sha256
from scripts.validate_model_zoo import dynamic_input_flags, InputShapeMismatch


def verify(path, reference, report=None):
    report = report or {}
    preferred = report.get('weight_granularity', 'per-channel')
    error = None
    for granularity in dict.fromkeys([preferred, 'per-channel', 'per-tensor']):
        try:
            precision = check_native_precision(path, 'int4', 'int8', granularity)
            break
        except ValueError as caught:
            error = caught
    else:
        raise error
    structure = compare_structure(path, reference)
    if not structure['matched']:
        raise ValueError(f"Artifact does not match reference: {structure['differences'][:3]}")
    actual = inventory(path)
    if actual['float_compute_ops']:
        raise ValueError('Artifact retains floating-point compute operators')
    return precision, structure, actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validated-summary', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--specialize-to-reference', action='store_true')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args(argv)
    previous = json.loads(args.validated_summary.read_text())
    reference_root, target, work_root = args.reference_root.absolute(), args.output_dir.resolve(), args.work_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    names = Counter(record['model'] for record in previous['models'])
    records = []
    for old in previous['models']:
        start = time.monotonic()
        source, reference = Path(old['onnx']), Path(old['reference'])
        relative = reference.absolute().relative_to(reference_root)
        output = target / relative.parent / (old['model'] + '_w4a8.tflite')
        existing_flat = target / output.name
        if names[old['model']] == 1 and existing_flat.is_file():
            output = existing_flat
        record = {'model': old['model'], 'onnx': str(source), 'reference': str(reference.resolve()),
                  'output': str(output), 'status': 'failed', 'previous_status': old['status']}
        try:
            report_file = output.with_suffix('.quantization.json')
            report = json.loads(report_file.read_text()) if report_file.is_file() else {}
            if output.is_file():
                precision, structure, actual = verify(output, reference, report)
                action = 'reused_existing_target'
            elif old['status'] in ('passed', 'structure_passed') and Path(old['output']).is_file():
                original = Path(old['output']).resolve()
                report = json.loads(original.with_suffix('.quantization.json').read_text())
                precision, _, _ = verify(original, reference, report)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, output)
                precision, structure, actual = verify(output, reference, report)
                action = 'reused_validated_output'
                record['reused_from'] = str(original)
            else:
                model_path = source
                try:
                    shape_flags = dynamic_input_flags(source, reference, check_fixed=True)
                except InputShapeMismatch:
                    if not args.specialize_to_reference:
                        raise
                    model_path = work_root / 'specialized' / relative.parent / (old['model'] + '.onnx')
                    specialization_file = model_path.with_suffix('.specialization.json')
                    if model_path.is_file() and specialization_file.is_file():
                        specialization = json.loads(specialization_file.read_text())
                        if specialization['original_source_sha256'] != sha256(source):
                            raise ValueError('The original source changed since input specialization')
                        dynamic_input_flags(model_path, reference, check_fixed=True)
                    else:
                        specialization = specialize_to_reference(source, reference, model_path)
                    record['input_specialization'] = specialization
                    shape_flags = []
                work = work_root / (old['model'] + '_work_' + str(time.time_ns()))
                log = work_root / (old['model'] + '.log')
                command = ['bash', str(ROOT / 'run_w4a8.sh'), str(model_path), '-o', str(output),
                           '--weight-granularity', 'per-channel', '--structure-reference', str(reference),
                           '--validation-mode', 'structure', '--reference-padding-policy', 'baseline',
                           '--keep-workdir', str(work), *shape_flags]
                record.update(command=command, log=str(log), workdir=str(work))
                if execute(command, log, args.timeout):
                    raise ValueError(log.read_text(errors='replace')[-4000:])
                report = json.loads(report_file.read_text())
                precision, structure, actual = verify(output, reference, report)
                action = 'converted_missing_model'
            if old['status'] == 'skipped_input_shape':
                declared = report.get('input_onnx')
                specialization_file = Path(declared).with_suffix('.specialization.json') if declared else None
                if specialization_file is not None and specialization_file.is_file():
                    record['input_specialization'] = json.loads(specialization_file.read_text())
            report.update(output_tflite=str(output), validation_mode='structure',
                          runtime_validation='not_requested', onnx_parity_validation='not_requested',
                          weight_granularity=precision['weight_granularity'],
                          structure_comparison=structure, output_inventory=actual)
            # Older pre-existing files may have no exporter report; describe
            # only checks actually performed, rather than invent calibration.
            report.setdefault('input_onnx', str(source))
            report.setdefault('mode', 'native')
            report.setdefault('requested', {'weight_dtype': 'int4', 'activation_dtype': 'int8',
                                           'weight_granularity': precision['weight_granularity']})
            report.setdefault('backend_details', {})['precision_verification'] = precision
            report['deployment'] = {'action': action, 'validated_summary': str(args.validated_summary.resolve()),
                                    'original_onnx': str(source), 'input_specialization': record.get('input_specialization')}
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            record.update(status='structure_passed', action=action, precision=precision, structure=structure,
                          output_sha256=sha256(output), output_bytes=output.stat().st_size,
                          runtime_requirements=report['backend_details'].get('runtime_requirements', []))
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            record['error'] = str(error)
        record['seconds'] = round(time.monotonic() - start, 2)
        records.append(record)
        summary = {'total': len(previous['models']), 'completed': len(records),
                   'structure_passed': sum(r['status'] == 'structure_passed' for r in records),
                   'failed': sum(r['status'] == 'failed' for r in records),
                   'validation_scope': 'structure_and_actual_W4A8_precision', 'output_dir': str(target),
                   'models': records}
        (target / 'validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[{len(records)}/{summary['total']}] {record['status']}: {old['model']} ({record.get('action', '')})", flush=True)
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
