"""Export every ONNX in multiple roots through run_w4a8.sh with honest statuses."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_model_zoo import dynamic_input_flags, InputShapeMismatch


def discover_sources(roots):
    """Keep distinct source paths and unambiguous output namespaces."""
    labels = [root.name if root.name != 'onnx' else root.parent.name for root in roots]
    if len(set(labels)) != len(labels):
        raise ValueError('ONNX roots must have distinct directory names for output namespaces')
    records, seen = [], set()
    for root, label in zip(roots, labels):
        if not root.is_dir():
            raise ValueError(f'ONNX root is not a directory: {root}')
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() != '.onnx' or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            relative = path.relative_to(root)
            records.append({'source_root': str(root), 'source': str(path.resolve()),
                            'source_group': label,
                            'relative_output': str(Path(label) / relative.parent / (path.stem + '_w4a8.tflite'))})
    return records


def reference_index(roots):
    index = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'Reference root is not a directory: {root}')
        for path in sorted(root.rglob('*.tflite')):
            match = re.fullmatch(r'(.+)_(?:uint8|int8)(?:_.*)?', path.stem, re.IGNORECASE)
            if match:
                index.setdefault(match.group(1).casefold(), []).append((root, path))
    return index


def select_reference(source, source_root, index):
    candidates = index.get(source.stem.casefold(), [])
    if not candidates:
        return None, []
    scored = []
    for root, path in candidates:
        # A reference co-located under a source root takes precedence. Within
        # model_zoo, retain the explicit onnx1/onnx2/... deployment grouping.
        local = int(root.resolve() == source_root.resolve())
        a, b = source.relative_to(source_root).parent.parts, path.relative_to(root).parent.parts
        common = 0
        for left, right in zip(a, b):
            if left != right:
                break
            common += 1
        exact = int(path.stem.casefold() == source.stem.casefold() + '_uint8')
        scored.append(((local, common, exact), path))
    best = max(score for score, _ in scored)
    selected = sorted({path for score, path in scored if score == best})
    if len(selected) > 1:
        from onnx_edge_quantizer.structure import graph_contract
        contract = graph_contract(selected[0])
        if any(graph_contract(path) != contract for path in selected[1:]):
            raise ValueError(f'Ambiguous, structurally different references: {selected}')
    return selected[0], [str(path) for path in selected]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def execute(command, log, timeout):
    """Timeout the entire conversion process group, including onnx2tf workers."""
    env = os.environ.copy()
    env['W4A8_PYTHON'] = sys.executable
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-root', type=Path, action='append', required=True)
    parser.add_argument('--reference-root', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--weight-granularity', choices=['per-tensor', 'per-channel'], default='per-channel')
    parser.add_argument('--reference-padding-policy', choices=['preserve', 'baseline'], default='preserve')
    parser.add_argument('--integerize-reference-islands', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse successful outputs only when source and output SHA256 still match')
    parser.add_argument('--fallback-without-reference', action='store_true',
                        help='After a reference structure mismatch, try full ONNX/runtime validation without that reference')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout < 1:
        parser.error('workers and timeout must be positive')
    try:
        models = discover_sources([p.resolve() for p in args.onnx_root])
        references = reference_index([p.resolve() for p in args.reference_root])
    except ValueError as error:
        parser.error(str(error))
    if not models:
        parser.error('No ONNX files found')
    output_root, work_root = args.output_dir.resolve(), args.work_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    run_id = str(time.time_ns())
    previous = {}
    summary_path = output_root / 'export_summary.json'
    if args.resume and summary_path.is_file():
        previous = {item['source']: item for item in json.loads(summary_path.read_text())['models']}

    def convert(item):
        source = Path(item['source'])
        relative = Path(item['relative_output'])
        output = output_root / relative
        folder = work_root / run_id / relative.parent
        folder.mkdir(parents=True, exist_ok=True)
        work = folder / (source.stem + '_work')
        log = folder / (source.stem + '.log')
        record = {**item, 'output': str(output), 'status': 'planned', 'log': str(log), 'workdir': str(work)}
        start = time.monotonic()
        try:
            old = previous.get(str(source))
            if (args.resume and old and old['status'].startswith('generated_') and output.is_file()
                    and old.get('output_sha256') == sha256(output)
                    and old.get('source_sha256') == sha256(source)):
                record = {**old, 'reused_verified_output': True}
                return record
            if output.exists() or output.with_suffix('.quantization.json').exists():
                if not args.overwrite:
                    record['status'] = 'skipped_existing_output'
                    return record
            reference, alternatives = select_reference(source, Path(item['source_root']), references)
            flags = []
            if reference is not None:
                record['original_reference'] = str(reference)
                record['reference_candidates'] = alternatives
                flags.extend(dynamic_input_flags(source, reference, check_fixed=True))
                shape_flags = flags.copy()
                if args.integerize_reference_islands and not args.dry_run:
                    from onnx_edge_quantizer.runtime import inventory
                    if any(op['op'] == 'DEPTHWISE_CONV_2D' for op in inventory(reference)['float_compute_ops']):
                        from onnx_edge_quantizer.reference_integerization import integerize_depthwise_reference
                        corrected = folder / (source.stem + '_integer_reference.tflite')
                        record['reference_integerization'] = integerize_depthwise_reference(reference, corrected)
                        reference = corrected
                record['reference'] = str(reference)
                record['validation_scope'] = 'structure_and_integer_precision'
                flags.extend(['--structure-reference', str(reference), '--validation-mode', 'structure',
                              '--reference-padding-policy', args.reference_padding_policy])
            else:
                # No input sizes or external graph are guessed. Fixed-shape
                # models still get full ONNX/FP32/runtime validation.
                from onnx_edge_quantizer.data import input_specs
                input_specs(source, {})
                record['reference'] = None
                record['validation_scope'] = 'onnx_fp32_parity_and_native_runtime_without_w8_reference'
            command = ['bash', str(ROOT / 'run_w4a8.sh'), str(source), '-o', str(output),
                       '--weight-granularity', args.weight_granularity,
                       '--keep-workdir', str(work), *flags]
            if args.overwrite:
                command.append('--overwrite')
            record['command'] = command
            if args.dry_run:
                return record
            record['exit_code'] = execute(command, log, args.timeout)
            fallback = False
            if (record['exit_code'] != 0 and reference is not None and args.fallback_without_reference
                    and 'Deployment structure mismatch' in log.read_text(errors='replace')):
                comparison = work / 'structure_comparison.json'
                record['rejected_reference_comparison'] = json.loads(comparison.read_text()) if comparison.is_file() else None
                record['structure_attempt_log'] = str(log)
                work = folder / (source.stem + '_full_validation_work')
                log = folder / (source.stem + '_full_validation.log')
                command = ['bash', str(ROOT / 'run_w4a8.sh'), str(source), '-o', str(output),
                           '--weight-granularity', args.weight_granularity,
                           '--keep-workdir', str(work), *shape_flags]
                if args.overwrite:
                    command.append('--overwrite')
                record.update(command=command, log=str(log), workdir=str(work),
                              validation_scope='onnx_fp32_parity_and_native_runtime_reference_structure_mismatched')
                record['exit_code'] = execute(command, log, args.timeout)
                fallback = True
            if record['exit_code'] != 0:
                record['status'] = 'failed_conversion'
                record['error_tail'] = log.read_text(errors='replace')[-4000:]
                return record
            report = json.loads(output.with_suffix('.quantization.json').read_text())
            if report['requested']['weight_dtype'] != 'int4' or report['requested']['activation_dtype'] != 'int8':
                raise ValueError('The output report does not describe W4A8')
            record['status'] = ('generated_full_verified_reference_mismatch' if fallback else
                                'generated_structure_verified' if reference else 'generated_full_verified')
            record['precision'] = report['backend_details']['precision_verification']
            record['structure_comparison'] = report['structure_comparison']
            record['reference_semantics_changes'] = report.get('reference_semantics_changes', [])
            record['runtime_validation'] = report['runtime_validation']
            record['runtime_requirements'] = report['backend_details'].get('runtime_requirements', [])
            record['output_bytes'] = output.stat().st_size
            record['output_sha256'] = sha256(output)
            record['source_sha256'] = sha256(source)
        except InputShapeMismatch as error:
            record.update(status='skipped_input_shape', reason=str(error))
        except subprocess.TimeoutExpired:
            record.update(status='timeout', reason=f'Conversion exceeded {args.timeout} seconds; process group stopped')
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            message = str(error)
            status = 'skipped_dynamic_shape' if 'dynamic/invalid shape' in message else 'failed_preflight'
            record.update(status=status, reason=message)
        finally:
            record['seconds'] = round(time.monotonic() - start, 2)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(convert, model) for model in models]
        for future in as_completed(tasks):
            record = future.result()
            records.append(record)
            counts = Counter(item['status'] for item in records)
            summary = {'source_roots': [str(p.resolve()) for p in args.onnx_root],
                       'output_dir': str(output_root), 'work_dir': str(work_root),
                       'total': len(models), 'completed': len(records),
                       'generated': sum(count for status, count in counts.items() if status.startswith('generated_')),
                       'status_counts': dict(counts), 'dry_run': args.dry_run,
                       'models': sorted(records, key=lambda r: r['source'])}
            staged = output_root / '.export_summary.json.tmp'
            staged.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(staged, output_root / 'export_summary.json')
            print(f"[{len(records)}/{len(models)}] {record['status']}: {Path(record['source']).name}", flush=True)
    return 0 if args.dry_run or all(item['status'].startswith('generated_') for item in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
