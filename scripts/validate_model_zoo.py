"""Convert paired ONNX/UINT8 models and require matching W4A8 deployment graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dynamic_input_flags(onnx_path, reference):
    """Use explicit reference interface plus the deployment axis convention."""
    import onnx
    from onnx_edge_quantizer.runtime import read_model
    model = onnx.load(str(onnx_path), load_external_data=False)
    constants = {t.name for t in model.graph.initializer}
    inputs = [t for t in model.graph.input if t.name not in constants]
    graph = read_model(reference).subgraphs[0]
    flags = []
    for i, tensor in enumerate(inputs):
        shape = [int(d.dim_value) if d.HasField('dim_value') else None
                 for d in tensor.type.tensor_type.shape.dim]
        if all(d is not None and d > 0 for d in shape):
            continue
        if len(inputs) != 1 or len(graph.inputs) != 1:
            raise ValueError('Dynamic multi-input models require explicit shape/name mappings')
        deployment = graph.tensors[int(graph.inputs[i])].shape.tolist()
        expected = [deployment[j] for j in [0, 3, 1, 2]] if len(shape) == 4 else deployment
        if len(shape) != len(expected) or any(d is not None and d > 0 and d != e for d, e in zip(shape, expected)):
            raise ValueError('Reference input shape is incompatible with the ONNX/deployment axis contract')
        flags.extend(['--input-shape', tensor.name + '=' + ','.join(map(str, expected))])
    return flags


def discover(onnx_root, reference_root):
    sources = {}
    for path in sorted(onnx_root.rglob('*.onnx')):
        sources.setdefault(path.stem.casefold(), []).append(path)
    pairs = []
    for reference in sorted(reference_root.rglob('*_uint8.tflite')):
        name = reference.stem.removesuffix('_uint8')
        candidates = sources.get(name.casefold(), [])
        if not candidates:
            continue
        relative = reference.relative_to(reference_root)
        same_group = [p for p in candidates if p.relative_to(onnx_root).parts[0] == relative.parts[0]]
        candidates = same_group or candidates
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous ONNX source for {reference}: {candidates}')
        pairs.append((name, candidates[0], reference, relative))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', help='Optional exact model names, case insensitive')
    parser.add_argument('--failed-from', type=Path,
                        help='Retry only failed reference paths from a previous validation_summary.json')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    pairs = discover(args.onnx_root.resolve(), args.reference_root.resolve())
    if args.failed_from:
        previous = json.loads(args.failed_from.read_text(encoding='utf-8'))
        failed = {str(Path(item['reference']).resolve()) for item in previous['models']
                  if item['status'] != 'passed'}
        pairs = [pair for pair in pairs if str(pair[2].resolve()) in failed]
    if args.models:
        selected = {n.casefold() for n in args.models}
        pairs = [p for p in pairs if p[0].casefold() in selected]
        missing = selected - {p[0].casefold() for p in pairs}
        if missing:
            parser.error(f'No paired model for {sorted(missing)}')
    if args.workers < 1 or args.timeout < 1 or not pairs:
        parser.error('Require paired models, positive workers and timeout')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run(pair):
        name, onnx, reference, relative = pair
        folder = args.output_dir / relative.parent
        output = folder / f'{name}_w4a8.tflite'
        record = {'model': name, 'onnx': str(onnx), 'reference': str(reference),
                  'output': str(output), 'status': 'planned'}
        if args.dry_run:
            return record
        folder.mkdir(parents=True, exist_ok=True)
        work = folder / f'{name}_work_{time.time_ns()}'
        log = folder / f'{name}.log'
        command = [sys.executable, '-m', 'onnx_edge_quantizer', str(onnx), '-o', str(output),
                   '--weight-dtype', 'int4', '--weight-granularity', 'per-channel',
                   '--structure-reference', str(reference), '--keep-workdir', str(work), '--overwrite']
        start = time.monotonic()
        try:
            command.extend(dynamic_input_flags(onnx, reference))
            with log.open('w') as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=args.timeout)
            record['exit_code'] = process.returncode
            record['status'] = 'passed' if process.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            record['status'] = 'timeout'
        except (ValueError, OSError) as error:
            record['status'] = 'failed'
            log.write_text(str(error), encoding='utf-8')
        record['seconds'] = round(time.monotonic() - start, 2)
        record['log'], record['workdir'] = str(log), str(work)
        comparison = work / 'structure_comparison.json'
        if comparison.is_file():
            record['structure'] = json.loads(comparison.read_text())
        if record['status'] == 'passed':
            report = json.loads(output.with_suffix('.quantization.json').read_text())
            record['precision'] = report['backend_details']['precision_verification']
            record['validation'] = report['validation']
        else:
            record['error_tail'] = log.read_text(errors='replace')[-3000:]
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(run, pair) for pair in pairs]
        for future in as_completed(tasks):
            record = future.result()
            records.append(record)
            summary = {'models': sorted(records, key=lambda r: (r['model'], r['reference'])),
                       'passed': sum(r['status'] == 'passed' for r in records),
                       'total': len(pairs)}
            # Persist after every model so interrupted sweeps remain reviewable.
            (args.output_dir / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f"{record['status']}: {record['model']}", flush=True)
    return 0 if args.dry_run or all(r['status'] == 'passed' for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
