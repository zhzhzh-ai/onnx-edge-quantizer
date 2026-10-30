"""Compare deployment topology independently of tensor names and quantization.

Operator contracts, data edges, tensor shapes and control constants must match.
Weight/bias values, integer signedness, scales and opcode versions may differ.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .runtime import modules, read_model
from .tflite_optimizations import compact_model


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, '__dict__'):
        return {key: _plain(v) for key, v in vars(value).items()
                if v is not None and key not in ('quantizedBiasType', 'quantSpec')}
    return value


def graph_contract(path):
    model = deepcopy(path) if hasattr(path, 'subgraphs') else read_model(Path(path))
    _, s = modules()
    compact_model(model)
    names = {value: name for name, value in vars(s.BuiltinOperator).items() if isinstance(value, int)}
    control_slots = {
        s.BuiltinOperator.TRANSPOSE: {1}, s.BuiltinOperator.RESHAPE: {1},
        s.BuiltinOperator.PAD: {1}, s.BuiltinOperator.PADV2: {1},
        s.BuiltinOperator.GATHER: {1}, s.BuiltinOperator.MEAN: {1},
        s.BuiltinOperator.SUM: {1}, s.BuiltinOperator.SLICE: {1, 2},
        s.BuiltinOperator.STRIDED_SLICE: {1, 2, 3},
        s.BuiltinOperator.SPLIT: {0}, s.BuiltinOperator.SPLIT_V: {1, 2},
        s.BuiltinOperator.RESIZE_BILINEAR: {1}, s.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: {1},
    }
    graphs = []
    for graph in model.subgraphs:
        def shape_of(index):
            shape = graph.tensors[int(index)].shape
            return shape.tolist() if shape is not None else []

        refs = {int(t): ['input', i] for i, t in enumerate(graph.inputs)}
        for i, op in enumerate(graph.operators):
            refs.update({int(t): ['op', i, port] for port, t in enumerate(op.outputs)})
        records = []
        for op in graph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
            inputs = []
            for position, tensor_id in enumerate(op.inputs):
                if tensor_id < 0:
                    inputs.append({'optional': True})
                    continue
                tensor = graph.tensors[int(tensor_id)]
                item = {'shape': shape_of(tensor_id)}
                if int(tensor_id) in refs:
                    item['edge'] = refs[int(tensor_id)]
                else:
                    item['constant'] = True
                    if position in control_slots.get(builtin, set()):
                        dtype = {s.TensorType.INT32: '<i4', s.TensorType.INT64: '<i8'}.get(tensor.type)
                        data = model.buffers[tensor.buffer].data
                        if dtype is None or data is None:
                            raise ValueError('Structure comparison requires static integer control tensors')
                        item['value'] = np.frombuffer(bytes(data), dtype=dtype).tolist()
                        if builtin == s.BuiltinOperator.RESHAPE and position == 1 and -1 in item['value']:
                            # A fixed deployment shape has one unique inferred
                            # dimension; normalize the equivalent literal -1.
                            requested = list(item['value'])
                            if requested.count(-1) == 1:
                                source_shape = shape_of(op.inputs[0])
                                total = int(np.prod(source_shape))
                                product = -int(np.prod(requested))
                                if product > 0 and total % product == 0:
                                    requested[requested.index(-1)] = total // product
                                    item['value'] = requested
                inputs.append(item)
            options = _plain(op.builtinOptions) or {}
            if builtin == s.BuiltinOperator.RESHAPE and len(op.inputs) > 1:
                options.pop('newShape', None)  # Shape input takes precedence.
            if builtin in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D,
                           s.BuiltinOperator.MAX_POOL_2D, s.BuiltinOperator.AVERAGE_POOL_2D):
                source_shape = graph.tensors[int(op.inputs[0])].shape
                target_shape = graph.tensors[int(op.outputs[0])].shape
                if builtin in (s.BuiltinOperator.CONV_2D, s.BuiltinOperator.DEPTHWISE_CONV_2D):
                    weight_shape = graph.tensors[int(op.inputs[1])].shape
                    kh, kw = weight_shape[1:3]
                else:
                    kh, kw = options['filterHeight'], options['filterWidth']
                padding = []
                for axis, kernel, stride, dilation in [
                    (1, kh, options['strideH'], options.get('dilationHFactor', 1)),
                    (2, kw, options['strideW'], options.get('dilationWFactor', 1)),
                ]:
                    total = max((int(target_shape[axis]) - 1) * stride
                                + (int(kernel) - 1) * dilation + 1 - int(source_shape[axis]), 0)
                    padding.extend([total // 2, total - total // 2] if options['padding'] == s.Padding.SAME else [0, 0])
                options['padding'] = padding
            records.append({'op': names.get(builtin, str(builtin)),
                            'custom_code': code.customCode.decode() if code.customCode else None,
                            'inputs': inputs,
                            'outputs': [shape_of(t) for t in op.outputs],
                            'options': options})
        graphs.append({'inputs': [shape_of(t) for t in graph.inputs],
                       'outputs': [refs.get(int(t), ['constant', shape_of(t)]) for t in graph.outputs],
                       'operators': records})
    return graphs


def compare_structure(candidate, reference):
    actual, expected = graph_contract(candidate), graph_contract(reference)
    differences = []

    def diff(a, b, path):
        if type(a) is not type(b):
            differences.append({'path': path, 'candidate': a, 'reference': b})
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append({'path': f'{path}.{key}', 'candidate': a.get(key), 'reference': b.get(key)})
                else:
                    diff(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({'path': path + '.length', 'candidate': len(a), 'reference': len(b)})
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f'{path}[{i}]')
        elif a != b:
            differences.append({'path': path, 'candidate': a, 'reference': b})

    diff(actual, expected, 'subgraphs')
    return {'matched': not differences, 'candidate': str(Path(candidate).resolve()),
            'reference': str(Path(reference).resolve()),
            'candidate_op_counts': dict(Counter(op['op'] for g in actual for op in g['operators'])),
            'reference_op_counts': dict(Counter(op['op'] for g in expected for op in g['operators'])),
            'difference_count': len(differences), 'differences': differences}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = compare_structure(args.candidate, args.reference)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(content + '\n', encoding='utf-8')
    else:
        print(content)
    raise SystemExit(0 if result['matched'] else 1)
