from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from examples.make_demo import make_demo
from onnx_edge_quantizer.runtime import modules, write_model


@pytest.fixture
def demo(tmp_path):
    """Independent hand-built FP32 LiteRT fixture matching the ONNX demo."""
    weights, bias = make_demo(tmp_path)
    _, s = modules()
    model = s.ModelT()
    model.version, model.description = 3, b"Test dense float fixture"
    model.buffers, model.operatorCodes, model.subgraphs, model.signatureDefs = [s.BufferT()], [], [], []
    graph = s.SubGraphT()
    graph.name, graph.tensors, graph.operators = b"main", [], []
    for name, shape, values in [(b"input", [1, 4], None), (b"weight", [3, 4], weights),
                                (b"bias", [3], bias), (b"dense", [1, 3], None), (b"output", [1, 3], None)]:
        tensor = s.TensorT()
        tensor.name, tensor.shape, tensor.type, tensor.buffer = name, np.array(shape, dtype=np.int32), s.TensorType.FLOAT32, 0
        if values is not None:
            buf = s.BufferT()
            buf.data = np.frombuffer(values.astype("<f4").tobytes(), dtype=np.uint8)
            model.buffers.append(buf)
            tensor.buffer = len(model.buffers) - 1
        graph.tensors.append(tensor)
    for builtin, inputs, output in [(s.BuiltinOperator.FULLY_CONNECTED, [0, 1, 2], 3), (s.BuiltinOperator.RELU, [3], 4)]:
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = builtin
        code.version = 1
        model.operatorCodes.append(code)
        op = s.OperatorT()
        op.opcodeIndex = len(model.operatorCodes) - 1
        op.inputs, op.outputs = np.array(inputs, dtype=np.int32), np.array([output], dtype=np.int32)
        if builtin == s.BuiltinOperator.FULLY_CONNECTED:
            op.builtinOptionsType = s.BuiltinOptions.FullyConnectedOptions
            op.builtinOptions = s.FullyConnectedOptionsT()
        graph.operators.append(op)
    graph.inputs, graph.outputs = np.array([0], dtype=np.int32), np.array([4], dtype=np.int32)
    model.subgraphs.append(graph)
    signature = s.SignatureDefT()
    signature.signatureKey, signature.subgraphIndex = b"serving_default", 0
    entry = s.TensorMapT()
    entry.name, entry.tensorIndex = b"input", 0
    signature.inputs = [entry]
    entry = s.TensorMapT()
    entry.name, entry.tensorIndex = b"output", 4
    signature.outputs = [entry]
    model.signatureDefs = [signature]
    write_model(model, tmp_path / "float.tflite")
    return tmp_path
