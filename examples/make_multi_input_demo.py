"""Generate a two-input ONNX model and named calibration data outside [0, 1]."""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_multi_input_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = np.array([[0.11, -0.52, 0.83, 0.24], [-0.31, 0.67, -0.92, 0.43],
                        [0.75, 0.12, -0.23, 0.54]], dtype=np.float32)
    bias = np.array([0.12, -0.13, 0.27], dtype=np.float32)
    graph = helper.make_graph([
        helper.make_node("Add", ["left", "right"], ["combined"]),
        helper.make_node("Gemm", ["combined", "weight", "bias"], ["dense"], transB=1),
        helper.make_node("Relu", ["dense"], ["output"]),
    ], "multi_input_demo", [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4]) for name in ["left", "right"]],
       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
       [numpy_helper.from_array(weights, "weight"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "multi_input.onnx")
    rng = np.random.default_rng(42)
    for filename, count in [("calibration.npz", 32), ("validation.npz", 8)]:
        np.savez(directory / filename,
                 left=rng.uniform(-5, 5, (count, 1, 4)).astype(np.float32),
                 right=rng.uniform(10, 100, (count, 1, 4)).astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multi-input"))
    args = parser.parse_args()
    make_multi_input_demo(args.output_dir)
    print(args.output_dir / "multi_input.onnx")
