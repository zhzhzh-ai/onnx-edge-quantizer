"""Create a reproducible FP32 ONNX dense network and calibration inputs."""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = np.array([[0.11, -0.52, 0.83, 0.24], [-0.31, 0.67, -0.92, 0.43],
                        [0.75, 0.12, -0.23, 0.54]], dtype=np.float32)
    bias = np.array([0.12, -0.13, 0.27], dtype=np.float32)
    graph = helper.make_graph([
        helper.make_node("Gemm", ["input", "weight", "bias"], ["dense"], transB=1),
        helper.make_node("Relu", ["dense"], ["output"]),
    ], "dense_demo", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
       [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
       [numpy_helper.from_array(weights, "weight"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "demo.onnx")
    rng = np.random.default_rng(42)
    np.savez(directory / "calibration.npz", input=rng.uniform(-1, 1, (32, 1, 4)).astype(np.float32))
    np.savez(directory / "validation.npz", input=rng.uniform(-1, 1, (8, 1, 4)).astype(np.float32))
    return weights, bias


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    args = p.parse_args()
    make_demo(args.output_dir)
    print(args.output_dir / "demo.onnx")
