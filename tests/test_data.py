import numpy as np
import pytest
from onnx_edge_quantizer.data import input_specs, load_samples


def test_calibration_shape_and_determinism(demo):
    _, specs = input_specs(demo / "demo.onnx", {})
    samples = load_samples(demo / "calibration.npz", specs)
    assert len(samples) == 32 and samples[0]["input"].shape == (1, 4)
    a, b = load_samples(None, specs), load_samples(None, specs)
    np.testing.assert_array_equal(a[0]["input"], b[0]["input"])
    assert len(a) == 1


def test_multi_input_and_count_rejection(tmp_path):
    specs = [{"name": "a", "shape": [1, 2]}, {"name": "b", "shape": [1, 3]}]
    path = tmp_path / "multi.npz"
    np.savez(path, a=np.zeros((5, 1, 2), np.float32), b=np.zeros((5, 1, 3), np.float32))
    assert len(load_samples(path, specs)) == 5
    np.savez(path, a=np.zeros((5, 1, 2), np.float32), b=np.zeros((4, 1, 3), np.float32))
    with pytest.raises(ValueError, match="same nonzero"):
        load_samples(path, specs)


def test_dynamic_shapes_require_override(demo):
    import onnx
    path = demo / "demo.onnx"
    model = onnx.load(path)
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"
    onnx.save(model, path)
    with pytest.raises(ValueError, match="dynamic"):
        input_specs(path, {})
    _, specs = input_specs(path, {"input": [1, 4]})
    assert specs[0]["shape"] == [1, 4]
