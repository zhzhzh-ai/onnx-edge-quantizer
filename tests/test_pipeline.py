import json
import numpy as np
import pytest
from onnx_edge_quantizer.backends import reference_export
from onnx_edge_quantizer.cli import Config, main
from onnx_edge_quantizer.data import input_specs, load_samples
from onnx_edge_quantizer.quantization import DTYPES, minmax_params, quantize, dequantize
from onnx_edge_quantizer.runtime import calibrate, make_interpreter, map_details, read_model, run


@pytest.mark.parametrize("weight_dtype", DTYPES)
@pytest.mark.parametrize("activation_dtype", DTYPES)
@pytest.mark.parametrize("granularity", ["per-tensor", "per-channel"])
def test_all_25_combinations_execute_and_match_numpy(demo, weight_dtype, activation_dtype, granularity):
    _, specs = input_specs(demo / "demo.onnx", {})
    samples = load_samples(demo / "calibration.npz", specs)
    ranges, _ = calibrate(demo / "float.tflite", specs, samples)
    config = Config(weight_dtype, activation_dtype, weight_granularity=granularity)
    output = demo / "reference.tflite"
    report = reference_export(demo / "float.tflite", output, ranges, config)
    assert not report["integer_arithmetic"]
    original = read_model(demo / "float.tflite")
    weights = np.frombuffer(bytes(original.buffers[1].data), dtype="<f4").reshape(3, 4)
    bias = np.frombuffer(bytes(original.buffers[2].data), dtype="<f4")
    if granularity == "per-channel":
        restored = []
        for row in weights:
            wp = minmax_params(float(row.min()), float(row.max()), weight_dtype, True)
            restored.append(dequantize(quantize(row, wp), wp))
        weights = np.stack(restored)
        assert report["weights"]["1"]["axis"] == 0
        assert len(report["weights"]["1"]["scales"]) == 3
    else:
        wp = minmax_params(float(weights.min()), float(weights.max()), weight_dtype, True)
        weights = dequantize(quantize(weights, wp), wp)
    instance = make_interpreter(output)
    mapping = map_details(specs, instance.get_input_details())
    for sample in samples[:3]:
        def rounded(value, index):
            p = minmax_params(*ranges[index], activation_dtype, False)
            return dequantize(quantize(value, p), p)
        expected = rounded(np.maximum(rounded(rounded(sample["input"], 0) @ weights.T + bias, 3), 0), 4)
        actual = run(instance, sample, mapping)["output"]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


def test_cli_external_and_synthetic_and_overwrite(demo):
    output = demo / "output.tflite"
    args = [str(demo / "demo.onnx"), "-o", str(output), "--float-tflite", str(demo / "float.tflite"), "--mode", "reference"]
    assert main(args) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert report["synthetic_calibration"] and report["calibration_samples"] == 1
    previous = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == previous
    assert main(args + ["--overwrite", "--calibration", str(demo / "calibration.npz"), "--validation-data", str(demo / "validation.npz")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert not report["synthetic_calibration"] and report["validation_samples"] == 8


def test_wrong_float_conversion_never_published(demo):
    from onnx_edge_quantizer.runtime import write_model
    model = read_model(demo / "float.tflite")
    model.buffers[1].data[:] = 0
    write_model(model, demo / "wrong.tflite")
    output = demo / "should_not_exist.tflite"
    assert main([str(demo / "demo.onnx"), "-o", str(output), "--mode", "reference", "--float-tflite", str(demo / "wrong.tflite")]) == 2
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--weight-granularity", "--granularity"])
def test_cli_per_channel_and_report(demo, flag):
    output = demo / "per_channel.tflite"
    assert main([str(demo / "demo.onnx"), "-o", str(output), "--mode", "reference",
                 "--float-tflite", str(demo / "float.tflite"), flag, "per-channel"]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert report["weight_granularity"] == report["requested"]["weight_granularity"] == "per-channel"
    assert report["activation_granularity"] == "per-tensor"
    assert len(report["backend_details"]["weights"]["1"]["scales"]) == 3


def test_native_unsupported_config_fails_before_conversion(demo):
    output = demo / "bad.tflite"
    assert main([str(demo / "demo.onnx"), "-o", str(output), "--weight-dtype", "uint4"]) == 2
    assert not output.exists()
