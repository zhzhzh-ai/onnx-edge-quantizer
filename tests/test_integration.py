"""Real conversion tests, skipped only when an optional backend is absent."""

import importlib.util
import json
import pytest
from examples.make_demo import make_demo
from onnx_edge_quantizer.cli import main


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None, reason="onnx2tf optional dependency is absent")
@pytest.mark.parametrize("mode", ["reference", "native"])
def test_multi_input_calibration_unrestricted_range(tmp_path, mode):
    if mode == "native" and importlib.util.find_spec("ai_edge_quantizer") is None:
        pytest.skip("AI Edge Quantizer optional dependency is absent")
    from examples.make_multi_input_demo import make_multi_input_demo
    make_multi_input_demo(tmp_path)
    output = tmp_path / f"multi_{mode}.tflite"
    assert main([str(tmp_path / "multi_input.onnx"), "-o", str(output), "--mode", mode,
                 "--calibration", str(tmp_path / "calibration.npz"),
                 "--validation-data", str(tmp_path / "validation.npz")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert {i["name"] for i in report["inputs"]} == {"left", "right"}
    assert report["calibration_samples"] == 32 and report["validation_samples"] == 8
    assert min(r["min"] for r in report["activation_ranges"].values()) < 0
    assert max(r["max"] for r in report["activation_ranges"].values()) > 1


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None or importlib.util.find_spec("ai_edge_quantizer") is None,
                    reason="Both conversion and quantization backends are required")
@pytest.mark.parametrize("granularity", ["per-tensor", "per-channel"])
def test_onnx_w4a8_full_pipeline(tmp_path, granularity):
    make_demo(tmp_path)
    output = tmp_path / "w4a8.tflite"
    assert main([str(tmp_path / "demo.onnx"), "-o", str(output), "--weight-dtype", "int4",
                 "--activation-dtype", "int8", "--weight-granularity", granularity,
                 "--calibration", str(tmp_path / "calibration.npz"),
                 "--validation-data", str(tmp_path / "validation.npz")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert not report["output_inventory"]["float_compute_ops"]
    assert any(t["type"] == "INT4" and t["scales"] for t in report["output_inventory"]["tensors"])
    assert any(t["type"] == "INT8" and t["scales"] for t in report["output_inventory"]["tensors"])


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None, reason="onnx2tf optional dependency is absent")
def test_onnx_to_tflite_reference(tmp_path):
    make_demo(tmp_path)
    output = tmp_path / "converted.tflite"
    assert main([str(tmp_path / "demo.onnx"), "-o", str(output), "--mode", "reference",
                 "--calibration", str(tmp_path / "calibration.npz"), "--validation-data", str(tmp_path / "validation.npz"),
                 "--keep-workdir", str(tmp_path / "work")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert report["validation_samples"] == 8
    assert all(v["onnx_vs_float_tflite"]["max_abs_error"] < 1e-4 for v in report["validation"])


@pytest.mark.skipif(importlib.util.find_spec("ai_edge_quantizer") is None, reason="AI Edge Quantizer optional dependency is absent")
@pytest.mark.parametrize("weight_dtype,activation_dtype,scheme", [("int8", "int8", "asymmetric"), ("int8", "int16", "symmetric"), ("int4", "int8", "asymmetric")])
@pytest.mark.parametrize("granularity", ["per-tensor", "per-channel"])
def test_native_quantization(demo, weight_dtype, activation_dtype, scheme, granularity):
    output = demo / "native.tflite"
    assert main([str(demo / "demo.onnx"), "-o", str(output), "--float-tflite", str(demo / "float.tflite"),
                 "--weight-dtype", weight_dtype, "--activation-dtype", activation_dtype, "--activation-scheme", scheme,
                 "--weight-granularity", granularity,
                 "--calibration", str(demo / "calibration.npz")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert not report["output_inventory"]["float_compute_ops"]
    assert any(t["type"] == weight_dtype.upper() and t["scales"] for t in report["output_inventory"]["tensors"])
    weight_tensors = [t for t in report["output_inventory"]["tensors"] if t["type"] == weight_dtype.upper() and "weight" in t["name"] and t["scales"]]
    assert weight_tensors
    assert all(len(t["scales"]) == (3 if granularity == "per-channel" else 1) for t in weight_tensors)
    from onnx_edge_quantizer.runtime import check_native_precision
    with pytest.raises(ValueError, match="granularity"):
        check_native_precision(output, weight_dtype, activation_dtype, "per-tensor" if granularity == "per-channel" else "per-channel")


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None or importlib.util.find_spec("ai_edge_quantizer") is None,
                    reason="Both conversion and quantization backends are required")
def test_native_full_pipeline_and_temporary_cleanup(tmp_path):
    make_demo(tmp_path)
    output = tmp_path / "native_from_onnx.tflite"
    assert main([str(tmp_path / "demo.onnx"), "-o", str(output),
                 "--calibration", str(tmp_path / "calibration.npz")]) == 0
    assert output.is_file()


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None, reason="onnx2tf optional dependency is absent")
@pytest.mark.parametrize("mode", ["reference", "native"])
@pytest.mark.parametrize("granularity", ["per-tensor", "per-channel"])
@pytest.mark.parametrize("depthwise", [False, True])
def test_full_cnn_nchw_pipeline(tmp_path, mode, granularity, depthwise):
    if mode == "native" and importlib.util.find_spec("ai_edge_quantizer") is None:
        pytest.skip("AI Edge Quantizer optional dependency is absent")
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    rng = np.random.default_rng(12)
    channels = 6 if depthwise else 2
    weights = rng.uniform(-0.5, 0.5, (channels, 1 if depthwise else 3, 3, 3)).astype(np.float32)
    graph = helper.make_graph([
        helper.make_node("Conv", ["images", "weight"], ["conv"], kernel_shape=[3, 3], group=3 if depthwise else 1),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["scores"]),
    ], "cnn", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
       [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, channels, 1, 1])],
       [numpy_helper.from_array(weights, "weight")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), tmp_path / "cnn.onnx")
    np.savez(tmp_path / "data.npz", images=rng.uniform(-1, 1, (4, 1, 3, 8, 8)).astype(np.float32))
    output = tmp_path / f"cnn_{mode}.tflite"
    assert main([str(tmp_path / "cnn.onnx"), "-o", str(output), "--mode", mode,
                 "--weight-granularity", granularity,
                 "--calibration", str(tmp_path / "data.npz"), "--keep-workdir", str(tmp_path / "work")]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text(encoding="utf-8"))
    assert report["inputs"][0]["shape"] == [1, 3, 8, 8]
    assert max(v["onnx_vs_float_tflite"]["max_abs_error"] for v in report["validation"]) < 1e-5
    if granularity == "per-channel":
        axis = 3 if depthwise else 0
        if mode == "reference":
            assert any(w["axis"] == axis and len(w["scales"]) == channels for w in report["backend_details"]["weights"].values())
        else:
            assert any(t["quantized_dimension"] == axis and len(t["scales"]) == channels and t["type"] == "INT8"
                       for t in report["output_inventory"]["tensors"])


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None,
                    reason="onnx2tf optional dependency is absent")
@pytest.mark.parametrize("mode", ["native", "reference"])
@pytest.mark.parametrize("height,width", [(6, 10), (4, 4)])
@pytest.mark.parametrize("input_channels,keep_layout", [(1, False), (3, False), (4, False), (3, True)])
def test_spatial_cnn_output_layout(tmp_path, mode, height, width, input_channels, keep_layout):
    """Dense SR outputs retain H/W; converter metadata must align NHWC to NCHW.

    Equal channel/spatial dimensions must also work without guessing axes from
    shapes or searching permutations for the best numerical match.
    """
    if mode == "native" and importlib.util.find_spec("ai_edge_quantizer") is None:
        pytest.skip("AI Edge Quantizer optional dependency is absent")
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from onnx_edge_quantizer.data import input_specs, load_samples
    from onnx_edge_quantizer.runtime import modules, read_model
    from onnx_edge_quantizer.validation import validate
    rng = np.random.default_rng(23)
    graph = helper.make_graph([
        helper.make_node("Conv", ["image", "w1"], ["conv"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("Conv", ["relu", "w2"], ["pixels"], pads=[1, 1, 1, 1]),
    ], "spatial_cnn", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, input_channels, height, width])],
       [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 4, height, width])],
       [numpy_helper.from_array(rng.uniform(-0.5, 0.5, shape).astype(np.float32), name)
        for name, shape in [("w1", (8, input_channels, 3, 3)), ("w2", (4, 8, 3, 3))]])
    model_path = tmp_path / "spatial.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model_path)
    data_path = tmp_path / "data.npz"
    np.savez(data_path, image=rng.uniform(-1, 1, (2, 1, input_channels, height, width)).astype(np.float32))
    output = tmp_path / "spatial.tflite"
    work = tmp_path / "work"
    flags = ["--keep-onnx-input-layout"] if keep_layout else []
    assert main([str(model_path), "-o", str(output), "--mode", mode,
                 "--weight-dtype", "int4", "--weight-granularity", "per-channel",
                 "--calibration", str(data_path), "--keep-workdir", str(work), *flags]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text())
    assert report["output_layouts"]["pixels"]["tflite_to_onnx_permutation"] == [0, 3, 1, 2]
    assert report["output_layouts"]["pixels"]["source"] == "onnx2tf_tensor_correspondence_report"
    assert all(v["output_shapes"] == {"onnx": [1, 4, height, width], "tflite": [1, height, width, 4]}
               for v in report["validation"])
    assert max(v["onnx_vs_float_tflite"]["max_abs_error"] for v in report["validation"]) < 1e-5
    _, schema = modules()
    expected_input = [1, input_channels, height, width] if keep_layout else [1, height, width, input_channels]
    assert report["tflite_inputs"] == [{"name": "image", "shape": expected_input}]
    assert bool(report["input_layouts"]) is not keep_layout
    if not keep_layout:
        layout = report["input_layouts"]["image"]
        assert layout["onnx_to_tflite_permutation"] == [0, 2, 3, 1]
        assert layout["removed_adapter"] == ("RESHAPE" if input_channels == 1 else "TRANSPOSE")
    for path in [work / "converted/source_float32.tflite", output]:
        tflite = read_model(path)
        graph = tflite.subgraphs[0]
        assert graph.tensors[graph.inputs[0]].shape.tolist() == expected_input
        assert tflite.signatureDefs[0].inputs[0].tensorIndex == graph.inputs[0]
        adapters = [op for op in graph.operators
                    if max(tflite.operatorCodes[op.opcodeIndex].builtinCode,
                           tflite.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode)
                    in (schema.BuiltinOperator.RESHAPE, schema.BuiltinOperator.TRANSPOSE)]
        assert bool(adapters) is keep_layout
    if mode == "native":
        assert not report["output_inventory"]["float_compute_ops"]
        assert any(t["type"] == "INT4" and len(t["scales"]) == 4
                   for t in report["output_inventory"]["tensors"])

    # The same mismatch without converter evidence must still be rejected.
    _, specs = input_specs(model_path, {})
    with pytest.raises(ValueError, match="shape/layout mismatch|parity failed"):
        validate(model_path, work / "converted/source_float32.tflite", output,
                 specs, load_samples(data_path, specs), {}, {}, 1e-4, 1e-3,
                 input_layouts=report["input_layouts"])


@pytest.mark.skipif(importlib.util.find_spec("onnx2tf") is None,
                    reason="onnx2tf optional dependency is absent")
@pytest.mark.parametrize("mode", ["native", "reference"])
def test_image_and_vector_inputs_keep_onnx_calibration_layout(tmp_path, mode):
    if mode == "native" and importlib.util.find_spec("ai_edge_quantizer") is None:
        pytest.skip("AI Edge Quantizer optional dependency is absent")
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    rng = np.random.default_rng(41)
    graph = helper.make_graph([
        helper.make_node("Conv", ["image", "weight"], ["conv"]),
        helper.make_node("GlobalAveragePool", ["conv"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Add", ["flat", "offset"], ["scores"]),
    ], "mixed_inputs", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 6, 8]),
                         helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 2])],
       [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
       [numpy_helper.from_array(rng.uniform(-0.5, 0.5, (2, 3, 3, 3)).astype(np.float32), "weight")])
    model_path = tmp_path / "mixed.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), model_path)
    data_path = tmp_path / "data.npz"
    np.savez(data_path, image=rng.uniform(-3, 3, (4, 1, 3, 6, 8)).astype(np.float32),
             offset=rng.uniform(10, 100, (4, 1, 2)).astype(np.float32))
    output = tmp_path / "mixed.tflite"
    assert main([str(model_path), "-o", str(output), "--mode", mode,
                 "--calibration", str(data_path)]) == 0
    report = json.loads(output.with_suffix(".quantization.json").read_text())
    assert set(report["input_layouts"]) == {"image"}
    assert {d["name"]: d["shape"] for d in report["tflite_inputs"]} == {
        "image": [1, 6, 8, 3], "offset": [1, 2]}
    assert report["calibration_samples"] == 4
    assert max(v["onnx_vs_float_tflite"]["max_abs_error"] for v in report["validation"]) < 1e-5
