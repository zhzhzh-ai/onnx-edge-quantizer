"""Guard against installing into the wrong Python when Conda is activated."""

from scripts import install


def test_named_conda_is_recognized_when_python_prefixes_are_equal(monkeypatch, tmp_path):
    prefix = str(tmp_path / "env")
    monkeypatch.setattr(install.sys, "prefix", prefix)
    monkeypatch.setattr(install.sys, "base_prefix", prefix)
    monkeypatch.setenv("CONDA_PREFIX", prefix)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "onnx-edge-quantizer")
    assert install.isolated_environment()


def test_conda_base_and_wrong_interpreter_are_rejected(monkeypatch, tmp_path):
    prefix = str(tmp_path / "env")
    monkeypatch.setattr(install.sys, "prefix", prefix)
    monkeypatch.setattr(install.sys, "base_prefix", prefix)
    monkeypatch.setenv("CONDA_PREFIX", prefix)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    assert not install.isolated_environment()
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "onnx-edge-quantizer")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "other"))
    assert not install.isolated_environment()
    monkeypatch.delenv("CONDA_PREFIX")
    assert not install.isolated_environment()


def test_venv_remains_supported(monkeypatch):
    monkeypatch.setattr(install.sys, "prefix", "venv")
    monkeypatch.setattr(install.sys, "base_prefix", "system")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert install.isolated_environment()
