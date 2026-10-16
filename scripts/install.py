"""Install into a Python 3.12+ Conda env or venv, preserving LiteRT overlay order."""

from pathlib import Path
import os
import subprocess
import sys


def isolated_environment():
    if sys.prefix != sys.base_prefix:
        return True
    # Conda Python generally has prefix == base_prefix, unlike venv Python.
    prefix = os.environ.get("CONDA_PREFIX")
    return bool(prefix and os.environ.get("CONDA_DEFAULT_ENV") != "base"
                and os.path.normcase(os.path.realpath(prefix)) == os.path.normcase(os.path.realpath(sys.prefix)))


def main():
    if sys.version_info < (3, 12):
        raise SystemExit("Use Python 3.12+ for the onnx2tf backend.")
    if not isolated_environment():
        raise SystemExit("Activate a named Conda environment or a venv first; see README.md.")
    root = Path(__file__).resolve().parents[1]
    commands = [
        ["-e", f"{root}[convert,test]", "onnx2tf==2.6.9"],
        ["ai-edge-quantizer-nightly==0.10.0.dev20261016"],
        ["--force-reinstall", "--no-deps", "ai-edge-litert-nightly==2.4.0.dev20261014"],
    ]
    for packages in commands:
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
    subprocess.run([sys.executable, "-m", "onnx_edge_quantizer", "--doctor"], check=True)


if __name__ == "__main__":
    main()
