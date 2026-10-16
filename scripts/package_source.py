"""Create an allowlisted source archive without environments, models or user data."""

from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts" / "onnx-edge-quantizer-source.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [root / name for name in ["pyproject.toml", "README.md", "Dockerfile", ".dockerignore", ".gitignore",
                                     "quantize.ps1", "requirements-windows-tested.txt"]]
    for folder in ["onnx_edge_quantizer", "tests", "examples", "scripts"]:
        files.extend((root / folder).rglob("*.py"))
    files.extend((root / ".github").rglob("*.yml"))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "onnx-edge-quantizer/" + path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
