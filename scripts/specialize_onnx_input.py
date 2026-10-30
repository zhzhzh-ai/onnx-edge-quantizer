"""Create an ONNX copy at the reference deployment input size."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_edge_quantizer.input_specialization import specialize_to_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(specialize_to_reference(args.source, args.reference, args.output), indent=2))


if __name__ == '__main__':
    main()
