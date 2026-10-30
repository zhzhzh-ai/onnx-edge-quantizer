"""Create a separate W8A8 reference from a mixed FP32 depthwise island."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_edge_quantizer.reference_integerization import integerize_depthwise_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = integerize_depthwise_reference(args.source, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
