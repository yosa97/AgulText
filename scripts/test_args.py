#!/usr/bin/env python3
"""Diagnostic helper: print all CLI arguments with type inference.

Useful for debugging how shell argument quoting and escaping behaves
when launching training sub-processes from text_trainer.py.
"""
import sys


def _infer_type(value: str) -> str:
    """Return a short type label for a string argument."""
    if value.lower() in ("true", "false"):
        return "bool"
    try:
        int(value)
        return "int"
    except ValueError:
        pass
    try:
        float(value)
        return "float"
    except ValueError:
        pass
    return "str"


def main():
    print(f"Script : {sys.argv[0]}")
    print(f"Args   : {len(sys.argv) - 1}")
    print()

    if len(sys.argv) > 1:
        print("All arguments:")
        for i, arg in enumerate(sys.argv[1:], 1):
            kind = _infer_type(arg)
            print(f"  [{i:>3}] ({kind:<5}) {arg}")
    else:
        print("No arguments provided.")


if __name__ == "__main__":
    main()
