#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run TIGER project tests")
    parser.add_argument(
        "--scope",
        choices=["tiger", "all"],
        default="tiger",
        help="Which tests to run (default: tiger)",
    )
    args = parser.parse_args()

    # Ensure we're at repo root
    # Ensure we run from this project's folder
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    if args.scope == "tiger":
        paths = ["tests"]
    else:
        # Run everything under this project folder
        paths = ["."]

    cmd = [sys.executable, "-m", "pytest", "-q", *paths]
    print("Running:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("pytest not found. Please install it: python -m pip install pytest", file=sys.stderr)
        sys.exit(127)


if __name__ == "__main__":
    main()
