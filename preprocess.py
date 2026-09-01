"""Compatibility entry point for the unified preprocessing command."""

from __future__ import annotations

import sys

from TrainEnsemble import main


if __name__ == "__main__":
    raise SystemExit(main(["preprocess", *sys.argv[1:]]))
