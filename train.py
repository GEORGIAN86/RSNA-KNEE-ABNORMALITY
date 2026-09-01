"""Compatibility entry point for the unified training command."""

from __future__ import annotations

import sys

from TrainEnsemble import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
