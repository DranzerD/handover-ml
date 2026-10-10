"""Logging setup shared by the CLI and worker processes."""

from __future__ import annotations

import logging
import os


def setup_logging(verbose: bool = False) -> None:
    level = (
        logging.DEBUG
        if verbose
        else getattr(logging, os.environ.get("CELLSELECT_LOG", "INFO").upper(), logging.INFO)
    )
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
