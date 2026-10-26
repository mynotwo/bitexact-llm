"""Compatibility entry point for the strict v1 reproduction CLI.

Old unversioned reports must be recorded again; they do not identify the complete experiment.
See docs/REPRODUCING.md. The canonical entry point is python -m bitexact.repro.
"""

import sys

from bitexact.repro import PROMPTS, compare, main, record

__all__ = ["PROMPTS", "compare", "main", "record"]

if __name__ == "__main__":
    sys.exit(main())
