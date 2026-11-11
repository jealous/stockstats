"""Make the single-module ``stockstats`` package importable.

In the repository the tests live in ``tests/`` next to ``stockstats.py``,
so the repository root (the parent of this directory) is added to
``sys.path``. No production code is touched.
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
