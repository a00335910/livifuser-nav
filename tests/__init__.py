"""Test package.

Puts ``src/`` on the import path so the suite runs identically whether or not
`livifuser_nav` is installed. On Windows it is available through the uv editable
install; on the ROS host it is not, and the tests must not depend on the caller
remembering to set PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
