"""
Compatibility shim for node imports.

Main currently carries `_lg_nodes_deprecated.py`; this module re-exports it
so imports like `from .lg_nodes import ...` remain valid.
"""

from ._lg_nodes_deprecated import *  # noqa: F401,F403

