#!/usr/bin/env python3
"""Repository entrypoint for the SkyPlan inference runner.

This shim keeps local tests and external validators aligned by delegating to
`AgentEnv.inference` while being resilient to different working directories.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path


def _bootstrap_agentenv_path() -> None:
    """Add a repository root containing AgentEnv to sys.path when needed."""

    current_file = Path(__file__).resolve()
    search_roots = [current_file.parent, *current_file.parents]

    for root in search_roots:
        if (root / "AgentEnv" / "__init__.py").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return


_bootstrap_agentenv_path()

_impl = importlib.import_module("AgentEnv.inference")

if __name__ == "__main__":
    asyncio.run(_impl.main(sys.argv[1:]))
else:
    sys.modules[__name__] = _impl
