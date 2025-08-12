from __future__ import annotations

import importlib

def import_by_path(path: str):
    """'a.b.c.Class' -> Class"""
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)
