"""Compatibility shim for app.rag.vector_store_manager."""

from importlib import import_module
import sys

_module = import_module("app.rag.vector_store_manager")
sys.modules[__name__] = _module
globals().update(_module.__dict__)
