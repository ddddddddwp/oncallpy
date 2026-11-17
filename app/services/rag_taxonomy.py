"""Compatibility shim for app.rag.rag_taxonomy."""

from importlib import import_module
import sys

_module = import_module("app.rag.rag_taxonomy")
sys.modules[__name__] = _module
globals().update(_module.__dict__)
