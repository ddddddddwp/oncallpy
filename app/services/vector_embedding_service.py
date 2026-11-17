"""Compatibility shim for app.rag.vector_embedding_service."""

from importlib import import_module
import sys

_module = import_module("app.rag.vector_embedding_service")
sys.modules[__name__] = _module
globals().update(_module.__dict__)
