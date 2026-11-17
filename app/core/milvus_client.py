"""Compatibility shim for app.rag.milvus_client."""

from importlib import import_module
import sys

_module = import_module("app.rag.milvus_client")
sys.modules[__name__] = _module
globals().update(_module.__dict__)
