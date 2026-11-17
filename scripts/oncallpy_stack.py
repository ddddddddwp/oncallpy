#!/usr/bin/env python3
"""Compatibility entry point for the renamed EnergyAgent stack manager."""

if __package__ == "scripts" and __name__ == "scripts.oncallpy_stack":
    from .energyagent_stack import main
else:
    import importlib.util
    from pathlib import Path

    _canonical_path = Path(__file__).resolve().with_name("energyagent_stack.py")
    _canonical_spec = importlib.util.spec_from_file_location(
        "_energyagent_stack_compat", _canonical_path
    )
    if _canonical_spec is None or _canonical_spec.loader is None:
        raise ImportError(f"cannot load canonical stack manager from {_canonical_path}")
    _canonical_module = importlib.util.module_from_spec(_canonical_spec)
    _canonical_spec.loader.exec_module(_canonical_module)
    main = _canonical_module.main

if __name__ == "__main__":
    raise SystemExit(main())
