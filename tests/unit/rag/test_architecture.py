from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BANNED_RUNTIME_MODULES = ("langchain_milvus", "pymilvus", "openai")
BANNED_IMPORT_PREFIXES = (
    "app.rag.infrastructure",
    "langchain",
    "pymilvus",
    "openai",
)
LEGACY_ENTRYPOINT_PATHS = (
    "app/rag/vector_index_service.py",
    "app/rag/rag_retrieval_service.py",
    "app/services/vector_index_service.py",
    "app/services/rag_retrieval_service.py",
)
LEGACY_ENTRYPOINT_MODULES = (
    "app.rag.vector_index_service",
    "app.rag.rag_retrieval_service",
    "app.services.vector_index_service",
    "app.services.rag_retrieval_service",
)
RAG_PRODUCTION_CALLERS = (
    "app/api/file.py",
    "app/tools/knowledge_tool.py",
)
QUERY_FORBIDDEN_IMPORT_PREFIXES = ("langchain", "fastapi", "pymilvus", "app.agent")


@pytest.mark.parametrize(
    "module_name",
    [
        "app.rag.models",
        "app.rag.ports",
        "app.rag.indexing.contracts",
        "app.rag.query.contracts",
    ],
)
def test_domain_and_contract_imports_do_not_load_external_runtimes(
    module_name: str,
) -> None:
    script = (
        "import importlib, json, sys; "
        f"importlib.import_module({module_name!r}); "
        f"banned={BANNED_RUNTIME_MODULES!r}; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in banned)))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "relative_path",
    [
        "app/rag/indexing/indexing_service.py",
        "app/rag/query/query_service.py",
        "app/rag/query/query_router.py",
        "app/rag/query/retrieval_fusion.py",
        "app/rag/query/reranker.py",
    ],
)
def test_use_cases_do_not_import_infrastructure_or_frameworks(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    violations = sorted(
        module
        for module in imported_modules
        if module.startswith(BANNED_IMPORT_PREFIXES)
    )
    assert violations == []


def test_legacy_rag_entrypoints_are_removed() -> None:
    remaining_paths = [
        relative_path
        for relative_path in LEGACY_ENTRYPOINT_PATHS
        if (PROJECT_ROOT / relative_path).exists()
    ]

    assert remaining_paths == []


@pytest.mark.parametrize("relative_path", RAG_PRODUCTION_CALLERS)
def test_rag_callers_do_not_import_legacy_entrypoints(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    violations = sorted(
        module for module in imported_modules if module in LEGACY_ENTRYPOINT_MODULES
    )
    assert violations == []


def test_query_core_does_not_depend_on_framework_or_agent_layers() -> None:
    """Keep the query core replaceable; LangChain is only an infrastructure adapter."""
    query_root = PROJECT_ROOT / "app/rag/query"
    violations: list[str] = []
    for path in query_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module)
                if node.module == "app":
                    modules.extend(f"app.{alias.name}" for alias in node.names)
            for module in modules:
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in QUERY_FORBIDDEN_IMPORT_PREFIXES
                ):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: {module}")
    assert violations == []
