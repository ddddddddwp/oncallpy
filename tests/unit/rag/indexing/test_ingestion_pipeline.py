from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.documents import Document

import app.rag.indexing.indexing_service as indexing_service_module
from app.rag.indexing.contracts import (
    IngestionDecision as NewIngestionDecision,
    LoadedDocument as NewLoadedDocument,
)
from app.rag.indexing.deduplicator import RagDeduplicator as NewRagDeduplicator
from app.rag.indexing.document_loader import RagDocumentLoader as NewRagDocumentLoader
from app.rag.indexing.indexing_service import IndexingService
from app.rag.indexing.metadata_builder import RagMetadataBuilder as NewRagMetadataBuilder
from app.rag.models import Document as DomainDocument
from app.rag.rag_deduplicator import RagDeduplicator
from app.rag.rag_document_loader import RagDocumentLoader
from app.rag.rag_ingestion_models import IngestionDecision, LoadedDocument
from app.rag.rag_metadata_builder import RagMetadataBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_indexing_collaborators_reexport_new_types() -> None:
    assert LoadedDocument is NewLoadedDocument
    assert IngestionDecision is NewIngestionDecision
    assert RagDocumentLoader is NewRagDocumentLoader
    assert RagMetadataBuilder is NewRagMetadataBuilder
    assert RagDeduplicator is NewRagDeduplicator


def _run_isolated_json_probe(
    tmp_path: Path,
    probe: str,
    *,
    config_text: str = "{}\n",
) -> dict[str, bool]:
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text(config_text, encoding="utf-8")
    environment = {
        "ENERGY_AGENT_CONFIG_FILES": str(fake_config),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(PROJECT_ROOT),
    }

    assert Path(sys.executable).is_absolute()
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    assert set(payload.values()) <= {True, False}
    return payload


def _run_isolated_rag_settings_probe(tmp_path: Path) -> dict[str, bool]:
    return _run_isolated_json_probe(
        tmp_path,
        """
import json
from app.config import Settings, config

explicit = Settings(_env_file=None)
payload = {
    "global_index_defaults": (
        config.milvus_collection_name == "biz"
        and config.rag_schema_version == 1
        and config.rag_aiops_top_k >= config.rag_top_k
        and config.rag_vector_metric_type == "L2"
    ),
    "explicit_index_defaults": (
        explicit.milvus_collection_name == "biz"
        and explicit.rag_schema_version == 1
        and explicit.rag_aiops_top_k >= explicit.rag_top_k
        and explicit.rag_vector_metric_type == "L2"
    ),
    "global_sentence_defaults": (
        config.rag_sentence_window_enabled is False
        and config.rag_sentence_window_size >= 1
        and config.rag_sentence_min_unit_chars >= 1
        and config.rag_route_retrieval_enabled is True
        and config.rag_route_top_k >= 1
    ),
    "explicit_sentence_defaults": (
        explicit.rag_sentence_window_enabled is False
        and explicit.rag_sentence_window_size >= 1
        and explicit.rag_sentence_min_unit_chars >= 1
        and explicit.rag_route_retrieval_enabled is True
        and explicit.rag_route_top_k >= 1
    ),
}
print(json.dumps(payload, sort_keys=True))
""",
    )


def test_importing_app_does_not_load_config_or_logger(tmp_path: Path) -> None:
    result = _run_isolated_json_probe(
        tmp_path,
        """
import json
import sys
import app

payload = {
    "app_version_present": bool(app.__version__),
    "app_config_present": "app.config" in sys.modules,
    "app_logger_present": "app.utils.logger" in sys.modules,
}
print(json.dumps(payload, sort_keys=True))
""",
    )

    assert result == {
        "app_version_present": True,
        "app_config_present": False,
        "app_logger_present": False,
    }


def test_rag_pipeline_imports_and_explicit_builder_do_not_load_app_config(
    tmp_path: Path,
) -> None:
    result = _run_isolated_json_probe(
        tmp_path,
        """
import json
import sys
from app.rag.rag_deduplicator import RagDeduplicator
from app.rag.rag_document_loader import RagDocumentLoader
from app.rag.rag_ingestion_models import IngestionDecision, LoadedDocument
from app.rag.rag_metadata_builder import RagMetadataBuilder

_ = (RagDeduplicator, RagDocumentLoader, IngestionDecision, LoadedDocument)
after_top_level_imports = "app.config" in sys.modules
RagMetadataBuilder(schema_version=7, collection_name="IsolatedCollection")
payload = {
    "app_config_present_after_top_level_imports": after_top_level_imports,
    "app_config_present_after_explicit_builder": "app.config" in sys.modules,
}
print(json.dumps(payload, sort_keys=True))
""",
    )

    assert result == {
        "app_config_present_after_top_level_imports": False,
        "app_config_present_after_explicit_builder": False,
    }


def test_rag_metadata_builder_default_constructor_loads_isolated_config(
    tmp_path: Path,
) -> None:
    result = _run_isolated_json_probe(
        tmp_path,
        """
import json
import sys
from app.rag.rag_metadata_builder import RagMetadataBuilder

app_config_absent_before_default = "app.config" not in sys.modules
builder = RagMetadataBuilder()
payload = {
    "app_config_absent_before_default": app_config_absent_before_default,
    "app_config_present_after_default": "app.config" in sys.modules,
    "isolated_defaults_match": (
        builder.schema_version == 73
        and builder.collection_name == "IsolatedCollection"
    ),
}
print(json.dumps(payload, sort_keys=True))
""",
        config_text=("rag_schema_version: 73\n" "milvus_collection_name: IsolatedCollection\n"),
    )

    assert result == {
        "app_config_absent_before_default": True,
        "app_config_present_after_default": True,
        "isolated_defaults_match": True,
    }


def test_rag_metadata_builder_empty_collection_name_falls_back_to_config(
    tmp_path: Path,
) -> None:
    result = _run_isolated_json_probe(
        tmp_path,
        """
import json
import sys
from app.rag.rag_metadata_builder import RagMetadataBuilder

app_config_absent_before_builder = "app.config" not in sys.modules
builder = RagMetadataBuilder(schema_version=0, collection_name="")
payload = {
    "app_config_absent_before_builder": app_config_absent_before_builder,
    "app_config_present_after_builder": "app.config" in sys.modules,
    "explicit_schema_preserved": builder.schema_version == 0,
    "empty_collection_falls_back": builder.collection_name == "IsolatedCollection",
}
print(json.dumps(payload, sort_keys=True))
""",
        config_text=("rag_schema_version: 73\n" "milvus_collection_name: IsolatedCollection\n"),
    )

    assert result == {
        "app_config_absent_before_builder": True,
        "app_config_present_after_builder": True,
        "explicit_schema_preserved": True,
        "empty_collection_falls_back": True,
    }


def test_rag_metadata_builder_missing_schema_falls_back_to_config(
    tmp_path: Path,
) -> None:
    result = _run_isolated_json_probe(
        tmp_path,
        """
import json
import sys
from app.rag.rag_metadata_builder import RagMetadataBuilder

app_config_absent_before_builder = "app.config" not in sys.modules
builder = RagMetadataBuilder(schema_version=None, collection_name="ExplicitCollection")
payload = {
    "app_config_absent_before_builder": app_config_absent_before_builder,
    "app_config_present_after_builder": "app.config" in sys.modules,
    "missing_schema_falls_back": builder.schema_version == 73,
    "explicit_collection_preserved": builder.collection_name == "ExplicitCollection",
}
print(json.dumps(payload, sort_keys=True))
""",
        config_text=("rag_schema_version: 73\n" "milvus_collection_name: IsolatedCollection\n"),
    )

    assert result == {
        "app_config_absent_before_builder": True,
        "app_config_present_after_builder": True,
        "missing_schema_falls_back": True,
        "explicit_collection_preserved": True,
    }


def test_loader_computes_stable_file_identity(tmp_path: Path):
    file_path = tmp_path / "cpu.md"
    file_path.write_bytes("# CPU告警\n处理步骤".encode())

    loaded = RagDocumentLoader().load(str(file_path))

    assert loaded.normalized_path == file_path.resolve().as_posix()
    assert loaded.file_id
    assert loaded.file_hash
    assert loaded.file_size == len(file_path.read_bytes())
    assert loaded.extension == ".md"
    assert loaded.text == "# CPU告警\n处理步骤"
    assert loaded.metadata == {}


def test_loader_parses_frontmatter_metadata(tmp_path: Path):
    file_path = tmp_path / "invt.md"
    file_path.write_text(
        "---\n"
        "manufacturer: 英威腾\n"
        "device_type: inverter\n"
        "model: XD5\n"
        "---\n"
        "# 英威腾逆变器\n"
        "寄存器说明",
        encoding="utf-8",
    )

    loaded = RagDocumentLoader().load(str(file_path))

    assert loaded.metadata == {
        "manufacturer": "英威腾",
        "device_type": "inverter",
        "model": "XD5",
    }
    assert loaded.text == "# 英威腾逆变器\n寄存器说明"


def test_metadata_builder_adds_stable_chunk_metadata(tmp_path: Path):
    file_path = tmp_path / "cpu.md"
    file_path.write_text("# CPU告警\n## 处理方案\n扩容", encoding="utf-8")
    loaded = RagDocumentLoader().load(str(file_path))
    docs = [
        Document(
            page_content="## 处理方案\n扩容",
            metadata={
                "alert_name": "CPU告警",
                "runbook_section": "resolution",
                "h1": "CPU告警",
                "h2": "处理方案",
                "_original_text": "扩容",
                "_context_window": "## 处理方案\n扩容 checkout 实例",
                "_index_granularity": "sentence_window",
            },
        )
    ]

    enriched = RagMetadataBuilder(schema_version=7, collection_name="biz_v7").build(
        loaded=loaded,
        documents=docs,
        parser="markdown",
    )

    assert isinstance(enriched[0], DomainDocument)
    metadata = enriched[0].metadata
    assert metadata["_schema_version"] == 7
    assert metadata["_collection_name"] == "biz_v7"
    assert metadata["_file_id"] == loaded.file_id
    assert metadata["_file_hash"] == loaded.file_hash
    assert metadata["_chunk_id"]
    assert metadata["_chunk_hash"]
    assert metadata["_chunk_index"] == 0
    assert metadata["_chunk_count"] == 1
    assert metadata["alert_name"] == "CPU告警"
    assert metadata["runbook_section"] == "resolution"
    assert metadata["document_type"] == "runbook"
    assert metadata["heading_path"] == "CPU告警 > 处理方案"
    assert metadata["route_key"] == "CPU告警"
    assert metadata["_original_text"] == "扩容"
    assert metadata["_context_window"] == "## 处理方案\n扩容 checkout 实例"
    assert metadata["_index_granularity"] == "sentence_window"


def test_metadata_builder_normalizes_device_taxonomy_for_filtering(tmp_path: Path):
    file_path = tmp_path / "huawei-inverter.md"
    file_path.write_text("# PV告警\n## 处理方案\n复位逆变器", encoding="utf-8")
    loaded = RagDocumentLoader().load(str(file_path))
    docs = [
        Document(
            page_content="复位逆变器",
            metadata={
                "alert_name": "PVOverVoltage",
                "runbook_section": "resolution",
                "设备类型": "逆变器",
                "厂家": "华为",
            },
        )
    ]

    enriched = RagMetadataBuilder(schema_version=7, collection_name="biz_v7").build(
        loaded=loaded,
        documents=docs,
        parser="markdown",
    )

    metadata = enriched[0].metadata
    assert metadata["device_type"] == "inverter"
    assert metadata["manufacturer"] == "huawei"
    assert metadata["route_key"] == "inverter/huawei/PVOverVoltage"


def test_metadata_builder_infers_invt_inverter_from_file_context(tmp_path: Path):
    file_path = tmp_path / "英威腾逆变器寄存器协议.md"
    file_path.write_text("# 英威腾逆变器\n## 工作表：新列表\n寄存器说明", encoding="utf-8")
    loaded = RagDocumentLoader().load(str(file_path))
    docs = [
        Document(
            page_content="## 工作表：新列表\n0x3601 告警信息",
            metadata={"h1": "英威腾逆变器", "h2": "工作表：新列表"},
        )
    ]

    enriched = RagMetadataBuilder(schema_version=7, collection_name="biz_v7").build(
        loaded=loaded,
        documents=docs,
        parser="markdown",
    )

    metadata = enriched[0].metadata
    assert metadata["device_type"] == "inverter"
    assert metadata["manufacturer"] == "invt"
    assert metadata["document_type"] == "protocol"
    assert metadata["register_address"] == "0x3601"
    assert metadata["route_key"] == "inverter/invt/0x3601"


def test_metadata_builder_extracts_register_protocol_metadata(tmp_path: Path):
    file_path = tmp_path / "英威腾逆变器Modbus寄存器协议.md"
    file_path.write_text(
        "# 英威腾逆变器 Modbus 寄存器协议\n## 遥测寄存器\n",
        encoding="utf-8",
    )
    loaded = RagDocumentLoader().load(str(file_path))
    docs = [
        Document(
            page_content=(
                "地址: 0x3601\n"
                "名称: 告警状态\n"
                "功能码: 03\n"
                "数据类型: uint16\n"
                "单位: -\n"
                "倍率: 1\n"
                "读写: R"
            ),
            metadata={"h1": "英威腾逆变器 Modbus 寄存器协议", "h2": "遥测寄存器"},
        )
    ]

    enriched = RagMetadataBuilder(schema_version=7, collection_name="biz_v7").build(
        loaded=loaded,
        documents=docs,
        parser="markdown",
    )

    metadata = enriched[0].metadata
    assert metadata["document_type"] == "protocol"
    assert metadata["device_type"] == "inverter"
    assert metadata["manufacturer"] == "invt"
    assert metadata["protocol_type"] == "modbus"
    assert metadata["register_address"] == "0x3601"
    assert metadata["register_name"] == "告警状态"
    assert metadata["function_code"] == "03"
    assert metadata["data_type"] == "uint16"
    assert metadata["read_write"] == "R"
    assert metadata["route_key"] == "inverter/invt/0x3601"


class FakeVectorStore:
    def __init__(self, has_same_hash=False, existing_file=False, legacy_ids=None):
        self.has_same_hash = has_same_hash
        self.existing_file = existing_file
        self.legacy_ids = list(legacy_ids or [])
        self.captured_sources = []
        self.deleted_document_ids = []
        self.deleted_file_ids = []
        self.deleted_sources = []

    def has_file_hash(self, file_id, file_hash):
        return self.has_same_hash

    def has_file_id(self, file_id):
        return self.existing_file

    def delete_by_file_id(self, file_id):
        self.deleted_file_ids.append(file_id)
        return 3

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        return 0

    def get_legacy_ids_by_source(self, source):
        self.captured_sources.append(source)
        return list(self.legacy_ids)

    def delete_by_ids(self, document_ids):
        self.deleted_document_ids.append(list(document_ids))
        return len(document_ids)


def _loaded(path: str = "/tmp/cpu.md") -> LoadedDocument:
    return LoadedDocument(
        path=path,
        normalized_path=path,
        file_name="cpu.md",
        extension=".md",
        file_size=8,
        modified_time=1.0,
        raw_bytes=b"content",
        text="content",
        file_id="file-id",
        file_hash="file-hash",
    )


def test_deduplicator_skips_unchanged_file():
    store = FakeVectorStore(has_same_hash=True)

    decision = RagDeduplicator(store).prepare(_loaded())

    assert decision.action == "skip"
    assert decision.deleted_count == 0


def test_deduplicator_defers_changed_file_cleanup_until_new_generation_is_ready():
    store = FakeVectorStore(existing_file=True)

    decision = RagDeduplicator(store).prepare(_loaded())

    assert decision.action == "replace"
    assert decision.deleted_count == 0
    assert store.deleted_file_ids == []


def test_deduplicator_captures_legacy_ids_without_deleting_them():
    store = FakeVectorStore(legacy_ids=["legacy-1", "legacy-2"])

    decision = RagDeduplicator(store).prepare(_loaded())

    assert decision.action == "insert"
    assert decision.reason == "legacy_source_replaced"
    assert decision.legacy_ids == ["legacy-1", "legacy-2"]
    assert store.deleted_sources == []
    assert store.deleted_document_ids == []


class FakeLoader:
    def load(self, file_path):
        return _loaded(file_path)


class FakeDeduplicator:
    def __init__(self, decision):
        self.decision = decision

    def prepare(self, loaded):
        return self.decision


class FakeSplitter:
    def split_document(self, content, file_path):
        return [Document(page_content="content", metadata={})]


class FakeRunbookSplitter:
    def split_document(self, content, file_path):
        return [
            Document(
                page_content="CPUHigh 排查步骤",
                metadata={
                    "document_type": "runbook",
                    "alert_name": "CPUHigh",
                    "runbook_section": "troubleshooting",
                    "business_domain": "checkout",
                    "route_key": "CPUHigh",
                },
            )
        ]


class FakeDeviceRunbookSplitter:
    def split_document(self, content, file_path):
        return [
            Document(
                page_content="华为逆变器 PVOverVoltage 处理步骤",
                metadata={
                    "document_type": "runbook",
                    "alert_name": "PVOverVoltage",
                    "runbook_section": "resolution",
                    "设备类型": "逆变器",
                    "厂家": "华为",
                },
            )
        ]


class FakeInverterSplitter:
    def split_document(self, content, file_path):
        return [
            Document(
                page_content="# 英威腾逆变器\n## 工作表：新列表\n0x3601 告警信息",
                metadata={"h1": "英威腾逆变器", "h2": "工作表：新列表"},
            )
        ]


class FakeProtocolSplitter:
    def split_document(self, content, file_path):
        return [
            Document(
                page_content="0x3601 告警状态 功能码 03 uint16",
                metadata={
                    "document_type": "protocol",
                    "device_type": "inverter",
                    "manufacturer": "invt",
                    "protocol_type": "modbus",
                    "register_address": "0x3601",
                    "register_name": "告警状态",
                    "function_code": "03",
                    "data_type": "uint16",
                },
            )
        ]


class FakeMetadataBuilder:
    def build(self, loaded, documents, parser):
        return documents


class FakeStore:
    def __init__(self):
        self.added = []
        self.optimized = False

    def add_documents(self, docs):
        self.added.extend(docs)
        return ["id-1"]

    def optimize_collection(self):
        self.optimized = True
        return {"optimized": True}

    def delete_file_generations(self, file_id, keep_file_hash):
        return 0


class FakeSparseIndex:
    def __init__(self):
        self.upserted = []

    def upsert_documents(self, docs):
        self.upserted.extend(docs)


class FailingSplitter:
    def split_document(self, content, file_path):
        raise RuntimeError("splitter failed")


class EmptySplitter:
    def split_document(self, content, file_path):
        return []


class ReplacementStore:
    def __init__(self, add_error: str | None = None):
        self.add_error = add_error
        self.calls = []

    def has_file_hash(self, file_id, file_hash):
        return False

    def has_file_id(self, file_id):
        return True

    def delete_by_file_id(self, file_id):
        self.calls.append(("delete_all", file_id))
        return 2

    def delete_by_source(self, source):
        return 0

    def add_documents(self, docs):
        self.calls.append(("add", len(docs)))
        if self.add_error:
            raise RuntimeError(self.add_error)
        return [doc.metadata["_chunk_id"] for doc in docs]

    def delete_by_file_generation(self, file_id, file_hash):
        self.calls.append(("rollback", file_id, file_hash))
        return 0

    def delete_file_generations(self, file_id, keep_file_hash):
        self.calls.append(("cleanup", file_id, keep_file_hash))
        return 2

    def optimize_collection(self):
        self.calls.append(("optimize",))
        return {"optimized": True}


class RetryReplacementStore(ReplacementStore):
    def __init__(self):
        super().__init__()
        self.current_generation_present = False

    def has_file_hash(self, file_id, file_hash):
        return self.current_generation_present

    def add_documents(self, docs):
        ids = super().add_documents(docs)
        self.current_generation_present = True
        return ids


class CleanupRetryStore(RetryReplacementStore):
    def __init__(self):
        super().__init__()
        self.cleanup_attempts = 0

    def delete_file_generations(self, file_id, keep_file_hash):
        self.cleanup_attempts += 1
        self.calls.append(("cleanup", file_id, keep_file_hash))
        if self.cleanup_attempts == 1:
            raise RuntimeError("dense cleanup failed")
        return 2


class CurrentGenerationLookupFailingStore(ReplacementStore):
    def __init__(self, add_error: str):
        super().__init__(add_error=add_error)
        self.current_generation_present = True
        self.add_attempts = 0
        self.rollback_attempts = 0

    def has_file_hash(self, file_id, file_hash):
        raise RuntimeError("generation lookup unavailable")

    def add_documents(self, docs):
        self.add_attempts += 1
        return super().add_documents(docs)

    def delete_by_file_generation(self, file_id, file_hash):
        self.rollback_attempts += 1
        self.current_generation_present = False
        return 1


class FileIdLookupFailingStore(ReplacementStore):
    def __init__(self):
        super().__init__(add_error="dense write should not run")
        self.add_attempts = 0
        self.rollback_attempts = 0

    def has_file_id(self, file_id):
        raise RuntimeError("file id lookup unavailable")

    def add_documents(self, docs):
        self.add_attempts += 1
        return super().add_documents(docs)

    def delete_by_file_generation(self, file_id, file_hash):
        self.rollback_attempts += 1
        return 1


class RetryableSparseIndex:
    def __init__(self):
        self.fail_next_replace = True
        self.current_hashes = set()
        self.calls = []

    def contains_file_hash(self, file_id, file_hash):
        self.calls.append(("contains", file_id, file_hash))
        return (file_id, file_hash) in self.current_hashes

    def replace_file_documents(self, file_id, docs):
        self.calls.append(("replace", file_id, len(docs)))
        if self.fail_next_replace:
            self.fail_next_replace = False
            raise RuntimeError("sparse write failed")
        file_hash = docs[0].metadata["_file_hash"]
        self.current_hashes.add((file_id, file_hash))
        return 1

    def upsert_documents(self, docs):
        raise RuntimeError("sparse write failed")


class AlwaysFailingSparseIndex:
    def replace_file_documents(self, file_id, docs):
        raise RuntimeError("sparse write failed")


class LegacyStore(ReplacementStore):
    def __init__(self, add_error: str | None = None):
        super().__init__(add_error=add_error)
        self.active_ids = {"legacy-1", "legacy-2"}
        self.deleted_id_batches = []
        self.deleted_source_calls = []

    def has_file_id(self, file_id):
        return False

    def get_legacy_ids_by_source(self, source):
        return [
            document_id
            for document_id in ("legacy-1", "legacy-2")
            if document_id in self.active_ids
        ]

    def delete_by_source(self, source):
        self.deleted_source_calls.append(source)
        deleted = len(self.active_ids.intersection({"legacy-1", "legacy-2"}))
        self.active_ids.difference_update({"legacy-1", "legacy-2"})
        return deleted

    def add_documents(self, docs):
        ids = super().add_documents(docs)
        self.active_ids.update(ids)
        return ids

    def delete_by_ids(self, document_ids):
        batch = list(document_ids)
        self.deleted_id_batches.append(batch)
        deleted = len(self.active_ids.intersection(batch))
        self.active_ids.difference_update(batch)
        return deleted


class StaticLoader:
    def __init__(self, loaded):
        self.loaded = loaded

    def load(self, file_path):
        return self.loaded


class StaticSplitter:
    def __init__(self, documents):
        self.documents = documents

    def split_document(self, content, file_path):
        return list(self.documents)


class DuplicateRejectingStore(ReplacementStore):
    def __init__(self, occupied_ids):
        super().__init__()
        self.occupied_ids = set(occupied_ids)
        self.added_ids = []

    def add_documents(self, docs):
        ids = [doc.metadata["_chunk_id"] for doc in docs]
        if self.occupied_ids.intersection(ids):
            raise RuntimeError("duplicate primary key")
        self.calls.append(("add", len(docs)))
        self.occupied_ids.update(ids)
        self.added_ids.extend(ids)
        return ids


def make_indexing_service(
    *,
    store,
    splitter=None,
    loader=None,
    metadata_builder=None,
    deduplicator=None,
    sparse_index=None,
):
    return IndexingService(
        loader=loader or FakeLoader(),
        splitter=splitter or FakeSplitter(),
        metadata_builder=metadata_builder or FakeMetadataBuilder(),
        deduplicator=deduplicator
        or FakeDeduplicator(IngestionDecision(action="insert", reason="new")),
        vector_writer=store,
        sparse_writer=sparse_index or FakeSparseIndex(),
    )


class FakeInverterLoader:
    def load(self, file_path):
        return LoadedDocument(
            path=file_path,
            normalized_path=file_path,
            file_name="英威腾逆变器寄存器协议.md",
            extension=".md",
            file_size=8,
            modified_time=1.0,
            raw_bytes=b"content",
            text="content",
            file_id="invt-file-id",
            file_hash="invt-file-hash",
        )


def test_new_indexing_service_accepts_protocol_dependencies(tmp_path):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    sparse_index = FakeSparseIndex()
    service = IndexingService(
        loader=FakeLoader(),
        splitter=FakeSplitter(),
        metadata_builder=FakeMetadataBuilder(),
        deduplicator=FakeDeduplicator(
            IngestionDecision(action="insert", reason="new")
        ),
        vector_writer=store,
        sparse_writer=sparse_index,
    )

    outcome = service.index_single_file(str(path))

    assert outcome.inserted_chunks == 2
    assert sparse_index.upserted == store.added


def test_index_single_file_skips_unchanged_file(tmp_path, monkeypatch):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(
        store=store,
        deduplicator=FakeDeduplicator(IngestionDecision(action="skip", reason="unchanged")),
    )

    outcome = service.index_single_file(str(path))

    assert outcome.skipped is True
    assert store.added == []


def test_index_single_file_adds_enriched_documents(tmp_path, monkeypatch):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(store=store)

    outcome = service.index_single_file(str(path))

    assert outcome.inserted_chunks == 2
    assert len(store.added) == 2
    assert store.optimized is True
    assert any(doc.metadata.get("document_type") == "route_summary" for doc in store.added)


def test_index_single_file_syncs_sparse_index_after_vector_insert(tmp_path, monkeypatch):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    sparse_index = FakeSparseIndex()
    service = make_indexing_service(
        store=store,
        sparse_index=sparse_index,
    )

    outcome = service.index_single_file(str(path))

    assert outcome.inserted_chunks == 2
    assert sparse_index.upserted == store.added


def _replacement_service(store, splitter):
    return IndexingService(
        loader=FakeLoader(),
        splitter=splitter,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=RagDeduplicator(store),
        vector_writer=store,
        sparse_writer=FakeSparseIndex(),
    )


def test_replacement_splitter_failure_keeps_old_dense_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = ReplacementStore()
    service = _replacement_service(store, FailingSplitter())

    with pytest.raises(RuntimeError, match="splitter failed"):
        service.index_single_file(str(path))

    assert not any(call[0] == "delete_all" for call in store.calls)


@pytest.mark.parametrize(
    "failure",
    ["embedding failed", "dense write failed"],
    ids=["embedding", "dense-write"],
)
def test_replacement_dense_failure_keeps_old_generation(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = ReplacementStore(add_error=failure)
    service = _replacement_service(store, FakeSplitter())

    with pytest.raises(RuntimeError, match=failure):
        service.index_single_file(str(path))

    assert not any(call[0] == "delete_all" for call in store.calls)
    assert ("rollback", "file-id", "file-hash") in store.calls


@pytest.mark.parametrize(
    "add_error",
    ["embedding failed", "dense write failed"],
    ids=["embedding", "dense-write"],
)
def test_generation_lookup_failure_never_adds_or_rolls_back_current_generation(
    tmp_path, monkeypatch, add_error
):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = CurrentGenerationLookupFailingStore(add_error=add_error)
    service = _replacement_service(store, FakeSplitter())

    with pytest.raises(RuntimeError) as exc_info:
        service.index_single_file(str(path))

    assert "generation lookup unavailable" in str(exc_info.value)
    assert store.add_attempts == 0
    assert store.rollback_attempts == 0
    assert store.current_generation_present is True


def test_file_id_lookup_failure_never_starts_dense_write_or_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FileIdLookupFailingStore()
    service = _replacement_service(store, FakeSplitter())

    with pytest.raises(RuntimeError) as exc_info:
        service.index_single_file(str(path))

    assert "file id lookup unavailable" in str(exc_info.value)
    assert store.add_attempts == 0
    assert store.rollback_attempts == 0


def test_unverified_ingestion_decision_cannot_rollback_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = ReplacementStore(add_error="dense write failed")
    service = make_indexing_service(
        store=store,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=FakeDeduplicator(IngestionDecision(action="insert", reason="unverified")),
    )

    with pytest.raises(RuntimeError, match="dense write failed"):
        service.index_single_file(str(path))

    assert not any(call[0] == "rollback" for call in store.calls)


def test_replacement_empty_chunks_keeps_old_generation_and_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = ReplacementStore()
    service = _replacement_service(store, EmptySplitter())

    with pytest.raises(RuntimeError, match="no indexable chunks"):
        service.index_single_file(str(path))

    assert not any(call[0] == "delete_all" for call in store.calls)


def test_replacement_cleans_old_dense_generation_only_after_new_write(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = ReplacementStore()
    service = _replacement_service(store, FakeSplitter())

    outcome = service.index_single_file(str(path))

    assert outcome.deleted_count == 2
    assert [call[0] for call in store.calls] == ["add", "cleanup", "optimize"]
    assert store.calls[1] == ("cleanup", "file-id", "file-hash")


def test_unchanged_retry_repairs_sparse_before_cleaning_old_dense_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = RetryReplacementStore()
    sparse_index = RetryableSparseIndex()
    service = make_indexing_service(
        store=store,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=RagDeduplicator(store),
        sparse_index=sparse_index,
    )

    with pytest.raises(RuntimeError, match="sparse write failed"):
        service.index_single_file(str(path))

    assert not any(call[0] == "cleanup" for call in store.calls)

    outcome = service.index_single_file(str(path))

    assert outcome.skipped is False
    assert outcome.reason == "sparse_repaired"
    assert sparse_index.contains_file_hash("file-id", "file-hash") is True
    assert [call[0] for call in store.calls].count("add") == 1
    assert [call[0] for call in store.calls].count("cleanup") == 1


def test_unchanged_retry_retries_failed_obsolete_dense_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = CleanupRetryStore()
    service = make_indexing_service(
        store=store,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=RagDeduplicator(store),
    )

    with pytest.raises(RuntimeError, match="dense cleanup failed"):
        service.index_single_file(str(path))

    outcome = service.index_single_file(str(path))

    assert outcome.skipped is True
    assert outcome.deleted_count == 2
    assert store.cleanup_attempts == 2
    assert [call[0] for call in store.calls].count("add") == 1


@pytest.mark.parametrize("failure_stage", ["splitter", "dense", "sparse"])
def test_legacy_rows_survive_ingestion_failures(tmp_path, monkeypatch, failure_stage):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = LegacyStore(add_error="dense write failed" if failure_stage == "dense" else None)
    splitter = FailingSplitter() if failure_stage == "splitter" else FakeSplitter()
    sparse_index = AlwaysFailingSparseIndex() if failure_stage == "sparse" else FakeSparseIndex()
    service = make_indexing_service(
        store=store,
        splitter=splitter,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=RagDeduplicator(store),
        sparse_index=sparse_index,
    )

    with pytest.raises(RuntimeError):
        service.index_single_file(str(path))

    assert {"legacy-1", "legacy-2"}.issubset(store.active_ids)
    assert store.deleted_source_calls == []
    assert store.deleted_id_batches == []


def test_successful_ingestion_deletes_only_captured_legacy_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = LegacyStore()
    service = make_indexing_service(
        store=store,
        metadata_builder=RagMetadataBuilder(schema_version=1, collection_name="test"),
        deduplicator=RagDeduplicator(store),
    )

    outcome = service.index_single_file(str(path))

    assert outcome.deleted_count == 2
    assert store.deleted_source_calls == []
    assert store.deleted_id_batches == [["legacy-1", "legacy-2"]]
    assert set(outcome.document_ids).issubset(store.active_ids)
    assert "legacy-1" not in store.active_ids
    assert "legacy-2" not in store.active_ids


def test_partial_overlap_replacement_uses_generation_scoped_primary_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indexing_service_module.config, "rag_route_retrieval_enabled", False
    )
    path = tmp_path / "cpu.md"
    path.write_text("new content", encoding="utf-8")
    old_loaded = replace(_loaded(str(path)), file_hash="old-file-hash")
    new_loaded = replace(_loaded(str(path)), file_hash="new-file-hash", text="new content")
    old_chunks = [
        Document(page_content="unchanged first chunk", metadata={}),
        Document(page_content="old second chunk", metadata={}),
    ]
    new_chunks = [
        Document(page_content="unchanged first chunk", metadata={}),
        Document(page_content="new second chunk", metadata={}),
    ]
    builder = RagMetadataBuilder(schema_version=1, collection_name="test")
    old_ids = {
        doc.metadata["_chunk_id"]
        for doc in builder.build(old_loaded, old_chunks, parser="markdown")
    }
    expected_new_ids = [
        doc.metadata["_chunk_id"]
        for doc in builder.build(new_loaded, new_chunks, parser="markdown")
    ]
    retry_ids = [
        doc.metadata["_chunk_id"]
        for doc in builder.build(new_loaded, new_chunks, parser="markdown")
    ]
    store = DuplicateRejectingStore(old_ids)
    service = make_indexing_service(
        store=store,
        loader=StaticLoader(new_loaded),
        splitter=StaticSplitter(new_chunks),
        metadata_builder=builder,
        deduplicator=RagDeduplicator(store),
    )

    outcome = service.index_single_file(str(path))

    assert outcome.inserted_chunks == 2
    assert expected_new_ids == retry_ids
    assert old_ids.isdisjoint(expected_new_ids)
    assert store.added_ids == expected_new_ids
    assert ("cleanup", "file-id", "new-file-hash") in store.calls


def test_index_single_file_adds_file_route_summary_without_runbook_metadata(tmp_path, monkeypatch):
    path = tmp_path / "general.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(store=store)

    service.index_single_file(str(path))

    route_doc = next(
        doc for doc in store.added if doc.metadata.get("document_type") == "route_summary"
    )
    assert route_doc.metadata["target_document_type"] == "document"
    assert route_doc.metadata["route_key"] == "cpu.md"
    assert "cpu.md" in route_doc.page_content


def test_index_single_file_adds_route_summary_document(tmp_path, monkeypatch):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(
        store=store,
        splitter=FakeRunbookSplitter(),
    )

    outcome = service.index_single_file(str(path))

    route_docs = [
        doc for doc in store.added if doc.metadata.get("document_type") == "route_summary"
    ]
    assert outcome.inserted_chunks == 2
    assert len(route_docs) == 1
    assert route_docs[0].metadata["target_document_type"] == "runbook"
    assert route_docs[0].metadata["alert_name"] == "CPUHigh"
    assert route_docs[0].metadata["business_domain"] == "checkout"
    assert route_docs[0].metadata["route_key"] == "CPUHigh"
    assert "CPUHigh" in route_docs[0].page_content


def test_index_single_file_adds_device_taxonomy_to_route_summary(tmp_path, monkeypatch):
    path = tmp_path / "pv.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(
        store=store,
        splitter=FakeDeviceRunbookSplitter(),
    )

    service.index_single_file(str(path))

    route_doc = next(
        doc for doc in store.added if doc.metadata.get("document_type") == "route_summary"
    )
    assert route_doc.metadata["target_document_type"] == "runbook"
    assert route_doc.metadata["device_type"] == "inverter"
    assert route_doc.metadata["manufacturer"] == "huawei"
    assert route_doc.metadata["route_key"] == "inverter/huawei/PVOverVoltage"
    assert "device_type: inverter" in route_doc.page_content
    assert "manufacturer: huawei" in route_doc.page_content


def test_index_single_file_infers_file_taxonomy_for_route_summary(tmp_path, monkeypatch):
    path = tmp_path / "英威腾逆变器寄存器协议.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(
        store=store,
        loader=FakeInverterLoader(),
        splitter=FakeInverterSplitter(),
    )

    service.index_single_file(str(path))

    route_doc = next(
        doc for doc in store.added if doc.metadata.get("document_type") == "route_summary"
    )
    assert route_doc.metadata["target_document_type"] == "protocol"
    assert route_doc.metadata["device_type"] == "inverter"
    assert route_doc.metadata["manufacturer"] == "invt"
    assert route_doc.metadata["route_key"] == "inverter/invt/0x3601"
    assert route_doc.metadata["register_address"] == "0x3601"
    assert "file_name: 英威腾逆变器寄存器协议.md" in route_doc.page_content
    assert "manufacturer: invt" in route_doc.page_content
    assert "device_type: inverter" in route_doc.page_content


def test_index_single_file_adds_protocol_route_summary(tmp_path, monkeypatch):
    path = tmp_path / "英威腾逆变器Modbus寄存器协议.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = make_indexing_service(
        store=store,
        loader=FakeInverterLoader(),
        splitter=FakeProtocolSplitter(),
    )

    service.index_single_file(str(path))

    route_doc = next(
        doc for doc in store.added if doc.metadata.get("document_type") == "route_summary"
    )
    assert route_doc.metadata["target_document_type"] == "protocol"
    assert route_doc.metadata["route_key"] == "inverter/invt/0x3601"
    assert route_doc.metadata["register_address"] == "0x3601"
    assert route_doc.metadata["register_name"] == "告警状态"
    assert "protocol_type: modbus" in route_doc.page_content
    assert "register_address: 0x3601" in route_doc.page_content


def test_rag_index_defaults_are_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", str(tmp_path / "parent-only.yaml"))
    monkeypatch.setenv("MILVUS_COLLECTION_NAME", "ParentCollection")
    monkeypatch.setenv("RAG_SCHEMA_VERSION", "99")
    result = _run_isolated_rag_settings_probe(tmp_path)

    assert result["global_index_defaults"] is True
    assert result["explicit_index_defaults"] is True


def test_rag_sentence_window_defaults_are_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", str(tmp_path / "parent-only.yaml"))
    monkeypatch.setenv("RAG_SENTENCE_WINDOW_ENABLED", "true")
    monkeypatch.setenv("RAG_ROUTE_RETRIEVAL_ENABLED", "false")
    result = _run_isolated_rag_settings_probe(tmp_path)

    assert result["global_sentence_defaults"] is True
    assert result["explicit_sentence_defaults"] is True
