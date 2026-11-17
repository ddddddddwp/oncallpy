from types import SimpleNamespace

from app.rag.infrastructure.dashscope_embeddings import (
    DashScopeEmbeddings as InfrastructureDashScopeEmbeddings,
)
from app.rag.vector_embedding_service import DashScopeEmbeddings


def test_legacy_embedding_class_uses_infrastructure_implementation() -> None:
    assert issubclass(DashScopeEmbeddings, InfrastructureDashScopeEmbeddings)


class _FakeEmbeddingsResource:
    def __init__(self):
        self.batch_sizes = []

    def create(self, *, model, input, dimensions, encoding_format):
        if len(input) > 10:
            raise AssertionError("embedding batch exceeded provider limit")

        self.batch_sizes.append(len(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index)] * dimensions)
                for index, _ in enumerate(input)
            ]
        )


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsResource()


def test_embed_documents_batches_provider_requests_to_ten_inputs():
    embeddings = DashScopeEmbeddings(api_key="sk-test", dimensions=4)
    fake_client = _FakeClient()
    embeddings.client = fake_client

    result = embeddings.embed_documents([f"text-{index}" for index in range(25)])

    assert fake_client.embeddings.batch_sizes == [10, 10, 5]
    assert len(result) == 25
    assert all(len(vector) == 4 for vector in result)
