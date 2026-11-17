"""向量检索服务模块"""

# 这是直接调用 pymilvus 的旧兼容检索入口；新查询主流程优先使用 QueryService + VectorSearcher。
from typing import Any

from loguru import logger
from pymilvus import Collection

from app.config import config
from app.rag.milvus_client import milvus_manager
from app.rag.vector_embedding_service import vector_embedding_service


class SearchResult:
    """搜索结果类"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: dict[str, Any],
    ):
        # 简单 DTO 隔离 pymilvus hit.entity，让上层只接触稳定字段。
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """向量检索服务 - 负责从 Milvus 中搜索相似向量"""

    def __init__(self):
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    def search_similar_documents(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回最相似的K个结果

        Returns:
            List[SearchResult]: 搜索结果列表

        Raises:
            RuntimeError: 搜索失败时抛出
        """
        try:
            logger.info(f"开始搜索相似文档, 查询: {query}, topK: {top_k}")

            # 1. 将查询文本向量化
            # 查询 embedding 必须与 collection 中的文档向量使用同一模型和 1024 维度。
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功, 维度: {len(query_vector)}")

            # 2. 获取 collection
            collection: Collection = milvus_manager.get_collection()

            # 3. 构建搜索参数
            # metric_type 与索引一致；nprobe 越大通常召回更充分但查询成本也更高。
            search_params = {
                "metric_type": config.rag_vector_metric_type,
                "params": {"nprobe": config.rag_vector_nprobe},
            }

            # 4. 执行搜索
            # data 是批量接口，即使单条查询也需包成长度为 1 的向量列表。
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            # 5. 解析搜索结果
            search_results = []
            # 外层对应每条查询向量，内层是该查询按距离排序的命中项。
            for hits in results:
                for hit in hits:
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # L2 距离，越小越相似
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成, 找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e


# 全局单例
# 兼容模块保留共享 service；它本身无连接状态，collection 由 milvus_manager 管理。
vector_search_service = VectorSearchService()
