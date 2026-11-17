"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口"""

# 该适配器把 DashScope 的 OpenAI 兼容 API 包装成 LangChain/本项目端口可调用的 embedding 能力。
from langchain_core.embeddings import Embeddings
from loguru import logger
from openai import OpenAI


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding (OpenAI 兼容模式)

    实现 LangChain 标准 Embeddings 接口:
    - embed_documents(texts: List[str]) → List[List[float]]: 批量嵌入文档
    - embed_query(text: str) → List[float]: 嵌入单个查询
    """

    # 服务端对单次输入数量有限制，分批发送也能控制请求体与失败重试的影响范围。
    MAX_BATCH_SIZE = 10

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        """
        初始化 DashScope Embeddings

        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称
            dimensions: 向量维度
        """
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        # DashScope 提供 OpenAI 兼容 endpoint，因此复用标准客户端而不向上游暴露供应商 SDK。
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimensions

        # 打印初始化信息
        # 日志只显示掩码后的 key，既能辅助确认配置来源，又避免完整凭据泄露。
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API Key 用于日志"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档列表 (LangChain 标准接口)

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            # 按原输入顺序逐批追加，最终向量下标仍与 texts 下标一一对应。
            embeddings = []
            for start in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[start : start + self.MAX_BATCH_SIZE]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                # API 返回的 data 顺序构成批内映射；extend 再保持批次间顺序。
                embeddings.extend(item.embedding for item in response.data)

            logger.debug(f"批量嵌入完成, 维度: {len(embeddings[0])}")

            return embeddings

        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询文本 (LangChain 标准接口)

        Args:
            text: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            # 查询向量必须与文档使用同一模型和 dimensions，Milvus 距离才有意义。
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )

            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e
