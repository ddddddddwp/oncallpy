"""知识检索工具 - 从向量数据库中检索相关信息"""


from copy import deepcopy

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.rag.bootstrap import get_query_service
from app.rag.models import SearchQuery
from app.rag.query.errors import safe_error_summary


@tool(response_format="content_and_artifact")
def retrieve_knowledge(
    query: str,
    profile: str = "chat",
    top_k: int | None = None,
) -> tuple[str, list[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询
        profile: 检索策略，chat 用于普通聊天，aiops 用于运维诊断，protocol 用于厂商寄存器协议
        top_k: 返回结果数量，默认使用 profile 对应配置

    Returns:
        tuple[str, list[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}', profile={profile}, top_k={top_k}")

        domain_docs = get_query_service().retrieve(
            SearchQuery(
                text=query,
                profile=profile,
                top_k=top_k,
            )
        )
        docs = [
            Document(
                page_content=document.page_content,
                metadata=deepcopy(document.metadata),
                **({"id": document.id} if document.id else {}),
            )
            for document in domain_docs
        ]

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as exc:
        message = safe_error_summary(exc)
        logger.error(f"知识检索工具调用失败: {message}")
        return f"检索知识时发生错误: {message}", []


def format_docs(docs: list[Document]) -> str:
    """
    格式化文档列表为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
