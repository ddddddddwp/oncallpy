"""RAG Agent 服务 - 基于 LangGraph 的智能代理"""

# 这是 RAG 用例之外的 Agent 入口：负责模型、工具、会话 checkpoint 和流式事件协议。
from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import config
from app.core.checkpoint import (
    close_checkpointer,
    create_redis_checkpointer,
    delete_checkpoint_thread,
    get_latest_checkpoint,
)
from app.core.llm_factory import llm_factory
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry

class AgentState(TypedDict):
    """Agent 状态"""
    # add_messages reducer 让 LangGraph 按消息 ID 合并状态，而不是简单覆盖整个列表。
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    # 中间件只在历史超过阈值时返回状态补丁；None 表示保持原消息序列。
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    # 奇偶分支尽量保留完整问答轮次，避免窗口从孤立的 assistant/tool 消息开始。
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    # 先发送 REMOVE_ALL_MESSAGES，再追加保留项，符合 LangGraph reducer 的删除协议。
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        # 模型由统一 factory 创建，服务只依赖标准 chat model 行为。
        self.model = llm_factory.create_chat_model(
            model=config.rag_model,
            streaming=streaming,
        )
        self.model_name = getattr(self.model, "model", self.model_name)

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # Redis checkpointer 延迟初始化，避免模块导入时建立网络连接
        self.checkpointer = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成, model={self.model_name}, streaming={streaming}")

    async def _initialize_checkpointer(self):
        """初始化 Redis checkpointer。"""
        # Redis 连接推迟到首次会话操作，构造服务不会立即访问网络。
        if self.checkpointer is None:
            self.checkpointer = await create_redis_checkpointer()

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        # 初始化标志保证同一服务实例不会重复加载 MCP tools 或创建 Agent graph。
        if self._agent_initialized:
            return

        await self._initialize_checkpointer()

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具
        # MCP 工具列表在运行时发现，可与内置知识检索/时间工具共同交给模型选择。
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        # checkpointer 使用 session_id/thread_id 隔离会话状态，tools 则构成 Agent 可执行能力集。
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        # 局部导入仅在构建 prompt 时需要，dedent 让源码缩进不进入实际系统消息。
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            # 非流式与流式入口共享同一个延迟初始化 Agent 和 checkpoint 状态。
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # thread_id 是 Redis checkpoint 的会话键，相同 session 可延续历史状态。
            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                # 不同模型可能返回纯字符串或 reasoning/text 结构块，这里统一提取最终文本。
                raw_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                # 处理结构化 content_blocks（如 codex-MiniMax-M2.7 返回的 reasoning + text 块）
                if isinstance(raw_content, list):
                    text_parts = []
                    # reasoning 等非 text 块不直接暴露给最终用户，只拼接 type=text 的内容。
                    for block in raw_content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    answer = "\n".join(text_parts) if text_parts else str(raw_content)
                else:
                    answer = raw_content

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # messages 模式逐个产出模型消息块，同时附带当前 LangGraph node 元数据。
            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                # 只向客户端转发 AI 文本；tool/内部状态消息由 Agent 自己消费。
                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            # complete 是显式流终止事件，客户端无需依赖网络断开判断完成。
            yield {"type": "complete"}

        except Exception as e:
            # 先产出结构化 error 事件，再重新抛出，兼顾流消费者与服务端错误处理。
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    async def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 Redis checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            await self._initialize_checkpointer()
            config_dict = {"configurable": {"thread_id": session_id}}
            # 读取最新 checkpoint，而不是重新调用 Agent，历史查询不会消耗模型请求。
            checkpoint_data = await get_latest_checkpoint(self.checkpointer, config_dict)

            if not checkpoint_data:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 从检查点中提取消息
            # LangGraph 把 reducer 后的消息列表保存在 channel_values.messages。
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history = []
            # 前端历史只需要 user/assistant；系统 prompt 属于服务配置，不展示给用户。
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                # 旧消息没有 timestamp 时用当前时间补齐，维持前端数据形状稳定。
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 Redis checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            await self._initialize_checkpointer()
            # 按 thread 删除 checkpoint，不影响其他 session 或共享 Agent 实例。
            await delete_checkpoint_thread(self.checkpointer, session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            # MCP 客户端由外部全局管理；本服务只关闭自己持有的 Redis checkpointer。
            if self.checkpointer is not None:
                await close_checkpointer(self.checkpointer)
                self.checkpointer = None
                self.agent = None
                self._agent_initialized = False
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
# 保留历史模块级服务入口；真正的外部连接仍由异步初始化延迟建立。
rag_agent_service = RagAgentService(streaming=True)
