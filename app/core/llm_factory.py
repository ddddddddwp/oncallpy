"""LLM 工厂类

统一创建聊天模型实例，默认使用 MiniMax，支持按配置切换。

默认行为：
- CHAT_PROVIDER=minimax（默认）
- 对应使用 MiniMax 的 OpenAI 兼容接口
- 当 provider=dashscope 时使用阿里云 DashScope 兼容接口
- 当 provider=deepseek 时使用 DeepSeek OpenAI 兼容接口
"""

from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    # 阿里云 DashScope OpenAI 兼容模式 URL
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    PROVIDER_DASHSCOPE = "dashscope"
    PROVIDER_MINIMAX = "minimax"
    PROVIDER_DEEPSEEK = "deepseek"
    SUPPORTED_PROVIDERS = {
        PROVIDER_DASHSCOPE,
        PROVIDER_MINIMAX,
        PROVIDER_DEEPSEEK,
    }

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> ChatOpenAI:
        provider = (provider or config.chat_provider).strip().lower()
        if provider not in LLMFactory.SUPPORTED_PROVIDERS:
            logger.warning(f"不支持的 CHAT_PROVIDER={provider}，回退到 minimax")
            provider = LLMFactory.PROVIDER_MINIMAX

        if provider == LLMFactory.PROVIDER_MINIMAX:
            model = model or config.minimax_model
            base_url = base_url or config.minimax_base_url or LLMFactory.MINIMAX_BASE_URL
            api_key = api_key or config.minimax_api_key
        elif provider == LLMFactory.PROVIDER_DASHSCOPE:
            model = model or config.dashscope_model
            base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
            api_key = api_key or config.dashscope_api_key
        else:
            model = model or config.deepseek_model
            base_url = base_url or config.deepseek_base_url or LLMFactory.DEEPSEEK_BASE_URL
            api_key = api_key or config.deepseek_api_key

        extra_body = {"stream": streaming} if provider == LLMFactory.PROVIDER_DASHSCOPE else None

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
            timeout=timeout,
        )

        return llm

# 全局 LLM 工厂实例
llm_factory = LLMFactory()
