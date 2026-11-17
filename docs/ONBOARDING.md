# EnergyAgent 项目学习文档

本文用于帮助新加入项目的人快速理解 EnergyAgent 的整体结构、运行方式和主要业务流程。文档只讲架构、模块职责和数据流，不包含具体源码片段。

## 1. 项目一句话概括

EnergyAgent 是一个面向智能 OnCall 场景的后端应用，核心能力是：

- 通过 RAG 知识库回答用户问题。
- 通过 AIOps Agent 自动执行运维诊断流程。
- 通过 MCP 接入日志查询和监控查询工具。
- 通过 Web 页面提供聊天、文件上传和一键智能运维入口。

从用户视角看，它像一个智能运维助手；从工程视角看，它是 FastAPI + LangChain/LangGraph + Milvus + MCP 的组合应用。

## 2. 当前技术栈

| 类别 | 使用内容 | 作用 |
| --- | --- | --- |
| Web 后端 | FastAPI、Uvicorn、sse-starlette | 暴露 HTTP API、SSE 流式响应和静态页面 |
| Agent 编排 | LangChain、LangGraph、langchain-qwq | 构建 RAG Agent 和 Plan-Execute-Replan 工作流 |
| 大模型 | 阿里云 DashScope / 通义千问 | 聊天、规划、执行、重规划、生成报告 |
| Embedding | DashScope text-embedding-v4 | 将知识库文档和用户查询转换为向量 |
| 向量数据库 | Milvus | 存储和检索知识库分片 |
| 工具接入 | MCP、FastMCP、langchain-mcp-adapters | 将日志、监控等能力暴露给 Agent 调用 |
| 前端 | 原生 HTML/CSS/JavaScript | 聊天页面、流式渲染、文件上传、AIOps 入口 |
| 日志 | Loguru | 控制台日志、按天轮转文件日志 |
| 开发工具 | ruff、black、isort、mypy、pyright、pytest 配置 | 代码格式、静态检查和测试配置 |

项目配置文件 `pyproject.toml` 要求 Python 版本为 `>=3.11,<3.14`。README 中写的 Python 3.10+ 和配置文件不完全一致，实际开发应以 `pyproject.toml` 为准。

## 3. 运行时组件总览

项目运行时大致分成六块：

| 组件 | 代表位置 | 职责 |
| --- | --- | --- |
| FastAPI 主应用 | `app/main.py` | 创建应用、注册路由、挂载静态页面、启动和关闭 Milvus 连接 |
| API 路由层 | `app/api/` | 对外暴露聊天、文件上传、AIOps、健康检查等接口 |
| 业务服务层 | `app/services/` | 实现 RAG、AIOps、向量索引、向量检索、文档切分等业务逻辑 |
| Agent 层 | `app/agent/` | 管理 MCP 客户端和 AIOps 的 planner、executor、replanner 节点 |
| 基础设施层 | `app/core/` | 管理 LLM 工厂和 Milvus 连接/collection |
| Web 前端 | `static/` | 提供浏览器聊天界面、历史记录、上传入口和 AIOps 按钮 |

额外还有两类辅助资源：

- `mcp_servers/`：本地 MCP 服务，用 mock 数据模拟日志查询和监控查询。
- `aiops-docs/`：内置运维经验文档，用于上传到向量库，辅助 RAG 和 AIOps 规划。

## 4. 目录结构怎么读

### `app/main.py`

这是 FastAPI 入口。它做四件事：

- 创建 FastAPI 应用实例。
- 配置 CORS。
- 注册各个 API router。
- 挂载 `static/` 下的前端页面。
- 在应用生命周期里连接和关闭 Milvus。

理解项目时应先看它，因为它告诉你外部请求会进入哪些模块。

### `app/api/`

这是 HTTP 边界层，只负责接收请求、调用服务、包装响应。

| 文件 | 主要职责 |
| --- | --- |
| `chat.py` | 普通聊天、流式聊天、清空会话、查询会话历史 |
| `aiops.py` | 一键 AIOps 诊断，返回 SSE 流 |
| `file.py` | 上传 txt/md 文件，并触发向量索引 |
| `health.py` | 检查应用和 Milvus 连接状态 |

重要点：健康检查路由在代码中注册为 `/health`，不是 `/api/health`。README 的接口表和实际代码存在差异，学习和调试时以代码为准。

### `app/services/`

这是项目最重要的业务层。

| 文件 | 主要职责 |
| --- | --- |
| `rag_agent_service.py` | 构建 RAG Agent，处理普通问答和流式问答 |
| `aiops_service.py` | 构建 Plan-Execute-Replan 状态图，驱动 AIOps 诊断 |
| `document_splitter_service.py` | 将上传文档切分成适合向量化的分片 |
| `vector_embedding_service.py` | 调 DashScope embedding 接口生成 1024 维向量 |
| `vector_store_manager.py` | 封装 LangChain Milvus VectorStore 的写入、删除和相似度检索 |
| `vector_index_service.py` | 读取文件、删除旧索引、切分文档、写入 Milvus |
| `vector_search_service.py` | 直接使用 pymilvus 搜索相似文档 |

`rag_agent_service.py` 和 `aiops_service.py` 是两条主业务线；向量服务则支撑知识库能力。

### `app/agent/`

这里负责 Agent 的工具接入和 AIOps 节点。

| 文件 | 主要职责 |
| --- | --- |
| `mcp_client.py` | 创建全局 MCP 客户端，给工具调用加重试拦截 |
| `aiops/planner.py` | 根据任务、知识库经验和工具列表生成诊断计划 |
| `aiops/executor.py` | 执行当前计划步骤，按需调用工具 |
| `aiops/replanner.py` | 判断继续执行、重规划或生成最终报告 |
| `aiops/state.py` | 定义 Plan-Execute-Replan 工作流状态 |
| `aiops/utils.py` | 格式化工具描述等辅助逻辑 |

### `app/tools/`

本地工具目前比较少：

- `retrieve_knowledge`：从 Milvus 知识库检索相关文档。
- `get_current_time`：返回指定时区当前时间。

这些本地工具会和 MCP 工具合并，一起给 Agent 使用。

### `app/core/`

基础设施封装：

- `milvus_client.py`：连接 Milvus，创建/加载 `biz` collection，检查向量维度，创建向量索引。
- `llm_factory.py`：统一的聊天模型工厂，默认走 MiniMax 的 OpenAI 兼容接口，支持按配置切换到 DashScope。

### `static/`

这是原生前端，不依赖 React/Vue。

| 文件 | 主要职责 |
| --- | --- |
| `index.html` | 页面结构，包含侧边栏、聊天区、输入框、上传入口和 AIOps 按钮 |
| `app.js` | 前端状态管理、接口调用、SSE 解析、Markdown 渲染、localStorage 历史记录 |
| `styles.css` | 页面样式 |

前端调用后端接口时主要走 `/api/chat`、`/api/chat_stream`、`/api/upload` 和 `/api/aiops`。

### `mcp_servers/`

本地 MCP 服务用于模拟真实运维系统：

| 文件 | 服务 | 默认端口 | 能力 |
| --- | --- | --- | --- |
| `cls_server.py` | CLS 日志服务 | 8003 | 主题查询、日志搜索、服务日志查询、日志模式分析等 |
| `monitor_server.py` | Monitor 监控服务 | 8004 | CPU、内存、进程、服务信息、历史工单等 |

当前 MCP 服务返回 mock 数据。要接入生产环境，需要把这些工具内部的数据来源替换为真实日志平台、监控平台或工单系统。

### `aiops-docs/`

这是内置运维知识库文档，包含 CPU、内存、磁盘、服务不可用、慢响应等告警处理方案。项目初始化时可以把这些文档上传到向量库，供 RAG 和 AIOps Planner 检索参考。

## 5. 对外 API 总览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 返回前端首页；如果静态文件不存在，则返回 API 欢迎信息 |
| GET | `/health` | 健康检查，包含 Milvus 连接状态 |
| POST | `/api/chat` | 普通问答，一次性返回完整回答 |
| POST | `/api/chat_stream` | 流式问答，通过 SSE 返回内容片段 |
| POST | `/api/chat/clear` | 清空指定会话历史 |
| GET | `/api/chat/session/{session_id}` | 查询指定会话历史 |
| POST | `/api/upload` | 上传 txt/md 文件，并自动建立向量索引 |
| POST | `/api/index_directory` | 索引指定目录下的 txt/md 文件 |
| POST | `/api/aiops` | 启动 AIOps 诊断，通过 SSE 返回计划、步骤、报告和完成事件 |

注意：前端允许选择 `.markdown` 文件，但后端只允许 `txt` 和 `md`。前端文件大小限制是 50MB，后端限制是 10MB。这是一个前后端约束不一致的地方。

## 6. 启动流程

应用启动时主要发生这些事：

1. Uvicorn 加载 `app.main:app`。
2. `app/__init__.py` 引入日志配置，Loguru 开始接管日志输出。
3. FastAPI lifespan 执行启动逻辑。
4. 应用读取 `.env` 和默认配置。
5. 应用连接 Milvus。
6. 如果 `biz` collection 不存在，则创建 collection 和向量索引。
7. 如果 collection 已存在，会检查向量维度是否为 1024；不匹配时会删除旧 collection 并重建。
8. collection 加载到内存。
9. 路由和静态文件开始对外服务。

这意味着：只要启动主服务，就必须能访问 Milvus；如果 Milvus 不可用，启动或健康检查会失败。

## 7. 文档上传与向量索引流程

用户从前端上传文件，或调用上传接口后，流程如下：

1. 请求进入 `/api/upload`。
2. 后端检查文件名、扩展名和大小。
3. 文件保存到 `uploads/` 目录。
4. 如果同名文件已存在，会覆盖旧文件。
5. `VectorIndexService` 读取文件内容。
6. `VectorStoreManager` 先删除该文件以前写入 Milvus 的旧分片。
7. `DocumentSplitterService` 按文件类型切分内容：
   - Markdown 先按一级、二级标题切分，再按字符长度二次切分，并合并过小分片。
   - 普通文本直接按字符递归切分。
8. `DashScopeEmbeddings` 批量生成 1024 维向量。
9. `VectorStoreManager` 把分片内容、向量和元数据写入 Milvus 的 `biz` collection。
10. 接口返回上传结果。

文档分片元数据会记录来源文件、扩展名和文件名。后续检索时，这些元数据会帮助回答里展示参考来源。

## 8. RAG 问答流程

RAG 问答有两种模式：普通模式和流式模式。

### 普通问答

1. 前端调用 `/api/chat`。
2. API 层把用户问题和会话 ID 交给 `RagAgentService.query`。
3. `RagAgentService` 首次使用时会延迟初始化 Agent。
4. 初始化时加载本地工具和 MCP 工具。
5. Agent 使用 `LLMFactory` 创建的聊天模型（默认 MiniMax）。
6. 会话 ID 被用作 LangGraph 的 `thread_id`，用于在 Redis checkpointer 中区分会话。
7. Agent 根据问题自主决定是否调用知识库工具、时间工具或 MCP 工具。
8. 最终返回完整回答。

### 流式问答

1. 前端调用 `/api/chat_stream`。
2. 后端用 SSE 返回消息流。
3. `RagAgentService.query_stream` 使用 Agent 的流式输出。
4. 当前实现主要把模型文本块作为 `content` 事件返回。
5. 前端逐步追加内容，并用 Markdown 渲染展示。
6. 结束时发送完成事件。

### 会话历史

服务端会话历史基于 LangGraph Redis checkpointer。默认本地连接是 `redis://:your-redis-password@localhost:6379/0`，可通过 `REDIS_URL` 覆盖。只要 Redis 数据仍可用，FastAPI 进程重启后服务端 checkpoint 仍可继续读取。

前端也在浏览器 localStorage 里保存聊天历史。前端历史和后端 Redis checkpoint 是两套机制：前端主要用于页面体验，后端主要用于 Agent 上下文。

## 9. AIOps 诊断流程

AIOps 是项目里最重要的智能运维能力。它基于 Plan-Execute-Replan 模式实现。

### 用户入口

前端有一个 AI Ops 按钮。点击后：

1. 前端新建对话。
2. 显示“分析中”状态。
3. 调用 `/api/aiops`。
4. 后端通过 SSE 持续返回诊断过程。
5. 前端把过程和最终报告渲染到聊天窗口。

### 后端工作流

`AIOpsService` 创建了一个 LangGraph 状态图，节点顺序是：

1. Planner：制定计划。
2. Executor：执行当前步骤。
3. Replanner：评估是否继续、重规划或生成报告。
4. 如果还有必要步骤，回到 Executor。
5. 如果信息足够，结束并输出最终报告。

### Planner 做什么

Planner 接收一个固定的 AIOps 诊断任务描述。它会先尝试从知识库检索相关经验，例如 CPU、内存、磁盘或服务不可用处理方案。然后它获取本地工具和 MCP 工具列表，结合这些信息生成诊断步骤。

Planner 本身不执行工具，它只产出计划。

### Executor 做什么

Executor 每次只取计划中的第一步执行。它把本地工具和 MCP 工具绑定给模型，让模型判断是否需要调用工具。若模型发起工具调用，则由 LangGraph 工具节点执行，再让模型基于工具结果生成该步骤的执行结果。

执行完后，该步骤会从剩余计划中移除，结果会追加到 `past_steps`。

### Replanner 做什么

Replanner 读取原始任务、剩余计划和已执行步骤。它有三种选择：

- 继续执行当前计划。
- 替换剩余计划。
- 信息足够，生成最终报告。

当前实现倾向于尽早结束，而不是无限追求更完整的数据。它还设置了最大执行步骤限制，避免诊断循环过长。

### AIOps 返回事件

后端会把工作流节点输出转换成适合前端消费的事件：

| 类型 | 含义 |
| --- | --- |
| `plan` | 诊断计划已生成 |
| `step_complete` | 某个执行步骤完成 |
| `status` | 普通状态更新 |
| `report` | 最终报告已生成 |
| `complete` | 诊断流程结束 |
| `error` | 诊断过程中出现异常 |

最终报告要求是 Markdown 文本，且必须基于工具查询结果，不应该编造数据。

## 10. MCP 工具体系

项目中的 Agent 工具有两类：

| 类型 | 来源 | 示例能力 |
| --- | --- | --- |
| 本地工具 | `app/tools/` | 知识库检索、当前时间 |
| MCP 工具 | `mcp_servers/` | 日志查询、监控指标、进程列表、服务信息、历史工单 |

MCP 客户端由 `app/agent/mcp_client.py` 统一创建。它使用全局单例，避免每次请求重复初始化。工具调用还包了一层重试拦截器，失败时会进行指数退避重试；如果多次失败，返回错误结果，而不是直接让整个 Agent 崩掉。

配置项位于 `app/config.py`：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `mcp_cls_url` | `http://localhost:8003/mcp` | CLS 日志 MCP 服务地址 |
| `mcp_monitor_url` | `http://localhost:8004/mcp` | Monitor 监控 MCP 服务地址 |
| `mcp_cls_transport` | `streamable-http` | CLS 传输方式 |
| `mcp_monitor_transport` | `streamable-http` | Monitor 传输方式 |

如果 MCP 服务没启动，RAG Agent 或 AIOps Agent 初始化工具时可能失败或降级，具体取决于调用点的异常处理。

## 11. Milvus 和向量库设计

项目统一使用 Milvus collection：`biz`。

| 字段 | 作用 |
| --- | --- |
| `id` | 文档分片 ID，项目写入时生成 UUID |
| `vector` | 1024 维 embedding 向量 |
| `content` | 分片正文 |
| `metadata` | JSON 元数据，包含来源文件等信息 |

向量索引使用 L2 距离。collection 初始化时会创建向量索引。检索时既有 LangChain VectorStore 的 retriever，也有直接使用 pymilvus 的 `VectorSearchService`。

需要特别注意：如果已有 collection 的向量维度不是 1024，启动时会删除旧 collection 并重建。这在开发环境很方便，但在生产环境应谨慎，因为这意味着旧数据可能被清空。

## 12. 配置和端口

### 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 必填。聊天模型和 embedding 都需要 |
| `DASHSCOPE_MODEL` | 默认聊天模型 |
| `DASHSCOPE_EMBEDDING_MODEL` | 默认 embedding 模型 |
| `MILVUS_HOST` | Milvus 地址，默认 localhost |
| `MILVUS_PORT` | Milvus 端口，默认 19530 |
| `RAG_TOP_K` | 知识库检索返回数量 |
| `CHUNK_MAX_SIZE` | 文档切分大小 |
| `CHUNK_OVERLAP` | 文档切分重叠长度 |

`DashScopeEmbeddings` 在模块导入时就会创建全局单例。如果 `DASHSCOPE_API_KEY` 缺失或仍是占位值，应用导入相关服务时就可能失败。

### 常用端口

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| FastAPI | 9900 | 主后端和前端页面 |
| Milvus | 19530 | 向量数据库 |
| Attu | 8000 | Milvus Web UI |
| MinIO | 9000 / 9001 | Milvus 对象存储依赖 |
| CLS MCP | 8003 | 日志工具服务 |
| Monitor MCP | 8004 | 监控工具服务 |

## 13. 本地启动方式

推荐先理解 Makefile 的目标：

| 命令 | 作用 |
| --- | --- |
| `make init` | 启动 Milvus、启动服务、等待健康、上传内置文档 |
| `make up` | 启动 Milvus 相关 Docker 容器 |
| `make start` | 启动 CLS MCP、Monitor MCP 和 FastAPI |
| `make stop` | 停止 FastAPI 和 MCP 服务 |
| `make check` | 检查 FastAPI 健康状态 |
| `make status-mcp` | 查看 MCP 服务状态 |
| `make upload` | 上传 `aiops-docs/` 下的内置知识库文档 |
| `make dev` | 前台热重载启动 FastAPI |
| `make test` | 运行测试 |
| `make lint` / `make format` / `make type-check` | 质量检查和格式化 |

当前仓库没有看到 `tests/` 目录，但 `pyproject.toml` 已经配置了 pytest、coverage 和相关开发依赖。后续补测试时可以直接沿用这套配置。

## 14. 前端交互模型

前端是单页静态应用。它主要维护这些状态：

- 当前会话 ID。
- 当前对话模式：快速或流式。
- 当前是否正在流式响应。
- 当前聊天记录。
- localStorage 中的历史会话列表。

主要用户动作和后端关系：

| 用户动作 | 前端行为 | 后端接口 |
| --- | --- | --- |
| 输入问题并发送 | 根据模式选择普通或流式请求 | `/api/chat` 或 `/api/chat_stream` |
| 上传文件 | FormData 上传文件 | `/api/upload` |
| 点击 AI Ops | 新建会话并发起诊断 | `/api/aiops` |
| 新建对话 | 保存当前历史，生成新 session ID | 通常不直接调用后端 |
| 清空会话 | 请求后端清理会话 | `/api/chat/clear` |

前端会用 Markdown 渲染模型回答，因此 AIOps 最终报告可以直接以 Markdown 格式展示。

## 15. 学习顺序建议

如果你刚开始接手，建议按以下顺序学习：

1. 先读 `README.md`，了解项目目标和启动方式。
2. 读 `app/main.py`，理解应用入口、路由注册和 Milvus 生命周期。
3. 读 `app/config.py`，理解所有关键环境变量。
4. 读 `app/api/chat.py`、`app/api/file.py`、`app/api/aiops.py`，理解对外接口。
5. 读 `app/services/vector_index_service.py`、`document_splitter_service.py`、`vector_store_manager.py`，理解知识库写入链路。
6. 读 `app/tools/knowledge_tool.py`，理解知识库如何被 Agent 调用。
7. 读 `app/services/rag_agent_service.py`，理解 RAG 问答和会话管理。
8. 读 `app/services/aiops_service.py`，理解 AIOps 工作流图。
9. 读 `app/agent/aiops/planner.py`、`executor.py`、`replanner.py`，理解诊断智能体内部三阶段。
10. 读 `app/agent/mcp_client.py` 和 `mcp_servers/`，理解工具从哪里来。
11. 最后读 `static/app.js`，把前端按钮、接口和后端流程串起来。

这个顺序的好处是先看边界，再看业务链路，最后看智能体内部细节。

## 16. 修改功能时怎么定位

| 你想改什么 | 优先看哪里 |
| --- | --- |
| 新增 API | `app/api/` 和 `app/main.py` |
| 改聊天回答逻辑 | `app/services/rag_agent_service.py` |
| 改知识库检索方式 | `app/tools/knowledge_tool.py`、`app/services/vector_store_manager.py` |
| 改文档上传和索引 | `app/api/file.py`、`app/services/vector_index_service.py` |
| 改文档切分策略 | `app/services/document_splitter_service.py` |
| 改 AIOps 计划生成 | `app/agent/aiops/planner.py` |
| 改 AIOps 步骤执行 | `app/agent/aiops/executor.py` |
| 改 AIOps 何时结束 | `app/agent/aiops/replanner.py` |
| 接入真实日志平台 | `mcp_servers/cls_server.py` |
| 接入真实监控平台 | `mcp_servers/monitor_server.py` |
| 改页面交互 | `static/app.js`、`static/index.html`、`static/styles.css` |

## 17. 当前实现里值得注意的点

### 1. 健康检查路径和 README 不一致

代码注册的是 `/health`，README 表格里写的是 `/api/health`。实际调试应以 `/health` 为准。

### 2. Python 版本说明不一致

README 写 Python 3.10+，`pyproject.toml` 要求 Python 3.11 到 3.13。安装依赖和运行项目时应以 `pyproject.toml` 为准。

### 3. 前后端上传限制不一致

前端允许 `.markdown` 和 50MB，后端只允许 `txt/md` 和 10MB。实际可用范围以后端为准。

### 4. 会话历史不是持久化数据库

后端使用 Redis checkpointer 存储 LangGraph checkpoint。默认本地连接是 `redis://:your-redis-password@localhost:6379/0`，通过 `REDIS_URL` 覆盖。只要 Redis 数据还在，FastAPI 进程重启后服务端会话 checkpoint 仍可继续读取。前端 localStorage 只影响当前浏览器体验，不等于服务端 checkpoint 存储。

### 5. MCP 服务当前是 mock

AIOps 的诊断看起来能查日志和监控，但数据来源是本地模拟逻辑。接生产时需要替换为真实数据源。

### 6. Milvus collection 维度不匹配会触发重建

这对开发友好，但生产环境要避免误删数据。上线前应明确是否允许自动删除旧 collection。

### 7. CORS 当前允许所有来源

这方便本地开发，但生产环境应限制可信域名。

### 8. 当前缺少可见测试目录

项目配置了 pytest 和 coverage，但当前没有看到 `tests/` 目录。后续改核心链路时，建议补测试覆盖上传、切分、索引、RAG 工具和 AIOps 状态流。

## 18. 一个完整请求如何串起来

### 知识库问答

浏览器发送问题到聊天接口。API 层把问题交给 RAG 服务。RAG 服务初始化 Agent，加载本地工具和 MCP 工具。Agent 判断是否需要查知识库。如果需要，知识库工具通过 Milvus retriever 找到相关文档分片。模型把问题、工具结果和上下文组合成回答。后端返回普通 JSON 或 SSE 流。前端把回答渲染成聊天消息。

### 上传知识文档

浏览器选择文件。后端保存到 `uploads/`。索引服务读取文件，切分成多个文档分片。embedding 服务把分片转成 1024 维向量。向量存储管理器把分片写进 Milvus。后续问答或 AIOps 规划时，就能检索这些文档。

### AIOps 诊断

浏览器点击 AI Ops。后端创建固定诊断任务。Planner 先查知识库经验，再结合工具列表制定计划。Executor 每次执行一个步骤，并通过 MCP 调用日志或监控工具。Replanner 判断是否继续。最终模型根据执行历史生成 Markdown 报告。前端通过 SSE 持续接收计划、步骤状态和最终报告。

## 19. 建议的后续改进方向

| 方向 | 价值 |
| --- | --- |
| 增加测试目录和核心链路测试 | 降低重构 RAG/AIOps 时的回归风险 |
| 统一 README、pyproject 和 config 的版本/路径说明 | 降低新人启动成本 |
| 统一前后端上传限制 | 避免用户侧显示允许但后端拒绝 |
| 将会话历史持久化 | 支持服务重启后恢复对话 |
| 为 MCP 服务增加真实数据源适配层 | 从 demo 诊断走向生产可用 |
| 生产环境收紧 CORS 和鉴权 | 避免接口暴露风险 |
| 为 Milvus collection 重建加保护开关 | 防止生产数据被意外删除 |
| 增加 Agent 工具调用可观测性 | 更容易排查 AIOps 诊断质量问题 |

## 20. 总结

EnergyAgent 的核心不是单纯聊天，而是“知识库 + 工具调用 + 运维诊断工作流”的组合。理解它时可以抓住三条主线：

- 知识库主线：上传文档、切分、embedding、写入 Milvus、检索。
- RAG 主线：聊天请求、Agent、工具、会话、回答。
- AIOps 主线：Planner、Executor、Replanner、MCP 工具、最终报告。

只要把这三条主线串起来，项目整体就比较清晰了。后续新增能力时，优先判断它属于 API 边界、知识库链路、Agent 逻辑、MCP 工具还是前端交互，再进入对应目录修改。
