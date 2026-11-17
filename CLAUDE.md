# EnergyAgent 开发规则

## 基本原则

- Python 代码应保持接近 Go 的显式、清晰和可维护风格。
- 优先使用明确的数据结构、类型和接口，避免依赖隐式约定。
- 每个模块、类型和函数只承担一种职责，不为方便而混合复用。
- 保持实现简单；不要引入当前需求不需要的抽象或兼容层。

## 数据结构

- 所有业务数据必须定义为具名类型，并明确声明每个字段。
- API 请求和响应使用 Pydantic Model；领域值对象优先使用 `dataclass`。
- 请求模型、响应模型、领域模型和持久化模型应分别定义，不得混用。
- 不要用 `dict[str, Any]`、`Any` 或 `object` 代替已知的业务结构。
- 不要定义 `Model | dict | object` 这类混合参数；在系统边界立即完成类型转换。
- 可空字段必须显式标注为 `T | None`，不得依赖隐式 `None`。

## 函数和方法

- 函数参数和返回值必须有准确的类型注解。
- 返回值应使用明确的具名类型，不要返回结构不固定的字典或元组。
- 一个参数只表达一种语义；不同输入场景应使用不同方法或不同请求类型。
- 公共函数不得返回 `Any`，也不得通过宽泛联合类型逃避类型设计。
- 类型注解必须与实际运行时行为一致，包括异常和可空返回值。

## 接口和依赖

- 需要解耦实现时，优先使用 `typing.Protocol` 定义小而专一的接口。
- 接口由使用方按实际需求定义，避免大型通用接口。
- 业务逻辑依赖接口，不直接依赖数据库、HTTP 客户端、向量库或第三方 SDK。
- 基础设施代码负责实现接口，并在应用启动或组合根中注入。
- 只有确实需要共享运行时行为或状态时才使用抽象基类。

## 分层边界

- API 层负责协议转换、输入校验和响应组装，不承载业务规则。
- 领域层保存业务模型、规则和领域错误，不依赖 FastAPI、数据库或外部 SDK。
- 服务层编排用例，通过接口访问外部能力。
- 基础设施层负责数据库、Milvus、Redis、HTTP 和第三方服务适配。
- 跨层传递数据时使用明确模型和显式转换，不共享可变字典。

## 错误处理

- 使用具体、可识别的领域异常，不使用裸 `Exception` 表达业务失败。
- 不静默吞掉异常；捕获异常后必须处理、转换或带上下文重新抛出。
- API 层统一把领域错误映射为协议错误。

## 质量要求

- 新功能和缺陷修复必须包含相应测试。
- 修改后运行相关测试，并按需运行 `uv run pytest`、`uv run pyright` 和 `uv run ruff check .`。
- 遵循 `pyproject.toml` 中的格式化、Lint 和类型检查配置。
- 不修改与当前任务无关的代码、配置或依赖。

## 代码搜索

- 本仓库已由 CodeGraph 索引（`.codegraph/`），理解或定位代码时优先使用它，再进行 grep/读文件。
- 优先用 MCP 工具 `codegraph_explore`，或 shell 命令 `codegraph explore "<符号名或问题>"`，一次获取相关符号源码与调用路径。
- 需要跨多轮 grep + 读文件才能回答的问题（如调用链、动态分发），应先用 CodeGraph。
- grep/glob 用于精确的文本或文件名匹配，尽量并行批量发起搜索。
- 不要修改 `.codegraph/` 目录内容。

## 推荐示例

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ManualEscalationRequest:
    reason: str
    target_team: str
    requested_by: str


@dataclass(frozen=True, slots=True)
class AlertEscalation:
    alert_id: str
    target_team: str
    status: str


class AlertEscalator(Protocol):
    def escalate_alert(
        self,
        alert_id: str,
        request: ManualEscalationRequest,
    ) -> AlertEscalation:
        ...
```

<comet-ambient-resume>
<!-- Managed by Comet. Edits inside this block may be replaced by comet init/update. -->
<!-- Contract: comet.resume_probe.v2 -->

## Comet Ambient Resume

在这个仓库中，开始处理需要改动或调查的任务前，如果可能存在活跃 Comet workflow，把当前用户请求传入只读探针：`comet resume-probe . --stdin --json`。

- 只信任返回的 `workflow`、`skill` 和 `entrySource`；它们只由项目配置或无配置兼容回退决定。不得扫描或切换另一套 workflow。
- 如果 probe 返回 `auto_resume`，简短说明选中的 active change，并进入 `nextCommand` 指向的永久入口。不要把状态命令当作恢复入口直接推进。
- 如果 probe 返回 `ask_user`，只问一个简短问题并等待用户回复。
- 如果 probe 返回 `out_of_scope` 或 `none`，不要进入 Comet workflow。
- 如果配置或状态无效且没有 `nextCommand`，停止并报告原因；不要猜测另一个 workflow。
- 不能只因为存在 active change 就把无关任务挂到该 change。Native 的未提交改动由 Native 入口检查，不由探针自动归因。
</comet-ambient-resume>
