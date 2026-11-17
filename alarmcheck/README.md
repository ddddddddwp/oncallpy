# alarmcheck

`alarmcheck` 用于本地验收 oncall 的完整告警自动处置链路：

1. 启动内置模拟云端 HTTP 服务。
2. 向 oncall `POST /api/alerts/ingest` 发送原始告警。
3. 调用 `POST /api/alerts/{id}/auto-remediate`。
4. 校验知识库处置码、运维指令、云端返回和告警最终状态。

端口约定：

| 服务 | 默认地址 | 说明 |
| --- | --- | --- |
| oncall 后端 | `http://127.0.0.1:6872` | 接收告警、自动处置、更新状态 |
| alarmcheck 模拟云端 | `http://127.0.0.1:19090` | 接收 oncall 发出的运维指令，并提供 Web 查看页 |

模拟云端 Web 页面：

```text
http://127.0.0.1:19090
```

模拟云端指令接收接口：

```text
http://127.0.0.1:19090/api/cloud/commands
```

默认验证 5 个海索逆变器故障码：

```bash
cd /root/code/oncall/alarmcheck
go run . -oncall-url http://127.0.0.1:6872 -fault-codes 010A,010B,010C,010D,010E -timeout 60s -keep-server
```

加 `-keep-server` 后，验收完成不会立即退出，浏览器可以继续打开 `http://127.0.0.1:19090` 查看模拟云端收到的指令。

可指定其他故障码：

```bash
go run . -oncall-url http://127.0.0.1:6872 -fault-codes 0116,013B,1159 -timeout 60s
```

只启动模拟云端，不发告警：

```bash
go run . -serve-only
```

更换模拟云端端口：

```bash
go run . -cloud-addr 127.0.0.1:19190 -serve-only
```

说明：

- 上报告警不会携带解决方法或处置码，只携带原始故障信息。
- 非内置故障码会按文档规则使用“故障码最后一位十六进制数字 % 6”推导预期处置码。
- 工具会自动启动固定端口模拟云端，oncall 自动处置接口会向这个模拟云端发送 POST。
- 模拟云端 Web 会展示接收时间、故障码、设备、告警 ID、指令、操作人和完整请求 payload。
