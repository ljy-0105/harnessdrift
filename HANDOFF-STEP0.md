# 第 0 步交接说明

## 已完成

- 项目根：`C:\Users\HP\Downloads\RSI\harnessdrift`
- venv：`harnessdrift\.venv`（Python 3.13.15）
- 代码：`src/harnessdrift/{paths,edits,guards}.py`
- 测试：`tests/test_contracts.py`，**17 passed**
- 已验证：往返 5 轮不漂移；`write_tree` 产物无 CRLF

## 我的选择与理由

| 决定 | 理由 |
|---|---|
| 用 `### ADD/REPLACE/DELETE <path>` 而不是 unified diff | LLM 写 unified diff 的上下文行经常对不齐；声明式改动不需要对上下文 |
| 用 `os.replace` 而不是移植 `fcntl` | 单机单进程只需"读者看不到半个文件"；跨平台移植文件锁是过度设计 |
| 一律抛错，不做宽容解析 | 静默成功会让错的提案看起来像对的 |
| `render_patch` 用单换行分隔块 | 空行会被解析成正文首行，导致每轮多一个前导 `\n` |

## 第 0 步踩到的三个坑（面试可讲）

1. **`{"a.md": "1", r"a.md": "2"}` 这个测试永远绿** —— Python 字面量先把重复 key 合并了，
   根本测不到归一化撞车。改用 `list[tuple]` 才能真正触发。
2. **`DELETE` 带正文被静默吞掉** —— `body = None if action == "DELETE" else ...`
   在构造函数校验之前就把内容丢弃了，校验形同虚设。必须在 flush 处拦。
3. **块间空行导致正文漂移** —— `"\n\n".join(blocks)` 渲染出的空行，解析时会被当成正文首行，
   每往返一次正文前面多一个 `\n`。这个坑只有在做往返测试时才会暴露 ——
   **往返测试（render → parse → apply == 原树）是这一层的核心回归手段**，别删。

## 下一步需要你决定的

第 1 步开工前，需要你定两件事：

1. **客服场景选哪个**（电商售后 / SaaS 技术支持 / 内部 IT 帮助台）—— 20 条任务要你自己写
2. **API key 和 base_url** —— 写进 `harnessdrift/.env`（不要提交到 git）