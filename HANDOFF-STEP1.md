# 第 1 步交接：客服 Agent 基线

## 现在有什么

```
harnessdrift/
├── src/harnessdrift/
│   ├── paths.py / edits.py / guards.py    ← 第 0 步的三条契约
│   ├── tools.py       电商售后工具层（确定性，零网络）
│   ├── tasks.py       任务加载，强制每条任务都有 why
│   ├── llm.py         OpenAI 兼容适配器（DeepSeek/Qwen/中转通用）
│   ├── agent.py       工具循环 + harness→提示 渲染
│   ├── evaluate.py    确定性评测器（L1 硬指标 + L2 编造检查）
│   └── cli.py         run / show-prompt / init
├── harness/base/      基线 harness 树（systemprompt + tool_descriptions + 1 个 skill）
├── data/tasks_ecommerce_after_sales.json   20 条任务
└── tests/             **39 passed**
```

已验证：`pytest` 39 全绿；`show-prompt` 能正确渲染；政策边界（第 7/8 天、家居 15 天、
数码激活例外、食品不可无理由、质量问题豁免）全部有单测。

## 你只需要做的两件事

```bash
cd C:\Users\HP\Downloads\RSI\harnessdrift
copy .env.example .env
# 打开 .env，把 LLM_API_KEY= 后面填上你的 key
```

然后先花几分钱验证链路：

```bash
.\.venv\Scripts\python.exe -m harnessdrift.cli run --limit 3 --verbose
```

通了再跑全量（20 条，约 60 次模型调用）：

```bash
.\.venv\Scripts\python.exe -m harnessdrift.cli run --out runs/base.json
```

把 `pass_rate` 和失败项分布贴给我，我据此决定第 2 步先修哪一块。

## 我这步踩到的四个坑（都能写进面试稿）

1. **评测器读副作用而不是读轨迹**：我一开始用 `len(toolkit.returns)` 判断"有没有开退货单"。
   但 `returns` 是副作用，测试里可以直接往里 append —— 评测器会被自己伪造的状态骗过去。
   改成只信 `Toolkit.dispatch` 写进 `trace` 的调用。
2. **`RunResult.trace` 是快照**：调用方先 dispatch、再构造 RunResult 时它是空的。
   现在 judge 合并 `result.trace` 和 `toolkit.trace` 并按 id 去重 —— 快照和活状态都能用。
3. **订单号大小写**：T17 里客户写的是 `ab66778890`，模型复述小写完全合理，
   却被"没查过"判死。加了 `normalize_orders_in()`，只在比对副本上大写化，不改答复原文。
4. **断言写宽了**：`any("期望动作" in c.name ...)` 因为名字里有个 `=` 永远匹配不上，
   测试其实是空转的。改成 `startswith`。

## 设计上刻意没做的

- **没有反思/记忆/多 agent**。基线必须简单到你敢说"这个分数就是这棵树的能力"，
  否则第 2 步之后涨了也不知道是谁的功劳。
- **不给部分分**。`TaskScore.value` 只有 0/1。部分分会让自改进学会"修一半"，
  而"答错但礼貌"和"答错"在客户体验里没有区别。要细粒度就在检查项里拆细。
- **不给全局查订单的工具**。只能按 order_id 精确查 —— 逼 agent 在信息不足时追问，
  T11（没给单号）就是要它学会这件事。

## 第 2 步预告

基线分数出来之后，第 2 步是 **PROPOSE**：让改进器读失败用例的轨迹，
吐出一份 `### ADD/REPLACE/DELETE` 补丁，改 `harness/round1/`——不改 `base`。
`render_patch` 会让每一次提案都可回放、可回滚。