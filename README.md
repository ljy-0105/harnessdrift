# HarnessDrift — 自改进 agent harness（RSI 骨架）

一个让**提示词自己迭代**的研究原型：agent 在 38 道带确定性判分的售后题上跑分，
失败样本交给改进器产出补丁，补丁经守卫校验后应用到新树，再评测、再迭代。
**不训练模型，只让 harness（提示词 / 技能文档 / 工具描述）在评测反馈下演化。**
曲线上的每个点都带 ±，`runs/` 里是原始 json，README 的每个数字都能对回去。

思路来自 [google-research/harness2](https://github.com/google-research/harness2)（Propose / Probe / Compose 骨架）
与 [google-research/dream-rsi](https://github.com/google-research/dream-rsi)（探测器调度）。

**实测：31.67 → 36.00（+4.33），38 题 × 3 次 × 5 轮，20 分钟。**
成本约 ¥1.4 —— 这是按实测 5,087 token/题 × 570 题估的，不是账单；早期曲线的 `curve.json`
里还没有用量字段，`total_tokens` 是后来才接的（见「实测曲线」一节的说明）。

三个能独立讲清楚的结论：

1. **回退规则**。只看均值的接受规则会收下"涨 1 分但卖掉一道安全题"的补丁。第 2、3 轮均值都高于基线，
   却被拦下 —— 它们各自把一道原本稳定通过的题变成稳定挂掉（T07/T34、T27/T34）。
   **拿旧能力换新能力不算改进。**
2. **原子归因**。第 1 轮 +4.00 里，最强单点只贡献 +2.33，**+1.67 是四条一起才有的交互效应** ——
   只报总分的展示说不清这件事。
3. **消融**。三档接受规则跑同一套题：`always` +3.00 / `strict` +2.67 / `nonregress` +0.00。
   默认规则一分没涨，因为它为了保 T12 拒绝了两轮 +4 分的补丁。报告里写的是
   "这张表只支持接受规则需要论证，不支持排名"。

## 为什么先做这一层

两个仓库里各有一处会在 Windows / 小规模复现时咬人的设计，第 0 步先把它们处理掉，
免得等业务代码写完了才发现日志对不上：

| 上游的做法 | 问题 | 这里的做法 |
|---|---|---|
| `read_tree()` 用 `str(Path.relative_to())` 做 key | Windows 上是 `scripts\summarize.py`，提案写 `scripts/summarize.py`，补丁**静默不命中** | 所有 key 进出都走 `normalize_rel()` 强制 posix |
| `core/schedule.py` 直接 `import fcntl` | 该模块在 Windows 上不存在，import 就炸 | 用 `os.replace` 原子写替代，单机场景语义等价 |
| 评测产物可能混进 harness | 带 `grader` 字样的文件一旦进入后续轮次，就会污染证据 | `assert_no_grader_verdicts()` 在物化**之前**扫描 |

## 三条契约

### 1. 路径（`paths.py`）

树 = `dict[posix 相对路径, 文件内容]`。绝对路径、`..` 逃逸、归一化后撞车，全部抛错。
物化用 `newline=""` 写，杜绝 Windows 把 `\n` 改成 `\r\n` 导致内容哈希跨平台对不上。

### 2. 编辑（`edits.py`）

提案只输出 diff，**没提到的路径原样继承**（omitted is inherited）：

```
### ADD skills/refund/SKILL.md
    handle refunds
### REPLACE systemprompt.md
    new prompt
### DELETE skills/none/SKILL.md
```

`ADD` 已存在、`REPLACE`/`DELETE` 不存在、同一路径写两次、`DELETE` 带正文 —— 全部抛错。
不做宽容处理：静默成功会让一次错误的提案看起来像一次成功的提案。
`render_patch()` 能把任意一对树渲染成可重放的提案，日志因此可回滚、可回归测试。

### 3. 反作弊（`guards.py`）

全部抄自上游，不是想象：

- `assert_prompt_describes_work` ← `core/spec.py` 的 `BenchPrompts.__post_init__`：
  面 agent 的提示词里出现 `benchmark / evaluation / grader / rubric / score` 即抛错
- `assert_no_grader_verdicts` ← `core/context.py`：文件名和正文都不许有评分标记
- `is_leaked` ← `spec.leak_globs`：执行沙箱屏蔽答案文件

## 跑起来

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests -q
```

预期：**101 passed**（全部离线，不访问网络、不花 token）。

```bash
# 单次基线（先 --limit 3 验证链路）
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli run --repeat 3 --out runs/base.json

# 一条完整的自改进曲线
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli loop \
    --reps 3 --max-rounds 4 --patience 2 --out runs/curve1

# 归因：这轮补丁里是哪一条在起作用（N 个原子 = N+2 次全量评测，最贵的一步）
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli ablate \
    --parent harness/base --cand runs/curve1/round1 --reps 3 --out runs/attr

# 消融：几条曲线并排看
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli compare \
    --curves runs/ablate_strict,runs/ablate_nonregress,runs/ablate_always \
    --out runs/ABLATION.md
```

`loop` 的产物在 `runs/curve1/`：`REPORT.md`（人看）、`curve.json`（机器读）、
`roundN/`（每轮接受的树）、`_rejected/`（被拒候选 + 原因）、`best/`（最佳树）。

## 看板

![看板](docs/preview.png)

`runs/` 里的东西是给愿意读代码的人看的；这一页是给不打算读代码的人看的。

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli export \
    --run runs/curve3 --attr runs/attr_curve3 \
    --compare runs/ablate_strict,runs/ablate_nonregress,runs/ablate_always \
    --out dist/index.html
```

产物是**一个自包含的 HTML**：数据内联在 `<script>` 里，没有后端、没有构建、没有 CDN，
双击就能开（`file://` 不受 CORS 限制）。`dist/index.html` 已提交，
也可以把 `dist/` 直接挂到 GitHub Pages。

页面五块：曲线（被拒的轮次也画）、每轮补丁、38 题矩阵、原子归因、接受规则消融。
**看板是证据的投影，不是新的数据源** —— 每个数字都能在 `runs/` 的 json 里对回出处。

`web/smoke.cjs` 在 node 里跑一遍渲染逻辑（不需要浏览器），CI 里会跑，
防止"导出成功但页面白屏"这种只有真人打开才发现的问题。

## 实测曲线

两条，同一条改进器、同一个题库，差别只在接受规则。用来看"规则本身值多少钱"。

**`runs/curve3/`（当前默认规则：不退化 + 不改坏旧题）**

```
 round  kind        score     ±  solved
     0  base        31.67  0.47    30/38
     1  accepted    36.00  2.16    33/38 +     ← 修好 T02/T14/T31/T32，零回退
     2  rejected    32.33  0.47    32/38 x     ← 改坏 T07/T34
     3  rejected    34.00  0.00    33/38 x     ← 改坏 T27/T34
 停止：patience=2（连续两轮没有实质改进）       总耗时 1223s ≈ 20 分钟
```

基线 30 → 33 条稳定通过，`期望动作=return_created` 类失败从 12 次降到 1 次。
被拒的两轮不是"分数低了"，而是**分数还想往上走、但代价是踩掉本来能过的题**：
第 3 轮 34.00 分比第 2 轮的 32.33 还高，照样拒，因为它让 T27/T34 从"每轮都过"
变成"每轮都挂"。评测集是固定题库，那 2 条是真回退，不是抖动。

**`runs/curve2/`（早期规则：只看均值不退化）**

```
     0  base        32.67  1.70    31/38
     1  accepted    34.00  0.82    32/38 ~     ← 修好 T14/T18，改坏 T12/T24
     2  rejected    32.00  1.63    29/38 x
```

同一套东西，第 1 轮 +1.33 就被收了 —— 代价是 T12、T24 从此稳定挂掉。
这就是加回退规则的直接理由，`tests/test_loop.py::test_broke_stable_*` 锁着这条行为。

### 读这两条曲线时必须承认的三件事

1. **两次跑的基线不一样**（32.67 / 31.67）。远程模型没有固定 seed，`reps=3`
   估出来的 std 也就 0.47~1.70 地飘。所以"±噪声"这个判据本身是带噪声的 ——
   真正要下结论，`reps` 至少得 9，这不是现在能负担的。
2. **第 1 轮的 +4.33 里有一部分来自抖动**。候选 std 2.16 比父树 0.47 大得多，
   说明它是"有些题开始随机过"，不是纯能力上升。报告第 5 节把抖动项单独列了出来，
   就是为了不让这部分混进结论。
3. **20 分钟的曲线只说明闭环能跑通**，不说明改进器有多强。（token 成本已在第 4 步
   接进 `curve.json` 的 `total_tokens`，早于这一步跑的曲线没有这个字段。）
   它现在会提"往 SKILL.md 里加一条规则"这类补丁，因为失败样本就是这么喂的。
   要它提结构性改动（改工具描述、拆技能），得给它更强的失败归因。

## 归因：这轮补丁里是哪一条在起作用

`harnessdrift ablate --parent <父树> --cand <候选树>` 把一轮补丁拆成原子，
每个原子单独合回父树各评一次，产出 `ATTRIBUTION.md`。

`runs/attr_curve3/` 是对 curve3 第 1 轮那 5 处改动（合并后 4 个原子）的归因：

| # | 改动 | 单独得分 | Δ | 结论 |
|---:|---|---:|---:|---|
| 0 | `skills/return_eligibility/SKILL.md` — 质量问题的运费由平台承担 | 35.33 | +2.33 | **有效** |
| 1 | `systemprompt.md` — "客户在要求"的触发词 | 33.33 | +0.33 | 测不出 |
| 2 | `systemprompt.md` — "要凭证不等于不开单" | 34.67 | +1.67 | **有效** |
| 3 | `systemprompt.md` — 换货次数硬上限 | 33.33 | +0.33 | 测不出 |

父树 33.00 / 全量 37.00 / 最强单点 +2.33 / **交互效应 +1.67**。

也就是说：这轮 +4 的增益里，两条单点各值 +2.33 和 +1.67，剩下 1.67 分是四条
**一起**才有的 —— 单看每一条，第 1、3 条根本没被测出来。这就是"归因"和
"看总分"的区别，也是面试里能讲的东西：**一次改五处然后说分数涨了，等于什么都没说。**

## 消融：接受规则是不是一个变量

`harnessdrift compare --curves runs/ablate_strict,runs/ablate_nonregress,runs/ablate_always`
把三条曲线并排，产出 `runs/ABLATION.md`。

```
                       基线    历史最佳  最后留下的树    Δ       稳定通过
 strict         32.67±0.47    35.33       35.33     +2.67     30→34   丢了 T12
 nonregress     32.00±0.82    32.00       32.00     +0.00     30→30   —
 always         32.67±0.94    36.67       35.67     +3.00     31→33   —
```

三件事值得单独说：

1. **`nonregress` 拒了两个 +4 分的补丁，最后交的还是基线。** 被拦下的第 2 轮
   36.33 分，比任何一条被接受的树都高 —— 它被拦是因为 T12 挂了。T12 是
   「拿别人的订单号要求退款」，必须 `clarify` 而不能直接开单。**这条拦得对**：
   分数 +4 的代价是"开始给别人的订单开退货单"，那不该算改进。
   但这套规则不区分"丢了安全题"和"丢了普通题"，它只是拒绝一切回退 ——
   这是它最该被质疑的地方，也是下一层要做的事（按任务权重分档）。
2. **`strict` 收了 +2.67 的那一轮，代价正好是 T12。** 只看均值的规则会把安全题
   当普通题卖掉，而且卖掉之后不会再有人发现 —— 后面每轮都在缺了 T12 的树上改。
3. **`always` 最高到过 36.67，停下来时留下 35.67。** 朴素 RSI 不挑补丁，
   中间掉分的那几轮照样替换，只是恰好没掉到基线以下。它这一批没踩坏题是运气，
   不是规则保护的结果。

`ABLATION.md` 里还写了一条硬话：**每档只跑了一次**，模型没有固定 seed，
两次跑同一档基线就差 1 分。所以这张表只支持"接受规则是个需要论证的选择"，
**不支持**"哪一档高 0.5 分所以更好"。

## 目录

```
src/harnessdrift/
  paths.py      路径归一化 / 物化 / 原子写
  edits.py      diff 提案的解析与应用
  guards.py     反作弊三件套
  tools.py      模拟电商售后系统（订单 / 退货 / 转人工）
  tasks.py      38 条任务，每条带 why
  agent.py      「接单 → 调工具 → 答复」循环
  evaluate.py   L1 确定性评测（期望动作 / 工具调用 / 编造 / 禁语）
  llm.py        OpenAI 兼容适配器，重试在这一层
  runner.py     跑一棵树收成 EvalResult（不做判断）
  propose.py    PROPOSE：失败样本 → 补丁 → 守卫 → 新树
  loop.py       多轮调度：接受规则 + 五个停止条件
  ablate.py     归因：把一轮补丁拆成原子，逐条单独评测
  compare.py    消融：几条曲线并排看
  report.py     生成 REPORT.md
  cli.py        run / loop / ablate / compare / show-prompt / init
tests/
  test_contracts.py       路径、编辑、反作弊契约
  test_tools_and_judge.py 工具层与判分
  test_propose.py         坏补丁必须被拦住
  test_loop.py            循环的判定逻辑（离线，脚本化假 LLM）
  test_ablate.py          拆原子、合回去、交互效应
  test_compare.py         三档对照报告
  test_tokens.py          用量统计（第一次评测也要记到、记的是增量）
```

## 分步交接文档

- `HANDOFF-STEP0.md` 骨架与三条契约
- `HANDOFF-STEP1.md` 客服 agent + 评测器
- `HANDOFF-STEP3.md` 闭环：接受规则、停止条件、报告
- `HANDOFF-STEP4.md` 归因与消融：原子怎么切、对照怎么读

## 下一步（第 5 步）

1. **L3 软质量评审**：现在只有 L1 确定性判分（期望动作 / 工具调用 / 编造 / 禁语），
   能判"做没做对"，判不了"说得好不好"。语气、政策解释是否清楚、是不是把用户当傻子，
   这些得单独一个判分模型，并且要先证明它和人的判断一致（否则只是换了个噪声源）。
2. **任务分档**：现在的回退规则把所有题一视同仁，于是它为了 T12（安全题）拒绝了
   +4 分。合理做法是给任务标 `safety` / `normal`，回退时按档位不同处理。
3. **多条独立轨迹**：`reps` 是题内重复，不是轨迹重复。要排名得跑 3~5 条独立曲线取均值。
4. **成本核算**：`total_tokens` 已经落进 `curve.json`，还差把 token 换算成钱，
   并回答"多跑一轮值不值"。