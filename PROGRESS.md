# 进度快照

> 保存于 2026-10-28 19:50 · 126 tests passed · 最新 commit 见 `git log -1`
> 这份文件是给人看"现在停在哪、下一步从哪接"的。技术细节在各 `docs/internal/HANDOFF-STEP*.md`，数字出处在下面对应的 `runs/`。

## 一句话

不训练模型，只让 harness（系统提示 / 技能文档 / 工具描述）在评测反馈下自我演化；
每一轮都留下曲线、补丁全文、归因，并且**规则要能被数据打脸**。

## 已完成的四层

| 步骤 | 内容 | 落点 |
|---|---|---|
| 0 | 骨架 + 三条契约（路径归一化 / 补丁格式 / 反作弊） | `docs/internal/HANDOFF-STEP0.md` |
| 1 | 电商售后客服 agent + 38 题评测器（L1 确定性判分） | `docs/internal/HANDOFF-STEP1.md` |
| 2 | PROPOSE：失败样本 → 补丁 → 守卫 → 新树 | `src/harnessdrift/propose.py` |
| 3 | 闭环：接受规则 + 五个停止条件 + REPORT.md | `docs/internal/HANDOFF-STEP3.md` |
| 4 | 归因（拆原子）+ 消融（多曲线对照） | `docs/internal/HANDOFF-STEP4.md` |
| 5 | 静态看板 export（单文件 HTML，CI 里跑渲染冒烟） | `web/template.html` |
| 6 | **安全题分档**：`tier` + 锁定名单 + 一票否决 | `docs/internal/HANDOFF-STEP3.md` §2 |
| 7 | **留出集**：12 条 H 题只计分、不进提案端、不参与决策 | `README.md` §留出集 |
| 8 | **硬预算**：`--max-tokens`，每轮开始与评测前各查一次 | `loop.py` |
| 9 | **一键回滚**：整树替换 + 先备份 + 拒绝回退到被拒轮 | `rollback.py` |

## 现在的数字（全部可复现，全在 `runs/`）

| 跑批 | 规则 | 基线 | 最终 | Δ | 备注 |
|---|---|---|---|---|---|
| `runs/curve2` | 旧判据 | 32.67 | 34.00 | +1.33 | 反例：均值涨了但踩坏 T12/T24 没被发现 |
| `runs/curve3` | nonregress + 回退守卫 | 31.67 | 36.00 | +4.33 | 第 1 轮全绿；第 2、3 轮踩坏 T07/T34 被拦 |
| `runs/ablate_strict` | strict，无安全档 | 32.67 | 35.33 | +2.67 | 代价是 T12 被换掉 |
| `runs/strict_veto` | strict + v1 守卫 | 32.00 | 37.00 | +5.00 | **v1 有洞**：T12 变 `[0,1,0]` 后被永久豁免 |
| `runs/strict_lock` | strict + v2 锁定 | 32.00 | 32.00 | +0.00 | 两轮（36.33 / 35.67）被一票否决 |
| `runs/holdout1` | strict + 留出集（38 训练 / 12 留出） | 42.33 | 46.67 | +4.33 | **训练列 +4.33、留出列 +0.00**：涨的是这批题 |
| `runs/noise_b1..3` | 同一棵树重跑三次 | — | — | — | 42.67 / 41.00 / 42.33，**极差 3.33 分** = 均值噪声底 |
| `runs/placebo1` | 安慰剂：失败样本不进提案端 | 44.33 | — | — | Δ 在 **−4.67 ~ +2.67** 之间乱摆（单轮最大 4.67） |
| `runs/attr_feed1` | 喂了"别弄坏 T12"的提案端 | 41.67 | 41.67 | +0.00 | 两轮 +4.00 / +3.67 **都被一票否决**（T12 仍 0/3） |
| `runs/probe_noavc` | 人工删掉毒性文件后重测 | — | 46.00 | — | 与候选同分，但 **T12 回到 3/3** |
| `runs/rescue_replay` | 用代码自动抢救同一棵被否决的树 | 42.67 | 46.00 | — | 6 个原子试删 4 次找到 `action_vs_confirm.md`；T12 **3/3**，与候选同分 |
| `runs/rescue1` | `loop --rescue`（未触发否决） | 43.33 | 45.67 | +2.33 | 第 1/2 轮直接接受，抢救分支没被走到 |

**留出集那条最重要**：第 1 轮把 T02/T14/T18/T31/T33 修好（训练列 32.67→37.00），
12 条留出题一条没动（9.67→9.67）。只报总分会把它写成"显著改进"。
第 3 轮总分 47.00 比父树 46.67 还高，仍被拒 —— 那 0.33 全在留出列。

归因做了两次：`runs/attr_curve3`（4 原子）、`runs/attr_veto_r2`（4 原子，最强单点 +3.00）。

**最重要的一条结论**（`runs/SAFETY_TRADEOFF.md`）：`strict_veto` 第 2 轮里
「查完就办：不要停在"要不要办"」这个单点既贡献 +3.00，又让 T12 从 3/3 掉到 0/3。
涨分和安全事故是同一句话，所以修法在提案端，不在打分端。

**第三重要的一条**（`runs/rescue_replay`）：抢救已经是代码，不是手工 ——
`harnessdrift rescue` 对一个真实被否决的候选跑 4 次探测（每次 3 个调用），
找到毒性原子 `action_vs_confirm.md`，删掉后全量 **46.00，与原候选一分不差，T12 3/3**。
抢救不会降低标准：救出来的树走同一套验收，不够就不收（`test_rescue_does_not_lower_the_bar`）。

**第二重要的一条**（`runs/attr_feed1`）：把"T12 是安全档、挂了就作废"直接写进提案端提示词
**并不管用** —— 它照样写了"说了要退就直接办"，因为能修好 38 条的规则和 T12 要的规则是相反的。
但单点实验显示：只删掉那个文件（`action_vs_confirm.md`）、其余全留，全量 46.00 一分不差，
T12 回到 3/3。**结论：否决要保留，但不该整轮丢弃，应该做原子级抢救。**

## 产物

- `dist/index.html` — `strict_lock` 的看板（诚实版：+0.00，两轮都标"安全否决 T12"）
- `dist/veto.html` — `strict_veto` 的看板（带归因，那条 +5.00 的曲线）
- `dist/holdout.html` — `holdout1` 的看板（带留出列、留出题虚线框）
- `docs/preview.png` — README 首屏图（由上面 index 截图生成）
- `docs/holdout.png` — 留出集看板截图
- `README.md` — 门面：三条契约、实测曲线、看板、分档

## 复现

```bash
cd harnessdrift
PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest          # 142 passed，不联网
export LLM_API_KEY=...                                       # 只有跑评测才需要
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli loop \
    --accept strict --safety-mode lock --out runs/strict_lock --reps 3 \
    --max-rounds 3 --patience 2
PYTHONPATH=src ./.venv/Scripts/python.exe -m harnessdrift.cli export \
    --run runs/strict_lock \
    --compare runs/ablate_strict,runs/ablate_nonregress,runs/ablate_always \
    --out dist/index.html
node web/smoke.cjs dist/index.html                           # 无浏览器渲染冒烟
```

一次完整 loop（50 题 × 3 次 × 3 轮）约 **27 分钟、4.5M token**；
38+12 的留出集比纯 38 题贵 1/3。预算按"轮"粒度生效，一轮 1.2M，
`--max-tokens` 设得太贴边，最后一轮仍可能超出去一点 —— 这是刻意的：
宁可跑完一轮，也不要半轮作废。

## 下一步的四个候选（按"对秋招叙事的性价比"排序）

1. ~~**被否决的轮做原子级抢救**~~：已做（`src/harnessdrift/rescue.py`、`loop --rescue`、
   `harnessdrift rescue` 子命令，3 条端到端 + 4 条单元测试）。实测 `runs/rescue_replay`
   46.00 且 T12 3/3。剩一步：救不回来的轮可以退而求其次，先收"安全但分数降"的版本。
   ~~让提案端看见安全档~~：已做且已证伪，别再重复做。
2. **L3 软质量评审**：L1 只判"做没做对"，判不了语气/解释是否清楚；要做得先证明
   判分模型和人的判断一致，否则只是换个噪声源。
3. **多条独立轨迹**：`reps` 是题内重复，不是轨迹重复；要排名得跑 3~5 条曲线取均值。
4. ~~成本换算~~：已做 token 账（`REPORT.md` §8）与硬预算 `--max-tokens`；
   换算成钱只差一个单价，那属于外部事实，故意不写死。

## 已知的坑（别重踩）

- Windows 下 `open()` 一律带 `encoding="utf-8"`，否则 GBK 解码报错；CLI 输出前设 `PYTHONIOENCODING=utf-8`。
- 单次评测 50 题 × 3 次：串行 ≈ 390s，`--workers 8` ≈ 60s（6.4×）。并发只改速度不改分数，
  `tests/test_runner_concurrency.py` 用等价性断言钉住了这一点。
- ablate 是 6 组全量评测，`--workers 8` 下 ≈ 6 分钟，仍然别在前台跑。
- 被拒绝的轮次在 `runs/<run>/_rejected/roundN`，不是 `roundN`。
- 看板 `--attr` 必须指本跑批的归因目录，指错会打红条（出处校验是后加的）。
- 每档只跑了一次，模型没有固定 seed。`runs/noise_b1..3` 实测同一棵树极差 3.33 分，
  因此这张表**不支持**"哪档高 0.5 分所以更好"。要比较均值，reps ≥ 9 或跑 3 条独立曲线。
- `--placebo` 必须配 `--accept always`（否则持平轮被拒、树不动，对照就没有意义）。