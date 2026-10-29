# 参与进来

这个项目**最欢迎的贡献是"你证明了某个结论是错的"**。
它现在写下的每一条结论（回退规则、噪声区间、原子级抢救）都带着可复现的命令，
所以推翻它也有明确的做法：跑一遍、把 json 贴出来、指出 README 哪句话该改。

## 三条硬规矩

1. **不许改历史数字。** `runs/` 里的曲线是证据，不是草稿。跑错了就新增一个目录，
   不要改旧 json 里的分数。README 里写"这条曲线跑在 38 题上"而不是把分母改成 50 ——
   改数字能让结论变好看，也会让整个仓库失去可信度。
2. **每个新结论要有一条能失败的测试。** 比如"抢回被否决的轮"这个功能，
   测试里必须有一条断言"救不回来的时候要老实报失败"（`test_rescue.py`）。
3. **离线优先。** 测试一律不碰网络：注入假 LLM，跑在 3 秒内。
   需要真实模型的实验放 `runs/` 里，并在文档里写清 token 成本。

## 想加东西时

| 想加什么 | 从哪儿下手 | 别忘了 |
|---|---|---|
| 一道新题 | `data/tasks_ecommerce_after_sales.json`（带 `why` / `tier` / `split`） | 判分要确定性；安全题标 `tier: safety` |
| 一类新的失败检测 | `evaluate.py` 的 checks | 加一条"故意写错必须被抓"的测试 |
| 一种新的接受规则 | `loop.py` 的 `_accepted` / `_is_improved` | 至少两条曲线证明它和现有规则不一样 |
| 一种新的归因粒度 | `ablate.py` 的 `atoms_of` | 归因的分数必须能被 `runs/` 复现 |
| 前端 | `web/template.html` + `web/smoke.cjs` | `node web/smoke.cjs dist/index.html` 要过 |

## 本地检查（提交前跑这四个）

```bash
ruff check src tests                                   # 风格
pytest                                                 # 142 条，全离线
harnessdrift export --run runs/curve3 --out dist/index.html && node web/smoke.cjs dist/index.html
git status --short                                     # 不该出现 .env 或新的 run 目录
```

## 不适合放进来的东西

- **真实客户数据**：题库是合成的，规则来自公开的电商售后知识库。别把线上工单塞进来。
- **只在某台机器上跑得通的东西**：Windows 路径、GBK 编码、绝对路径截图，
  这个项目已经在这三处各踩过一次坑，`tests/` 里都有对应的回归测试。
- **没有对照组的效果声明**：说"改了之后涨了 3 分"之前，先给出噪声区间。

## 许可

MIT。用就行了，出了事别找我（这句话在 `LICENSE` 里写得更正式）。