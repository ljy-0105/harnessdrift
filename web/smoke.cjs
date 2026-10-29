// 在 node 里跑一遍看板的渲染逻辑，确认没有运行时错误、产出非空。
// 用法：node web/smoke.cjs dist/index.html
const fs = require("fs");

const file = process.argv[2] || "dist/index.html";
const html = fs.readFileSync(file, "utf8");

// 1) 取出数据块：从 window.__DATA__ = 开始，到它所在的 </script> 之前
const start = html.indexOf("window.__DATA__ = ");
if (start < 0) throw new Error("没找到 window.__DATA__");
const endTag = html.indexOf("</script>", start);
const stmt = html.slice(start, endTag).trim().replace(/;$/, "");
const D = JSON.parse(stmt.replace(/^window\.__DATA__ = /, ""));
console.log("数据 OK：", D.rounds.length, "轮,", D.tasks.length, "题,",
  (D.attribution?.atoms || []).length, "原子,", D.curves.length, "条对照");

// 2) 取出渲染逻辑块（最后一个 <script>）
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const logic = blocks[blocks.length - 1];

// 3) 最小 DOM stub
let mounted = "";
const stub = { innerHTML: "" };
global.window = { __DATA__: D };
global.document = { querySelector: () => stub };

// 4) 执行
new Function(logic)();
mounted = stub.innerHTML;
if (!mounted || mounted.length < 2000) throw new Error("渲染输出太短：" + mounted.length);

// 5) 结构断言：必有的四段 + "有数据才该出现"的两段
//    归因和消融是可选的（一次没有归因的跑批不该硬凑一段空表格），
//    但只要数据里有，渲染就必须把那一节画出来 —— 否则就是静默丢内容。
const need = ["自改进曲线", "每一轮改了什么", "题矩阵", "kpi"]
  .concat((D.attribution?.atoms || []).length ? ["归因"] : [])
  .concat((D.curves || []).length ? ["消融"] : []);
const missing = need.filter((k) => !mounted.includes(k));
if (missing.length) throw new Error("缺少段落：" + missing.join(","));
for (const bad of ["undefined", "NaN", "object Object"]) {
  if (mounted.includes(bad)) throw new Error("页面里出现了 " + bad);
}
console.log("渲染 OK：", mounted.length, "字符");
console.log("段落：", need.join(" / "));
