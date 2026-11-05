"""Render the MetisCraft v0.1 report (index.html) from src/data/*.json, the registry and the figures.

Every number on the page comes from src/data/ (written by collect.py from real runs) or from the
repository itself; nothing is typed by hand.  python build_report.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[3]
DATA = HERE / "data"

RECIPES = [  # (key, display name)
    ("boolq/qwen3-0.6b-base_answer-token", "BoolQ"),
    ("banking77/qwen3-0.6b-base_answer-token", "Banking77"),
    ("banking77/qwen3-0.6b-base_candidate-scalar", "Banking77"),
    ("clinc150/qwen3-0.6b-base_candidate-scalar", "CLINC150"),
    ("sst5/qwen3-0.6b-base_answer-token", "SST-5"),
    ("go_emotions/qwen3-0.6b-base_candidate-scalar", "GoEmotions"),
    ("banking77_support/qwen3-0.6b-base_candidate-scalar", "客服分流"),
]
READOUT = {"answer_token": "①", "candidate_scalar": "③"}
KIND = {"noul": "是非", "choice": "选择", "score": "等级", "multi_label": "多标签", "ranking": "排序（v0.2）"}
HEADLINE = {"noul": ("accuracy", "准确率"), "choice": ("accuracy", "准确率"),
            "score": ("accuracy", "准确率"), "multi_label": ("macro_f1", "macro-F1")}
KEYS = {
    "noul": ["accuracy", "auc", "nll", "brier", "ece"],
    "choice": ["accuracy", "macro_f1", "nll", "brier", "ece"],
    "score": ["accuracy", "mae_expected", "nll", "brier", "ece"],
    "multi_label": ["micro_f1", "macro_f1", "exact_match", "nll", "brier", "ece"],
}
LABELS = {"accuracy": "准确率", "auc": "AUC", "nll": "NLL", "brier": "Brier", "ece": "ECE",
          "macro_f1": "macro-F1", "micro_f1": "micro-F1", "exact_match": "完全匹配",
          "mae_expected": "期望分 MAE"}
METHODS = {"model": "模型", "model + temperature": "模型 + 温度", "prior": "先验"}
TIERS = {"must": "必收", "optional": "可选", "eval_only": "只评测"}
TERMS = {"attribution": "署名", "share_alike": "相同方式共享", "license_unspecified": "未声明许可证",
         "research_only": "仅限研究", "pii_possible": "可能含个人信息", "non_commercial": "非商用",
         "eval_only_no_train": "只评测", "teacher_unknown": "教师模型未知", "synthetic": "合成数据",
         "third_party_text_unverified": "第三方文本未核实", "mixed_per_row": "逐行许可证不同",
         "soft_attributes": "软标签属性"}

CSS = """
:root { --ink:#273444; --muted:#657080; --line:#DADDE3; --paper:#F7FAFC; --blue:#A7C7E7; --blue-ink:#5878A6;
        --coral:#F4A7AE; --coral-ink:#C18489; --green:#A6D0C6; --green-ink:#7B9A93; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); background:#fff; line-height:1.7; font-size:15px;
       font-family:"Noto Sans CJK SC","Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif; }
nav { position:fixed; top:0; left:0; bottom:0; width:220px; padding:28px 18px; background:var(--paper);
      border-right:1px solid var(--line); overflow-y:auto; }
nav .brand { font-weight:700; font-size:17px; margin-bottom:2px; }
nav .sub { color:var(--muted); font-size:12px; margin-bottom:20px; }
nav a { display:block; color:var(--muted); text-decoration:none; padding:5px 10px; border-left:3px solid transparent;
        font-size:14px; }
nav a:hover, nav a.on { color:var(--blue-ink); border-left-color:var(--blue); background:#fff; }
main { margin-left:220px; padding:36px 56px 80px; max-width:1080px; }
h1 { font-size:26px; margin:0 0 6px; }
.lede { color:var(--muted); margin:0 0 26px; }
h2 { font-size:20px; margin:46px 0 14px; padding-left:12px; border-left:4px solid var(--blue); }
h3 { font-size:16px; margin:26px 0 8px; }
p { margin:8px 0; }
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 8px; }
.card { border:1px solid var(--line); border-radius:6px; padding:12px 14px; background:#fff; }
.card b { display:block; font-size:24px; color:var(--blue-ink); font-variant-numeric:tabular-nums; }
.card span { color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; margin:10px 0 6px; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--paper); font-weight:600; white-space:nowrap; }
th.n { text-align:right; }
td.nw { white-space:nowrap; }
td.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
tr.prior td { color:var(--muted); }
.better { color:var(--green-ink); font-weight:600; }
.pending { color:var(--muted); font-style:italic; }
figure { margin:14px 0 10px; }
figure img { max-width:100%; border:1px solid var(--line); border-radius:4px; }
figcaption { color:var(--muted); font-size:13px; margin-top:4px; }
pre { background:var(--paper); border:1px solid var(--line); border-radius:4px; padding:10px 12px; overflow-x:auto;
      font-size:12.5px; line-height:1.5; font-family:"JetBrains Mono","Fira Code",Menlo,Consolas,monospace; }
code { font-family:"JetBrains Mono","Fira Code",Menlo,Consolas,monospace; font-size:0.92em; background:var(--paper);
       padding:0 3px; border-radius:3px; }
pre code { background:none; padding:0; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.note { color:var(--muted); font-size:13px; }
ul { padding-left:22px; } li { margin:3px 0; }
footer { margin-top:56px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:10px; }
@media (max-width: 900px) { nav { position:static; width:auto; } main { margin-left:0; padding:20px; }
  .cards, .two { grid-template-columns:1fr 1fr; } }
"""

JS = """
const links = [...document.querySelectorAll('nav a')];
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) {
  links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); } }),
  { rootMargin: '-30% 0px -60% 0px' });
document.querySelectorAll('section[id]').forEach(s => io.observe(s));
"""


def esc(x) -> str:
    return html.escape(str(x))


def num(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:,}" if isinstance(v, int) else f"{v:.{digits}f}"


def code(text: str, lang: str = "") -> str:
    return f'<pre><code class="{lang}">{esc(text.rstrip())}</code></pre>'


def table(head: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), row_class=None) -> str:
    out = ["<table><thead><tr>" + "".join(f'<th{" class=n" if j in numeric else ""}>{h}</th>'
                                          for j, h in enumerate(head)) + "</tr></thead><tbody>"]
    for i, row in enumerate(rows):
        cls = f' class="{row_class(i)}"' if row_class and row_class(i) else ""
        out.append(f"<tr{cls}>" + "".join(f'<td{" class=n" if j in numeric else ""}>{c}</td>'
                                          for j, c in enumerate(row)) + "</tr>")
    return "".join(out) + "</tbody></table>"


def figure(src: str, caption: str, svg: str | None = None) -> str:
    link = f'（<a href="{svg}">SVG</a>）' if svg else ""
    return f'<figure><a href="{src}"><img src="{src}" alt="{esc(caption)}"></a><figcaption>{caption}{link}</figcaption></figure>'


def load(name: str, default):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def kinds_of(result: dict) -> list[str]:
    test = result["metrics"].get("test", {})
    return [k for k in KEYS if k in test.get("model", {})]


def headline_rows(results: dict) -> list[list[str]]:
    rows = []
    for key, name in RECIPES:
        r = results.get(key)
        readout = "③" if "candidate-scalar" in key else "①"
        if r is None:
            rows.append([f"{name} {readout}", '<span class="pending">待验收</span>', "", "", "", "", "", "", ""])
            continue
        test = r["metrics"]["test"]
        for kind in kinds_of(r):
            metric, label = HEADLINE[kind]
            m, t, p = test["model"][kind], test.get("model + temperature", {}).get(kind, {}), test["prior"][kind]
            temp = r["temperatures"].get(f"*|{kind}")
            rows.append([f"{name} {readout}", KIND[kind], f"{label}", num(m["n"]), num(t.get(metric, m[metric])),
                         num(p[metric]), f"{num(m['nll'])} → {num(t.get('nll'))}",
                         f"{num(m['ece'])} → <span class=better>{num(t.get('ece'))}</span>", num(temp, 2)])
    return rows


def dirty_note(r: dict) -> str:
    if not r["dirty"]:
        return ""
    if r.get("tracked_changes"):
        return "（有未提交改动，见运行目录的 git.diff）"
    return "（工作区有未跟踪文件，已跟踪的代码与提交一致）"


def detail_tables(results: dict) -> str:
    out = []
    for key, name in RECIPES:
        r = results.get(key)
        readout = "③ 逐候选标量头" if "candidate-scalar" in key else "① 答案 token"
        out.append(f"<h3>{name} · {readout}</h3>")
        if r is None:
            out.append('<p class="pending">待验收。</p>')
            continue
        temps = "，".join(f"{KIND[k.split('|')[1]]} {v:.2f}" for k, v in r["temperatures"].items() if k.startswith("*|"))
        out.append(f'<p class="note">运行 <code>{esc(r["run"])}</code> · 提交 <code>{esc(r["commit"])}</code>'
                   f'{dirty_note(r)} · 数据 <code>{esc(r["data_version"])}</code> · '
                   f'1× {esc(r["device"])} · {r["steps"]:,} 步，训练 {r["train_minutes"]:.0f} 分钟 · '
                   f'损失 {r["loss_first"]:.2f} → {r["loss_last"]:.2f} · 温度 {temps}</p>')
        for split in ("test", "dev"):
            methods = r["metrics"].get(split)
            if not methods:
                continue
            for kind in [k for k in KEYS if k in methods.get("model", {})]:
                rows = [[METHODS.get(m, m)] + [num(v[kind].get(x)) for x in KEYS[kind]]
                        for m, v in methods.items() if kind in v]
                out.append(f'<p class="note">{split} · {KIND[kind]} · n={methods["model"][kind]["n"]:,}</p>')
                out.append(table(["方法"] + [LABELS[x] for x in KEYS[kind]], rows,
                                 numeric=set(range(1, len(KEYS[kind]) + 1)),
                                 row_class=lambda i, rows=rows: "prior" if rows[i][0] == "先验" else ""))
        nb = r.get("null_breakdown")
        if nb:
            out.append(f'<p class="note">test 拆开看（按 argmax）：范围内 {nb["ins"]:,} 条准确率 {nb["ins_ok"] / nb["ins"]:.3f}；'
                       f'范围外（应答 null）{nb["null"]:,} 条召回 {nb["null_ok"] / nb["null"]:.3f}；'
                       f'判为 null 的 {nb["pred_null"]:,} 条精确率 {nb["pred_null_ok"] / max(nb["pred_null"], 1):.3f}。</p>')
    return "".join(out)


CRITERIA = [  # (category, standard from design.md §12, tests that check it)
    ("单元测试", "pytest 在 CPU 上用微型随机初始化模型全部通过，不下载权重", []),
    ("等价性", "noul 的 ① 读出与两行 lm_head 之差的线性头一致；list 与 pairs 布局共享前缀的隐藏状态一致；产物在新进程加载后 logits 一致",
     ["test_noul_answer_token_equals_linear_head", "test_list_and_pairs_share_prefix_states",
      "test_predictor_reload_matches_training_process"]),
    ("可组合性", "2 种架构 × 2 种读出只改配置就能跑通训练、校准、评测", ["test_composability_matrix"]),
    ("梯度累积", "batch=b、GAS=k 与 batch=k·b、GAS=1 的损失和梯度在容差内一致",
     ["test_gradient_accumulation_matches_one_big_batch"]),
    ("校准", "温度缩放前后 argmax 不变；只在 calib 划分拟合", ["test_temperature_recovers_known_scale_and_keeps_argmax"]),
    ("数据", "首批转换器在样例上通过；SHA256SUMS 校验；产出去重报告",
     ["test_prepare_every_converter", "test_tampering_is_detected", "test_eval_first_dedup_in_pipeline"]),
    ("反馈接口", "奖励函数按 TRL reward_funcs 的签名形状调用，返回逐样本标量", ["test_reward_has_the_trl_shape"]),
    ("服务", "HTTP 服务对官方格式的请求返回合法响应，概率与 Predictor 一致", ["test_http_service_speaks_the_official_format"]),
    ("可追溯", "报告里每个数字都能追到运行目录；产物带 SHA256SUMS 与 NOTICE",
     ["test_candidate_scalar_run_is_complete_and_traceable", "test_answer_token_run_marks_research_only_data"]),
]


def criteria_table(results: dict, tests: dict) -> str:
    source = "\n".join(p.read_text(encoding="utf-8") for p in (REPO / "tests").rglob("test_*.py"))
    rows = []
    for category, standard, names in CRITERIA:
        missing = [n for n in names if f"def {n}(" not in source]
        if missing:
            raise SystemExit(f"criteria table names tests that do not exist: {missing}")
        if category == "单元测试":
            evidence = f"{tests['passed']} 个通过（提交 <code>{esc(tests['commit'])}</code>）" if tests else "待运行"
        else:
            evidence = "<br>".join(f"<code>{n}</code>" for n in names)
        rows.append([category, esc(standard), evidence])
    done = [name for key, name in RECIPES if key in results]
    rows.append(["验收训练", "Qwen3-0.6B 上 BoolQ、Banking77（CLINC150 作对照）、SST-5、GoEmotions 各跑通一次，"
                 "报告含校准前后与基线；Banking77 上出 ① 与 ③ 的对照；不设分数门槛",
                 f"{len(done)}/{len(RECIPES)} 份配方跑完，见下表"])
    return table(["类别", "标准（design.md §12）", "证据"], rows)


def registry_table() -> tuple[str, int]:
    reg = yaml.safe_load((REPO / "src/metiscraft/data/registry.yaml").read_text(encoding="utf-8"))["datasets"]
    rows = []
    for name, e in reg.items():
        judg = e.get("judgment") or []
        judg = judg if isinstance(judg, list) else [judg]
        repo = (e.get("hf") or "").split("@")[0]
        terms = "、".join(TERMS.get(t, t) for t in e.get("terms") or []) or "–"
        rows.append([f"<code>{esc(name)}</code>", f'<span style="white-space:nowrap">{TIERS.get(e.get("tier"), esc(e.get("tier", "")))}</span>',
                     "、".join(KIND.get(j, esc(j)) for j in judg), esc(repo), esc(e.get("license", "")), terms])
    return table(["注册名", "档位", "判断类型", "上游", "许可证", "条款"], rows), len(rows)


def build() -> str:
    results = load("acceptance.json", {})
    snippets = load("snippets.json", {})
    tests = load("tests.json", {})
    reg_html, n_datasets = registry_table()
    done = sum(1 for k, _ in RECIPES if k in results)
    nav = [("overview", "一页看懂"), ("pipeline", "全流程"), ("data", "数据"), ("readouts", "判断与读出"),
           ("results", "验收结果"), ("scenario", "垂直场景"), ("trace", "可追溯"), ("next", "局限与下一步"),
           ("reproduce", "复现")]
    parts = [f'<nav><div class="brand">MetisCraft</div><div class="sub">v0.1 交付报告 · 2026-11-05</div>'
             + "".join(f'<a href="#{i}">{t}</a>' for i, t in nav) + "</nav><main>"]

    parts.append('<section id="overview"><h1>MetisCraft v0.1 交付报告</h1>'
                 '<p class="lede">训练 Jev 类（System One）决策模型的工具包：一次前向给出类型化判断和校准过的概率。'
                 '既支持通用 judge，也能把垂直业务场景做成自己的 Jev。</p>')
    cards = [(n_datasets, "注册的数据集"), (4, "判断类型（是非、选择、等级、多标签）"), (2, "可对照的读出（① 与 ③）"),
             (f"{done}/{len(RECIPES)}", "跑完的验收配方")]
    if tests.get("passed"):
        cards[3] = (f"{done}/{len(RECIPES)}", f"验收配方；CPU 测试 {tests['passed']} 个通过")
    parts.append('<div class="cards">' + "".join(f"<div class=card><b>{v}</b><span>{t}</span></div>" for v, t in cards)
                 + "</div>")
    parts.append("<h3>验收一览（test 划分）</h3>")
    parts.append(table(["配方", "类型", "主指标", "n", "模型 + 温度", "先验", "NLL 校准前 → 后", "ECE 校准前 → 后", "温度"],
                       headline_rows(results), numeric={3, 4, 5, 8}))
    parts.append('<p class="note">基座都是 Qwen3-0.6B-Base 加 LoRA，单卡训练，每份运行限时 2 小时。'
                 "① 是答案 token 读出，③ 是逐候选标量头。温度只在 calib 划分上拟合；先验是只看训练集类别频率、不看输入的基线。"
                 "不设分数门槛，结果照实报告。</p></section>")

    parts.append('<section id="pipeline"><h2>全流程</h2>'
                 + figure("figures/01-run-pipeline.png", "图 1　一次运行：从注册表到 /v1/systemone", "figures/01-run-pipeline.svg")
                 + "<ol><li><b>数据</b>：<code>metiscraft data prepare</code> 按注册表里固定的上游 revision 下载，"
                   "转成 Jev 记录（<code>state</code> + 问题 + 目标），先按评测集去重，再写进带 <code>SHA256SUMS</code> 的版本目录。</li>"
                   "<li><b>训练</b>：配方选定数据、基座、读出和目标；训练循环按全局问题数归一损失，梯度累积与多卡下数值一致。</li>"
                   "<li><b>封存</b>：产物含 LoRA adapter、读出头、许可证 <code>NOTICE</code> 和 <code>SHA256SUMS</code>。</li>"
                   "<li><b>校准与评测</b>：按 任务|类型 各拟合一个温度，只用 calib；dev 与 test 上同表报告模型、模型 + 温度和先验。</li>"
                   "<li><b>部署</b>：<code>metiscraft serve</code> 用官方 <code>/v1/systemone</code> 请求格式回答。</li></ol>")
    if snippets.get("recipe"):
        parts.append("<h3>一份真实配方</h3>" + code(snippets["recipe"], "yaml"))
    parts.append("</section>")

    parts.append(f'<section id="data"><h2>数据</h2><p>注册表收录 {n_datasets} 个数据集，分必收、可选、只评测三档；'
                 "每条都固定上游 revision，并带许可证与条款标记。用条款受限数据训练的产物，<code>NOTICE</code> 会写明不可商用。</p>"
                 + reg_html)
    if snippets.get("dataset_tree"):
        parts.append("<div class=two><div><h3>版本目录（真实）</h3>" + code("\n".join(snippets["dataset_tree"])) +
                     "</div><div><h3>说明</h3><ul><li><code>raw/</code> 是上游原样文件，只读。</li>"
                     "<li><code>derived/</code> 是转换后的 Jev 记录；<code>eval_only/</code> 里的 test 只在评测时读取。</li>"
                     "<li><code>manifests/</code> 记录算法输入、只评测集合和去重报告。</li>"
                     "<li>已有版本只校验、不覆盖；准备过程在临时目录完成。</li></ul></div></div>")
    if snippets.get("records"):
        parts.append("<h3>同一条消息的两条记录（真实）</h3>"
                     + code(json.dumps(snippets["records"], ensure_ascii=False, indent=2), "json"))
    parts.append("</section>")

    parts.append('<section id="readouts"><h2>判断类型与读出</h2>'
                 + table(["类型", "输出", "损失", "主要指标"], [
                     ["是非 noul", "一个概率 p(是)", "二元交叉熵", "准确率、AUC、NLL、Brier、ECE"],
                     ["选择 choice", "选项上的分布，可含 null 选项", "交叉熵", "准确率、macro-F1、NLL、Brier、ECE"],
                     ["等级 score", "等级上的分布与期望分", "交叉熵（可用序数软目标）", "准确率、期望分 MAE、NLL、ECE"],
                     ["多标签 multi_label", "每个候选独立的概率", "逐候选二元交叉熵", "micro/macro-F1、完全匹配、逐候选 ECE"]])
                 + "<div class=two><div>"
                 + figure("../../readouts/readout-1-lm-token.png", "① 答案 token：在答案位置读选项码的 logits，不加新层")
                 + "</div><div>"
                 + figure("../../readouts/readout-3-candidate-scalar.png", "③ 逐候选标量头：每个候选末 token 的隐藏状态经 MLP 打分")
                 + "</div></div><p>两种读出只差配方里的 <code>readout.kind</code> 一行，其余设置相同，所以能在同一协议下对照。"
                   "① 受限于选项码（单 token 的码表），③ 没有选项数限制，也能做多标签。七类读出的完整张量说明见 "
                   '<a href="../../readouts/README.md">readouts</a>。</p></section>')

    parts.append('<section id="results"><h2>验收结果</h2><h3>验收标准与证据</h3>' + criteria_table(results, tests))
    if (ROOT / "figures/03-acceptance-results.png").is_file():
        parts.append(figure("figures/03-acceptance-results.png", "图 3　test 划分上的主指标与校准", "figures/03-acceptance-results.svg"))
    parts.append(detail_tables(results)
                 + '<p class="note">ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。</p></section>')

    parts.append('<section id="scenario"><h2>垂直场景：把客服业务做成 Jev</h2>'
                 + figure("figures/02-support-scenario.png", "图 2　客服分流：映射表把 77 个意图改写成两道业务题", "figures/02-support-scenario.svg"))
    ex = snippets.get("api_example")
    if ex:
        parts.append(f"<div class=two><div><h3>请求（示例消息为构造）</h3>{code(json.dumps(ex['request'], ensure_ascii=False, indent=2), 'json')}</div>"
                     f"<div><h3>验收产物 <code>{esc(ex['run'])}</code> 的实际响应</h3>"
                     f"{code(json.dumps(ex['response'], ensure_ascii=False, indent=2), 'json')}</div></div>")
    parts.append('<p>换成自己的业务，只要在注册表里写一张映射表和题目说明，其余流程不变。详见 '
                 '<a href="../../../recipes/banking77_support/README.md">配方说明</a>。</p></section>')

    parts.append('<section id="trace"><h2>可追溯</h2><p>每次运行一个目录，记下完整配置、git 提交（含未提交改动）、环境、'
                 "数据版本与基座身份；报告里的每个数字都能追到运行 id。</p>")
    if snippets.get("run_tree"):
        parts.append("<div class=two><div><h3>运行目录（真实）</h3>" + code("\n".join(snippets["run_tree"])) +
                     "</div><div><h3>产物里的 NOTICE（真实）</h3>" + code(snippets["notice"]) + "</div></div>")
    parts.append("</section>")

    parts.append('<section id="next"><h2>局限与下一步</h2><ul>'
                 "<li>只验收了 Qwen3-0.6B-Base 一个基座；1.7B、4B 和第二个基座家族放到后续版本。</li>"
                 "<li>没有做 encoder 强基线的同协议对照，参照数字都是作者自报、协议不同。</li>"
                 "<li>排序（ranking）判断类型、ms-swift / HF 适配层、vLLM / SGLang / llama.cpp 引擎导出放到 v0.2。</li>"
                 "<li>评测推理沿用训练的批量，候选多的任务（如 CLINC150）评测偏慢，可单独设评测批量。</li>"
                 "<li>每份配方只跑了一个种子，没有扫学习率。</li></ul></section>")

    parts.append('<section id="reproduce"><h2>复现</h2>'
                 + code("pip install -e \".[serve]\"\n"
                        "export METISCRAFT_DATA_ROOT=… METISCRAFT_MODELS_ROOT=… METISCRAFT_RUNS_ROOT=…\n"
                        "metiscraft data prepare banking77_support\n"
                        "metiscraft train recipes/banking77_support/qwen3-0.6b-base_candidate-scalar.yaml\n"
                        "metiscraft serve <运行目录>/artifact", "bash")
                 + '<p class="note">本页由 <code>src/build_report.py</code> 生成，数字来自 <code>src/data/</code>'
                   "（<code>src/collect.py</code> 从运行目录收集）；流程图规格在 <code>src/*.json</code>，检查记录在 <code>checks/</code>。</p></section>")
    parts.append("<footer>MetisCraft v0.1 · Shichao</footer></main>")
    return ("<!DOCTYPE html><html lang=zh-CN><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>MetisCraft v0.1 交付报告</title><style>{CSS}</style></head><body>"
            + "".join(parts) + f"<script>{JS}</script></body></html>\n")


if __name__ == "__main__":
    (ROOT / "index.html").write_text(build(), encoding="utf-8")
    print(ROOT / "index.html")
