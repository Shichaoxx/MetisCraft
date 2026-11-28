"""Build the English/Chinese site from canonical repository Markdown, without copies to maintain."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "documentation"
SITE = BUILD / "site"
ORIGIN = "https://shichaoxx.github.io"
DEFAULT_PREFIX = "/MetisCraft/"
# Set METIS_SITE_PREFIX to build and validate the same sources under another Pages path.
PREFIX = os.environ.get("METIS_SITE_PREFIX", DEFAULT_PREFIX)
if not re.fullmatch(r"/(?:[A-Za-z0-9_-][A-Za-z0-9._-]*/)*", PREFIX):
    raise ValueError("METIS_SITE_PREFIX must be a slash-delimited Pages path, e.g. /Metis/review/")
SOURCE_REF = "91259c6d07ece19248a02a4b56b43dd3a77d81e5"
REPO = f"https://github.com/Shichaoxx/MetisCraft/blob/{SOURCE_REF}/"


def legacy_destination(name):
    """Resolve a repository source to its established public URL."""
    path = Path(name)
    if path.parts[0] == "docs":
        path = Path(*path.parts[1:])
        if path.parts[0] not in {"en", "zh"}:
            path = Path("zh") / path
    elif path.parts[0] == "recipes":
        path = Path("zh") / path
    else:
        lang = "zh" if path.stem.endswith("_zh") else "en"
        path = Path(lang) / (
            "contributing.md" if name.startswith("CONTRIBUTING") else "changelog.md"
        )
    if path.name in {"README.md", "README_en.md", "README_zh.md"}:
        path = path.with_name("index.md")
    return path


def public_destinations(policy):
    """A language pair has one logical URL, independent of source file placement."""
    destinations = {name: legacy_destination(name) for name in policy["pages"]}
    pairs = policy.get("language_pairs", {})
    for pair in pairs.values():
        if set(pair) != {"en", "zh"}:
            raise ValueError(f"A public language pair must contain en and zh: {pair}")
        canonical = legacy_destination(pair["zh"])
        for lang in ("en", "zh"):
            name = pair[lang]
            if name not in destinations:
                raise ValueError(f"Language pair references a non-public source: {name}")
            destinations[name] = Path(lang) / Path(*canonical.parts[1:])
    if len(set(destinations.values())) != len(destinations):
        raise ValueError("Public page routes collide")
    return destinations


def toml(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {toml(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def page_url(path):
    path = path.with_suffix("")
    if path.name == "index":
        path = path.parent
    return PREFIX + path.as_posix().rstrip("/") + "/"


def site_navigation(lang):
    def item(chinese, english, target):
        return {chinese if lang == "zh" else english: target}

    return [
        item("首页", "Home", "index.md"),
        {
            "1. JevLike": [
                item("决策模型研究", "Decision model research", "jevlike/index.md"),
                item(
                    "基础原理",
                    "Foundations",
                    [
                        item(
                            "任务与答案空间",
                            "Tasks and answer spaces",
                            "tensor-analysis/introduction.md",
                        ),
                        item("接口与输出语义", "Contracts and outputs", "tensor-analysis/index.md"),
                        item(
                            "张量、轴与索引",
                            "Tensors, axes and indices",
                            "tensor-analysis/notation.md",
                        ),
                    ],
                ),
                item(
                    "方法与开源实现",
                    "Methods and implementations",
                    [
                        item("架构图谱", "Architecture atlas", "tensor-analysis/architectures.md"),
                        item(
                            "答案 token 与固定槽位",
                            "Answer tokens and fixed slots",
                            "tensor-analysis/architectures-token.md",
                        ),
                        item(
                            "标量、指针与联合头",
                            "Scalar, pointer and joint heads",
                            "tensor-analysis/architectures-structured.md",
                        ),
                        {"Tev1-4B": "jevlike/tev1.md"},
                        {"JevForge": "jevlike/jevforge.md"},
                        {"Kev-4B": "jevlike/kev.md"},
                        item(
                            "编码器与逐对重排",
                            "Encoders and pair rerankers",
                            "tensor-analysis/architectures-encoders.md",
                        ),
                        item(
                            "答案 token 读出",
                            "Answer-token readout",
                            "tensor-analysis/answer-token.md",
                        ),
                        item("标量头读出", "Scalar readouts", "tensor-analysis/scalar-heads.md"),
                    ],
                ),
                item(
                    "综合比较",
                    "Comparative analysis",
                    [
                        item(
                            "候选打分与概率",
                            "Candidate scores and probabilities",
                            "jevlike/candidate-scores.md",
                        ),
                        item(
                            "读出机制比较",
                            "Comparing readouts",
                            "tensor-analysis/readout-comparison.md",
                        ),
                        item(
                            "训练、校准与输出",
                            "Training, calibration and outputs",
                            "tensor-analysis/training-calibration.md",
                        ),
                        item("实现与证据矩阵", "Implementations and evidence", "support-matrix.md"),
                    ],
                ),
            ]
        },
        {
            "2. MetisCraft": [
                item("开发手册", "Developer manual", "metiscraft/index.md"),
                item("安装", "Installation", "installation.md"),
                item("快速开始", "Quickstart", "quickstart.md"),
                item("模型与产物", "Model and artifacts", "metiscraft/model-and-artifacts.md"),
                item("数据与配方", "Data and recipes", "data.md"),
                item("训练与校准", "Training and calibration", "training.md"),
                item("预测与服务", "Prediction and serving", "serving.md"),
                item(
                    "技术参考",
                    "Reference",
                    [
                        item("接口约定", "Interface contracts", "interfaces.md"),
                        item("配方字段", "Recipe fields", "cookbook/recipe-reference.md"),
                        item("兼容范围", "Compatibility", "compatibility.md"),
                        item("验证范围", "Evidence and limitations", "limitations.md"),
                        item("贡献指南", "Contribution guide", "contributing.md"),
                        item("版本说明", "Changelog", "changelog.md"),
                    ],
                ),
            ]
        },
        {
            "3. Cookbook": [
                item("实验与教程", "Experiments and tutorials", "cookbook/index.md"),
                item("客服分流示例", "Support routing example", "cookbook/decision.md"),
                item("CPU 快速开始", "First model on CPU", "cookbook/first-model.md"),
                item("自定义任务", "Custom tasks", "cookbook/own-task.md"),
                item("张量计算示例", "Tensor walkthrough", "tensor-analysis/walkthrough.md"),
                item("建模原则", "Modeling principles", "cookbook/principles.md"),
                item("实践建议", "Best practices", "cookbook/best-practices.md"),
                item(
                    "结果与复现",
                    "Results and reproductions",
                    [
                        item("Tev1 复现", "Tev1 reproduction", "reproductions/tev1.md"),
                        {"Banking77": "recipes/banking77/index.md"},
                        item("客服分流", "Customer routing", "recipes/banking77_support/index.md"),
                        {"BoolQ": "recipes/boolq/index.md"},
                        {"CLINC150": "recipes/clinc150/index.md"},
                        {"GoEmotions": "recipes/go_emotions/index.md"},
                        {"SST-5": "recipes/sst5/index.md"},
                        item("Tev1 训练配方", "Tev1 training recipe", "recipes/tev1/index.md"),
                    ],
                ),
            ]
        },
    ]


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)  # Only this script's ignored, reproducible build output.
    SITE.mkdir(parents=True)
    policy = json.loads((ROOT / "docs/public-pages.json").read_text())
    destinations = public_destinations(policy)
    mapping = {}
    sources = [ROOT / name for name in policy["pages"]]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"Missing public page: {source.relative_to(ROOT)}")
        mapping[source.resolve()] = destinations[source.relative_to(ROOT).as_posix()]
    for alias, target in policy["aliases"].items():
        mapping[(ROOT / alias).resolve()] = mapping[(ROOT / target).resolve()]
    partners = {}
    for pair in policy.get("language_pairs", {}).values():
        for name in pair.values():
            partners[(ROOT / name).resolve()] = pair
    source_snapshots = {}

    def rewrite(value, source):
        url = urlsplit(value)
        if not url.scheme and not url.netloc and value.startswith(DEFAULT_PREFIX):
            return PREFIX + value.removeprefix(DEFAULT_PREFIX)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return value
        target = (source.parent / unquote(url.path)).resolve()
        suffix = ("?" + url.query if url.query else "") + (
            "#" + url.fragment if url.fragment else ""
        )
        if target.is_dir():
            for name in ("README.md", "index.md"):
                if (target / name).exists():
                    target = target / name
                    break
        if target in mapping:
            # Legacy shared sources resolve to the reader's language. Explicit en/zh
            # links remain explicit, including the language selector in Markdown.
            resolved = target
            canonical_name = policy["aliases"].get(target.relative_to(ROOT).as_posix())
            if canonical_name:
                resolved = (ROOT / canonical_name).resolve()
            rel = target.relative_to(ROOT)
            explicit = (
                rel.parts[0] == "docs" and len(rel.parts) > 1 and rel.parts[1] in {"en", "zh"}
            ) or rel.stem.endswith(("_en", "_zh"))
            if resolved in partners and not explicit:
                source_lang = mapping.get(source.resolve(), Path("zh/index.md")).parts[0]
                target = (ROOT / partners[resolved][source_lang]).resolve()
            return page_url(mapping[target]) + suffix
        if not target.is_relative_to(ROOT) or not target.exists():
            raise ValueError(f"Broken local link in {source.relative_to(ROOT)}: {value}")
        relative = target.relative_to(ROOT)
        if target.suffix == ".md" and relative.parts[0] == "docs":
            raise ValueError(f"Non-public page linked from {source.relative_to(ROOT)}: {value}")
        if target.suffix in {
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".html",
            ".json",
            ".yaml",
            ".tex",
            ".zip",
        }:
            output = SITE / "assets" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, output)
            if target.suffix == ".svg":
                companion = target.with_name(target.stem + ".dark.svg")
                if companion.is_file():
                    shutil.copyfile(companion, output.with_name(companion.name))
            # Standalone report HTML has its own local figures and styles.
            if target.suffix == ".html":
                rendered = re.sub(
                    r'((?:src|href)=")([^"]+)(")',
                    lambda m: m[1] + rewrite(m[2], target) + m[3],
                    target.read_text(encoding="utf-8"),
                )
                output.write_text(rendered, encoding="utf-8")
                for folder in ("figures", "assets"):
                    if (target.parent / folder).is_dir():
                        shutil.copytree(
                            target.parent / folder, output.parent / folder, dirs_exist_ok=True
                        )
            return PREFIX + "assets/" + quote(relative.as_posix()) + suffix
        if target.is_file():
            if relative not in source_snapshots:
                try:
                    original = subprocess.check_output(
                        ["git", "show", f"{SOURCE_REF}:{relative.as_posix()}"],
                        cwd=ROOT,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
                    original = None
                source_snapshots[relative] = original == target.read_bytes()
            if not source_snapshots[relative]:
                # Raw Markdown downloads are source text, not another published
                # page. A shallow checkout may not contain SOURCE_REF, so this
                # fallback must also pass the documentation engine's link check.
                download = (
                    relative.with_name(relative.name + ".txt")
                    if relative.suffix == ".md"
                    else relative
                )
                output = SITE / "assets" / download
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, output)
                return PREFIX + "assets/" + quote(download.as_posix()) + suffix
        return REPO + quote(relative.as_posix()) + suffix

    for source in sources:
        dest = mapping[source.resolve()]
        content = source.read_text(encoding="utf-8")
        # Keep copyable tutorial commands on the same deployment as their data.
        content = content.replace(ORIGIN + DEFAULT_PREFIX, ORIGIN + PREFIX)
        content = re.sub(
            r"(!?\[[^\]\n]*\]\()([^\s)]+)(\))",
            lambda m, source=source: m[1] + rewrite(m[2], source) + m[3],
            content,
        )
        content = re.sub(
            r"(?m)^(\[[^\]]+\]:\s*)(\S+)",
            lambda m, source=source: m[1] + rewrite(m[2], source),
            content,
        )
        content = re.sub(
            r'((?:src|href|srcset)=")([^"]+)(")',
            lambda m, source=source: m[1] + rewrite(m[2], source) + m[3],
            content,
        )
        output = BUILD / "sources" / dest
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")

    for lang in ("en", "zh"):
        zh = lang == "zh"
        nav = site_navigation(lang)
        project = {
            "site_name": "Metis",
            "site_url": f"{ORIGIN}{PREFIX}{lang}/",
            "site_description": (
                "JevLike 决策模型研究、MetisCraft 开发手册与可复现实验。"
                if zh
                else "JevLike model research, the MetisCraft manual and reproducible experiments."
            ),
            "docs_dir": f"sources/{lang}",
            "site_dir": f"site/{lang}",
            "repo_url": "https://github.com/Shichaoxx/MetisCraft",
            "repo_name": "MetisCraft",
            "nav": nav,
            "extra_css": [
                PREFIX + "assets/site.css",
                PREFIX + "assets/documentation.css",
                PREFIX + "assets/home.css",
                PREFIX + "assets/katex/katex.min.css",
            ],
            "extra_javascript": [
                PREFIX + "assets/katex/katex.min.js",
                PREFIX + "assets/math.js",
                PREFIX + "assets/figures.js",
                PREFIX + "assets/language-map.js",
                PREFIX + "assets/documentation.js",
            ],
        }
        config = "[project]\n" + "\n".join(f"{k} = {toml(v)}" for k, v in project.items())
        config += """
[project.theme]
language = "LANG"
font = false
logo = "SITE_PREFIXassets/docs/visuals/figures/mark.svg"
favicon = "SITE_PREFIXassets/docs/visuals/figures/mark.svg"
features = ["content.code.copy", "navigation.top", "search.highlight"]
[[project.theme.palette]]
media = "(prefers-color-scheme)"
scheme = "auto"
primary = "custom"
accent = "custom"
toggle.icon = "lucide/moon"
toggle.name = "Dark mode"
[[project.theme.palette]]
media = "(prefers-color-scheme: dark)"
scheme = "slate"
primary = "custom"
accent = "custom"
toggle.icon = "lucide/sun"
toggle.name = "Light mode"
[[project.theme.palette]]
media = "(prefers-color-scheme: light)"
scheme = "default"
primary = "custom"
accent = "custom"
toggle.icon = "lucide/monitor"
toggle.name = "Use system theme"
[project.extra]
scope = "SITE_PREFIX"
alternate = [
 { name = "English", link = "SITE_PREFIXen/", lang = "en" },
 { name = "简体中文", link = "SITE_PREFIXzh/", lang = "zh" }
]
[project.markdown_extensions]
admonition = {}
attr_list = {}
md_in_html = {}
toc.permalink = true
toc.permalink_title = "Permanent link"
toc.slugify = { kwds = { case = "lower" } }
pymdownx.highlight = {}
pymdownx.superfences = {}
pymdownx.details = {}
pymdownx.tasklist.custom_checkbox = true
pymdownx.arithmatex.generic = true
""".replace('language = "LANG"', f'language = "{lang}"').replace("SITE_PREFIX", PREFIX)
        if zh:
            config = (
                config.replace('toggle.name = "Dark mode"', 'toggle.name = "深色主题"')
                .replace('toggle.name = "Light mode"', 'toggle.name = "浅色主题"')
                .replace('toggle.name = "Use system theme"', 'toggle.name = "跟随系统"')
                .replace(
                    'toc.permalink_title = "Permanent link"', 'toc.permalink_title = "章节链接"'
                )
            )
        path = BUILD / f"zensical-{lang}.toml"
        path.write_text(config, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "zensical", "build", "-f", str(path), "--strict"],
            cwd=ROOT,
            check=True,
        )
        # Full page navigation is used here. Pages without equations do not need
        # the math runtime or fonts; math.js remains a harmless shared entry.
        for page in (SITE / lang).rglob("*.html"):
            html = page.read_text()
            html = re.sub(
                r'<img\b(?![^>]*\bloading=)([^>]*\bsrc="[^"]*/(?:tensor-analysis/figures|visuals/figures)/(?:pipeline|judgments|[^"]+\.(?:en|zh))[^"]*\.svg"[^>]*)>',
                r'<img loading="lazy" decoding="async"\1>',
                html,
            )
            if 'class="arithmatex"' not in html:
                html = re.sub(
                    r'<script\b[^>]*src="[^"]*/katex/katex\.min\.js"[^>]*></script>', "", html
                )
                html = re.sub(r'<link\b[^>]*href="[^"]*/katex/katex\.min\.css"[^>]*>', "", html)
            page.write_text(html)
    # Old tutorial URLs are redirects, not a second maintained tutorial or search page.
    for alias, target in policy["aliases"].items():
        old = legacy_destination(alias)
        destination = page_url(mapping[(ROOT / target).resolve()])
        if page_url(old) == destination:
            continue
        output = (
            SITE / old.parent / "index.html"
            if old.stem == "index"
            else SITE / old.with_suffix("") / "index.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f'<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url={destination}">'
            f'<link rel="canonical" href="{destination}"><title>MetisCraft</title>'
            f"<script>location.replace({json.dumps(destination)}+location.search+location.hash)</script>"
            f'<a href="{destination}">继续阅读文档</a></html>',
            encoding="utf-8",
        )
    assets = SITE / "assets" / "docs" / "visuals" / "figures"
    assets.mkdir(parents=True, exist_ok=True)
    # Page-linked figures were already copied by rewrite(); the header and favicon
    # additionally need these canonical theme variants, never the design candidates.
    for name in ("mark.svg", "mark-light.svg", "mark-dark.svg"):
        shutil.copyfile(ROOT / "docs/visuals/figures" / name, assets / name)
    shutil.copyfile(ROOT / "docs/site.css", SITE / "assets/site.css")
    shutil.copytree(ROOT / "docs/assets", SITE / "assets", dirs_exist_ok=True)
    language_map = {}
    for pair in policy.get("language_pairs", {}).values():
        links = {lang: page_url(destinations[name]) for lang, name in pair.items()}
        for link in links.values():
            language_map[link] = links
    for destination in destinations.values():
        url = page_url(destination)
        links = language_map.get(url)
        if not links:
            continue
        output = (
            SITE / destination.parent / "index.html"
            if destination.stem == "index"
            else SITE / destination.with_suffix("") / "index.html"
        )
        if output.is_file():

            def alternate(match, links=links):
                tag = match[0]
                lang = re.search(r'hreflang="(en|zh)"', tag)[1]
                tag = re.sub(r'href="[^"]*"', f'href="{links[lang]}"', tag)
                return tag[:-1] + ' target="_self">'

            text = re.sub(
                # Zensical treats head alternates as roots for sitemap discovery;
                # pair only the visible links, with explicit native navigation.
                r'<a\b[^>]*hreflang="(?:en|zh)"[^>]*>',
                alternate,
                output.read_text(encoding="utf-8"),
            )
            if destination.parts[0] == "zh":
                text = re.sub(
                    r'(<div class="md-copyright">\s*)Made with\b',
                    r"\1构建工具：",
                    text,
                )
            output.write_text(text, encoding="utf-8")
    (SITE / "assets/language-map.js").write_text(
        "window.METIS_DOCUMENTATION_LANGUAGES="
        + json.dumps(language_map, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    (SITE / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=zh/"><title>Metis</title>'
        '<a href="zh/" lang="zh-CN">简体中文</a> · <a href="en/" lang="en">English</a></html>',
        encoding="utf-8",
    )
    print(f"Documentation built: {SITE}")


if __name__ == "__main__":
    main()
