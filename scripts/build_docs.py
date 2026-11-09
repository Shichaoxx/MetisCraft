"""Build the English/Chinese site from canonical repository Markdown, without copies to maintain."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "documentation"
SITE = BUILD / "site"
PREFIX = "/MetisCraft/"
SOURCE_REF = "6b9701da7814111b13d8882f48288f4df46dc983"
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
                output = SITE / "assets" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, output)
                return PREFIX + "assets/" + quote(relative.as_posix()) + suffix
        return REPO + quote(relative.as_posix()) + suffix

    for source in sources:
        dest = mapping[source.resolve()]
        content = source.read_text(encoding="utf-8")
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
        nav = [
            {
                "开始使用" if zh else "Get started": [
                    {"概览" if zh else "Overview": "index.md"},
                    {"安装" if zh else "Installation": "installation.md"},
                    {"快速上手" if zh else "Quickstart": "quickstart.md"},
                ]
            },
            {
                "Cookbook": [
                    {"阅读路线" if zh else "Choose a path": "cookbook/index.md"},
                    {"认识一次判断" if zh else "Understand a decision": "cookbook/decision.md"},
                    {"跑通第一个模型" if zh else "Your first model": "cookbook/first-model.md"},
                    {"接入自己的任务" if zh else "Your own task": "cookbook/own-task.md"},
                ]
            },
            {
                "原理与张量解析" if zh else "Concepts & Tensors": [
                    {
                        "从 System One 到 MetisCraft"
                        if zh
                        else "System One to MetisCraft": "tensor-analysis/index.md"
                    },
                    {
                        "张量、轴与索引"
                        if zh
                        else "Tensors, axes & indices": "tensor-analysis/notation.md"
                    },
                    {
                        "答案 token 读出"
                        if zh
                        else "Answer-token readout": "tensor-analysis/answer-token.md"
                    },
                    {
                        "标量头读出"
                        if zh
                        else "Scalar-head readouts": "tensor-analysis/scalar-heads.md"
                    },
                    {
                        "训练、校准与输出" if zh else "Training, calibration & output": (
                            "tensor-analysis/training-calibration.md"
                        )
                    },
                    {
                        "七类读出对照"
                        if zh
                        else "Seven readout mechanisms": "tensor-analysis/readout-comparison.md"
                    },
                    {
                        "五分钟技术导览"
                        if zh
                        else "Five-minute walkthrough": "tensor-analysis/walkthrough.md"
                    },
                ]
            },
            {"数据与配方" if zh else "Data & recipes": "data.md"},
            {"训练与校准" if zh else "Train & calibrate": "training.md"},
            {"预测与服务" if zh else "Predict & serve": "serving.md"},
            {
                "结果与复现" if zh else "Results & reproductions": [
                    {"验证范围" if zh else "Evidence & limitations": "limitations.md"},
                    {"Tev1": "reproductions/tev1.md"},
                    {"支持矩阵" if zh else "Ecosystem support": "support-matrix.md"},
                ]
            },
            {
                "参考" if zh else "Reference": [
                    {"原理" if zh else "Principles": "cookbook/principles.md"},
                    {"最佳实践" if zh else "Best practices": "cookbook/best-practices.md"},
                    {"接口约定" if zh else "Interface contracts": "interfaces.md"},
                    {"配方字段" if zh else "Recipe fields": "cookbook/recipe-reference.md"},
                ]
            },
            {
                "参与贡献" if zh else "Contribute": [
                    {"贡献指南" if zh else "Contribution guide": "contributing.md"},
                    {"版本说明" if zh else "Changelog": "changelog.md"},
                ]
            },
        ]
        project = {
            "site_name": "MetisCraft",
            "site_url": f"https://shichaoxx.github.io/MetisCraft/{lang}/",
            "site_description": (
                "JevLike 模型工具包：训练、复现、校准与服务。"
                if zh
                else "A JevLike model toolkit. Train, reproduce, calibrate and serve."
            ),
            "docs_dir": f"sources/{lang}",
            "site_dir": f"site/{lang}",
            "repo_url": "https://github.com/Shichaoxx/MetisCraft",
            "repo_name": "MetisCraft",
            "nav": nav,
            "extra_css": [
                PREFIX + "assets/site.css",
                PREFIX + "assets/documentation.css",
                PREFIX + "assets/katex/katex.min.css",
            ],
            "extra_javascript": [
                PREFIX + "assets/katex/katex.min.js",
                PREFIX + "assets/math.js",
                PREFIX + "assets/language-map.js",
                PREFIX + "assets/documentation.js",
            ],
        }
        config = "[project]\n" + "\n".join(f"{k} = {toml(v)}" for k, v in project.items())
        config += """
[project.theme]
language = "LANG"
font = false
logo = "/MetisCraft/assets/docs/visuals/figures/mark.svg"
favicon = "/MetisCraft/assets/docs/visuals/figures/mark.svg"
features = ["content.code.copy", "navigation.sections", "navigation.top", "search.highlight"]
[[project.theme.palette]]
media = "(prefers-color-scheme: light)"
scheme = "default"
primary = "teal"
accent = "amber"
toggle.icon = "lucide/moon"
toggle.name = "Dark mode"
[[project.theme.palette]]
media = "(prefers-color-scheme: dark)"
scheme = "slate"
primary = "teal"
accent = "amber"
toggle.icon = "lucide/sun"
toggle.name = "Light mode"
[project.extra]
alternate = [
 { name = "English", link = "/MetisCraft/en/", lang = "en" },
 { name = "简体中文", link = "/MetisCraft/zh/", lang = "zh" }
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
""".replace('language = "LANG"', f'language = "{lang}"')
        if zh:
            config = (
                config.replace('toggle.name = "Dark mode"', 'toggle.name = "深色主题"')
                .replace('toggle.name = "Light mode"', 'toggle.name = "浅色主题"')
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
            output.write_text(text, encoding="utf-8")
    (SITE / "assets/language-map.js").write_text(
        "window.METIS_DOCUMENTATION_LANGUAGES="
        + json.dumps(language_map, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    (SITE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=en/"><title>MetisCraft</title>'
        '<a href="en/">English</a> · <a href="zh/">简体中文</a></html>',
        encoding="utf-8",
    )
    print(f"Documentation built: {SITE}")


if __name__ == "__main__":
    main()
