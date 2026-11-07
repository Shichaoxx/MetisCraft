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
REPO = "https://github.com/Shichaoxx/MetisCraft/blob/main/"


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
    mapping = {}
    sources = [ROOT / name for name in policy["pages"]]
    for source in sources:
        if not source.is_file():
            raise ValueError(f"Missing public page: {source.relative_to(ROOT)}")
        relative = source.relative_to(ROOT)
        if relative.parts[0] == "docs":
            relative = source.relative_to(ROOT / "docs")
            dest = relative if relative.parts[0] in {"en", "zh"} else Path("zh") / relative
        elif relative.parts[0] == "recipes":
            dest = Path("zh") / relative
        else:
            dest = Path("en") / (
                "contributing.md" if source.name == "CONTRIBUTING.md" else "changelog.md"
            )
        if dest.name == "README.md":
            dest = dest.with_name("index.md")
        mapping[source.resolve()] = dest
    for alias, target in policy["aliases"].items():
        mapping[(ROOT / alias).resolve()] = mapping[(ROOT / target).resolve()]

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
            {"数据与配方" if zh else "Data & recipes": "data.md"},
            {"训练与校准" if zh else "Train & calibrate": "training.md"},
            {"预测与服务" if zh else "Predict & serve": "serving.md"},
            {
                "结果与复现" if zh else "Results & reproductions": [
                    {"验证范围" if zh else "Evidence & limitations": "limitations.md"},
                    {"Tev1（中文）": PREFIX + "zh/reproductions/tev1/"},
                    {"支持矩阵（中文）": PREFIX + "zh/support-matrix/"},
                ]
            },
            {
                "参考" if zh else "Reference": [
                    {"原理与最佳实践（中文）": PREFIX + "zh/cookbook/principles/"},
                    {"接口约定（中文）": PREFIX + "zh/reference/"},
                    {"配方字段（中文）": PREFIX + "zh/cookbook/recipe-reference/"},
                    {"读出与张量图（中文）": PREFIX + "zh/readouts/"},
                ]
            },
            {
                "参与贡献" if zh else "Contribute": [
                    {"Contribution guide": PREFIX + "en/contributing/"},
                    {"Changelog": PREFIX + "en/changelog/"},
                ]
            },
        ]
        project = {
            "site_name": "MetisCraft",
            "site_url": f"https://shichaoxx.github.io/MetisCraft/{lang}/",
            "site_description": "Train. Calibrate. Decide. A toolkit for typed decision models.",
            "docs_dir": f"sources/{lang}",
            "site_dir": f"site/{lang}",
            "repo_url": "https://github.com/Shichaoxx/MetisCraft",
            "repo_name": "MetisCraft",
            "nav": nav,
            "extra_css": [PREFIX + "assets/site.css"],
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
toc.slugify = { kwds = { case = "lower" } }
pymdownx.highlight = {}
pymdownx.superfences = {}
pymdownx.details = {}
pymdownx.tasklist.custom_checkbox = true
""".replace('language = "LANG"', f'language = "{lang}"')
        path = BUILD / f"zensical-{lang}.toml"
        path.write_text(config, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "zensical", "build", "-f", str(path), "--strict"],
            cwd=ROOT,
            check=True,
        )
    # Old tutorial URLs are redirects, not a second maintained tutorial or search page.
    for alias, target in policy["aliases"].items():
        old = Path(alias).relative_to("docs")
        if old.parts[0] not in {"en", "zh"}:
            old = Path("zh") / old
        if old.name == "README.md":
            old = old.with_name("index.md")
        destination = page_url(mapping[(ROOT / target).resolve()])
        if page_url(old) == destination:
            continue
        output = SITE / old.with_suffix("") / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f'<!doctype html><html lang="zh"><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url={destination}">'
            f'<link rel="canonical" href="{destination}"><title>Cookbook</title>'
            f'<a href="{destination}">Continue to the Cookbook</a></html>',
            encoding="utf-8",
        )
    assets = SITE / "assets" / "docs" / "visuals" / "figures"
    shutil.copytree(ROOT / "docs/visuals/figures", assets, dirs_exist_ok=True)
    shutil.copyfile(ROOT / "docs/site.css", SITE / "assets/site.css")
    (SITE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=en/"><title>MetisCraft</title>'
        '<a href="en/">English</a> · <a href="zh/">简体中文</a></html>',
        encoding="utf-8",
    )
    print(f"Documentation built: {SITE}")


if __name__ == "__main__":
    main()
