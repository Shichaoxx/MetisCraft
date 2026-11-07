"""Audit public pages, search entries and distribution contents using stdlib only."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


def route(name):
    path = Path(name)
    if path.parts[0] == "docs":
        path = Path(*path.parts[1:])
        if path.parts[0] not in {"en", "zh"}:
            path = Path("zh") / path
    elif path.parts[0] == "recipes":
        path = Path("zh") / path
    else:
        path = Path("en") / ("contributing.md" if name == "CONTRIBUTING.md" else "changelog.md")
    return (
        path.parent / "index.html"
        if path.stem in {"README", "index"}
        else path.with_suffix("") / "index.html"
    ).as_posix()


def workflow(path):
    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- workflow:start -->") == text.count("<!-- workflow:end -->") == 1, path
    text = text.split("<!-- workflow:start -->")[1].split("<!-- workflow:end -->")[0]
    blocks = re.findall(r"```bash\s*\n(.*?)\n```", text, re.S)
    return [
        " ".join(x.split())
        for block in blocks
        for x in block.replace("\\\n", " ").splitlines()
        if x.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site", type=Path)
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    policy = json.loads((root / "docs/public-pages.json").read_text(encoding="utf-8"))
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    def blocked(name):
        path = Path(name)
        return "checks" in path.parts or any(
            path == Path(p) or Path(p) in path.parents for p in policy["development_only"]
        )

    pages = policy["pages"]
    check(len(pages) == len(set(pages)), "public page policy has duplicate sources")
    check(not [p for p in pages if blocked(p)], "public policy includes developer material")
    expected = {route(p) for p in pages} | {"index.html", "en/404.html", "zh/404.html"}
    redirects = {
        route(alias) for alias, target in policy["aliases"].items() if route(alias) != route(target)
    }
    expected |= redirects
    check(len({route(p) for p in pages}) == len(pages), "public page routes collide")
    template_root = root / "src/metiscraft/templates/quickstart"
    templates = {
        lang: template_root / ("README_zh.md" if lang == "zh" else "README.md")
        for lang in ("en", "zh")
    }
    commands = {lang: workflow(path) for lang, path in templates.items()}
    check(commands["en"] == commands["zh"], "packaged EN/ZH workflow commands differ")
    for lang in ("en", "zh"):
        source = root / f"docs/{lang}/cookbook/first-model.md"
        check(source.relative_to(root).as_posix() in pages, f"{lang} first-model is not public")
        check(
            source.read_text(encoding="utf-8").count("<!-- metiscraft:quickstart -->") == 1,
            f"{lang} tutorial does not include exactly one canonical workflow",
        )

    if args.site:
        site = args.site.resolve()
        # Standalone result reports are public only if linked by an approved source.
        for name in pages:
            text = (root / name).read_text(encoding="utf-8")
            for link in re.findall(r'(?:\]\(|(?:src|href)=["\x27])([^"\x27)\s]+)', text):
                url = urlsplit(link)
                if url.scheme or url.netloc or not url.path.endswith(".html"):
                    continue
                target = (root / name).parent.joinpath(unquote(url.path)).resolve()
                if target.is_relative_to(root):
                    rel = target.relative_to(root).as_posix()
                    check(not blocked(rel), f"public page links developer HTML: {name}: {rel}")
                    expected.add("assets/" + rel)
        actual = {p.relative_to(site).as_posix() for p in site.rglob("*.html")}
        check(
            actual == expected,
            f"HTML missing={sorted(expected - actual)}; extra={sorted(actual - expected)}",
        )
        internal_fields = re.compile(
            r"id-token\s*:|workflow_dispatch\s*:|PYPI_TOKEN|(?:Pending|Trusted) Publisher", re.I
        )
        for name in actual:
            body = unescape(re.sub("<[^>]+>", "", (site / name).read_text(encoding="utf-8")))
            check(not internal_fields.search(body), f"developer publishing fields in HTML: {name}")
        prefix = "/MetisCraft/"
        for lang in ("en", "zh"):
            choices = [site / lang / "search.json", site / lang / "search/search_index.json"]
            indexes = [p for p in choices if p.is_file()]
            check(bool(indexes), f"{lang} search payload missing")
            indexed = set()
            for index in indexes:
                payload = json.loads(index.read_text(encoding="utf-8"))
                entries = payload.get("items", payload.get("docs", []))
                check(bool(entries), f"empty search payload: {index}")
                for entry in entries:
                    location = entry.get("location")
                    if not isinstance(location, str):
                        check(False, f"search location missing: {entry.get('title')}")
                        continue
                    url = urlsplit(urljoin(f"https://audit.invalid{prefix}{lang}/", location))
                    rel = unquote(url.path.removeprefix(prefix))
                    rel = rel if rel.endswith(".html") else rel.rstrip("/") + "/index.html"
                    check(
                        url.netloc == "audit.invalid"
                        and url.path.startswith(prefix)
                        and rel in actual
                        and rel in expected,
                        f"unapproved search target: {location}",
                    )
                    indexed.add(rel)
                    body = unescape(re.sub("<[^>]+>", "", json.dumps(entry, ensure_ascii=False)))
                    check(
                        not internal_fields.search(body),
                        f"developer publishing fields in search: {location}",
                    )
            required = {
                p
                for p in expected
                if p.startswith(lang + "/") and not p.endswith("404.html") and p not in redirects
            }
            check(required <= indexed, f"{lang} search missing={sorted(required - indexed)}")
            guide = site / lang / "cookbook/first-model/index.html"
            if guide.is_file():
                body = re.sub("<[^>]+>", "", guide.read_text(encoding="utf-8"))
                plain = " ".join(unescape(body).replace("\\\n", " ").split())
                for command in commands[lang]:
                    check(command in plain, f"{lang} rendered tutorial omits command: {command}")

    if args.dist:
        wheels, sdists = sorted(args.dist.glob("*.whl")), sorted(args.dist.glob("*.tar.gz"))
        check(len(wheels) == len(sdists) == 1, "expected exactly one wheel and one sdist")
        required = {
            "LICENSE",
            "CONTRIBUTING.md",
            "README.md",
            "README_zh.md",
            "pyproject.toml",
            "hatch_build.py",
            "docs/public-pages.json",
        }
        for pattern in (
            "recipes/**/*.yaml",
            "recipes/**/README.md",
            "docs/examples/*.json",
            "docs/visuals/src/*",
            "docs/cookbook/src/*",
            "docs/reports/*/src/**/*",
            "docs/reports/*/index.html",
            "docs/reports/*/figures/*",
        ):
            required.update(
                p.relative_to(root).as_posix()
                for p in root.glob(pattern)
                if p.is_file()
                and not blocked(p.relative_to(root))
                and "__pycache__" not in p.parts
                and p.suffix != ".pyc"
            )
        for archive in wheels + sdists:
            if archive.suffix == ".whl":
                with zipfile.ZipFile(archive) as package:
                    members = {
                        x: package.read(x) for x in package.namelist() if not x.endswith("/")
                    }
                for lang, source in templates.items():
                    name = "metiscraft/templates/quickstart/" + source.name
                    check(
                        members.get(name) == source.read_bytes(),
                        f"{archive.name}: missing/changed {lang} template",
                    )
            else:
                with tarfile.open(archive) as package:
                    files = [x for x in package.getmembers() if x.isfile()]
                    tops = {Path(x.name).parts[0] for x in files}
                    check(len(tops) == 1, f"{archive.name}: ambiguous archive root")
                    members = {
                        Path(*Path(x.name).parts[1:]).as_posix(): package.extractfile(x).read()
                        for x in files
                    }
                required.update(p.relative_to(root).as_posix() for p in templates.values())
                for name in sorted(required):
                    check(
                        members.get(name) == (root / name).read_bytes(),
                        f"{archive.name}: missing/changed public source {name}",
                    )
            bad = [name for name in members if blocked(name)]
            check(not bad, f"{archive.name}: developer material present: {bad}")
            unsafe = [n for n in members if Path(n).is_absolute() or ".." in Path(n).parts]
            check(not unsafe, f"{archive.name}: unsafe member path")
    for error in errors:
        print(error)
    print(f"Public surface audit: {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
