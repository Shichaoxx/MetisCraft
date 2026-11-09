"""Check the published examples against the installed template and recorded response."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_blocks(text: str, kind: str) -> list[str]:
    return re.findall(r"```" + kind + r"\n(.*?)\n```", text, re.S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="refresh generated tutorial command regions"
    )
    args = parser.parse_args()
    template = ROOT / "src/metiscraft/templates/quickstart"
    english = (template / "README.md").read_text()
    chinese = (template / "README_zh.md").read_text()
    if code_blocks(english, "bash") != code_blocks(chinese, "bash"):
        raise SystemExit("Quickstart EN/ZH commands differ")
    for content in (english, chinese):
        for marker in ("<!-- workflow:start -->", "<!-- workflow:end -->"):
            if content.count(marker) != 1:
                raise SystemExit(f"Expected one {marker}")
    for name in ("decision", "own-task"):
        pages = [ROOT / f"docs/{lang}/cookbook/{name}.md" for lang in ("en", "zh")]
        for kind in ("bash", "python", "yaml", "json"):
            if code_blocks(pages[0].read_text(), kind) != code_blocks(pages[1].read_text(), kind):
                raise SystemExit(f"Cookbook {name}: {kind} differs between languages")
    for lang in ("en", "zh"):
        page = ROOT / f"docs/{lang}/cookbook/first-model.md"
        first = page.read_text()
        source = chinese if lang == "zh" else english
        workflow = (
            source.split("<!-- workflow:start -->", 1)[1]
            .split("<!-- workflow:end -->", 1)[0]
            .strip()
        )
        begin, end = "<!-- metiscraft:quickstart -->", "<!-- metiscraft:quickstart:end -->"
        generated = begin + "\n\n" + workflow + "\n\n" + end
        if args.write:
            if end in first:
                first = re.sub(
                    re.escape(begin) + r".*?" + re.escape(end),
                    lambda _, generated=generated: generated,
                    first,
                    flags=re.S,
                )
            else:
                first = first.replace(begin, generated)
            page.write_text(first)
        if first.count(begin) != 1 or first.count(end) != 1 or generated not in first:
            raise SystemExit(f"{lang} tutorial commands drifted; run check_examples.py --write")
    recorded = code_blocks((ROOT / "recipes/banking77_support/README.md").read_text(), "json")
    expected = [json.loads(block) for block in recorded[:2]]
    for index, name in enumerate(("support-request", "support-response")):
        if json.loads((ROOT / f"docs/examples/{name}.json").read_text()) != expected[index]:
            raise SystemExit(f"Recorded {name} differs from recipe")
    for lang in ("en", "zh"):
        blocks = code_blocks((ROOT / f"docs/{lang}/cookbook/decision.md").read_text(), "json")
        if [json.loads(block) for block in blocks[:2]] != expected:
            raise SystemExit(f"Recorded Cookbook JSON differs: {lang}")
    for name in ("README.md", "README_zh.md"):
        excerpt = json.loads(code_blocks((ROOT / name).read_text(), "json")[0])
        for key, answer in excerpt.items():
            for field, value in answer.items():
                if expected[1]["answers"][key][field] != value:
                    raise SystemExit(f"README response excerpt differs: {key}.{field}")
    print("Bilingual commands, template markers and recorded response examples match")


if __name__ == "__main__":
    main()
