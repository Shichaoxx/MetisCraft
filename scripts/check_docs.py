"""Check local rendered links, images and anchors under the Pages subpath."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "build/documentation/site"
ORIGIN = "https://shichaoxx.github.io"
PREFIX = "/MetisCraft/"


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for key in ("src", "href"):
            if attrs.get(key):
                self.links.append(attrs[key])


def main():
    pages = {p: Links(p.read_text(encoding="utf-8")) for p in SITE.rglob("*.html")}
    errors = []
    count = 0
    for source, page in pages.items():
        current = ORIGIN + PREFIX + source.relative_to(SITE).as_posix()
        for link in page.links:
            url = urlsplit(urljoin(current, link))
            if url.netloc != urlsplit(ORIGIN).netloc or not url.path.startswith(PREFIX):
                continue
            target = SITE / unquote(url.path.removeprefix(PREFIX))
            if target.is_dir():
                target /= "index.html"
            count += 1
            if not target.is_file():
                errors.append(f"{source.relative_to(SITE)}: missing {link}")
            elif (
                url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{source.relative_to(SITE)}: missing anchor {link}")
    for source in SITE.rglob("*.css"):
        current = ORIGIN + PREFIX + source.relative_to(SITE).as_posix()
        for link in re.findall(r"url\(['\"]?([^)'\"\s]+)", source.read_text(encoding="utf-8")):
            url = urlsplit(urljoin(current, link))
            if url.netloc != urlsplit(ORIGIN).netloc or not url.path.startswith(PREFIX):
                continue
            count += 1
            target = SITE / unquote(url.path.removeprefix(PREFIX))
            if not target.is_file():
                errors.append(f"{source.relative_to(SITE)}: missing CSS resource {link}")
    language_asset = SITE / "assets/language-map.js"
    if language_asset.is_file():
        text = language_asset.read_text(encoding="utf-8")
        mapping = json.loads(text.split("=", 1)[1].rstrip(";\n"))
        for current, pair in mapping.items():
            for lang, link in pair.items():
                target = SITE / link.removeprefix(PREFIX) / "index.html"
                if not target.is_file() or not link.startswith(PREFIX + lang + "/"):
                    errors.append(f"Invalid language partner: {current} → {link}")
                elif mapping.get(link) != pair:
                    errors.append(f"Asymmetric language partner: {current} → {link}")
    for error in sorted(set(errors)):
        print(error)
    print(f"Checked {len(pages)} HTML pages and {count} local references; {len(errors)} errors")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
