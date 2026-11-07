"""Use the canonical README on package indexes, resolving repository-local links."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from hatchling.metadata.plugin.interface import MetadataHookInterface

BASE = "https://github.com/Shichaoxx/MetisCraft"


def index_readme(text, ref):
    def resolve(value, image=False):
        url = urlsplit(value)
        if url.scheme or url.netloc or value.startswith("#"):
            return value
        if image:
            return f"https://raw.githubusercontent.com/Shichaoxx/MetisCraft/{ref}/{value}"
        return f"{BASE}/blob/{ref}/{value}"

    # PyPI does not render <picture>; use the light banner's standard image.
    text = re.sub(r"<picture>.*?(<img [^>]+>).*?</picture>", r"\1", text, flags=re.S)
    text = re.sub(
        r"(!?\[[^\]\n]*\]\()([^\s)]+)(\))",
        lambda m: m[1] + resolve(m[2], m[1].startswith("!")) + m[3],
        text,
    )
    return re.sub(
        r'((?:src|href)=")([^"]+)(")',
        lambda m: m[1] + resolve(m[2], m[1].startswith("src")) + m[3],
        text,
    )


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        # A published distribution belongs to its immutable release tag.
        ref = f"v{metadata['version']}"
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": index_readme(Path(self.root, "README.md").read_text(encoding="utf-8"), ref),
        }
