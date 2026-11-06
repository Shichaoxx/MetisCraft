"""Content identities for verified local base weights and tokenizer files.

The checksum manifest is verified afresh, not accepted as a fingerprint of the files.
An identity contains relative names and byte hashes, so moving a model directory does
not change it. Documentation and download bookkeeping are not part of the identity;
all files listed in SHA256SUMS are nevertheless checked. SOURCE.json is included even
when older acquisition tools wrote it after making SHA256SUMS.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from metiscraft.data.store import IntegrityError, sha256_file

SCHEMA = "metiscraft-base/1"
_SUMS = "SHA256SUMS"
_SOURCE = "SOURCE.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_LINE = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)\Z")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack")
_TOKENIZER_PAYLOADS = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tekken.json",
}


def _relative_name(value: Any) -> str:
    """Accept only unambiguous relative POSIX paths, on every host platform."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise IntegrityError(f"unsafe relative file path: {value!r}")
    return value


def _is_weight(name: str) -> bool:
    return name.endswith(_WEIGHT_SUFFIXES)


def _is_model_file(relative: str) -> bool:
    if relative.startswith(".cache/huggingface/download/"):
        return False
    name = PurePosixPath(relative).name
    return (
        _is_weight(name)
        or name.endswith((".json", ".model", ".tiktoken", ".jinja", ".j2", ".py"))
        or name.startswith(
            (
                "config",
                "generation_config",
                "tokenizer",
                "vocab",
                "merges",
                "special_tokens",
                "added_tokens",
            )
        )
        or name.startswith("chat_template")
        or "chat_templates" in PurePosixPath(relative).parts
    )


def _require_components(files: dict[str, str]) -> None:
    for required in (_SOURCE, "config.json"):
        if required not in files:
            raise IntegrityError(f"base identity is missing {required}")
    if not any(_is_weight(name) for name in files):
        raise IntegrityError("base identity is missing model weights")
    names = {PurePosixPath(name).name for name in files}
    if not any(
        name in _TOKENIZER_PAYLOADS
        or name.startswith("vocab")
        or name.endswith((".model", ".tiktoken"))
        or (name.startswith("tokenizer") and name.endswith(".vocab"))
        for name in names
    ):
        raise IntegrityError("base identity is missing tokenizer vocabulary or model")


def _inventory(root: Path) -> dict[str, Path]:
    # Reject links in ancestors too: resolving first would erase this evidence.
    for part in (root, *root.parents):
        if part.is_symlink():
            raise IntegrityError(f"symlinks are not allowed in a base path: {part}")
    if not root.is_dir():
        raise IntegrityError(f"{root}: base must be an existing local directory")

    def walk_error(error: OSError) -> None:
        raise IntegrityError(f"cannot inspect base directory: {error}") from error

    files = {}
    for directory, directories, filenames in os.walk(root, onerror=walk_error, followlinks=False):
        for name in directories + filenames:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise IntegrityError(f"symlinks are not allowed in a base directory: {path}")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise IntegrityError(f"base files must be regular files: {path}")
            relative = _relative_name(path.relative_to(root).as_posix())
            files[relative] = path
    return files


def _manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise IntegrityError(f"bad entry in {_SUMS}: {line!r}")
        digest, name = match.groups()
        name = _relative_name(name)
        if name == _SUMS:
            raise IntegrityError(f"{_SUMS} cannot list itself")
        if name in entries:
            raise IntegrityError(f"duplicate entry in {_SUMS}: {name}")
        entries[name] = digest.lower()
    if not entries:
        raise IntegrityError(f"{_SUMS} is empty")
    return entries


def _identity(files: dict[str, str]) -> dict[str, Any]:
    payload = {"schema": SCHEMA, "files": dict(sorted(files.items()))}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _check_indices(files: dict[str, Path]) -> None:
    for name, path in files.items():
        if not name.endswith((".safetensors.index.json", ".bin.index.json")):
            continue
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise IntegrityError(f"{name}: expected a nonempty weight_map")
        for shard in weight_map.values():
            shard = _relative_name(shard)
            if not _is_weight(shard) or shard not in files:
                raise IntegrityError(f"{name}: missing or invalid weight shard {shard!r}")


def capture_identity(path: str | Path) -> dict[str, Any]:
    """Verify local files against SHA256SUMS and return their portable content identity.

    Every actual weight, JSON configuration, tokenizer and chat-template file must be
    listed. All listed files must exist and match, including ancillary files. A newly
    added model file therefore cannot hide behind an otherwise unchanged manifest.
    This verifies local integrity, not the trustworthiness of the original publisher.
    """
    root = Path(path).expanduser().absolute()
    try:
        actual = _inventory(root)
        for required in (_SOURCE, _SUMS):
            if required not in actual:
                raise IntegrityError(f"{root}: missing {required}")
        source_bytes = actual[_SOURCE].read_bytes()
        if not isinstance(json.loads(source_bytes), dict):
            raise IntegrityError(f"{_SOURCE} must contain a JSON object")
        listed = _manifest(actual[_SUMS])
        selected = {name for name in actual if _is_model_file(name)} | {_SOURCE}
        missing = sorted(set(listed) - set(actual))
        unlisted = sorted(selected - set(listed) - {_SOURCE})
        if missing or unlisted:
            problems = [f"missing {name}" for name in missing]
            problems.extend(f"unlisted model file {name}" for name in unlisted)
            raise IntegrityError(f"{root}: " + "; ".join(problems))
        digests = {}
        for name, expected in listed.items():
            digest = sha256_file(actual[name])
            if digest != expected:
                raise IntegrityError(f"{root}: changed {name}")
            digests[name] = digest
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        if _SOURCE in digests and digests[_SOURCE] != source_digest:
            raise IntegrityError(f"{_SOURCE} changed during identity capture")
        digests[_SOURCE] = source_digest
        identity_files = {name: digests[name] for name in selected}
        _require_components(identity_files)
        _check_indices(actual)
        return _identity(identity_files)
    except (OSError, UnicodeError, ValueError) as error:
        raise IntegrityError(f"{root}: cannot verify base files: {error}") from error


def verify_identity(path: str | Path, expected: dict[str, Any]) -> None:
    """Require a valid recorded identity and an identical, freshly verified local base.

    Legacy-artifact opt-in belongs to the artifact loader. Missing or malformed identity
    metadata is never promoted to a verified identity by this function.
    """
    if not isinstance(expected, dict) or set(expected) != {"schema", "files", "sha256"}:
        raise IntegrityError("missing or malformed base identity")
    if expected["schema"] != SCHEMA:
        raise IntegrityError(f"unsupported base identity schema: {expected['schema']!r}")
    files = expected["files"]
    if not isinstance(files, dict) or not files:
        raise IntegrityError("base identity files must be a nonempty mapping")
    for name, digest in files.items():
        _relative_name(name)
        if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
            raise IntegrityError(f"invalid recorded SHA256 for {name}")
    _require_components(files)
    if expected["sha256"] != _identity(files)["sha256"]:
        raise IntegrityError("base identity summary does not match its recorded files")
    current = capture_identity(path)
    if current["files"] != files:
        problems = [f"missing {name}" for name in sorted(set(files) - set(current["files"]))]
        problems.extend(f"added {name}" for name in sorted(set(current["files"]) - set(files)))
        problems.extend(
            f"changed {name}"
            for name in sorted(set(files) & set(current["files"]))
            if files[name] != current["files"][name]
        )
        raise IntegrityError("base identity mismatch: " + "; ".join(problems))
