"""Generate bilingual TikZ, true vector PDF/SVG and a reproducible figure manifest.

Rebuild with Python 3.10+, Tectonic 0.17.0 and Poppler (pdftocairo/pdfimages)::

    python scripts/build_tensor_figures.py --tectonic /path/to/tectonic --qa-dir /tmp/tensor-qa

Each exported TeX file contains its own preamble, labels and geometry. It can be
compiled outside the checkout without the Python generator or shared source files.
Use --check for a read-only drift/vector check of the checked-in exports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/tensor-analysis/src"
OUTPUT = ROOT / "docs/tensor-analysis/figures"
TECTONIC_ARCHIVE_SHA256 = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"
TECTONIC_BUNDLE_URL = "https://relay.fullyjustified.net/default_bundle_v33.tar"
TECTONIC_BUNDLE_DIGEST = "6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_shapes(specs, module, geometry):
    """Keep every displayed axis edge tied to the shared shape ledger."""
    for figure in specs.values():
        for _, items in figure.rows:
            for item in items:
                if not isinstance(item, module.Tensor):
                    continue
                dims = item.shape.strip("[]").split(",")
                sizes = [int(d) if d.isdigit() else geometry["axes"][d] for d in dims]
                expected = (sizes[-1], 1) if len(sizes) == 1 else tuple(sizes[-2:])
                if (item.rows, item.cols) != expected:
                    raise SystemExit(
                        f"Axis geometry drift in {figure.slug}: {item.symbol} {item.shape}"
                    )


def load_specs():
    spec = importlib.util.spec_from_file_location("tensor_figure_specs", SOURCE / "figures.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plate_spec = importlib.util.spec_from_file_location("model_plates", SOURCE / "model_plates.py")
    plates = importlib.util.module_from_spec(plate_spec)
    sys.modules[plate_spec.name] = plates
    plate_spec.loader.exec_module(plates)
    module.model_plates = plates
    figures = module.figures()
    for slug in plates.available():
        figures[slug] = module.Figure(slug, "", "", "")
    figures["metiscraft-workflow"] = module.Figure("metiscraft-workflow", "", "", "")
    return module, figures


def document(figure, lang, module, labels, geometry, theme="light"):
    def label(key):
        return labels[key][lang]

    unit, offset, gutter = (geometry[k] for k in ("cell_cm", "sheet_offset_cm", "gutter_cm"))
    preamble = (SOURCE / "tensor-style.tex").read_text()
    if theme == "dark":
        palette = json.loads((SOURCE / "themes.json").read_text())["dark"]
        # PGF expands implicit tints against the original white. Name the canvas
        # explicitly so low-intensity panels stay dark and their text readable.
        preamble = preamble.replace("fill=white", "fill=Canvas")
        preamble = re.sub(
            r"(Ink|Msk|#6)!(\d+|\\level)",
            lambda m: m[0] + "!Canvas",
            preamble,
        )
        preamble += (
            "\n"
            + "\n".join(
                rf"\definecolor{{{name}}}{{HTML}}{{{value}}}" for name, value in palette.items()
            )
            + "\n"
            + r"\pagecolor{Canvas}"
            + "\n"
        )
    lines = [
        preamble,
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm,text=Ink]",
    ]
    if figure.slug in module.model_plates.available():
        body, bounds = module.model_plates.render(figure.slug, lang)
        module.model_plates.validate_ledger(bounds)
        return preamble + "\n" + r"\providecolor{Canvas}{HTML}{FFFFFF}" + "\n" + body, bounds
    if figure.slug == "metiscraft-workflow":
        body = (SOURCE / "architecture.tex").read_text()
        body = re.sub(r"@@([\w.]+)@@", lambda m: label(m[1]), body)
        return (
            preamble + "\n" + r"\providecolor{Canvas}{HTML}{FFFFFF}" + "\n" + body,
            [],
        )
    widths = []
    for _, items in figure.rows:
        widths.append(
            sum(
                x.cols * unit + x.depth * offset if isinstance(x, module.Tensor) else x.width
                for x in items
            )
            + gutter * (len(items) - 1)
        )
    canvas = max(21.0, max(widths) + 0.2)
    lines.append(
        rf"\node[font=\normalsize,align=center] at ({canvas / 2:.3f},4.0) "
        rf"{{$\displaystyle {figure.formula}$}};"
    )
    bounds = []
    for row_index, (title, items) in enumerate(figure.rows):
        y = -6.3 * row_index
        x = (canvas - widths[row_index]) / 2
        lines.append(rf"\node[stage] at (0,{y + 2.8:.3f}) {{{label(title)}}};")
        previous = None
        for i, item in enumerate(items):
            name = f"r{row_index}n{i}"
            if isinstance(item, module.Tensor):
                width = item.cols * unit + item.depth * offset
                center = x + item.cols * unit / 2
                support = {None: 0, "padding": 1, "candidate": 2, "causal": 3}[item.support]
                lines.append(rf"\renewcommand{{\BinarySupport}}{{{support}}}")
                lines.append(
                    rf"\Tensor{{{name}}}{{{center:.3f}}}{{{y:.3f}}}{{{item.rows}}}{{{item.cols}}}{{{item.role}}}{{{item.depth}}}{{{item.symbol}}}{{{item.shape}}}"
                )
                bottom = y - item.rows * unit / 2 - 0.85
                top = y + item.rows * unit / 2 + item.depth * offset
            elif isinstance(item, module.Operator):
                width = item.width
                center = x + width / 2
                lines.append(
                    rf"\node[font=\Large] ({name}) at ({center:.3f},{y:.3f}) {{$ {item.symbol} $}};"
                )
                lines.append(
                    rf"\coordinate ({name}-w) at ({x:.3f},{y:.3f});"
                    rf"\coordinate ({name}-e) at ({x + width:.3f},{y:.3f});"
                )
                bottom = y - item.height / 2
                top = y + item.height / 2
            else:
                width = item.width
                text = item.label if item.literal else label(item.label)
                center = x + width / 2
                lines.append(
                    rf"\node[operation,minimum width={width:.3f}cm,"
                    rf"minimum height={item.height:.3f}cm,text width={width - 0.35:.3f}cm] "
                    rf"({name}) at ({center:.3f},{y:.3f}) {{{text}}};"
                )
                lines.append(
                    rf"\coordinate ({name}-w) at ({x:.3f},{y:.3f});"
                    rf"\coordinate ({name}-e) at ({x + width:.3f},{y:.3f});"
                )
                bottom = y - item.height / 2
                top = y + item.height / 2
            bounds.append(
                {
                    "row": row_index,
                    "name": name,
                    "x": x,
                    "y": bottom,
                    "width": width,
                    "height": top - bottom,
                }
            )
            if (
                previous
                and row_index not in figure.no_connections
                and not isinstance(item, module.Operator)
                and not isinstance(items[i - 1], module.Operator)
            ):
                lines.append(rf"\draw[flow] ({previous}-e) -- ({name}-w);")
            previous = name
            x += width + gutter
    bottom = min(b["y"] for b in bounds) - 0.8
    lines.append(
        rf"\Meaning{{0}}{{{bottom:.3f}}}{{{canvas:.3f}}}{{{label('axes')}}}{{{label(figure.axes)}}}{{{label('meaning')}}}{{{label(figure.meaning)}}}"
    )
    lines.append(
        r"\node[font=\scriptsize,text=Muted,anchor=north,align=center] "
        rf"at ($(meaning.south)+(0,-.2)$) {{{label('illustration')}\\[3pt]"
        rf"MetisCraft: {figure.slug.replace('-', ' ')} $\cdot$ shichao}};"
    )
    lines.extend((r"\end{tikzpicture}", r"\end{document}"))
    return "\n".join(lines) + "\n", bounds


def build_one(slug, lang, theme, specs, module, labels, geometry, args):
    stem = slug + "." + lang + (".dark" if theme == "dark" else "")
    text, bounds = document(specs[slug], lang, module, labels, geometry, theme)
    tex = OUTPUT / (stem + ".tex")
    if args.check:
        if not tex.is_file() or tex.read_text() != text:
            raise SystemExit("Generated TeX drift: " + stem)
    else:
        tex.write_text(text)
        with tempfile.TemporaryDirectory(prefix="metiscraft-tensor-") as temp:
            folder = Path(temp)
            env = os.environ.copy()
            env["SOURCE_DATE_EPOCH"] = "1794182400"
            cmd = [
                str(args.tectonic),
                "--bundle",
                TECTONIC_BUNDLE_URL,
                "--untrusted",
                "--keep-logs",
                "--outdir",
                str(folder),
                str(tex),
            ]
            run = subprocess.run(
                cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if args.qa_dir:
                args.qa_dir.mkdir(parents=True, exist_ok=True)
                (args.qa_dir / (stem + ".log")).write_text(run.stdout)
            if run.returncode:
                raise SystemExit(run.stdout[-6000:])
            if re.search(r"Missing character|Overfull \\[hv]box", run.stdout):
                raise SystemExit("Typography warning in " + stem + "\n" + run.stdout[-3000:])
            shutil.copyfile(folder / (stem + ".pdf"), OUTPUT / (stem + ".pdf"))
        subprocess.run(
            ["pdftocairo", "-svg", str(OUTPUT / (stem + ".pdf")), str(OUTPUT / (stem + ".svg"))],
            check=True,
        )
    pdf = OUTPUT / (stem + ".pdf")
    svg = OUTPUT / (stem + ".svg")
    if not pdf.is_file() or not svg.is_file():
        raise SystemExit("Missing vector exports: " + stem)
    images = subprocess.check_output(["pdfimages", "-list", str(pdf)], text=True).splitlines()[2:]
    if any(x.strip() for x in images):
        raise SystemExit("Raster image in PDF: " + stem)
    svg_text = svg.read_text()
    if re.search(r"<(?:\w+:)?image\b|data:image", svg_text):
        raise SystemExit("Raster image in SVG: " + stem)
    if args.qa_dir and not args.check:
        subprocess.run(
            ["pdftocairo", "-png", "-r", "150", "-singlefile", str(pdf), str(args.qa_dir / stem)],
            check=True,
        )
        (args.qa_dir / (stem + ".geometry.json")).write_text(json.dumps(bounds, indent=2) + "\n")
    entry = {
        "slug": slug,
        "language": lang,
        "theme": theme,
        "files": {p.name: sha256(p) for p in (tex, pdf, svg)},
        "pdf_raster_images": 0,
        "svg_embedded_images": 0,
    }
    print(stem + ": PDF and SVG verified", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, required=True)
    parser.add_argument("--slug", action="append")
    parser.add_argument("--language", choices=("en", "zh"), action="append")
    parser.add_argument("--theme", choices=("light", "dark"), action="append")
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="verify committed artifacts without rebuilding"
    )
    parser.add_argument(
        "--record-manifest",
        action="store_true",
        help="record a complete verified artifact inventory, including in --check mode",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, 5),
        default=2,
        help="concurrent CPU-only figure builds (1--4)",
    )
    args = parser.parse_args()
    if args.record_manifest and (args.slug or args.language or args.theme):
        parser.error("--record-manifest requires all figures, languages and themes")
    module, specs = load_specs()
    labels = json.loads((SOURCE / "labels.json").read_text())
    geometry = json.loads((SOURCE / "geometry.json").read_text())
    validate_shapes(specs, module, geometry)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for slug in args.slug or specs:
        if slug not in specs:
            raise SystemExit("Unknown figure: " + slug)
    requests = [
        (slug, lang, theme)
        for slug in args.slug or specs
        for lang in args.language or ("en", "zh")
        for theme in args.theme or ("light", "dark")
    ]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        entries = list(
            pool.map(lambda item: build_one(*item, specs, module, labels, geometry, args), requests)
        )
    manifest = {
        "tectonic": subprocess.check_output([str(args.tectonic), "--version"], text=True).strip(),
        "tectonic_executable_sha256": sha256(args.tectonic),
        "tectonic_release_archive_sha256": TECTONIC_ARCHIVE_SHA256,
        "tectonic_release_url": "https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0",
        "tectonic_bundle_url": TECTONIC_BUNDLE_URL,
        "tectonic_bundle_digest": TECTONIC_BUNDLE_DIGEST,
        "poppler": subprocess.run(
            ["pdftocairo", "-v"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.splitlines()[0],
        "source_date_epoch": 1794182400,
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted(SOURCE.glob("*"))
            if p.is_file()
        },
        "builder_sha256": sha256(Path(__file__)),
        "figures": entries,
        "geometry": geometry,
    }
    if args.check and not args.record_manifest:
        saved = json.loads((OUTPUT / "manifest.json").read_text())
        if saved["source_sha256"] != manifest["source_sha256"]:
            raise SystemExit("Figure source inventory changed; rebuild or record the manifest")
        if saved["builder_sha256"] != manifest["builder_sha256"]:
            raise SystemExit("Figure builder changed; rebuild or record the manifest")
        expected = {(e["slug"], e["language"], e["theme"]): e["files"] for e in saved["figures"]}
        for entry in entries:
            key = (entry["slug"], entry["language"], entry["theme"])
            if expected.get(key) != entry["files"]:
                raise SystemExit(f"Vector export content drift: {key}")
    if args.record_manifest or (
        not args.check and not args.slug and not args.language and not args.theme
    ):
        (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified", len(entries), "localized figures")


if __name__ == "__main__":
    main()
