# MetisCraft documentation design

The English and Chinese sites use a warm paper ground for daytime reading and
charcoal for dark mode. Keep the three-plane MetisCraft mark and the existing
Cookbook illustrations. The website's reading palette is defined in
[site.css](docs/site.css).

| Role | CSS token | Use |
| --- | --- | --- |
| Page and panels | `--metis-paper`, `--metis-panel`, `--metis-surface` | Reading column, navigation and figure surfaces |
| Text | `--metis-ink`, `--metis-muted` | Body, secondary labels and captions |
| Terracotta | `--metis-accent`, `--metis-link` | Emphasis, actionable text and current location |
| Mineral blue and sage | `--metis-blue`, `--metis-sage` | Distinguish technical roles alongside labels and shapes |
| Structure | `--metis-rule`, `--metis-code` | Separators and code surfaces |

Use system serif faces for the page title and section headings, a local sans
serif for body text and monospace for literal fields and commands. Keep CJK
fallbacks in each stack. Pair every reported result with its denominator,
comparison and limitations. Distinguish a prediction's probability from accuracy
over a dataset.

The two home pages share [home.css](docs/assets/home.css) and the same section
order. For diagrams, give each stage a readable label, keep the mobile reading
order and provide a text equivalent. Update generated workflow and judgment
figures in [the renderer](docs/visuals/src/render.py). The tensor atlas maintains
its own [print and screen palettes](docs/tensor-analysis/src/themes.json).

Check both languages at desktop and phone widths in both themes. Exercise
the palette switch, language links and keyboard figure enlargement in the
[documentation browser checks](scripts/qa_documentation.cjs). Aim for at least
4.5:1 contrast for normal text on the surface where it is displayed.
