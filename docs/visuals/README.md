# MetisCraft visual assets

The primary identity is a plain MetisCraft wordmark and a compact M monogram. Teal `#087f78`, graphite `#243438` and a
small amount of warm gold `#bd8c36` connect the repository and documentation.

| Asset | Use |
|---|---|
| [Metis character](figures/avatar.png) | Cookbook reading guide |
| [M monogram](figures/mark.svg) | Documentation header, small icons and favicon |
| [Light wordmark](figures/wordmark-light.svg) · [dark wordmark](figures/wordmark-dark.svg) | README title and compact brand lockup |
| [Light banner](figures/hero-light.svg) · [dark banner](figures/hero-dark.svg) | Optional wide banner |
| [Social preview](figures/social-preview.png) | Repository sharing image, 1200 × 630 |
| [Workflow](figures/pipeline.svg) · [mobile workflow](figures/pipeline-mobile.svg) | Data → train → calibrate → evaluate → serve |
| [Mobile judgments](figures/judgments-mobile.svg) | Four illustrative examples in a readable vertical layout |
| [Judgment examples](figures/judgments.svg) | Constructed examples, explicitly not benchmark results |

The diagrams and marks have editable SVG sources. Rebuild them with
`python docs/visuals/src/render.py`; the script uses only Python's standard library.
PNG banner and social exports are rendered from those SVG files at their declared
canvas sizes; the favicon is a 64 × 64 export of the M monogram. Use an SVG renderer
(such as a browser or `rsvg-convert`) to refresh the PNGs after changing the SVGs.
Metis is a raster illustration generated with Imagegen, not an SVG. She appears
inside the cookbook as a reading guide; repository and documentation homepages use
the typographic identity.

## Character direction

The avatar depicts an original owl-themed anime character with teal hair, amber
eyes, feather tufts and a feather-edged capelet. The visual direction was informed
by community character references collected for
[DeepSeek](https://genderswapai.com/zh-CN/ai-gijinka/deepseek),
[GPT](https://genderswapai.com/zh-CN/ai-gijinka/gpt) and
[Claude](https://genderswapai.com/zh-CN/ai-gijinka/claude).
These third-party characters and images are not part of this repository's assets.
The generation prompt is preserved in [avatar-prompt.txt](avatar-prompt.txt).

## Small-screen use

Use the M monogram at favicon sizes. The cookbook guide uses the existing character
at a readable size within the document flow; it never floats over reading content. The workflow has a vertical variant
for narrow screens; diagrams should not be shrunk until their labels are unreadable.
