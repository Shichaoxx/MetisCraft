# MetisCraft visual assets

The owl is the project's visual motif. Teal `#087f78`, graphite `#243438` and a
small amount of warm gold `#bd8c36` connect the repository and documentation.

| Asset | Use |
|---|---|
| [Owl character](figures/avatar.png) | README introduction, community avatar and documentation |
| [Owl mark](figures/mark.svg) | Small icons and favicon |
| [Light wordmark](figures/wordmark-light.svg) · [dark wordmark](figures/wordmark-dark.svg) | README title and compact brand lockup |
| [Light banner](figures/hero-light.svg) · [dark banner](figures/hero-dark.svg) | Optional wide banner |
| [Social preview](figures/social-preview.png) | Repository sharing image, 1200 × 630 |
| [Workflow](figures/pipeline.svg) · [mobile workflow](figures/pipeline-mobile.svg) | Data → train → calibrate → evaluate → serve |
| [Mobile judgments](figures/judgments-mobile.svg) | Four illustrative examples in a readable vertical layout |
| [Judgment examples](figures/judgments.svg) | Constructed examples, explicitly not benchmark results |

The diagrams and marks have editable SVG sources. Rebuild them with
`python docs/visuals/src/render.py`; the script uses only Python's standard library.
PNG banner and social exports are rendered from those SVG files with `rsvg-convert`.
The owl character is a raster illustration generated with Imagegen, not an SVG.

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

Use the simple owl mark at favicon sizes. Use the character at avatar sizes where
its eyes and feather details remain legible. The workflow has a vertical variant
for narrow screens; diagrams should not be shrunk until their labels are unreadable.
