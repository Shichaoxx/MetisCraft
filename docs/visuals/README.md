# MetisCraft visual assets

The primary identity combines the MetisCraft wordmark with three slanted vertical planes, preserving the original project-showcase silhouette. Teal `#087f78`, graphite `#243438` and a
small amount of warm gold `#bd8c36` connect the repository and documentation.

| Asset | Use |
|---|---|
| [Metis character](figures/metis-guide-chibi.png) | Cookbook reading guide |
| [Adaptive three-plane mark](figures/mark.svg) | Documentation header, small icons and favicon |
| [Light wordmark](figures/wordmark-light.svg) · [dark wordmark](figures/wordmark-dark.svg) | README title and compact brand lockup |
| [Light banner](figures/hero-light.svg) · [dark banner](figures/hero-dark.svg) | Optional wide banner |
| [Social preview](figures/social-preview.png) | Repository sharing image, 1200 × 630 |
| [Workflow](figures/pipeline.svg) · [mobile workflow](figures/pipeline-mobile.svg) | Data → train → calibrate → evaluate → serve |
| [Mobile judgments](figures/judgments-mobile.svg) | Four illustrative examples in a readable vertical layout |
| [Judgment examples](figures/judgments.svg) | Constructed examples, explicitly not benchmark results |

The diagrams and marks have editable SVG sources. Rebuild them with
`python docs/visuals/src/render.py`; the script uses only Python's standard library.
PNG banner and social exports are rendered from those SVG files at their declared
canvas sizes; the favicon is a 64 × 64 export of the three-plane mark. Use an SVG renderer
(such as a browser or `rsvg-convert`) to refresh the PNGs after changing the SVGs.
Metis is a raster illustration generated with Imagegen, not an SVG. She appears
inside the cookbook as a reading guide; repository and documentation homepages use
the typographic identity.

## Character direction

Metis (小墨) is an original scholar character. Her identity combines shoulder-length,
wavy deep-teal hair with mint highlights, round amber-gold eyes, soft cheeks, a
small closed-mouth smile, a looped forelock and a gold feather clip. Feather tufts
are small accents. The guide wears an ivory blouse, a tailored navy waistcoat and
a teal tie with an astrolabe clasp, and holds a star-chart book.

Keep the face compact and gentle when drawing new poses. A slender body should
come from balanced anatomy, with a connected head, neck and shoulders. Preserve
the warm expression and avoid making every pose lean toward one shoulder.

The selected transparent chibi illustration is used for the guide; the previous half-body image remains as a source reference. Its complete generation
prompt and source hashes are preserved in [metis-guide-chibi-prompt.txt](metis-guide-chibi-prompt.txt).
The [initial portrait](figures/avatar.png) and its [prompt](avatar-prompt.txt) remain
as historical face references. Third-party character images are not included.

## Small-screen use

Use the three-plane mark at favicon sizes. The Cookbook displays the complete guide in a fixed square using contain sizing; the transparent image is retained without cropping.
The guide stays within the document flow and never floats over reading content. The workflow has a vertical variant
for narrow screens; diagrams should not be shrunk until their labels are unreadable.

## Theme variants

The [light mark](figures/mark-light.svg) is teal, the [dark mark](figures/mark-dark.svg) is silver-gray, and a [graphite monochrome export](figures/mark-ink.svg) supports neutral contexts. The site header and favicon follow the selected theme; the adaptive SVG also follows the system color scheme when opened independently. The GitHub wordmark uses its light/dark picture sources. These are project assets, not account-avatar changes.
