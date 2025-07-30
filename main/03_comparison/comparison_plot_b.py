from svgutils.compose import *

# Load the SVGs
svg1 = SVG("figures/bem_error.svg").scale(1.0)
svg2 = SVG("figures/pinns_error.svg").scale(1.0)

# Create a figure using known dimensions
Figure(
    631,  # total width (still no math if you avoid this by guessing)
    191,  # height (or just use one of them)
    Panel(svg1),
    Panel(svg2).move(245, 0)
).scale(1.3).save("figures/errors.svg")