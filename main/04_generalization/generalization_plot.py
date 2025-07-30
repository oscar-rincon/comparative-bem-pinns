from svgutils.compose import *

# Load the SVGs
svg1 = SVG("figures/generalization_bem.svg")
svg2 = SVG("figures/generalization_pinns.svg")

# Create a figure using known dimensions
Figure(
    586,  # total width (still no math if you avoid this by guessing)
    257,  # height (or just use one of them)
    Panel(svg1),
    Panel(svg2).move(295, 0)
).save("figures/generalization.svg")