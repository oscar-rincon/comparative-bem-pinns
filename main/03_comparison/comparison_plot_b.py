#%%%

# Standard library imports
import sys
import os
 
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

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
 
# Load the SVGs
svg1 = SVG("figures/rel_error_time.svg").scale(1.0)
svg2 = SVG("figures/errors.svg").scale(1.0)

# Create a figure using known dimensions
Figure(
    631,  # total width (still no math if you avoid this by guessing)
    366,  # height (or just use one of them)
    Panel(svg1).move(0, 0).scale(1.3),
    #Text("A", 5, 15, size=12, weight="bold",font="sans-serif"),
    Panel(svg2).move(-3, 175)#,Text("B", 5, 225, size=12, weight="bold", font="sans-serif")
).save("figures/comparison.svg")
# %%
