#%% ======================== IMPORTS ========================
import sys
import os

#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

#%% ======================== SVG LOADING AND COMPOSITION ========================
from svgutils.compose import *

# Load the SVGs
svg1 = SVG("figures/generalization_bem.svg")
svg2 = SVG("figures/generalization_pinns.svg")

# Create a figure using known dimensions
Figure(
    641,  # total width (still no math if you avoid this by guessing)
    449,  # height (or just use one of them)
    Panel(svg1).scale(1.30),
    Panel(svg2).move(255, 0).scale(1.30)
).save("figures/generalization.svg")
