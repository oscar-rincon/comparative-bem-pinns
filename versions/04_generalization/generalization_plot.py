
 # -*- coding: utf-8 -*-

import sys
import os

# Set the current directory and utilities path
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

from svgutils.compose import *

# Load the SVGs
svg1 = SVG("figures/generalization_bem.svg")
svg2 = SVG("figures/generalization_pinns.svg")

# Create a figure using known dimensions
Figure(
    638,  # total width (still no math if you avoid this by guessing)
    337,  # height (or just use one of them)
    Panel(svg1),
    Panel(svg2).move(250, 0).scale(1.30)
).save("figures/generalization.svg")