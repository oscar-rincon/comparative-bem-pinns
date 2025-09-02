#%% ======================== IMPORTS ========================
import sys
import os
import time
#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

#%%

# Record start time
start_time = time.time()

# Get script name
script_name = os.path.basename(__file__) 

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
    658,  # total width (still no math if you avoid this by guessing)
    573,  # height (or just use one of them)
    Panel(svg1).scale(1.30),
    Panel(svg2).move(255, 0).scale(1.30)
).save("figures/generalization.svg")

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

log_filename = os.path.splitext(script_name)[0] + "_log.txt"
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to {log_filename}") 
