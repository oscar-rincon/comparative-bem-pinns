#%%%

# Standard library imports
import sys
import os
import time
import cairosvg

# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

from svgutils.compose import *


#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define output file path
output_file = os.path.join(output_folder, f"{script_name}_log.txt")

#%%
# Load the SVGs
svg1 = SVG("figures/bem_error.svg").scale(1.0)
svg2 = SVG("figures/pinns_error.svg").scale(1.0)

# Create a figure using known dimensions
Figure(
    631,  # total width (still no math if you avoid this by guessing)
    191,  # height (or just use one of them)
    Panel(svg1),
    Panel(svg2).move(240, 0)
).scale(1.3).save("figures/errors.svg")
 
# Load the SVGs
svg1 = SVG("figures/rel_error_time.svg").scale(1.0)
svg2 = SVG("figures/errors.svg").scale(1.0)

# Create a figure using known dimensions
Figure(
    617,  # total width (still no math if you avoid this by guessing)
    348,  # height (or just use one of them)
    SVG("figures/comparison_base.svg").scale(3.79),
    Panel(svg1).move(0, 0).scale(1.3),
    #Text("A", 5, 15, size=12, weight="bold",font="sans-serif"),
    Panel(svg2).move(25, 175)#,Text("B", 5, 225, size=12, weight="bold", font="sans-serif")
).save("figures/05_accuracy_time_error_bem_pinns.svg") 

# Convert to PDF
cairosvg.svg2pdf(
    url="figures/05_accuracy_time_error_bem_pinns.svg", 
    write_to="figures/05_accuracy_time_error_bem_pinns.pdf"
)
 
#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time
 
# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Define log filename inside the logs folder
log_filename = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")
# %%
