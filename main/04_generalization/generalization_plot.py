#%% ======================== IMPORTS ========================
import sys
import os
import time
#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

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
 
# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Define log filename inside the logs folder
log_filename = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")
