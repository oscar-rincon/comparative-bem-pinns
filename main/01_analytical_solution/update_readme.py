import os
import re

# Paths
base_dir = os.path.dirname(__file__)
readme_file = os.path.join(base_dir, "README.md")
log_file = os.path.join(base_dir, "logs", "analytical_solution_log.txt")  # adjust if script name differs

# Extract elapsed time from log
elapsed_time = None
with open(log_file, "r") as f:
    for line in f:
        if line.startswith("Execution time"):
            elapsed_time = line.split(":")[1].strip()
            break

if not elapsed_time:
    raise RuntimeError("Could not find execution time in log file.")

# Read README content
with open(readme_file, "r") as f:
    content = f.read()

# Replace the estimated time line
new_line = f"The execution of this script takes approximately **{elapsed_time}** on a standard machine."
content = re.sub(
    r"The execution of this script takes approximately \*\*.*?\*\* on a standard machine\.",
    new_line,
    content,
)

# Write updated README
with open(readme_file, "w") as f:
    f.write(content)

print(f"README.md updated with execution time: {elapsed_time}")
