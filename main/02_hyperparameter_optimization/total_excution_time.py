import os
import re

# Folder with logs
base_dir = os.path.dirname(__file__)
logs_dir = os.path.join(base_dir, "logs")

total_time = 0.0

# Regex to match lines like "Execution time (s): 10.10"
pattern = re.compile(r"Execution time\s*\(s\):\s*([0-9]*\.?[0-9]+)")

for log_file in os.listdir(logs_dir):
    if log_file.endswith(".txt"):
        with open(os.path.join(logs_dir, log_file), "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    total_time += float(match.group(1))

print(f"Total execution time across all logs: {total_time:.2f} s")
