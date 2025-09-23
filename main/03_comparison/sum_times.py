#%%
import re

# List your specific txt files here
files = ["logs/pinns_training_evaluation_log.txt",
         "logs/bem_solution_log.txt",
         "logs/comparison_bem_log.txt",
         "logs/comparison_pinns_log.txt",
         "logs/comparison_plot_time_error_log.txt",
         "logs/comparison_figure_compose_log.txt"]

total_time = 0.0

for file in files:
    with open(file, 'r') as f:
        for line in f:
            # Match either "Execution time (s):" or "Total execution time (s):"
            match = re.search(r"(Execution time|Total execution time).*?:\s*([\d.]+)", line)
            if match:
                time_sec = float(match.group(2))
                total_time += time_sec
                print(f"Added {time_sec:.2f} sec from {file}")

total_time_min = total_time / 60  # convert seconds to minutes
total_hours = int(total_time // 3600)
total_minutes = int((total_time % 3600) // 60)
total_seconds = total_time % 60

# Print results
print(f"Total execution time (s): {total_time:.2f}")
print(f"Total execution time (min): {total_time_min:.2f}")
print(f"Total execution time (h:m:s): {total_hours}:{total_minutes:02d}:{total_seconds:05.2f}")

# Save results to a txt file
output_file = "logs/total_execution_time.txt"
with open(output_file, "w") as f:
    f.write(f"Total execution time (s): {total_time:.2f}\n")
    f.write(f"Total execution time (min): {total_time_min:.2f}\n")
    f.write(f"Total execution time (h:m:s): {total_hours}:{total_minutes:02d}:{total_seconds:05.2f}\n")

print(f"Results saved to {output_file}")
# %%
