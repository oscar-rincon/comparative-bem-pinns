#%%
# List your specific txt files here
files = ["logs/generalization_bem_log.txt",
         "logs/generalization_pinns_log.txt",
         "logs/generalization_figure_composition_log.txt"]

total_time = 0.0

for file in files:
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("Execution time (s):"):
                # Extract the number after the colon and convert to float
                time_sec = float(line.split(":")[1].strip())
                total_time += time_sec

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
