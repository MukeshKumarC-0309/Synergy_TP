import sys
import os
from visualise import (load_cleaned_data, plot_domain_average_score,
                       plot_attendance_vs_score, plot_submission_status_count,
                       write_plot_summary)

input_path = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

df = load_cleaned_data(input_path)

plot_domain_average_score(df, output_dir + "/domain_average_score.png")
plot_attendance_vs_score(df, output_dir + "/attendance_vs_score.png")
plot_submission_status_count(df, output_dir + "/submission_status_count.png")
write_plot_summary(output_dir + "/plot_summary.md")

print("All plots saved successfully.")