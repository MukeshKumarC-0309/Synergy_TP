import sys
import os
import json
from clean_data import (load_data, generate_summary, remove_duplicates,
                        standardize_domains, clean_attendance, clean_scores,
                        clean_study_hours, clean_height, clean_weight,
                        clean_submitted, handle_missing_values,
                        validate_cleaned_data, save_cleaned_data, write_report)
from validate import run_validation

input_path = sys.argv[1]
output_path = sys.argv[2]

os.makedirs("task_4/output", exist_ok=True)

df = load_data(input_path)
summary_before = generate_summary(df)

df = remove_duplicates(df)
df = standardize_domains(df)
df = clean_attendance(df)
df = clean_scores(df)
df = clean_study_hours(df)
df = clean_height(df)
df = clean_weight(df)
df = clean_submitted(df)
df = handle_missing_values(df)

summary_after = generate_summary(df)

with open("task_4/output/summary_before.json", "w") as f:
    json.dump(summary_before, f, indent=4)

with open("task_4/output/summary_after.json", "w") as f:
    json.dump(summary_after, f, indent=4)

save_cleaned_data(df, output_path)
write_report("task_4/output/cleaning_report.md")
run_validation(df)

is_valid = validate_cleaned_data(df)
print("All validation checks passed!" if is_valid else "Some validation checks failed.")