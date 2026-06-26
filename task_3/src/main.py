import sys
import os
import pandas as pd
from man_par import read_csv_manual, convert_types, calculate_summary,write_json_manual
from pan_par import calculate_summary_pandas,write_json_pandas

file_path = sys.argv[1]

os.makedirs("task_3/output", exist_ok=True)

rows = read_csv_manual(file_path)
rows = convert_types(rows)
manual_summary = calculate_summary(rows)
write_json_manual(manual_summary, "task_3/output/manual_summary.json")

df = pd.read_csv(file_path)
pandas_summary = calculate_summary_pandas(df)
write_json_pandas(pandas_summary, "task_3/output/pandas_summary.json")

match = manual_summary == pandas_summary

with open("task_3/output/comparison_report.md", "w") as f:
    f.write("# Comparison Report\n\n")
    f.write("## Do both outputs match?\n\n")
    f.write("Yes, both outputs match.\n\n" if (match==True) else "No, there are differences.\n\n")
    f.write("## Manual Summary\n\n")
    for k, v in manual_summary.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n## Pandas Summary\n\n")
    for k, v in pandas_summary.items():
        f.write(f"- **{k}**: {v}\n")