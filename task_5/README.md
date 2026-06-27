# Task 5: Matplotlib Visualization from Cleaned Data

## Objective
Generate three properly labeled plots from the cleaned dataset produced in Task 4 using matplotlib.

## Folder Structuretask_5/

README.md

output/

domain_average_score.png

attendance_vs_score.png

submission_status_count.png

plot_summary.md

src/

visualize.py

main.py

## Required Packages
- pandas
- matplotlib

Install with:pip install matplotlib

## Run Command
Run this from the root of the Synergy_TP repository:python task_5/src/main.py task_4/output/cleaned_students.csv task_5/output

## Expected Output Files
- task_5/output/domain_average_score.png
- task_5/output/attendance_vs_score.png
- task_5/output/submission_status_count.png
- task_5/output/plot_summary.md

## Logic Explanation
visualize.py contains one function per plot. Each function loads the cleaned data, generates a labeled plot using matplotlib, and saves it as a PNG file using savefig().
Then the main.py is used to run all commands.