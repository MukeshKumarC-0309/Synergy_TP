# Task 4: Messy CSV Cleaning

## Objective
Clean a messy CSV dataset using pandas and produce a clean dataset, validation checks, and a written cleaning report.

## Folder Structure
task_4/
README.md
data/
submissions.csv
output/
cleaned_students.csv
cleaning_report.md
summary_before.json
summary_after.json
src/
clean_data.py
validate_data.py
main.py

## Run Command
Run this from the root of the Synergy_TP repository:
python task_4/src/main.py task_4/data/messy_students.csv task_4/output/cleaned_students.csv

## Expected Output Files
- task_4/output/cleaned_students.csv
- task_4/output/summary_before.json
- task_4/output/summary_after.json
- task_4/output/cleaning_report.md

## Logic Explanation
clean_data.py handles all cleaning steps in separate functions. Each function targets one specific problem in the data and handles them seperately

validate_data.py checks that the cleaned data meets all required conditions — no duplicate IDs, numeric columns are numeric, attendance is between 0 and 100, and domain and submitted values are consistent.

main.py runs all cleaning functions in order, saves the before and after summaries as JSON, saves the cleaned CSV, and writes the cleaning report,and checks if everything is correct
