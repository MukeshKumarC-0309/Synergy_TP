# Task 3: Manual CSV Parser and Pandas Comparison

## Objective
Build a CSV parser manually, then repeat the same analysis using pandas. The goal is to understand how raw CSV text can be easily handled by python.

## Folder Structure
task_3/
README.md
data/
submissions.csv
output/
manual_summary.json
pandas_summary.json
comparison_report.md
src/
manual_parser.py
pandas_parser.py
main.py

## Required Packages
- pandas
Install with:pip install pandas in your virtual environment

## Setup Instructions
1. Navigate to the Synergy_TP repository through your terminal
2. Ensure submissions.csv is created at task_3/data/submissions.csv.
3. Install required packages using the command above and other libraries if necessary

## Run Command
Run this from the root of the repository:python task_3/src/main.py task_3/data/submissions.csv(The second is the output path given through the sys command)

## Expected Output Files
- task_3/output/manual_summary.json
- task_3/output/pandas_summary.json
- task_3/output/comparison_report.md

## Logic Explanation
manual_parser.py reads the CSV using only open() and basic string splitting — no pandas or csv module. Each row is stored as a dictionary, scores are converted to integers, and submitted values are normalized to yes/no. The summary is calculated using plain Python loops.

pandas_parser.py performs the exact same analysis using pandas, which handles type conversion and grouping in fewer lines using in built functions saving lots of memory and time.

main.py runs both type of parsing programs, saves their summaries as JSON files, and we write a comparison report comparing the 2 results and checking if they match.