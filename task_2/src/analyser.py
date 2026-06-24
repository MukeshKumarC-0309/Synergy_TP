

import csv
import json
import os

def read_submissions(filepath):
    students = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = int(row["score"])
            students.append(row)
    return students

def get_submitted_students(students):
    submitted = []
    for s in students:
        if s["submitted"] == "yes":
            submitted.append(s)
    return submitted

def calculate_average_score(students):
    total = 0
    for s in students:
        total += s["score"]
    return round(total / len(students), 2)

def get_domain_wise_average(students):
    domains = {}
    for s in students:
        domain = s["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(s["score"])

    averages = {}
    for domain, scores in domains.items():
        averages[domain] = round(sum(scores) / len(scores), 2)
    return averages

def get_missing_submissions(students):
    missing = []
    for s in students:
        if s["submitted"] == "no":
            missing.append(s["name"])
    return missing

def write_summary(summary, output_path):
    os.makedirs("task_2/output", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary written to {output_path}")