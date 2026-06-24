

import sys

from analyser import *

input_path  = sys.argv[1]
output_path = sys.argv[2]

students    = read_submissions(input_path)
submitted   = get_submitted_students(students)
missing     = get_missing_submissions(students)
avg         = calculate_average_score(students)
domain_avg  = get_domain_wise_average(students)

highest = max(submitted, key=lambda s: s["score"])
lowest  = min(submitted, key=lambda s: s["score"])
below_5 = [s["name"] for s in students if s["score"] < 5]

summary = {
    "total_students"         : len(students),
    "submitted_count"        : len(submitted),
    "missing_count"          : len(missing),
    "average_score"          : avg,
    "highest_scorer"         : highest["name"],
    "lowest_scorer_submitted": lowest["name"],
    "domain_wise_average"    : domain_avg,
    "missing_submissions"    : missing,
    "students_below_5"       : below_5
}

print("Total Students     :", summary["total_students"])
print("Submitted          :", summary["submitted_count"])
print("Missing            :", summary["missing_count"])
print("Average Score      :", summary["average_score"])
print("Highest Scorer     :", summary["highest_scorer"])
print("Missing Submissions:", summary["missing_submissions"])

write_summary(summary, output_path)