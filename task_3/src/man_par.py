import json

def read_csv_manual(file_path):
    rows = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    headers = lines[0].strip().split(",")
    for line in lines[1:]:
        line = line.strip()
        if line == "":
            continue
        values = line.split(",")
        if len(values) != len(headers):
            continue
        row = {}
        for i in range(len(headers)):
            row[headers[i]] = values[i]
        rows.append(row)
    return rows

def convert_types(rows):
    for row in rows:
        row["score"] = int(row["score"])
        row["submitted"] = row["submitted"].strip().lower()
        if row["submitted"] not in ["yes", "no"]:
            row["submitted"] = "no"
    return rows

def calculate_summary(rows):
    total = len(rows)
    submitted = [r for r in rows if r["submitted"] == "yes"]
    not_submitted = [r for r in rows if r["submitted"] == "no"]
    scores = [r["score"] for r in rows]
    avg_score = sum(scores) / total

    highest = rows[0]
    for r in rows:
        if r["score"] > highest["score"]:
            highest = r
    submitted_scores = [r for r in submitted]
    lowest = submitted[0]
    for r in submitted:
        if r["score"] < lowest["score"]:
            lowest = r
    

    domains = {}
    for r in rows:
        d = r["domain"]
        if d not in domains:
            domains[d] = []
        domains[d].append(r["score"])


    domain_avg = {}   
    for d,v in domains.items():
        domain_avg[d] = sum(v) / len(v)

    below_5 = [r["name"] for r in rows if r["score"] < 5]
    not_submitted_names = [r["name"] for r in not_submitted]

    return {
        "total_students": total,
        "submitted_count": len(submitted),
        "missing_submissions": len(not_submitted),
        "average_score": round(avg_score, 2),
        "highest_scorer": {"name": highest["name"], "score": highest["score"]},
        "lowest_scorer_among_submitted": {"name": lowest["name"], "score": lowest["score"]},
        "domain_average_score": {d: round(v, 2) for d, v in domain_avg.items()},
        "students_not_submitted": not_submitted_names,
        "students_below_5": below_5
    }

def write_json_manual(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)