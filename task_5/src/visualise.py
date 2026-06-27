import pandas as pd
import matplotlib.pyplot as plt

def load_cleaned_data(file_path):
    return pd.read_csv(file_path)

def plot_domain_average_score(df, output_path):
    domain_avg = df.groupby("domain")["score"].mean()
    plt.figure()
    plt.bar(domain_avg.index, domain_avg.values)
    plt.title("Average Score by Domain")
    plt.xlabel("Domain")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_attendance_vs_score(df, output_path):
    plt.figure()
    plt.scatter(df["attendance_percent"], df["score"])
    plt.title("Attendance vs Score")
    plt.xlabel("Attendance Percent")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_submission_status_count(df, output_path):
    counts = df["submitted"].value_counts()
    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.title("Submission Status Count")
    plt.xlabel("Submitted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def write_plot_summary(output_path):
    with open(output_path, "w") as f:
        f.write("# Plot Summary\n\n")
        f.write("## domain_average_score.png\n")
        f.write("This bar chart shows the average score for each domain. It helps identify which domain performed best overall.\n\n")
        f.write("## attendance_vs_score.png\n")
        f.write("This scatter plot shows the relationship between attendance percentage and score. It helps identify if higher attendance leads to better scores.\n\n")
        f.write("## submission_status_count.png\n")
        f.write("This bar chart shows how many students submitted and how many did not. It gives a quick overview of submission rates.\n")