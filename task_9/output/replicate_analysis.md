# Replicate Analysis

Each replicate group below is one (domain, condition, input_value) cell measured three times. Sample variance and sample standard deviation (ddof = 1) are used, and the 95% confidence interval uses the t-distribution with df = n - 1.

## Per-group summary

| domain | condition | input_value | replicate_count | mean_signal | standard_deviation_signal | standard_error_signal | confidence_interval_lower | confidence_interval_upper | coefficient_of_variation | stability_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Biochem | low_concentration | 0.1000 | 3 | 0.1200 | 0.0100 | 0.0058 | 0.0952 | 0.1448 | 0.0833 | moderate |
| Biochem | medium_concentration | 0.5000 | 3 | 0.5067 | 0.0153 | 0.0088 | 0.4687 | 0.5446 | 0.0301 | stable |
| Biochem | high_concentration | 1.0000 | 3 | 1.0033 | 0.0208 | 0.0120 | 0.9516 | 1.0550 | 0.0207 | stable |
| Electronics | low_load | 10.0000 | 3 | 4.9233 | 0.0252 | 0.0145 | 4.8608 | 4.9858 | 0.0051 | stable |
| Electronics | medium_load | 20.0000 | 3 | 4.6767 | 0.0252 | 0.0145 | 4.6142 | 4.7392 | 0.0054 | stable |
| Electronics | high_load | 40.0000 | 3 | 4.2167 | 0.1041 | 0.0601 | 3.9581 | 4.4752 | 0.0247 | stable |
| Mechanical | low_load | 50.0000 | 3 | 0.5100 | 0.0100 | 0.0058 | 0.4852 | 0.5348 | 0.0196 | stable |
| Mechanical | medium_load | 100.0000 | 3 | 1.0967 | 0.0153 | 0.0088 | 1.0587 | 1.1346 | 0.0139 | stable |
| Mechanical | high_load | 150.0000 | 3 | 1.8833 | 0.1893 | 0.1093 | 1.4131 | 2.3536 | 0.1005 | unstable |

## Answers

**Which replicate group is most stable?** Electronics low_load (input 10 ohm) — the lowest coefficient of variation in the dataset (0.0051). Its three readings sit almost on top of each other, so the mean is trustworthy.

**Which replicate group is most noisy?** Mechanical high_load (input 150 N) — the highest coefficient of variation (0.101). Two readings agree closely while a third (2.10 mm) pulls away, which inflates the spread.

**Which group has the widest confidence interval?** Mechanical high_load — the same outlier that drives its coefficient of variation also widens its confidence interval, so it is the group we are least sure about.

**Which group has the highest coefficient of variation?** Mechanical high_load (0.101). Because CoV is scale-free, it is the fair way to compare noise across groups whose signals live on very different magnitudes.

**Why is the mean alone not enough for judging reliability?** The mean can only indicate the center of the readings, not the spread. Two sets can share the same mean while one is consistent and the other is badly scattered — whether from general noise or a single outlier — and the mean looks identical in both cases, so it can't tell you which set to trust. Only a spread measure (standard deviation, coefficient of variation, confidence interval) reveals the difference.

**Why does replicate count affect confidence interval width?** The number of replicates (n) affects the width of the confidence interval through two effects. First, a higher n decreases the standard error (SE = SD/√n), giving a more stable estimate of the mean. Second, it lowers the t-multiplier: a higher n means more degrees of freedom, which makes the t-distribution narrower and the multiplier smaller. Both effects pull the same way — as n rises, SE and t both shrink, so the interval narrows and we become more confident about where the true mean lies. (The confidence interval describes certainty about the mean, not the spread of the data itself.)

**Which readings should be investigated before using the data for machine learning?** The Mechanical high-load group (M007–M009) should be investigated. Two of its readings agree closely (1.75, 1.80 mm) while the third (2.10 mm) pulls away — this outlier drove the group's coefficient of variation to 0.101, the highest in the dataset, which is why the pipeline flagged it unstable and excluded it. That reading should be judged for whether it is genuine — a real non-linear response at high load — or a measurement error, since that determines whether the automatic exclusion is correct or should be overridden.

**How do replicate statistics show reliability, and which domain is most/least reliable overall?** Replicate statistics turn repeated readings into a reliability judgement — the spread measures (standard deviation, standard error, confidence interval, coefficient of variation) show how consistent the repeats are, and low spread means the mean can be trusted. Rolling this up to the domain level, Electronics is the most stable domain (its groups have the lowest coefficients of variation, around 0.005), while Mechanical is the noisiest (its high-load group reaches 0.101, driven by the 2.10 mm outlier).
