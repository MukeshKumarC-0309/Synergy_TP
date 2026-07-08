# Correlation and Calibration Limitations

## Correlation and fit summary

| relationship | n_samples | pearson_correlation | spearman_correlation | slope | intercept | r_squared | mean_absolute_error | root_mean_squared_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Biochem: signal vs concentration | 9 | 0.9993 | 0.9487 | 0.9820 | 0.0196 | 0.9986 | 0.0112 | 0.0133 |
| Electronics: signal vs load | 9 | -0.9845 | -0.9487 | -0.0235 | 5.1533 | 0.9693 | 0.0379 | 0.0521 |
| Electronics: signal vs temperature | 9 | -0.9920 | -0.9748 | -0.0520 | 6.7645 | 0.9841 | 0.0267 | 0.0376 |
| Mechanical: signal vs load | 9 | 0.9841 | 0.9487 | 0.0137 | -0.2100 | 0.9684 | 0.0778 | 0.1013 |
| Mechanical: stress_mpa vs load | 9 | 0.9843 | 0.9487 | 1.6400 | -5.3333 | 0.9688 | 9.4074 | 12.0093 |

## Answers

**Does signal increase or decrease with input value?** It depends on the domain. In Biochem, absorbance increases with concentration (positive slope, Pearson ≈ 0.999). In Electronics, the measured voltage decreases as both load resistance and temperature increase (negative slope, Pearson ≈ -0.98). In Mechanical, both displacement and stress increase with load (positive slope).

**Which domain shows the strongest signal-input relationship?** Biochem — Pearson ≈ 0.999 and R² ≈ 0.999, an almost perfect straight-line calibration.

**Which domain shows the weakest or noisiest relationship?** The Mechanical fits carry the most error. The outlier in the high-load group inflates RMSE relative to MAE, so its calibration line explains the data less cleanly than the others.

**Does high correlation prove causation?** High correlation doesn't necessarily prove causation, because even though two variables move together, there could be a third external or hidden variable that influences both. In my own Electronics data, voltage and temperature are strongly correlated — but temperature isn't causing the voltage to change. The rising load drives both: it pushes the temperature up and the voltage down at the same time. The correlation is real, but reading it as "temperature causes the voltage change" would be the wrong causal story.

**Can correlation be trusted with small sample size?** No — correlation can't be trusted on its own with a small sample size. Each relationship here has only 3 distinct input levels, and with so few points a single stray reading can move the coefficient haphazardly and distort the apparent relationship between the two variables. The estimate also carries very wide uncertainty at that sample size, so a small-n correlation is suggestive at best, not conclusive.

**Can correlation miss nonlinear relationships?** Yes, correlation can miss a nonlinear relationship, particularly the Pearson coefficient. Pearson only measures how well the points fit a straight line, so if the real relationship is curved, it will understate it. That's why we also use Spearman correlation, which works on ranks and checks whether one variable moves consistently in the same direction (monotonically) as the other changes, even if the shape is curved. Comparing the two is a useful check: if Spearman is much higher than Pearson, the relationship is real but nonlinear, and a straight-line fit is missing part of it.

**How can outliers affect correlation?** Outliers affect correlation by dragging the fitted line toward themselves and distorting the actual fit. They also affect the correlation coefficient by either inflating or deflating it, depending on where the outlier sits. And they affect the error metrics — especially RMSE versus MAE: because RMSE squares each error, one distant point pushes RMSE far ahead of MAE. The common cause is that least-squares squares the distances, so a single far-off point counts disproportionately. This is exactly what the 2.10 mm reading in the Mechanical high-load group did — it pulled the fit, worsened the R², and drove RMSE above MAE, which is what flagged the group as noisy.

**How can temperature, load, material type, or experimental condition act as confounders?** A confounder is an external or hidden variable that drives both correlated variables, so the observed correlation may actually be the confounder's effect rather than a real link between the two. In our Electronics case we observed a correlation between temperature and voltage, but in reality the load drove both — as load increased, temperature rose and voltage dropped together. The problem with confounders is that they're extremely hard to separate out: here temperature and load never vary independently, so every high-temperature reading is also a high-load reading, and the data alone can't tell you which one is actually affecting the voltage.

**Why should mixed-domain correlation be avoided?** Mixed-domain correlation must be avoided because the domains have different units and different physical meanings, and each domain's signal responds to a different mechanism. Combining absorbance, voltage, and displacement into a single correlation produces a number that has no physical interpretation — it isn't measuring any real relationship, because the quantities being compared don't share a common meaning. The coefficient might even come out looking strong, but it would be meaningless.
