# Customer Satisfaction Analysis

A Python statistical analysis of survey-based customer satisfaction scores (120 customers, 10 variables). The goal is to find out what is associated with satisfaction and to surface insights that can support business decisions.

## Dataset

| Item | Detail |
|---|---|
| File | `dataset.csv` (not included in this repository, see [Data privacy](#data-privacy)) |
| Rows / columns | 120 / 10 |
| Columns | `Customer_ID`, `Group` (A/B), `Satisfaction_Score` (1-10), `Age`, `Gender`, `Location`, `Purchase_History`, `Support_Contacted`, `Loyalty_Level`, `Satisfaction_Factor` |

No missing values or duplicates were found, so all 120 rows were used.

## Repository contents

| File | Purpose |
|---|---|
| `Customer-satisfaction.py` | The analysis. 13 chunks (`# %%` cells), each following six steps: Library, Data Connection, Data Cleaning, Data Preparation, Analysis, Conclusion |
| `build_report.py` | Runs all chunks and builds `Customer_Satisfaction_Report.html` (printed results and charts) |
| `README.md` | This summary |

## How to run

```powershell
# 1. Create/activate a virtual environment, then install the packages
uv pip install --python .\.venv\Scripts\python.exe pandas numpy scipy statsmodels scikit-learn matplotlib seaborn

# 2. Put dataset.csv in the project folder (or edit DATA_PATH in the script)

# 3. Build the full report
.\.venv\Scripts\python.exe build_report.py
```

Open `Customer_Satisfaction_Report.html` in a browser. For a PDF, press `Ctrl+P` and choose **Save as PDF**. You can also run each chunk on its own with "Run Cell" in Positron or VS Code.

## Analyses performed

1. Descriptive statistics
2. Independent two-sample t-test
3. Paired t-test (skipped, see below)
4. One-way ANOVA
5. Chi-square test of independence
6. Mann-Whitney U test
7. Kruskal-Wallis test
8. Spearman rank correlation
9. Linear regression
10. Logistic regression
11. Factor analysis
12. Cluster analysis (K-Means and hierarchical)
13. Confidence intervals

## Results

| # | Analysis | Result | Conclusion |
|---|---|---|---|
| 1 | Descriptives | Mean score 5.35/10 (median 5, SD 3.03, range 1-10, most common score 1). Mean age 43.9 | Satisfaction is middling and very spread out |
| 2 | Independent t-test, Group A vs B | A = 5.12, B = 5.50, p = 0.508, Cohen's d = -0.12 | No difference |
| 3 | Paired t-test | Not run | The dataset has one score per customer, so there is no before/after pair |
| 4 | ANOVA by Loyalty Level | Low 5.84, Medium 4.47, High 5.65. F = 2.43, p = 0.092 | Not significant at 5% |
| 5 | Chi-square | Gender vs Purchase: p = 0.838. Loyalty vs Support contact: p = 0.182 | No significant association |
| 6 | Mann-Whitney U, Group A vs B | p = 0.490 | Confirms test 2: no difference |
| 7 | Kruskal-Wallis | By Loyalty: p = 0.086. **By Satisfaction Factor: H = 19.26, p = 0.023** | Only the satisfaction factor shows a significant difference |
| 8 | Spearman | Age vs Satisfaction: rho = 0.02 (p = 0.82). Loyalty vs Satisfaction: rho = -0.04 (p = 0.65). Age vs Loyalty: rho = -0.195 (p = 0.033) | Age and loyalty have a weak negative link. Neither relates to satisfaction |
| 9 | Linear regression | R² = 0.045, adjusted R² = -0.006, F-test p = 0.504 | The model explains almost nothing. No predictor is significant |
| 10 | Logistic regression | Purchase: p = 0.474, AUC = 0.61. Support contact: p = 0.576, AUC = 0.61 | Neither outcome can be predicted from age, gender, loyalty or satisfaction |
| 11 | Factor analysis | Eigenvalues are all close to 1 (1.71, 1.58, 1.44, 1.33, ...). Each retained factor loads on essentially one variable | No meaningful hidden structure |
| 12 | Clustering | Best k = 6, silhouette = 0.24 (weak). K-Means and hierarchical agree (ARI = 0.78) | Segments exist but are not sharply separated |
| 13 | Confidence intervals | Mean satisfaction 5.35, 95% CI [4.80, 5.90]. Purchased: 57.5% [48.6%, 66.0%]. Contacted support: 46.7% [38.0%, 55.6%] | Estimates are wide because the sample is small |

### Satisfaction by primary satisfaction factor (test 7)

| Factor | n | Mean | Median |
|---|---|---|---|
| Product Quality | 11 | 7.55 | 7.0 |
| Packaging | 15 | 7.00 | 7.0 |
| Product Variety | 10 | 6.10 | 6.0 |
| Customer Service | 11 | 5.91 | 6.0 |
| Delivery Speed | 12 | 5.17 | 5.5 |
| Brand Reputation | 14 | 4.79 | 4.0 |
| Support Availability | 11 | 4.45 | 4.0 |
| Price | 10 | 4.40 | 3.5 |
| Features | 12 | 4.08 | 4.0 |
| Ease Of Use | 14 | 4.07 | 3.5 |

### Cluster profiles (test 12)

| Cluster | n | Satisfaction | Avg age | Purchased | Contacted support |
|---|---|---|---|---|---|
| 6 | 15 | 7.80 | 47 | 100% | 0% |
| 4 | 27 | 6.37 | 46 | 100% | 100% |
| 3 | 20 | 5.45 | 42 | 0% | 15% |
| 1 | 24 | 5.00 | 50 | 0% | 50% |
| 2 | 13 | 3.77 | 31 | 46% | 100% |
| 5 | 21 | 3.57 | 41 | 100% | 5% |

## Overall conclusion

**Satisfaction in this sample is average (about 5.4 out of 10), varies widely between customers (SD about 3), and is largely not explained by who the customers are or what they do.**

- **No significant difference** between Group A and Group B (t-test and Mann-Whitney agree).
- **No significant relationship** between satisfaction and age, gender, loyalty level, purchase history or support contact. The regression models explain close to 0% of the variance.
- **One significant result:** satisfaction differs by the primary satisfaction factor (p = 0.023). Customers who cite **Product Quality** and **Packaging** report the highest scores (about 7 to 7.5). Customers who cite **Ease of Use, Features, Price and Support Availability** report the lowest (about 4 to 4.5).
- **Loyalty is borderline, not significant.** The Medium loyalty group scores lowest (4.47 vs 5.84 for Low and 5.65 for High, p about 0.09). It is worth re-checking with more data.
- **Customer segments** (clusters) are driven mostly by purchase and support-contact behaviour. The happiest segment (cluster 6, mean 7.8) purchased and never contacted support. The least satisfied segments are customers who contacted support (cluster 2, mean 3.8, youngest) and customers who purchased without contacting support (cluster 5, mean 3.6).

## Suggested actions

1. **Protect strengths:** keep Product Quality and Packaging performance steady, since customers who value them are the most satisfied.
2. **Investigate weak drivers:** look into Ease of Use, Features, Price and Support Availability, where satisfaction is lowest.
3. **Look at the Medium loyalty tier** to see why its satisfaction trends lower.
4. **Look at clusters 2 and 5** (mean satisfaction below 4) to understand what they have in common.
5. **Collect before/after scores** so the paired t-test can measure the effect of changes over time.

## Limitations

- **Small sample (n = 120).** Confidence intervals are wide, and each satisfaction-factor group has only 10 to 15 customers.
- **Many tests were run (about 20).** At a 5% threshold, about one significant result is expected by chance alone. The two significant results (satisfaction factor and Age vs Loyalty) should be treated as leads to confirm, not proven facts.
- **Post-hoc tests are not significant.** After Bonferroni correction, no individual pair of satisfaction factors differs significantly (smallest adjusted p = 0.16, Features vs Product Quality). With 45 pairs, this correction is strict.
- **Scores are not normally distributed** (Shapiro-Wilk p < 0.01), so the rank-based tests (Mann-Whitney, Kruskal-Wallis, Spearman) are the more reliable ones. The regression residuals are also non-normal.
- **Logistic regression accuracy** (59% and 56%) is barely above always guessing the majority class (57.5% and 53.3%).
- **Factor analysis** was run mostly on yes/no and category variables, so treat it as exploratory. The last eigenvalue is 0 because the satisfaction-factor dummy columns add up to 1.
- **Clusters are weak** (silhouette 0.24), so segment profiles are descriptive only.
- **Correlation is not causation.** The `Satisfaction_Factor` is the customer's own stated main driver, not a measured cause.

## Data privacy

`dataset.csv` contains customer IDs and is excluded through `.gitignore`. Add it only if you are sure it is safe to share publicly.
