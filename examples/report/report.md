# EAG model check

All checks passed: the data support the model. Compensation index kappa = 0.25 (95% CI 0.19 to 0.32); the structure is partly non-compensatory: weak components hold competence back.

| Layer | Check | Result | Verdict |
|---|---|---|---|
| Data | Number of students | 200 | pass |
| Data | Rated student-occasions | 800 with 2 rater(s) | pass |
| Data | Missing rubric scores | 0.0% | pass |
| Evidence | Indicator fit (infit and outfit mean squares) | 20 of 20 indicators within range; largest deviation PF3 (infit 0.79, outfit 0.78) | pass |
| Evidence | Use of rubric levels | 0 indicator(s) with a level used fewer than 10 times | pass |
| Evidence | Ordering of rubric thresholds | ordered in all components | pass |
| Evidence | Reliability of component proficiencies | Problem framing 0.86; System modelling 0.86; Measurement design 0.87; Inference design 0.86; Iterative refinement 0.87 | pass |
| Evidence | Distinctness of components | largest correlation between initial proficiencies 0.55 | pass |
| Evidence | Agreement between raters | mean correlation between raters 0.77 | pass |
| Evidence | Rater severity | range of severity 6.4 rating points (residual SD 8.4) | info |
| Computation | Importance sampling | smallest effective sample size 152 of 300 proposals | pass |
| Computation | Convergence of all stages | converged | pass |
| Aggregation | Precision of the compensation index | kappa = 0.25 (95% CI 0.19 to 0.32); rho = -2.03 | pass |
| Aggregation | Test of sum-score aggregation (kappa = 1) | likelihood ratio 95.7, df 1, p = 1.35e-22; structure partly non-compensatory: weak components hold competence back | info |
| Aggregation | Test of equal component weights | likelihood ratio 6.8, df 4, p = 0.145 | info |
| Aggregation | Functional form (residuals versus profile unevenness) | fitted model r = 0.00 (95% CI -0.14 to 0.14); sum-score model r = -0.21 | pass |
| Aggregation | Linearity of the rating link | Wald test of curvature 0.4, df 2, p = 0.817 | pass |
| Aggregation | Posterior predictive check (unevenness signature) | fitted model p = 0.56; sum-score model p = 0.00 | pass |
| Aggregation | 5-fold cross-validation against the sum-score model | difference in out-of-sample log predictive density 49.4 (SE 12.8); fitted model predicts better | info |
| Growth | Growth across design cycles | 5 of 5 components show reliable growth; mean growth 0.31 logits per cycle | info |
| Growth | Difference in growth between groups (intervention minus comparison) | 3 of 5 components differ reliably; mean difference 0.18 logits per cycle | info |
