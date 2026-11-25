# Protocol for the empirical validation of the EAG model

This protocol states how to test the model with real students. It is written so that independent teams can run the same
study and pool or compare results. Registering the protocol before data collection (for example on the Open Science
Framework) is strongly recommended.

## 1. Questions and hypotheses

| | Hypothesis | Evidence | The hypothesis is not supported if |
|---|---|---|---|
| H1 | Evidence function: indicators measure five distinguishable components | rubric fit, reliability, component correlations | more than 20% of indicators misfit after one revision, any reliability is below 0.50, or two components correlate above 0.90 |
| H2 | Aggregation function: competence is not fully compensatory (kappa < 1) | likelihood-ratio test, cross-validation, posterior predictive check | the 95% interval of kappa includes 1 and cross-validation shows no predictive gain |
| H3 | External validity: model-based competence predicts independent fair judgements better than rubric sum scores | scores from external judges who are blind to classroom data | the correlation with external scores is not higher than that of sum scores |
| H4 | Growth function: teaching the highest-leverage component accelerates growth | randomised comparison of leverage-informed and generic feedback | the difference in growth of competence is not positive at the 5% level |
| H5 | Fairness: parameters are invariant across gender and school location | subgroup fits | indicator difficulties differ by more than 0.5 logits or kappa differs reliably |

## 2. Design

**Phase 1 (structure, H1 to H3 and H5).** One semester, four design cycles, all students taught with the design-cycle sequence.

**Phase 2 (instruction, H4).** Two options.

* *Option A, individual randomisation within classes.* After every design review each student receives feedback that is
  either leverage-informed (the component named in `teach_next`, with a matched mini-lesson) or generic (whole-class
  feedback on all components). For a standardised difference of 0.30 on final competence, power 0.80, two-sided alpha 0.05
  and a baseline covariate explaining half of the variance: n per arm = 2 (1.96 + 0.84)^2 (1 - 0.5) / 0.30^2 = 88, that is
  about 210 students after 15% attrition.
* *Option B, cluster randomisation of classes.* Classes per arm = 2 (1.96 + 0.84)^2 [ICC (1 - R2c) + (1 - ICC)(1 - R2s) / m] / d^2.
  With 35 students per class, ICC = 0.20, R2c = R2s = 0.5 and d = 0.30 this gives 20 classes per arm.

Option A is cheaper and is recommended for a first study; Option B avoids contamination between students.

## 3. Participants and sample size

Upper-secondary students (grades 10 to 12) in at least four schools that differ in location and resources. For H1 to H3
plan at least 200 students with complete data, and 400 if the weights are of interest; use `eag power` with your own
design to refine these numbers. Record gender, grade, school and prior STEM-project experience.

## 4. Materials

* A low-cost AI and IoT kit per team of three: a microcontroller with wireless connectivity, at least four sensors
  (for example temperature, light, sound and soil moisture or CO2), and access to a tool for training small machine-learning models.
* Four design challenges of rising openness: cycle 0 "When should the classroom be ventilated?" (structured), cycle 1
  "Predicting when plants need water" (guided), cycle 2 "A noise map of the school" (guided, classification), cycle 3
  an open project chosen by the team.
* The rubric in `docs/rubric.md` and anchor portfolios for rater training.

## 5. Procedure

| Weeks | Activity |
|---|---|
| 1 to 2 | Teacher workshop (two days), rater calibration with anchor portfolios, consent and assent |
| 3 to 5 | Cycle 0: Frame, Model, Sense, Infer, Refine; design review |
| 6 to 8 | Cycle 1 and design review |
| 9 to 11 | Cycle 2 and design review |
| 12 to 15 | Cycle 3 (open project) and design review |
| 16 | School fair judged by external judges |

Each cycle takes six to eight lessons. Students submit a design brief, a design log, the IoT configuration with a data
sample, and an AI notebook or model card. Individual scores are based on individual briefs and logs even when hardware is shared.

## 6. Scoring

* **Analytic scoring.** The class teacher scores the 20 indicators (0 to 3) for every student at every cycle.
* **Holistic design review.** Two raters independently give a score from 0 to 100. Use a pool of at least six raters in a
  linked design: every rater shares at least 30 portfolios with at least two other raters, and at least 20% of portfolios
  are rated across schools from anonymised digital copies.
* **Rater training.** Four hours with anchor portfolios; raters qualify when their intraclass correlation with the anchor
  scores is at least 0.70.
* **External criterion.** Three external judges per project at the school fair, using the published ISEF criteria, with no
  access to classroom scores.

## 7. Analysis plan

1. Run `eag check` on the pooled data; report every check with its verdict.
2. H1: indicator fit, reliabilities, latent correlations. Revise or drop at most 20% of indicators once, then refit.
3. H2: kappa with 95% confidence interval (and the profile-likelihood interval), likelihood-ratio test against kappa = 1,
   five-fold cross-validation, posterior predictive check, and the linearity check of the rating link.
4. H3: Spearman correlation of external scores with (a) model-based competence at the last cycle, estimated without the
   external scores, and (b) the rubric sum score; compare by bootstrap over students.
5. H4: difference in growth between feedback conditions from the growth function, per component and for competence.
6. H5: refit by subgroup; compare indicator difficulties and kappa.
7. Sensitivity: refit with the holistic ratings from the last cycle only, and with each rater left out in turn.

## 8. Readiness threshold

A panel of six to eight experienced fair judges classifies 20 anchor portfolios as ready or not ready for a regional fair.
The rating at which half of the panel judges a portfolio ready, Y*, is converted to the competence scale by
C* = (Y* - alpha) / beta using the estimated rating link.

## 9. Ethics and data

Institutional ethics approval, parental consent and student assent are required. Participation does not affect grades.
Only scores are analysed; no images or recordings are needed. Students use AI tools that comply with age restrictions and
school policy, and every team has equal access to the kit. Share de-identified score tables and the analysis script, and
cite the software version used.

## 10. Applying the template to another competence

1. Define three to seven components from theory and an expert panel.
2. Write at least three indicators per component at four levels, with shared level descriptors.
3. Collect analytic scores and independent holistic ratings on at least two occasions, with at least two raters.
4. Run the checks; revise the rubric where the evidence function fails.
5. Interpret kappa and the weights; set the readiness threshold by a standard-setting panel.
6. Report all checks, share the data, and state which hypotheses were supported.
