# The EAG model

Students i = 1..N, components k = 1..K, indicators j = 1..J_k, occasions (design cycles) t, raters r.

## Evidence function

Indicator j of component k is scored x in {0, ..., m}. Given proficiency theta_ik(t),

    P(X = x | theta) = exp( x (theta - delta_j) - sum_{c<=x} tau_c ) / sum_h exp( h (theta - delta_j) - sum_{c<=h} tau_c )

with sum_j delta_j = 0 and sum_c tau_c = 0 within each component (rating scale model). Mastery is
pi = 1 / (1 + exp(-theta)); pi = 0.5 marks the midpoint of an indicator of average difficulty.

## Growth function

    theta_ik(t) = a_ik + b_ik t

The vector z_i = (a_i1, ..., a_iK, b_i1, ..., b_iK) is multivariate normal with mean mu (shifted by group) and covariance Sigma.

## Aggregation function

    C_i(t) = ( sum_k w_k pi_ik(t)^rho )^(1/rho),   w_k >= 0,  sum_k w_k = 1

rho = 1 gives the weighted arithmetic mean (sum score), rho -> 0 the weighted geometric mean, rho -> -infinity the
minimum. The compensation index is kappa = 1 / (2 - rho) for rho <= 1 and kappa = rho for rho > 1.

Holistic rating of rater r:  Y_ir(t) = alpha + beta C_i(t) - lambda_r + e,  e ~ N(0, sigma^2),  sum_r lambda_r = 0.

## Derived quantities

* Instructional leverage: dC / d theta_k = C s_k (1 - pi_k), with s_k = w_k pi_k^rho / sum_l w_l pi_l^rho.
* Readiness: P(C_i(T) >= C*), where C* = (Y* - alpha) / beta converts a rating standard Y* into a competence threshold.
* Cycles to readiness: the smallest t with C_i(t) >= C* at the student's estimated a_i and b_i.

## Estimation

1. **Rubric parameters** (delta, tau) per component by conditional maximum likelihood, which needs no assumption about the
   distribution of proficiency.
2. **Growth parameters** per component by marginal maximum likelihood. The two-dimensional integral over (a, b) is taken on
   a rectangular grid sheared along the mean occasion and refined once to match posterior widths; the prior is normalised
   on the grid. Standard errors use the outer product of per-student scores.
3. **Joint covariance** Sigma. Each student's rubric likelihood for a component is summarised by a Gaussian precision and
   natural mean. A closed-form EM for this Gaussian approximation is run to convergence, and an importance-sampling EM then
   corrects for non-Gaussian likelihood shapes (at most 15 iterations on a fixed set of 300 proposals per student, so the
   correction is partial). Proposals are a defensive mixture of a multivariate t around the Gaussian posterior
   approximation and independent t blocks around the stage-2 posteriors. A weak inverse-Wishart prior worth two
   pseudo-students, centred on independence between components, guards against singular solutions. The effective sample
   size of the importance weights is reported for every student. If it falls below 10 for any student, the importance
   sampling has broken down and the results should not be used (this happened in about 1% of simulated cohorts of
   100 students and very rarely in larger cohorts).
4. **Aggregation parameters** (w, kappa, alpha, beta, lambda, sigma) by simulated maximum likelihood over 150 resampled draws
   per student, with analytic gradients and a Berndt-Hall-Hall-Hausman ascent (L-BFGS-B as fallback). Confidence intervals
   use the outer product of scores on the log-kappa scale; the profile likelihood is available through
   `eagmodel.validate.profile_kappa`.

The sum-score hypothesis kappa = 1 is tested by a likelihood ratio (1 df) using the same draws for both models.
