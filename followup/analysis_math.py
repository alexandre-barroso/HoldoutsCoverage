"""Original finite-acquisition calculations. No data access, fitting or randomness."""
from fractions import Fraction
from math import comb

CERTIFICATES = (
    "A_STRICTLY_BETTER", "B_STRICTLY_BETTER", "EXACT_TIE",
    "A_WEAKLY_BETTER", "B_WEAKLY_BETTER", "BOTH_STRICT_ORDERINGS_FEASIBLE",
)
METRICS = (
    "estimator", "signed_error", "absolute_error", "squared_error",
    "LR_order_probability", "NB_order_probability", "tie_probability",
    "wrong_strict_order_probability", "best_regret", "worst_regret",
)


def certificate(known_difference, missing_disagreements):
    if missing_disagreements < 0:
        raise ValueError("Negative missing disagreement count")
    lo = known_difference - missing_disagreements
    hi = known_difference + missing_disagreements
    if hi < 0:
        return "A_STRICTLY_BETTER"
    if lo > 0:
        return "B_STRICTLY_BETTER"
    if lo == hi == 0:
        return "EXACT_TIE"
    if hi == 0:
        return "A_WEAKLY_BETTER"
    if lo == 0:
        return "B_WEAKLY_BETTER"
    return "BOTH_STRICT_ORDERINGS_FEASIBLE"


def exact_design(counts, k, target_n, true_difference_count,
                 total_disagreements, scale=Fraction(1)):
    """Multivariate hypergeometric law, exact integer weights and rational sums.

    counts orders categories -1,0,+1. scale is the estimator expansion factor,
    never a weight in the logical interval. The caller handles infeasible arms.
    """
    if len(counts) != 3 or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("Three nonnegative integer category counts required")
    if type(k) is not int or type(target_n) is not int or target_n <= 0:
        raise ValueError("Integer sample and positive target sizes required")
    minus, zero, plus = counts
    population = sum(counts)
    if population > target_n or not 0 <= total_disagreements <= target_n:
        raise ValueError("Eligible/target dimensions inconsistent")
    if abs(true_difference_count) > total_disagreements:
        raise ValueError("Target difference exceeds disagreements")
    remainder = total_disagreements - minus - plus
    residual_difference = true_difference_count - (plus-minus)
    if remainder < 0 or abs(residual_difference) > remainder or (residual_difference-remainder) % 2:
        raise ValueError("Binary loss categories inconsistent with target")
    scale = Fraction(scale)
    if scale < 0 or k < 0 or k > population:
        raise ValueError("Invalid scale or sample size")
    if population == 0:
        if k != 0 or total_disagreements != 0 or true_difference_count != 0 or scale != 0:
            raise ValueError("Only the empty-disagreement exact-tie case permits zero labels")
        metrics = {name:Fraction(0) for name in METRICS}
        metrics["tie_probability"] = Fraction(1)
        certificates = {name:Fraction(name == "EXACT_TIE") for name in CERTIFICATES}
        return _format(metrics, certificates, 1, 1, Fraction(0), Fraction(0), 1)
    if k == 0:
        raise ValueError("Nonempty eligible populations require positive samples")

    denominator = comb(population, k)
    weights = {name:0 for name in METRICS}
    certificates = {name:0 for name in CERTIFICATES}
    scale_num, scale_den = scale.numerator, scale.denominator
    estimate_denominator = scale_den*k
    error_denominator = estimate_denominator*target_n
    weight_sum = support_states = 0
    for a in range(min(minus, k)+1):
        for c in range(min(plus, k-a)+1):
            b = k-a-c
            if b > zero:
                continue
            weight = comb(minus,a)*comb(zero,b)*comb(plus,c)
            if not weight:
                continue
            weight_sum += weight
            support_states += 1
            observed_delta = c-a
            estimator_numerator = scale_num*observed_delta
            error_numerator = estimator_numerator*target_n - true_difference_count*estimate_denominator
            weights["estimator"] += weight*estimator_numerator
            weights["signed_error"] += weight*error_numerator
            weights["absolute_error"] += weight*abs(error_numerator)
            weights["squared_error"] += weight*error_numerator**2
            weights["LR_order_probability"] += weight*(estimator_numerator < 0)
            weights["NB_order_probability"] += weight*(estimator_numerator > 0)
            tied = estimator_numerator == 0
            wrong = estimator_numerator*true_difference_count < 0
            weights["tie_probability"] += weight*tied
            weights["wrong_strict_order_probability"] += weight*wrong
            weights["best_regret"] += weight*wrong*abs(true_difference_count)
            weights["worst_regret"] += weight*(wrong or tied)*abs(true_difference_count)
            missing = total_disagreements-a-c
            lo, hi = observed_delta-missing, observed_delta+missing
            if not lo <= true_difference_count <= hi:
                raise ArithmeticError("Oracle outside deterministic feasible interval")
            certificates[certificate(observed_delta, missing)] += weight
    if weight_sum != denominator:
        raise ArithmeticError("Hypergeometric probability mass does not sum exactly to one")
    metrics = {name:Fraction(value, denominator) for name,value in weights.items()}
    metrics["estimator"] /= estimate_denominator
    metrics["signed_error"] /= error_denominator
    metrics["absolute_error"] /= error_denominator
    metrics["squared_error"] /= error_denominator**2
    metrics["best_regret"] /= target_n
    metrics["worst_regret"] /= target_n
    certificates = {name:Fraction(value,denominator) for name,value in certificates.items()}
    mean = scale*Fraction(plus-minus,population)
    if metrics["estimator"] != mean:
        raise ArithmeticError("Estimator expectation differs from scaled category mean")
    truth = Fraction(true_difference_count,target_n)
    if metrics["signed_error"] != mean-truth:
        raise ArithmeticError("Expected signed error inconsistent")
    variance_population = Fraction(plus+minus,population)-Fraction(plus-minus,population)**2
    variance_expected = (scale**2*variance_population*Fraction(population-k,k*(population-1))
                         if population > 1 else Fraction(0))
    variance = metrics["squared_error"]-metrics["signed_error"]**2
    if variance != variance_expected or variance < 0:
        raise ArithmeticError("Finite-population variance identity failed")
    if sum(certificates.values()) != 1:
        raise ArithmeticError("Certificate probabilities do not sum to one")
    if sum(metrics[key] for key in ("LR_order_probability","NB_order_probability","tie_probability")) != 1:
        raise ArithmeticError("Order probabilities do not sum to one")
    return _format(metrics,certificates,weight_sum,denominator,mean,variance,support_states)


def _format(metrics, certificates, numerator, denominator, mean, variance, states):
    return {
        "expectation":{name:float(value) for name,value in metrics.items()},
        "expectation_rational":{name:str(value) for name,value in metrics.items()},
        "certificate_probability":{name:float(value) for name,value in certificates.items()},
        "certificate_probability_rational":{name:str(value) for name,value in certificates.items()},
        "scaled_population_mean":float(mean), "variance":float(variance),
        "scaled_population_mean_rational":str(mean),"variance_rational":str(variance),
        "probability_mass_numerator":str(numerator),"probability_mass_denominator":str(denominator),
        "probability_mass_exact":numerator == denominator,"nonzero_category_states":states,
    }


def realized_metrics(sample_difference_count, acquired_n, target_n,
                     true_difference_count, total_disagreements,
                     observed_disagreements, scale=Fraction(1)):
    if target_n <= 0 or acquired_n < 0 or observed_disagreements > acquired_n:
        raise ValueError("Invalid acquisition dimensions")
    if acquired_n == 0:
        if total_disagreements or true_difference_count or sample_difference_count or Fraction(scale) != 0:
            raise ValueError("Only empty-disagreement exact tie permits a zero-label estimate")
        estimate = Fraction(0)
    else:
        estimate = Fraction(scale)*Fraction(sample_difference_count, acquired_n)
    truth = Fraction(true_difference_count,target_n)
    error = estimate-truth
    wrong = estimate*truth < 0
    tied = estimate == 0
    missing = total_disagreements-observed_disagreements
    lo, hi = sample_difference_count-missing, sample_difference_count+missing
    if not lo <= true_difference_count <= hi:
        raise ArithmeticError("Oracle containment failed")
    return {
        "estimator":float(estimate),"signed_error":float(error),
        "absolute_error":float(abs(error)),"squared_error":float(error**2),
        "LR_order_probability":int(estimate < 0),"NB_order_probability":int(estimate > 0),
        "tie_probability":int(tied),"wrong_strict_order_probability":int(wrong),
        "best_regret":float(abs(truth) if wrong else 0),
        "worst_regret":float(abs(truth) if wrong or tied else 0),
        "known_difference_count":sample_difference_count,"missing_disagreements":missing,
        "lower_count":lo,"upper_count":hi,"lower":lo/target_n,"upper":hi/target_n,
        "certificate":certificate(sample_difference_count,missing),"oracle_containment":True,
    }


def composition_warning(target_fixed_count, selection_role, assessment_role,
                        maximum_selected_rank, assessment_start, aliases=0, target_n=1):
    """A declared sufficient condition, not arbitrary-source static analysis."""
    if target_n <= 0:
        raise ValueError("EMPTY_TARGET")
    if aliases or not 0 <= target_fixed_count <= target_n:
        raise ValueError("Invalid or inconsistent units/aliases")
    if (target_fixed_count > 0 and selection_role == assessment_role
            and maximum_selected_rank is not None and maximum_selected_rank < assessment_start):
        return "FORCED_STRATUM_ABSENCE"
    return "NO_FORCED_EXCLUSION_PROVED"
