from causalinference import CausalModel
from dowhy import CausalModel


def causalinference_analysis(df, treatment, outcome, confounders):
    X = df[confounders].values  # Confounders
    D = df[treatment].values  # Treatment (high_balance)
    Y = df[outcome].values  # Outcome (term deposit subscription)
    
    # Initialize causal model
    model = CausalModel(Y, D, X)

    # Perform matching and propensity score adjustments
    model.est_via_matching()  # Matching method
    model.est_via_weighting()  # Inverse propensity weighting

    # Output matching results
    print("Causal Inference via Matching:")
    print(model.estimates['matching'])

    # Output weighting results
    print("\nCausal Inference via Weighting:")
    print(model.estimates['weighting'])
    
    return model


def dowhy_analysis(df, treatment, outcome, confounders):
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders
    )
    
    # Identify causal effect
    identified_estimand = model.identify_effect()
    print("Identified Estimand:", identified_estimand)
    
    # Estimate the causal effect using propensity score matching
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
    print(f"Causal Estimate: {estimate.value}")
    
    # Perform a refutation test (placebo treatment)
    refute_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
    print(f"Refutation Results:\n{refute_results}")
    
    return model



def refutation_test(df, model, treatment, outcome, confounders):
    refutation = model.refute_estimate(
        model.identify_effect(),
        model.estimate_effect(model.identify_effect(), method_name="backdoor.propensity_score_matching"),
        method_name="placebo_treatment_refuter"
    )
    print(f"Refutation Test Results for {treatment}:\n{refutation}")


def sensitivity_analysis(df, model, treatment, outcome, confounders):
    sensitivity = model.refute_estimate(
        model.identify_effect(),
        model.estimate_effect(model.identify_effect(), method_name="backdoor.propensity_score_matching"),
        method_name="add_unobserved_common_cause"
    )
    print(f"Sensitivity Analysis Result for {treatment}:\n{sensitivity}")
