from dowhy import CausalModel
def run_causal_analysis(data, treatment, outcome, graph=None, target_units='ate', methods=["backdoor.linear_regression"]):
    """
    Run a causal analysis using the DoWhy library.
    
    Args:
        data (pandas.DataFrame): The input data.
        treatment (str): The name of the treatment variable.
        outcome (str): The name of the outcome variable.
        graph (networkx.DiGraph, optional): The causal graph, if available.
    
    Returns:
        tuple: A tuple containing the following:
            - estimates (list): A list of causal effect estimates using different methods.
        
        None: If an error is produced during the analysis.
    """

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph,
    )
    model.view_model()
    try:
        # Identify the causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand)
    except:
        print("No causal effect identified.")
        return None
    
    estimates = []
    refutes = []

    for method in methods:
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                target_units=target_units,
                confidence_intervals=True,
                method_params={"num_null_simulations": 100}
            )
            estimates.append(estimate)
            # print(f"Estimate using {method}:")
            # print(estimate)

            # Refute the results
            refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
            # print("Refutation Results:")
            # print(refute_results)
            refutes.append(refute_results)
        except Exception as e:
            print(f"Error with method {method}: {str(e)}")

    
    return identified_estimand, estimates, refutes