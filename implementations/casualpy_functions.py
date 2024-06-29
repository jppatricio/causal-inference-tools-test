############ CausalPy ##############################
import causalpy as cp
import numpy as np

def run_casualpy_syntheticcontrol(data, actual, intercept, treatment_time, covariates):
    
    # Prepare the formula
    if len(covariates) == 0:
        formula = f"{actual} ~ {intercept}"
    else:
        formula = f"{actual} ~ {intercept} + {' + '.join(covariates)}"
    print(formula)
    result = cp.pymc_experiments.SyntheticControl(
        data,
        treatment_time,
        formula=formula,
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={"target_accept": 0.95}
        ),
    )

    fig, _ = result.plot()
    return result, fig

def run_causalpy_ancova(
    data,
    outcome,
    pretreatment_variable_name,
    group_variable_name,
):
    # Prepare the formula
    formula = f"{outcome} ~ 1 + C({group_variable_name}) + {pretreatment_variable_name}"

    # Run ANCOVA
    result = cp.pymc_experiments.PrePostNEGD(
        data,
        formula,
        group_variable_name,
        pretreatment_variable_name,
        model=cp.pymc_models.LinearRegression(),
    )

    # Plot results
    fig, _ = result.plot()

    return result, fig


def run_casualpy_regressiondiscontinuity(
    data,
    outcome,
    running_variable_name,
    treatment_threshold=None,
    bandwidth=None,
    use_splines=False,
    spline_df=6,
    epsilon=0.01
):

    if treatment_threshold is None:
        treatment_threshold = data[running_variable_name].median()

    if bandwidth is None:
        bandwidth = np.inf

    # Prepare the formula
    if use_splines:
        formula = (
            f"{outcome} ~ 1 + bs({running_variable_name}, df={spline_df}) + {running_variable_name}"
        )
    else:
        formula = f"{outcome} ~ 1 + {running_variable_name} + {running_variable_name}"

    rd_result = cp.pymc_experiments.RegressionDiscontinuity(
        data,
        formula,
        running_variable_name=running_variable_name,
        model=cp.pymc_models.LinearRegression(),
        treatment_threshold=treatment_threshold,
        bandwidth=bandwidth,
        epsilon=epsilon,
    )

    rd_fig, _ = rd_result.plot()

    return rd_result, rd_fig


def run_casualpy_differenceindifferences(data, outcome, treatment, time_variable_name, group_variable_name):
    # Prepare the formula
    formula = f"{outcome} ~ 1 + {treatment} * {time_variable_name}"

    did_result = cp.pymc_experiments.DifferenceInDifferences(
        data,
        formula,
        time_variable_name=time_variable_name,
        group_variable_name=group_variable_name,
        model=cp.pymc_models.LinearRegression()
    )
    
    did_fig, _ = did_result.plot()

    return did_result, did_fig


def run_causalpy_iv(data, outcome, treatment, instruments, sample_kwargs=None):

    formula = f"{outcome} ~ 1 + {treatment}"
    instruments_formula = f"{treatment} ~ 1 + {' + '.join(instruments)}"

    iv_result = cp.pymc_experiments.InstrumentalVariable(
        instruments_data=data[[treatment, *instruments]],
        data=data[[outcome, treatment]],
        instruments_formula=instruments_formula,
        formula=formula,
        model=cp.pymc_models.InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    )

    return iv_result

