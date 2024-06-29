import multiprocessing
import pandas as pd
import sys
import warnings
import implementations.casualpy_functions as cpf
import matplotlib.pyplot as plt
from implementations.utils import create_dot_graph, parse_knowledge_file
from implementations.dowhy_functions import run_causal_analysis
import seaborn as sns
## For testing purposes
from implementations.utils import get_dataset_for_casualpy
def run_main():

    ########## Variables ###############################
    # Define treatment and outcome
    knowledge_file = './data/ground.truth/car-eval.knowledge.txt'
    data_file = './data/car_eval.csv'
    treatment = 'safety'  # replace with your actual treatment variable
    outcome = 'eval'    # replace with your actual outcome variable


    ######## Load and prepare data ######################

    data = pd.read_csv(data_file, sep='\t')

    # Convert categorical variables to numeric
    for col in data.columns:
        data[col] = pd.Categorical(data[col]).codes


    ######### Parse the knowledge file and create the dot graph

    # knowledge = parse_knowledge_file(knowledge_file)
    # dot_graph = create_dot_graph(knowledge)

    #####################################

    # ############ DoWhy ##############################
    # identified_estimand, estimates, refutes = run_causal_analysis(data, treatment, outcome, graph=dot_graph)
    # # Print results
    # for i, estimate in enumerate(estimates):
    #     print(f"\nEstimate {i+1}:")
    #     print(estimate)
    #     print(f"Refutation Results:")
    #     print(refutes[i])


    # ############ CausalPy ##############################

    # ############ Example for ANCOVA
    # df = get_dataset_for_casualpy('anova1')
    # result, _ = cpf.run_causalpy_ancova(df, 'post', 'pre', 'group')

    # plt.show()
    # result.summary()

    # ############ Example for Synthetic Control
    # df = get_dataset_for_casualpy('sc')
    # result, _ = cpf.run_casualpy_syntheticcontrol(df, 'actual', 0, 70, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    # plt.show()
    # result.summary()

    # ############ Example for Instrumental Variable
    # data, outcome, treatment, instruments = get_dataset_for_casualpy('iv')
    # result = cpf.run_causalpy_iv(data, outcome, treatment, instruments)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Ensure output encoding is set to UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    # Suppress warning
    warnings.filterwarnings(action='ignore')
    run_main()