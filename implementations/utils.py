import causalpy as cp
import numpy as np
import pandas as pd

def create_dot_graph(knowledge_dict, print_graph=True):
    """
    Generates a Graphviz dot graph representation of the knowledge structure defined in the `knowledge_dict` dictionary.
    
    The generated dot graph includes:
    - Nodes for each variable defined in the knowledge structure
    - Edges between nodes respecting the temporal order of the tiers
    - Invisible edges within tiers marked with an asterisk to prevent edges within the same tier
    - Dashed red edges for forbidden direct dependencies
    - Solid edges for required direct dependencies
    
    Args:
        knowledge_unparsed (dict): A dictionary representing the knowledge structure, as parsed from a configuration file.
    
    Returns:
        str: A string containing the Graphviz dot graph representation of the knowledge structure.
    """
    dot_graph = "digraph {\n"
    edges = set()
    tiers = {}
    
    if 'knowledge' in knowledge_dict:
        for section in knowledge_dict['knowledge']:
            if section[0] == 'addtemporal':
                continue
            
            tier_num = section[0].rstrip('*')
            variables = section[1:]
            tiers[int(tier_num)] = variables
            
            # If tier has an asterisk, add constraints for no edges within tier
            if section[0].endswith('*'):
                for var1 in variables:
                    for var2 in variables:
                        if var1 != var2:
                            dot_graph += f"    {var1} -> {var2} [style=invis];\n"
        
        # Create edges
        for i in range(1, len(tiers)+1):
            for j in range(i+1, len(tiers)+1):
                for var_from in tiers[i]:
                    for var_to in tiers[j]:
                        edges.add((var_from, var_to))
    
    # Process forbiddirect
    if 'forbiddirect' in knowledge_dict:
        for edge in knowledge_dict['forbiddirect']:
            if len(edge) == 2:
                edges[(edge[0], edge[1])] = '[style=dashed, color=red, constraint=false]'
    
    # Process requiredirect
    if 'requiredirect' in knowledge_dict:
        for edge in knowledge_dict['requiredirect']:
            if len(edge) == 2:
                edges.add((edge[0], edge[1]))
    
    # Add all edges to the dot graph
    for edge in edges:
        dot_graph += f"    {edge[0]} -> {edge[1]};\n"
    
    dot_graph += "}"

    # Create the graph
    if print_graph:
        print()
    return dot_graph

def parse_knowledge_file(file_path):
    """
    Parses a knowledge file at the given file path and returns a dictionary of causal relations.
    
    The knowledge file is expected to have the following format:
    - Lines starting with '/' denote a new section, with the section name following the '/'
    - Lines containing 'forbiddirect' or 'requiredirect' denote special sections for forbidden and required direct edges
    - All other lines are parsed as space-separated lists of variables, which represent causal relations
    
    The returned dictionary maps section names to lists of causal relations, where each causal relation is a list of variable names.
    
    Args:
        file_path (str): The path to the knowledge file to parse.
    
    Returns:
        dict: A dictionary of causal relations, where the keys are section names and the values are lists of causal relations.
    """
    causal_relations = {}
    current_section = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('/'):
                current_section = line[1:]
                causal_relations[current_section] = []
            elif line == 'forbiddirect' or line == 'requiredirect':
                current_section = line
                causal_relations[current_section] = []
            elif line and current_section:
                causal_relations[current_section].append(line.split())
    
    # Remove any empty lists
    causal_relations = {k: v for k, v in causal_relations.items() if v}
    
    return causal_relations

def get_dataset_for_casualpy(dataName):
    if(dataName == 'iv'):
        N = 100
        e1 = np.random.normal(0, 3, N)
        e2 = np.random.normal(0, 1, N)
        Z = np.random.uniform(0, 1, N)

        ## Ensure the endogeneity of the the treatment variable
        X = -1 + 4 * Z + e2 + 2 * e1
        y = 2 + 3 * X + 3 * e1

        return {
            'data': pd.DataFrame({"y": y, "X": X, "Z": Z}),
            'outcome': 'y',
            'treatment': 'X',
            'instruments': ['Z'],
        }
    
    return cp.load_data(dataName)