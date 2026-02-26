import pandas as pd
import itertools
import numpy as np


### IMPORTANT NOTE: In the comments that include "Input" and "Output" you should consider mostly the *structure*
### of the input and the output, rather than the specific inputs match to the output as you understand.
### In some cases, the output is divorced from the input above, to not make filling in the function too easy.
### Focus on the API through these comments, in the sense that, ensure that the input and the output of the functions is correct
### (and of course, semantically correct according to your understanding).

steps = [1, 2, 3, 4]

def get_included_steps(row, steps):
    """
    Get included steps as a tuple for each row.
    
    Example:
        Input: row = {'step1_present': 0, 'step2_present': 1, 'step3_present': 0, 'step4_present': 1},
               steps = [1, 2, 3, 4]
        Output: (2, 4)
    """
    included_steps = []
    ##############################################################################
    ###Question 4: INSERT CODE HERE: Get included steps as a tuple for each row.###
    ##############################################################################
    raise NotImplementedError("Get included steps as a tuple for each row.")   
    return tuple(sorted(included_steps))

def generate_all_subsets(steps):
    """
    Generate all possible subsets for the included steps.

    Example:
        Input: steps = [1, 2, 3]
        Output: [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    all_subsets_included = []
    for r in range(len(steps) + 1):
        subsets_r = list(itertools.combinations(steps, r))
        all_subsets_included.extend(subsets_r)
    return all_subsets_included

def compute_v_S(df, all_subsets_included):
    """
    Compute v(S) for all subsets of included steps.

    Example:
        Input: 
            df = pd.DataFrame({'present_steps': [(), (1,), (2,), (1, 2)], 'is_correct': [0.8, 0.7, 0.6, 0.5]})
            all_subsets_included = [(), (1,), (2,), (1, 2)]
        Output: 
            v_S = {(): 0.8, (1,): 0.7, (2,): 0.6, (1, 2): 0.5}
    """
    ##################################################################################
    ###Question 5: INSERT CODE HERE: Compute v(S) for all subsets of included steps.###
    ##################################################################################
    raise NotImplementedError("Implement the code to compute v(S) for all subsets of included steps.")    
    return v_S

def compute_marginal_contributions(steps, v_S):
    """
    Compute the marginal contributions for each step.

    Example:
        Input:
            steps = [1, 2]
            v_S = {(): 0.85, (1,): 0.67, (2,): 0.72, (1, 2): 0.75}
        Output:
            Delta_sum = {1: 0.08, 2: 0.12}, valid_permutations_count = 2
    """
    permutations = list(itertools.permutations(steps))
    Delta_sum = {i: 0.0 for i in steps}
    valid_permutations_count = 0

    total_steps_set = set(steps)

    for pi in permutations:
        valid_permutation = True
        for i in steps:
            idx_i = pi.index(i)
            #############################################################################################
            ###Question 6: INSERT CODE HERE: Retrieve S_i, S_i_union_i, S_i_sorted, included_S_i_sorted,
            ### included_S_i_union_i_sorted
            #############################################################################################
            raise NotImplementedError("Implement the retrieval of S_i, S_i_union_i, and their sorted tuples.")
            v_S_i = v_S.get(included_S_i_sorted, np.nan)
            v_S_i_union_i = v_S.get(included_S_i_union_i_sorted, np.nan)
            if np.isnan(v_S_i) or np.isnan(v_S_i_union_i):
                valid_permutation = False
                break
            else:
                ###############################################################################
                ###Question 6.1: INSERT CODE HERE: Compute the marginal contribution of step i###
                ###############################################################################
                raise NotImplementedError("Implement the computation of the marginal contribution of step i.")
        if valid_permutation:
            valid_permutations_count += 1
    return Delta_sum, valid_permutations_count

def compute_shapley_values(Delta_sum, valid_permutations_count, steps):
    """
    Compute the Shapley values for each step.

    Example:
        Input: 
            Delta_sum = {1: 0.08, 2: 0.12}
            valid_permutations_count = 2
            steps = [1, 2]
        Output: 
            Shapley_values = {1: 0.04, 2: 0.06}
    """
    ##############################################################
    ###Question 7: INSERT CODE HERE: Compute the Shapley values###
    ##############################################################
    raise NotImplementedError("INSERT CODE HERE: Compute the Shapley values")
    return shapley_values

def main():
    df = pd.read_csv('evaluation_with_steps.csv')

    df['present_steps'] = df.apply(get_included_steps, axis=1, args=(steps,))

    ######################################################################################
    # Question 7.1: INSERT CODE HERE: Generate all possible subsets for the included steps#
    ######################################################################################
    ###############################################
    # Question 7.2: INSERT CODE HERE: Compute v(S)#
    ###############################################
    #############################################################
    # Question 7.3: INSERT CODE HERE: Compute the Shapley values#
    #############################################################
    raise NotImplementedError("Implement the code to generate all possible subsets for the included steps.")
    raise NotImplementedError("Implement the code to compute v(S).")
    raise NotImplementedError("Implement the code to compute the Shapley values.")
if __name__ == "__main__":
    main()
