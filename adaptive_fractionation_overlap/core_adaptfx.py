"""
This is the core of adaptive fractionation that computes the optimal dose for each fraction
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd

from .helper_functions import (
    std_calc,
    get_state_space,
    probdist,
    max_action,
    penalty_calc_single,
    penalty_calc_matrix,
    penalty_calc_single_volume,
    benefit_calc_single,
    benefit_calc_single_volume,
    benefit_calc_matrix
)


def adaptive_fractionation_core(fraction: int, volumes: np.ndarray, accumulated_dose: float, steepness_penalty: float, steepness_benefit: float, number_of_fractions: int = 5, min_dose: float = 7.5, max_dose: float = 9.5, mean_dose:float  = 8, dose_steps: float = 0.25, alpha: float = 0.8909285040669036, beta:float = 0.522458969735114, minimum_benefit = 0):
    """The core function computes the optimal dose for a single fraction.
    The function optimizes the fractionation based on an objective function
    which aims to maximize the tumor coverage, i.e. minimize the dose when
    PTV-OAR overlap is large and maximize the dose when the overlap is small.

    Args:
        fraction (int): number of actual fraction
        volumes (np.ndarray): list of all volume overlaps observed so far
        accumulated_dose (float): accumulated physical dose in tumor
        number_of_fractions (int, optional): number of fractions given in total. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.
        alpha (float, optional): alpha value of gamma distribution. Defaults to 1.8380125313579265.
        beta (float, optional): beta value of gamma distribution. Defaults to 0.2654168553532238.

    Returns:
        numpy arrays and floats: returns 9 arrays: policies (all future policies), policies_overlap (policies of the actual overlaps),
        volume_space (all considered overlap volumes), physical_dose (physical dose to be delivered in the actual fraction),
        penalty_added (penalty added in the actual fraction if physical_dose is applied), values (values of all future fractions. index 0 is the last fraction),
        probabilits (probability of each overlap volume to occure), final_penalty (projected final penalty starting from the actual fraction)
    """
    steepness_penalty = np.abs(steepness_penalty)
    steepness_benefit = np.abs(steepness_benefit)
    goal = number_of_fractions * mean_dose #dose to be reached
    actual_volume = volumes[-1]
    if fraction == 1:
        accumulated_dose = 0
    minimum_future = accumulated_dose + min_dose 
    std = std_calc(volumes, alpha, beta)
    distribution = norm(loc = volumes.mean(), scale = std)
    volume_space = get_state_space(distribution)
    probabilities = probdist(distribution,volume_space) #produce probabilities of the respective volumes
    volume_space = volume_space.clip(0) #clip the volume space to 0cc as negative volumes do not exist
    dose_space = np.arange(min_dose * (fraction-1),goal, dose_steps) #spans the dose space delivered to the tumor
    dose_space_adapted = dose_space.copy()
    dose_space = np.concatenate((dose_space, [goal, goal + 0.05])) # add an additional state that overdoses and needs to be prevented
    bound = goal + 0.05
    delivered_doses = np.arange(min_dose,max_dose + 0.01,dose_steps) #spans the action space of all deliverable doses
    policies_overlap = np.zeros(len(volume_space))
    values = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space))) # 2d values list with first index being the accumulated dose and second being the overlap volume
    policies = np.zeros(((number_of_fractions - fraction), len(dose_space), len(volume_space)))
    if goal - accumulated_dose < (number_of_fractions + 1 - fraction) * min_dose:
        actual_policy = min_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - fraction), len(dose_space), len(volume_space))) * -1000000000000 
        actual_value = np.ones(1) * -1000000000000
    elif goal - accumulated_dose > (number_of_fractions + 1 - fraction) * max_dose:
        actual_policy = max_dose
        policies = np.ones(200)*actual_policy
        policies_overlap = np.ones(200)*actual_policy
        values = np.ones(((number_of_fractions - fraction), len(dose_space), len(volume_space))) * -1000000000000 
        actual_value = np.ones(1) * -1000000000000
    else:
        for state, fraction_state in enumerate(np.arange(number_of_fractions, fraction-1, -1)):
            if (state == number_of_fractions - 1):  # first fraction with no prior dose delivered so we dont loop through dose_space
                overlap_penalty = penalty_calc_matrix(delivered_doses, volume_space, min_dose, mean_dose, steepness_penalty) #This means only values over min_dose get a penalty. Values below min_dose do not get a reward
                actual_penalty = penalty_calc_single_volume(delivered_doses, min_dose, mean_dose, actual_volume, steepness_penalty)
                overlap_benefit = benefit_calc_matrix(delivered_doses, volume_space, mean_dose, steepness_benefit)
                actual_benefit = benefit_calc_single_volume(delivered_doses, mean_dose, actual_volume, steepness_benefit)
                future_values_func = interp1d(dose_space, (values[state - 1] * probabilities).sum(axis=1))
                future_values = future_values_func(delivered_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions
                values_actual_frac = -overlap_penalty + future_values + overlap_benefit
                policies_overlap = delivered_doses[values_actual_frac.argmax(axis = 1)]
                actual_value = -actual_penalty + future_values + actual_benefit
                actual_policy = delivered_doses[actual_value.argmax()]
                values_actual_frac = values_actual_frac.max(axis = 1) # this is the value of the actual fraction

            elif (fraction_state == fraction and fraction != number_of_fractions):  # actual fraction but not first fraction
                dose_space_adapted = np.arange(min_dose * (fraction-1),goal, dose_steps)
                values_actual_frac = np.zeros((len(dose_space_adapted), len(volume_space)))  # this is the value of the actual fraction
                policies_overlap = np.zeros((len(dose_space_adapted), len(volume_space)))  # this is the policy of the actual fraction
                for indexer, poss_accum_doses in enumerate(dose_space_adapted):  # loop through all possible accumulated doses
                    delivered_doses_clipped = delivered_doses[0 : max_action(poss_accum_doses, delivered_doses, goal)+1]
                    overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose, mean_dose, steepness_penalty) #This means only values over min_dose get a penalty.
                    actual_penalty = penalty_calc_single_volume(delivered_doses_clipped, min_dose, mean_dose, actual_volume, steepness_penalty)
                    overlap_benefit = benefit_calc_matrix(delivered_doses_clipped, volume_space, mean_dose, steepness_benefit)
                    actual_benefit = benefit_calc_single_volume(delivered_doses_clipped, mean_dose, actual_volume, steepness_benefit)
                    future_doses = poss_accum_doses + delivered_doses_clipped
                    future_doses[future_doses > goal] = bound
                    penalties = np.zeros(future_doses.shape)
                    penalties[future_doses > goal] = -1000000000000
                    future_values_func = interp1d(dose_space, (values[state - 1] * probabilities).sum(axis=1))
                    future_values = future_values_func(future_doses)  # for each dose and volume overlap calculate the penalty of the action and add the future value. We will only have as many future values as we have doses (not volumes dependent)
                    values_actual_frac_short = -overlap_penalty + future_values + penalties + overlap_benefit
                    policies_overlap_short = delivered_doses_clipped[values_actual_frac_short.argmax(axis = 1)]
                    actual_value =-actual_penalty + future_values + penalties + actual_benefit
                    policies_overlap[indexer] = policies_overlap_short
                    values_actual_frac[indexer] = values_actual_frac_short.max(axis = 1)  # this is the value array of the actual fraction
                delivered_doses_clipped = delivered_doses[0 : max_action(accumulated_dose, delivered_doses, goal)+1]
                overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose, mean_dose, steepness_penalty) #This means only values over min_dose get a penalty.
                actual_penalty = penalty_calc_single_volume(delivered_doses_clipped, min_dose, mean_dose, actual_volume, steepness_penalty)
                overlap_benefit = benefit_calc_matrix(delivered_doses_clipped, volume_space, mean_dose, steepness_benefit)
                actual_benefit = benefit_calc_single_volume(delivered_doses_clipped, mean_dose, actual_volume, steepness_benefit)
                future_doses = accumulated_dose + delivered_doses_clipped
                future_doses[future_doses > goal] = bound
                penalties = np.zeros(future_doses.shape)
                penalties[future_doses > goal] = -1000000000000
                future_values_func = interp1d(dose_space, (values[state - 1] * probabilities).sum(axis=1))
                future_values = future_values_func(future_doses)  # for each dose and volume overlap calculate the penalty of the action and add the future value. We will only have as many future values as we have doses (not volumes dependent)
                actual_value =-actual_penalty + future_values + penalties + actual_benefit 
                actual_policy = delivered_doses_clipped[actual_value.argmax()]


        
            elif (fraction == number_of_fractions):  #actual fraction is also the final fraction we do not need to calculate any penalty as the last action is fixed. 
                dose_space_adapted = np.arange(min_dose * (fraction-1),goal, dose_steps)
                values_actual_frac = np.zeros((len(dose_space_adapted), len(volume_space)))  # this is the value of the actual fraction
                policies_overlap = np.zeros((len(dose_space_adapted), len(volume_space)))  # this is the policy of the actual fraction
                for indexer, poss_accum_doses in enumerate(dose_space_adapted):  # loop through all possible accumulated doses
                    best_action = goal - poss_accum_doses
                    if poss_accum_doses > goal:
                        best_action = 0
                        actual_value = -1000000000000
                    if best_action < min_dose:
                        best_action = min_dose
                        actual_value = -1000000000000
                    if best_action > max_dose:
                        best_action = max_dose
                        actual_value = -1000000000000
                    else:
                        actual_value = 0
                    actual_policy = best_action
                    policies_overlap[indexer] = actual_policy
                    values_actual_frac[indexer] = actual_value
                best_action = goal - poss_accum_doses
                if accumulated_dose > goal:
                    best_action = 0
                    actual_value = -1000000000000
                if best_action < min_dose:
                    best_action = min_dose
                    actual_value = -1000000000000
                if best_action > max_dose:
                    best_action = max_dose
                    actual_value = -1000000000000
                else:
                    actual_value = 0
                actual_policy = best_action
        
            else: #any fraction that is not the actual one
                future_value_prob = (values[state - 1] * probabilities).sum(axis=1)
                future_values_func = interp1d(dose_space, future_value_prob)
                for tumor_index, tumor_value in enumerate(dose_space):  # this and the next for loop allow us to loop through all states
                    delivered_doses_clipped = delivered_doses[0 : max_action(tumor_value, delivered_doses, goal)+1]  # we only allow the actions that do not overshoot
                    overlap_penalty = penalty_calc_matrix(delivered_doses_clipped, volume_space, min_dose, mean_dose, steepness_penalty) #This means only values over min_dose get a penalty.
                    overlap_benefit = benefit_calc_matrix(delivered_doses_clipped, volume_space, mean_dose, steepness_benefit)
                    if state != 0:
                        future_doses = tumor_value + delivered_doses_clipped
                        future_doses[future_doses > goal] = bound #all overdosing doses are set to the penalty state
                        future_values = future_values_func(future_doses)  # for each action and sparing factor calculate the penalty of the action and add the future value we will only have as many future values as we have actions (not sparing dependent)
                        penalties = np.zeros(future_doses.shape)
                        penalties[future_doses > goal] = -1000000000000
                        vs = -overlap_penalty + future_values + penalties + overlap_benefit
                        best_action = delivered_doses_clipped[vs.argmax(axis=1)]
                        valer = vs.max(axis=1)

                    else:  # last fraction when looping, only give the final penalty
                        best_action = goal - tumor_value
                        if best_action > max_dose:
                            best_action = max_dose
                        if best_action < min_dose:
                            best_action = min_dose
                        future_accumulated_dose = tumor_value + best_action
                        last_penalty = penalty_calc_single(best_action, min_dose, mean_dose, volume_space, steepness_penalty)
                        last_benefit = benefit_calc_single(best_action, mean_dose, volume_space, steepness_benefit)
                        underdose_penalty = 0
                        overdose_penalty = 0
                        if np.round(future_accumulated_dose,2) < goal:
                            underdose_penalty = -1000000000000 #in theory one can change this such that underdosing is penalted linearly
                        if np.round(future_accumulated_dose,2) > goal:
                            overdose_penalty = -1000000000000 
                        valer = (- last_penalty + last_benefit + underdose_penalty * np.ones(volume_space.shape) + overdose_penalty * np.ones(volume_space.shape))  # gives the value of each action for all sparing factors. elements 0-len(sparingfactors) are the Values for
                    policies[state][tumor_index] = best_action
                    values[state][tumor_index] = valer
                
    physical_dose = np.round(actual_policy,2)
    penalty_added = penalty_calc_single(physical_dose, min_dose, mean_dose, actual_volume, steepness_penalty)
    benefit_added = benefit_calc_single(physical_dose, mean_dose, actual_volume, steepness_benefit)
    final_penalty = np.max(actual_value) - penalty_added + benefit_added
    return [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty, values_actual_frac, dose_space_adapted]
    
   
def adaptfx_full(volumes: list, number_of_fractions: int = 5, steepness_penalty: float = -0.5, steepness_benefit: float = -0.1, min_dose: float = 7.5, max_dose: float = 9.5, mean_dose: float = 8, dose_steps: float = 0.25, alpha: float = 0.8909285040669036, beta:float = 0.522458969735114, minimum_benefit: float = 0):
    """Computes a full adaptive fractionation plan when all overlap volumes are given.

    Args:
        volumes (list): list of all volume overlaps observed
        number_of_fractions (float, optional): number of fractions delivered. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.

    Returns:
        numpy arrays: physical dose (array with all optimal doses to be delivered),
        accumullated_doses (array with the accumulated dose in each fraction),
        total_penalty (final penalty after fractionation if all suggested doses are applied)
    """
    steepness_penalty = np.abs(steepness_penalty)
    steepness_benefit = np.abs(steepness_benefit)
    physical_doses = np.zeros(number_of_fractions)
    accumulated_doses = np.zeros(number_of_fractions)
    for index, frac in enumerate(range(1,number_of_fractions +1)):
        if frac != number_of_fractions:
            [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty,values_actual_frac, dose_space_adapted ]  = adaptive_fractionation_core(fraction = frac, volumes = volumes[:-number_of_fractions+frac], accumulated_dose = accumulated_doses[index], steepness_penalty = steepness_penalty, steepness_benefit= steepness_benefit, number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta, minimum_benefit = minimum_benefit)
            accumulated_doses[index+1] = accumulated_doses[index] + physical_dose
        else:
            [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty, values_actual_frac, dose_space_adapted]  = adaptive_fractionation_core(fraction = frac, volumes = volumes,accumulated_dose = accumulated_doses[index], steepness_penalty = steepness_penalty, steepness_benefit = steepness_benefit, number_of_fractions= number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta, minimum_benefit = minimum_benefit)
        physical_doses[index] = physical_dose
    total_penalty = 0
    for index, dose in enumerate(physical_doses):
        print(index, dose)
        print('benefit',benefit_calc_single(dose, mean_dose, volumes[-number_of_fractions+index] , steepness_benefit))
        total_penalty += benefit_calc_single(dose, mean_dose, volumes[-number_of_fractions+index] , steepness_benefit)
        print('penalty',penalty_calc_single(dose, min_dose, mean_dose, volumes[-number_of_fractions+index], steepness_penalty))
        total_penalty -= penalty_calc_single(dose, min_dose, mean_dose, volumes[-number_of_fractions+index], steepness_penalty)
    return physical_doses, accumulated_doses, total_penalty


def precompute_plan(fraction: int, volumes: np.ndarray, accumulated_dose: float, number_of_fractions: int = 5, steepness_penalty: float = -0.5, steepness_benefit: float = -0.1, min_dose: float = 7.5, max_dose: float = 9.5, mean_dose:float  = 8, dose_steps = 0.25, alpha: float = 0.8909285040669036, beta:float = 0.522458969735114, minimum_benefit:float = 0):
    """Precomputes all possible delivered doses in the next fraction by looping through possible
    observed overlap volumes. Returning a df and two lists with the overlap volumes and
    the respective dose that would be delivered.

    Args:
        fraction (int): number of actual fraction
        volumes (np.ndarray): list of all volume overlaps observed so far
        accumulated_dose (float): accumulated physical dose in tumor
        number_of_fractions (int, optional): number of fractions given in total. Defaults to 5.
        min_dose (float, optional): minimum phyical dose delivered in each fraction. Defaults to 7.5.
        max_dose (float, optional): maximum dose delivered in each fraction. Defaults to 9.5.
        mean_dose (int, optional): mean dose to be delivered over all fractions. Defaults to 8.
        alpha (float, optional): alpha value of gamma distribution. Defaults to 1.8380125313579265.
        beta (float, optional): beta value of gamma distribution. Defaults to 0.2654168553532238.

    Returns:
        pd.Dataframe, lists: Returns a dataframe with volumes and respective doses, and volumes and doses separated in two lists.
    """
    std = std_calc(volumes, alpha, beta)
    distribution = norm(loc = volumes.mean(), scale = std)
    volume_space = get_state_space(distribution)
    distribution_max = 6.5 if volume_space.max() < 6.5 else volume_space.max()
    volumes_to_check = np.arange(0,distribution_max,0.1)
    predicted_policies = np.zeros(len(volumes_to_check))
    for index, volume in enumerate(volumes_to_check):
        [policies, policies_overlap, volume_space, physical_dose, penalty_added, values, dose_space, probabilities, final_penalty] = adaptive_fractionation_core(fraction = fraction, volumes = np.append(volumes,volume), accumulated_dose = accumulated_dose, steepness_penalty = steepness_penalty, steepness_benefit = steepness_benefit, number_of_fractions = number_of_fractions, min_dose = min_dose, max_dose = max_dose, mean_dose = mean_dose, dose_steps = dose_steps, alpha = alpha, beta = beta, minimum_benefit = minimum_benefit)
        predicted_policies[index] = physical_dose
    data = {'volume': volumes_to_check,
            'dose': predicted_policies}
    volume_x_dose = pd.DataFrame(data)
    return volume_x_dose, volumes_to_check, predicted_policies
    