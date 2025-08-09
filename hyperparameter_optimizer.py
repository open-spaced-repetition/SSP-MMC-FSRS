import logging
import os
import signal
import warnings
import time

import numpy as np
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from colorama import Fore, Style

from script import SSPMMCSolver, LEARN_SPAN, LEARN_LIMIT_PER_DAY, REVIEW_LIMIT_PER_DAY, DECK_SIZE, PARALLEL, S_MAX, w
from simulator import (
    DEFAULT_REVIEW_COSTS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_REVIEW_RATING_PROB,
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    next_interval_torch,
    simulate
)

warnings.filterwarnings('ignore')

class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt')
        print(Fore.RED + 'Delaying KeyboardInterrupt')
        print(Style.RESET_ALL)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

review_costs = DEFAULT_REVIEW_COSTS
first_rating_prob = DEFAULT_FIRST_RATING_PROB
review_rating_prob = DEFAULT_REVIEW_RATING_PROB
first_rating_offsets = DEFAULT_FIRST_RATING_OFFSETS
first_session_lens = DEFAULT_FIRST_SESSION_LENS
forget_rating_offset = DEFAULT_FORGET_RATING_OFFSET
forget_session_len = DEFAULT_FORGET_SESSION_LEN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def simulate_policy(policy):
    (
        review_cnt_per_day,
        _,
        memorized_cnt_per_day,
        cost_per_day,
    ) = simulate(
        parallel=PARALLEL,
        w=w,
        policy=policy,
        device=DEVICE,
        deck_size=DECK_SIZE,
        learn_span=LEARN_SPAN,
        learn_limit_perday=LEARN_LIMIT_PER_DAY,
        review_limit_perday=REVIEW_LIMIT_PER_DAY,
        max_cost_perday=86400 / 2,  # 12 hours
        s_max=S_MAX)

    return review_cnt_per_day, cost_per_day, memorized_cnt_per_day


# Define a function with multiple objectives to optimize
def multi_objective_function(param_dict):
    solver = SSPMMCSolver(
        review_costs=DEFAULT_REVIEW_COSTS,
        first_rating_prob=DEFAULT_FIRST_RATING_PROB,
        review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
        first_rating_offsets=DEFAULT_FIRST_RATING_OFFSETS,
        first_session_lens=DEFAULT_FIRST_SESSION_LENS,
        forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
        forget_session_len=DEFAULT_FORGET_SESSION_LEN,
        w=w
    )

    cost_matrix, retention_matrix = solver.solve(param_dict)
    retention_matrix_tensor = torch.tensor(retention_matrix, device=DEVICE)

    def ssp_mmc_policy(s, d):
        d_index = solver.d2i_torch(d)
        s_index = solver.s2i_torch(s)
        # Handle array inputs by checking each element
        mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
        optimal_interval = torch.zeros_like(s)
        optimal_interval[~mask] = next_interval_torch(
            s[~mask], retention_matrix_tensor[d_index[~mask], s_index[~mask]], -w[20]
        )
        optimal_interval[mask] = np.inf
        return optimal_interval

    review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(ssp_mmc_policy)

    # reviews_average = review_cnt_per_day.mean()  # average number of reviews per day

    # time_average = cost_per_day.mean() / 60  # average time spent on reviews per day, minutes

    accum_cost = np.cumsum(cost_per_day, axis=-1)
    accum_time_average = accum_cost.mean() / 3600  # accumulated average time spent on reviews, hours

    memorized_average = memorized_cnt_per_day.mean()  # average of memorized cards on each day

    avg_accum_memorized_per_hour = memorized_average / accum_time_average  # efficiency

    print('')
    print(param_dict)
    print(f'Average memorized={memorized_average:.0f} cards')
    print(f'Average memorized/hours={avg_accum_memorized_per_hour:.1f} cards/hour')
    print('')
    return {'average_knowledge': (memorized_average, None),
            'average_knowledge_per_hour': (avg_accum_memorized_per_hour, None)}


# Custom loop
total_trials = 500
ax_seed = S_MAX

parameters = [
    # S ratio transformation
    {'name': 'a0', 'type': 'choice', 'values': ['no_log', 'log']},
    # S ratio power
    {'name': 'a1', 'type': 'range', 'bounds': [0.1, 10], 'log_scale': True, 'value_type': 'float', 'digits': 2},
    # D ratio power
    {'name': 'a2', 'type': 'range', 'bounds': [0.1, 10], 'log_scale': True, 'value_type': 'float', 'digits': 2},
    # constants
    {'name': 'a3', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a4', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    # coefficients for modifying cost based on S
    {'name': 'a5', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a6', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    # coefficients for modifying cost based on D
    {'name': 'a7', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a8', 'type': 'range', 'bounds': [-5, 5], 'log_scale': False, 'value_type': 'float', 'digits': 2},
]

# Maximize average knowledge on each day and knowledge acquisition rate
objectives = {'average_knowledge': ObjectiveProperties(minimize=False),
              'average_knowledge_per_hour': ObjectiveProperties(minimize=False)}

checkpoint_filename = os.path.abspath(f'SSP-MMC_Smax={ax_seed}_params={len(parameters)}_avg.json')

# Check if the checkpoint file exists and is valid
if os.path.isfile(checkpoint_filename):
    # Check file size
    file_size = os.path.getsize(checkpoint_filename)

    # Try to read the file content
    try:
        with open(checkpoint_filename, 'r') as f:
            content = f.read()
            # Check if it's valid JSON
            import json

            try:
                json_content = json.loads(content)
                # Check if it has expected Ax experiment structure
                if 'experiment' in json_content:
                    pass
                    # print("File appears to be a valid Ax experiment checkpoint")
                else:
                    print("WARNING: File doesn't contain 'experiment' key")
            except json.JSONDecodeError:
                print("ERROR: File does not contain valid JSON")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f'Checkpoint file does not exist')

loaded_flag = False
if os.path.isfile(checkpoint_filename):
    try:
        # Load the checkpoint
        print(f'Loading checkpoint from {checkpoint_filename}')
        with DelayedKeyboardInterrupt():
            ax = AxClient.load_from_json_file(checkpoint_filename)
        ax._random_seed = ax_seed  # currently ax doesn't preserve the seed because ax is a piece of junk
        assert ax._random_seed == ax_seed
        completed_trials = len(ax.experiment.trials)
        if completed_trials > 0:
            loaded_flag = True
        else:
            loaded_flag = False
        print(f'Successfully loaded experiment with {completed_trials} completed trials')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        quit()
else:
    ax = AxClient(random_seed=ax_seed, verbose_logging=False)
    ax.create_experiment(name='SSP-MMC, Bayesian search',
                         parameters=parameters,
                         objectives=objectives)
    completed_trials = 0
    ax.save_to_json_file(checkpoint_filename)

def pareto(frontier, calc_knee=False):
    print('')
    print('Pareto optimal points:')
    twod_list = []
    for number, dictionary in list(frontier.items()):
        params = dictionary[0]
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = params['a0'], params['a1'], params['a2'], params['a3'], params['a4'], \
            params['a5'], params['a6'], params['a7'], params['a8']
        average_knowledge, average_knowledge_per_hour = dictionary[1][0]['average_knowledge'], dictionary[1][0]['average_knowledge_per_hour']

        twod_list.append([a0, a1, a2, a3, a4, a5, a6, a7, a8, average_knowledge, average_knowledge_per_hour])

    # For finding the knee point
    x = []
    y = []
    hyperparams = []

    twod_list = sorted(twod_list, key=lambda x: x[-1])  # sort by average_knowledge_per_hour
    for minilist in twod_list:
        a0, a1, a2, a3, a4, a5, a6, a7, a8, average_knowledge, average_knowledge_per_hour = \
            minilist[0], minilist[1], minilist[2], minilist[3], minilist[4], minilist[5], minilist[6], minilist[7], \
                minilist[8], minilist[9], minilist[10]
        param_dict = {'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'a8': a8}
        print(f'    parameters={param_dict}, objectives=({average_knowledge:.0f}, {average_knowledge_per_hour:.1f})')

        if calc_knee:
            x.append(average_knowledge)
            y.append(average_knowledge_per_hour)
            hyperparams.append(param_dict)

    if len(x) > 2 and calc_knee:
        # Knee point calculation
        norm_x = [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]
        norm_y = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
        assert max(norm_x) == 1, f'{max(norm_x)}'
        assert min(norm_x) == 0, f'{min(norm_x)}'
        assert max(norm_y) == 1, f'{max(norm_y)}'
        assert min(norm_y) == 0, f'{min(norm_y)}'

        distances = []
        for norm_x_i, norm_y_i in zip(norm_x, norm_y):
            # Utopia point is (1, 1) if both objectives are being maximized
            distance_i = np.sqrt(np.power(norm_x_i - 1, 2) + np.power(norm_y_i - 1, 2))
            distances.append(distance_i)

        knee_index = distances.index(min(distances))
        print('Knee point:')
        print(f'    parameters={hyperparams[knee_index]}, objectives=({x[knee_index]:.0f}, '
              f'{y[knee_index]:.2f})')

    print('')

# Advantage-over-fixed-DR maximizer
# And it also has a thingy for proposing candidate hyperparameters
# I decided not to make a separate function for that. Maybe I should
# And I also added a way to print hyperparameters in a format that can be easily copy-pasted into script.py
def advantage_maximizer(frontier, propose_candidate=False, print_for_script=False):
    twod_list_ssp_mmc = []

    for number, dictionary in list(frontier.items()):
        params = dictionary[0]
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = params['a0'], params['a1'], params['a2'], params['a3'], params['a4'], \
            params['a5'], params['a6'], params['a7'], params['a8']
        average_knowledge, average_knowledge_per_hour = dictionary[1][0]['average_knowledge'], dictionary[1][0][
            'average_knowledge_per_hour']

        twod_list_ssp_mmc.append([{'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7,
                                   'a8': a8},
                                  average_knowledge, average_knowledge_per_hour])

    twod_list_ssp_mmc = sorted(twod_list_ssp_mmc, key=lambda x: x[1])  # sort by average_knowledge_per_hour

    # I decided to hard-code it because calculating this every time is slow
    # [DR, memorized cards (average), memorized per hour (average)]
    twod_list_dr = [
        [0.70, 5912, 31.0],
        [0.71, 5966, 30.3],
        [0.72, 6020, 30.3],
        [0.73, 6073, 29.7],
        [0.74, 6123, 29.3],
        [0.75, 6175, 29.0],
        [0.76, 6231, 28.5],
        [0.77, 6283, 28.0],
        [0.78, 6337, 27.5],
        [0.79, 6388, 27.0],
        [0.80, 6437, 26.6],
        [0.81, 6483, 26.2],
        [0.82, 6529, 25.6],
        [0.83, 6573, 25.0],
        [0.84, 6619, 24.4],
        [0.85, 6665, 23.8],
        [0.86, 6710, 23.1],
        [0.87, 6752, 22.3],
        [0.88, 6792, 21.4],
        [0.89, 6834, 20.4],
        [0.90, 6875, 19.5],
        [0.91, 6915, 18.3],
        [0.92, 6954, 17.4],
        [0.93, 6993, 16.0],
        [0.94, 7031, 14.5],
        [0.95, 7069, 12.9],
        [0.96, 7106, 11.1],
        [0.97, 7143, 9.0],
        [0.98, 7179, 6.7],
        [0.99, 7214, 3.8]]

    dr_differences = []
    dr_pairs = []
    # Indices of SSP-MMC configurations that do not provide more knowledge AND higher efficiency than fixed DR
    crappy_ssp_mmc_indices = []
    for ssp_mmc_list in twod_list_ssp_mmc:
        knowledge_differences = []
        efficiency_differences = []
        # Find two DR values
        # DR the gives the most similar total amount of memorized cards
        # And DR the gives the most similar amount of cards memorized per hour
        for dr_list in twod_list_dr:
            knowledge_diff = abs(ssp_mmc_list[1] - dr_list[1])
            efficiency_diff = abs(ssp_mmc_list[2] - dr_list[2])
            knowledge_differences.append(knowledge_diff)
            efficiency_differences.append(efficiency_diff)

            # Check is this SSP-MMC configuration is crappy
            if ssp_mmc_list[1] < dr_list[1] and ssp_mmc_list[2] < dr_list[2]:
                # Add its index to the list of crappy indices
                index_current = twod_list_ssp_mmc.index(ssp_mmc_list)
                if index_current not in crappy_ssp_mmc_indices:
                    crappy_ssp_mmc_indices.append(index_current)

            # print(f'DR={dr_list[0]}, average knowledge (SSP-MMC)={ssp_mmc_list[1]:.0f}, '
            #       f'average knowledge (DR)={dr_list[1]:.0f}, efficiency(SSP-MMC)={ssp_mmc_list[2]:.2f}, '
            #       f'efficiency(DR)={dr_list[2]:.2f}')

        closest_knowledge_dr_index = knowledge_differences.index(min(knowledge_differences))
        closest_efficiency_dr_index = efficiency_differences.index(min(efficiency_differences))
        closest_knowledge_dr = twod_list_dr[closest_knowledge_dr_index][0]
        closest_efficiency_dr = twod_list_dr[closest_efficiency_dr_index][0]
        dr_differences.append(closest_knowledge_dr - closest_efficiency_dr)
        dr_pairs.append([closest_knowledge_dr, closest_efficiency_dr])

    # Find hyperparameters that correspond to the biggest difference in the two DRs
    if max(dr_differences) <= 0:
        pass
    else:
        max_diff_index = dr_differences.index(max(dr_differences))
        max_diff_params = twod_list_ssp_mmc[max_diff_index]
        max_diff_drs = dr_pairs[max_diff_index]
        # Don't print if this function is called only to get the candidate
        if not propose_candidate:
            print(f'    Hyperparameters that provide the biggest advantage={max_diff_params[0]}')
            print(f'    You get the average knowledge of DR={100 * max_diff_drs[0]:.0f}%')
            print(f'    You get the efficiency of DR={100 * max_diff_drs[1]:.0f}%')
            print('')

    # For printing and then copy-pasting into script.py
    configurations = []
    if print_for_script:
        for dr_list in twod_list_dr:
            # Find a configuration that is the closest to this DR in terms of knowledge
            abs_knowledge_differences = []
            for ssp_mmc_list in twod_list_ssp_mmc:
                abs_knowledge_differences.append(abs(ssp_mmc_list[1] - dr_list[1]))

            closest_index = abs_knowledge_differences.index(min(abs_knowledge_differences))
            closest_params = twod_list_ssp_mmc[closest_index][0]

            # It's not guaranteed to add 'Balanced', which kinda sucks
            if closest_params == max_diff_params[0]:
                entry = [closest_params, 'Balanced']
            else:
                entry = [closest_params, None]

            # Do not append the same configuration more than once
            if entry not in configurations:
                configurations.append(entry)

        configurations.reverse()
        configurations[0][1] = 'Maximum knowledge'
        configurations[-1][1] = 'Maximum efficiency'
        print(f'list_of_dictionaries = {configurations}')

    if propose_candidate:
        if len(crappy_ssp_mmc_indices) == 0:  # No crappy configurations, yay!
            print('No need to manually propose a new candidate')
            return None
        worse_candidate = twod_list_ssp_mmc[min(crappy_ssp_mmc_indices)][0]
        better_candidate = twod_list_ssp_mmc[min(crappy_ssp_mmc_indices) - 1][0]
        all_keys = better_candidate.keys()

        # Average numerical values of two candidates if a0 is the same, otherwise mutate the best candidate
        if better_candidate.get('a0') == worse_candidate.get('a0'):
            strategy = 'average'
        else:
            strategy = 'mutate'

        # Ad hoc thingy for enhancing the hyperparameter optimizer to make it explore hyperparameters close to
        # good previously tested hyperparameters
        new_candidate = {}
        for key in all_keys:
            # To prevent new candidates from being the same
            np.random.seed(int(time.time()))
            if key == 'a0':
                # Use the value of the better candidate
                new_candidate.update({'a0': better_candidate.get(key)})
            else:
                if strategy == 'average':
                    better_param = better_candidate.get(key)
                    worse_param = worse_candidate.get(key)

                    random_weight_better = float(np.random.uniform(1.5, 4, 1))
                    random_weight_worse = float(np.random.uniform(0.7, 1, 1))

                    # Assign more weight to the parameter from the better candidate
                    w_avg_param = (random_weight_better * better_param + random_weight_worse * worse_param) \
                                  / (random_weight_better + random_weight_worse)
                    new_candidate.update({key: round(w_avg_param, 2)})
                elif strategy == 'mutate':
                    better_param = better_candidate.get(key)

                    mutation = float(np.random.normal(0, 0.1, 1))

                    # Apply parameter-specific clamps
                    if key in ['a1', 'a2']:
                        new_param = max(min(round(better_param * (1 + mutation), 2), 10.0), 0.1)
                    else:
                        new_param = max(min(round(better_param * (1 + mutation), 2), 5.0), -5.0)
                    new_candidate.update({key: new_param})
                else:
                    raise Exception('Unknown candidate generation strategy')

        print(f'Manually proposed new candidate: {new_candidate}')
        return new_candidate

printed_flag = False
if completed_trials < total_trials:
    for i in range(completed_trials, total_trials):
        # Print results when loading from a checkpoint
        if loaded_flag and not printed_flag:
            # Get Pareto frontier
            frontier = ax.get_pareto_optimal_parameters()
            pareto(frontier)
            advantage_maximizer(frontier)
            printed_flag = True

        # Print results after every 5 trials
        elif i > 0 and i%5 == 0:
            # Get Pareto frontier
            frontier = ax.get_pareto_optimal_parameters()
            pareto(frontier)
            advantage_maximizer(frontier)

        print(f'Starting trial {i}/{total_trials}')
        # Manually propose candidates sometimes, for better exploration
        if i >= 40 and i%10 == 0:
            frontier = ax.get_pareto_optimal_parameters()
            parameters = advantage_maximizer(frontier, propose_candidate=True)
            if parameters is not None:
                trial_indices = ax.attach_trial(parameters=parameters)
                trial_index = trial_indices[1]
            else:
                parameters, trial_index = ax.get_next_trial()
        else:
            parameters, trial_index = ax.get_next_trial()

        torch.cuda.empty_cache()  # just in case
        ax.complete_trial(trial_index=trial_index, raw_data=multi_objective_function(parameters))

        # Backup after each trial
        with DelayedKeyboardInterrupt():
            ax.save_to_json_file(checkpoint_filename)

# Get Pareto frontier
frontier = ax.get_pareto_optimal_parameters()
pareto(frontier)
advantage_maximizer(frontier, print_for_script=True)
