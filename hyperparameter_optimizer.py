import logging
import os
import signal
import warnings

import numpy as np
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from colorama import Fore, Style

from script import SSPMMCSolver
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


np.random.seed(42)

S_MIN = 0.1
S_MAX = 365 * 25
SHORT_STEP = np.log(2) / 20
LONG_STEP = 5

D_MIN = 1
D_MAX = 10
D_EPS = 0.1

R_MIN = 0.70
R_MAX = 0.99
R_EPS = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARALLEL = 8

LEARN_SPAN = 365 * 5

w = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542
]


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
        deck_size=10000,
        learn_span=LEARN_SPAN,
        s_max=S_MAX,
    )

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
    # reviews_total = review_cnt_per_day.sum()  # total number of reviews
    time_total = cost_per_day.sum() / 3600  # total time spent on reviews, hours
    memorized_total = memorized_cnt_per_day[:, -1]
    memorized_total_mean = np.mean(memorized_total)  # number of memorized cards at the end
    memorized_total_sem = np.std(memorized_total) / np.sqrt(len(memorized_total))

    memorized_per_hour = memorized_total / time_total
    memorized_per_hour_mean = np.mean(memorized_per_hour) # efficiency
    memorized_per_hour_sem = np.std(memorized_per_hour) / np.sqrt(len(memorized_per_hour))

    print('')
    print(param_dict)
    print(f'Memorized={memorized_total_mean:.1f}±{memorized_total_sem:.1f} cards')
    print(f'Average memorized/hour={memorized_per_hour_mean:.2f}±{memorized_per_hour_sem:.2f} cards/hour')
    print('')
    return {"knowledge": (memorized_total_mean, memorized_total_sem),
            "knowledge_per_hour": (memorized_per_hour_mean, memorized_per_hour_sem)}


# Custom loop
total_trials = 300
ax_seed = S_MAX

parameters = [
    {'name': 'a0', 'type': 'choice', 'values': ['no_log', 'log']},
    {'name': 'a1', 'type': 'range', 'bounds': [0.1, 10], 'log_scale': True, 'value_type': 'float', 'digits': 2},
    {'name': 'a2', 'type': 'range', 'bounds': [-1, 1], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a3', 'type': 'range', 'bounds': [-1, 1], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a4', 'type': 'range', 'bounds': [-1, 1], 'log_scale': False, 'value_type': 'float', 'digits': 2},
    {'name': 'a5', 'type': 'range', 'bounds': [-1, 1], 'log_scale': False, 'value_type': 'float', 'digits': 2},
]

# Ensure that the costs are positive
# a2 and a3 are for lapses, a4 and a5 are for successes
# parameter_constraints = ["a2 + a3 >= 0.0001", "a4 + a5 >= 0.0001"]

# Maximize total knowledge at the end and knowledge acquisition rate
objectives = {'knowledge': ObjectiveProperties(minimize=False),
              'knowledge_per_hour': ObjectiveProperties(minimize=False)}

checkpoint_filename = os.path.abspath(f'SSP-MMC_Smax={ax_seed}_params={len(parameters)}.json')

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


def print_pareto(frontier, calc_knee=True):
    print('')
    print('Pareto optimal points:')
    twod_list = []
    for number, dictionary in list(frontier.items()):
        params = dictionary[0]
        a0, a1, a2, a3, a4, a5 = params['a0'], params['a1'], params['a2'], params['a3'], params['a4'], \
            params['a5']
        knowledge, knowledge_per_hour = dictionary[1][0]['knowledge'], dictionary[1][0]['knowledge_per_hour']

        twod_list.append([a0, a1, a2, a3, a4, a5, knowledge, knowledge_per_hour])

    if calc_knee:
        x = []
        y = []
        hyperparams = []

    twod_list = sorted(twod_list, key=lambda x: x[-1])  # sort by knowledge_per_hour
    for minilist in twod_list:
        a0, a1, a2, a3, a4, a5, knowledge, knowledge_per_hour = \
            minilist[0], minilist[1], minilist[2], minilist[3], minilist[4], minilist[5], minilist[6], minilist[7]
        param_dict = {'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5}
        print(f'    parameters={param_dict}, objectives=({knowledge:.0f}, {knowledge_per_hour:.2f})')

        if calc_knee:
            x.append(knowledge)
            y.append(knowledge_per_hour)
            hyperparams.append(param_dict)

    if calc_knee:
        # Knee point calculation
        norm_x = [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]
        norm_y = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
        assert max(norm_x) == 1
        assert min(norm_x) == 0
        assert max(norm_y) == 1
        assert min(norm_y) == 0

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


printed_flag = False
if completed_trials < total_trials:
    for i in range(completed_trials+1, total_trials+1):
        # Print results when loading from a checkpoint
        if loaded_flag and not printed_flag:
            # Get experiment data
            experiment_data = exp_to_df(ax.experiment)
            # Get Pareto frontier
            frontier = ax.get_pareto_optimal_parameters()
            print_pareto(frontier)
            printed_flag = True

        # Print results after every 5 trials
        elif i%5 == 0:
            # Get experiment data
            experiment_data = exp_to_df(ax.experiment)
            # Get Pareto frontier
            frontier = ax.get_pareto_optimal_parameters()
            print_pareto(frontier)

        print(f'Starting trial {i}/{total_trials}')
        parameters, trial_index = ax.get_next_trial()
        torch.cuda.empty_cache()  # just in case
        ax.complete_trial(trial_index=trial_index, raw_data=multi_objective_function(parameters))

        # Backup after each trial
        with DelayedKeyboardInterrupt():
            ax.save_to_json_file(checkpoint_filename)

# Get Pareto frontier
frontier = ax.get_pareto_optimal_parameters()
print_pareto(frontier)

# Advantage-over-fixed-DR maximizer
def advantage_maximizer(frontier):
    twod_list_ssp_mmc = []
    
    # Anything below this is not good enough
    knowledge_threshold = 8900

    for number, dictionary in list(frontier.items()):
        params = dictionary[0]
        a0, a1, a2, a3, a4, a5 = params['a0'], params['a1'], params['a2'], params['a3'], params['a4'], \
            params['a5']
        knowledge, knowledge_per_hour = dictionary[1][0]['knowledge'], dictionary[1][0]['knowledge_per_hour']

        if knowledge >= knowledge_threshold:
            twod_list_ssp_mmc.append([{'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5}, knowledge, knowledge_per_hour])

    # I decided to hard-code it because calculating this every time is slow
    twod_list_dr = [
    [0.70, 8081, 2.85],
    [0.71, 8167, 2.78],
    [0.72, 8243, 2.77],
    [0.73, 8299, 2.73],
    [0.74, 8381, 2.68],
    [0.75, 8482, 2.66],
    [0.76, 8567, 2.63],
    [0.77, 8641, 2.57],
    [0.78, 8729, 2.53],
    [0.79, 8785, 2.47],
    [0.80, 8834, 2.44],
    [0.81, 8905, 2.40],
    [0.82, 8965, 2.35],
    [0.83, 9035, 2.30],
    [0.84, 9109, 2.23],
    [0.85, 9177, 2.18],
    [0.86, 9238, 2.12],
    [0.87, 9287, 2.04],
    [0.88, 9349, 1.95],
    [0.89, 9412, 1.86],
    [0.90, 9465, 1.78],
    [0.91, 9524, 1.66],
    [0.92, 9579, 1.58],
    [0.93, 9635, 1.45],
    [0.94, 9686, 1.31],
    [0.95, 9741, 1.16],
    [0.96, 9793, 0.99],
    [0.97, 9845, 0.80],
    [0.98, 9896, 0.59],
    [0.99, 9946, 0.33]]

    dr_differences = []
    dr_pairs = []
    for entry in twod_list_ssp_mmc:
        knowledge_differences = []
        efficiency_differences = []
        # Find two DR values
        # DR the gives the most similar total amount of memorized cards
        # And DR the gives the most similar amount of cards memorized per hour
        for dr_list in twod_list_dr:
            knowledge_diff = abs(entry[-2] - dr_list[-2])
            efficiency_diff = abs(entry[-1] - dr_list[-1])
            knowledge_differences.append(knowledge_diff)
            efficiency_differences.append(efficiency_diff)

        closest_knowledge_dr_index = knowledge_differences.index(min(knowledge_differences))
        closest_efficiency_dr_index = efficiency_differences.index(min(efficiency_differences))
        closest_knowledge_dr = twod_list_dr[closest_knowledge_dr_index][0]
        closest_efficiency_dr = twod_list_dr[closest_efficiency_dr_index][0]
        dr_differences.append(closest_knowledge_dr - closest_efficiency_dr)
        dr_pairs.append([closest_knowledge_dr, closest_efficiency_dr])

    # Find hyperparameters that correspond to the biggest difference in the two DRs
    max_diff_index = dr_differences.index(max(dr_differences))
    max_diff_hyperparams = twod_list_ssp_mmc[max_diff_index]
    max_diff_drs = dr_pairs[max_diff_index]
    print(f'Hyperparameters that provide the biggest advantage={max_diff_hyperparams[0]}')
    print(f'You get the total knowledge of DR={100*max_diff_drs[0]:.0f}%')
    print(f'You get the efficiency of DR={100*max_diff_drs[1]:.0f}%')

advantage_maximizer(frontier)
