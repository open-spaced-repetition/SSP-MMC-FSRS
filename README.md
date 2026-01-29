# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended version of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetition. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis and the memory state transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.

## Usage

Install dependencies with uv. The base install is enough to import the core package, while experiments need the extra dependencies.

```bash
uv sync
```

```bash
uv sync --extra experiments
```

Run the hyperparameter optimizer for a user (defaults to user 1). It will generate
a DR baseline at `outputs/checkpoints/user_<id>/dr_baseline.json` if missing, and
write policy configs to `outputs/checkpoints/user_<id>/policy_configs.json`.

```bash
uv run experiments/hyperparameter_optimizer.py --user-id 1
```

Run the hyperparameter optimizer across multiple users (aggregated by mean). Output
is written under `outputs/checkpoints/multi_users_<label>/`.

```bash
uv run experiments/hyperparameter_optimizer.py --user-ids 1-10 --aggregate mean
```

Generate SSP-MMC policies and surface plots for a user (required before simulating SSP-MMC).
Surface plots are written under `outputs/plots/user_<id>`.

```bash
uv run experiments/generate_ssp_mmc_policies.py --user-id 1
uv run experiments/generate_ssp_mmc_policies.py --user-id 1 --policy-configs outputs/checkpoints/multi_users_1-10/policy_configs.json
```

SSP-MMC-FSRS policy titles include a short hash derived from the hyperparameters so
simulation results don't overwrite each other. If you change hyperparameters or
pull updates that affect policy serialization, regenerate policies before running
simulations.

Run the main simulation (generates plots and simulations under
`outputs/plots/user_<id>` and `outputs/simulation/user_<id>`
and refreshes the user DR baseline JSON when DR policies are simulated):

```bash
uv run experiments/simulate.py --user-id 1
```

The simulation summary is written to `outputs/checkpoints/user_<id>/simulation_results.json`.

Optional flags for the experiment runner:

```bash
uv run experiments/simulate.py --simulation-type lim_time_unlim_reviews
uv run experiments/simulate.py --policies all
uv run experiments/simulate.py --policies ssp-mmc,memrise,anki-sm-2
uv run experiments/simulate.py --policies dr,interval
uv run experiments/simulate.py --policy-configs outputs/checkpoints/multi_users_1-10/policy_configs.json
uv run experiments/simulate.py --seed 123 --device cpu
uv run experiments/simulate.py --user-id 2
```

Run the convergence checks (by default loads per-user configs from
`outputs/checkpoints/user_<user>/policy_configs.json` and writes results to
`outputs/checkpoints/convergence_incremental_results.json` /
`outputs/checkpoints/unconverged_users.json`; pass `--policy-configs` for a shared config):

```bash
uv run experiments/converge.py --help
uv run experiments/converge.py --button-usage <path> --parameters <path>
uv run experiments/converge.py --policy-configs outputs/checkpoints/multi_users_1-10/policy_configs.json --button-usage <path> --parameters <path>
```

## Project layout

- `src/ssp_mmc_fsrs/`: core library (solver, policies, simulation, IO, config)
- `experiments/`: research scripts that drive simulations
- `outputs/`: generated plots, simulations, policies, and checkpoints
- `notebooks/`: exploratory analysis

## Scheduling policies

Scheduling Policy: how the intervals are calculated.

- SSP-MMC: the intervals are chosen so that the "cost" (in minutes of studying) is minimized. The tradeoff between time efficiency and knowledge (the number of cards memorized) can be controlled with hyperparameters. It uses values of memory stability and difficulty provided by FSRS. For every value of memory stability and difficulty, it finds the optimal next interval to minimize time spent on reviews.
- Fixed DR: fixed desired retention, meaning that the intervals correspond to the desired probability of recall, for example, 90%. This is how FSRS currently schedules reviews in Anki.
- Fixed intervals: intervals are always the same and do not depend on the rating (Again/Hard/Good/Easy) or anything else. The units are days.
- Memrise: the following sequence of intervals, in days: 1 -> 6 -> 12 -> 48 -> 96 -> 180. If the user presses Again, the interval is reset to 1, and the sequence starts all over again. If the user presses Hard, Good, or Easy, the next interval in the sequence is used.
- Anki-SM-2: Anki's version of SM-2. It's a relatively simple algorithm that keeps track of "ease" (which is based on ratings) and the length of the next interval is equal to the length of the last interval times ease; plus some bonuses and penalties related to Hard; plus some extra modifiers. Anki-specific settings (starting ease, interval modifier, etc.) are set to their default values as of Anki 24.11. 

To account for the core hypothesis of SSP-MMC, that cards will be remembered forever once the stability exceeds a certain threshold, for both DR and fixed intervals we allow the final interval to be shortened in order to reach the desired stability threshold on a "Good" review with the highest probability possible, and we allow cards that have reached the target stability threshold to never be scheduled again.

The probability that a card is recalled successfully is estimated by FSRS.

## Results

- Reviews per day (average): the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Minutes of studying per day (average): same as above, but minutes of studying are used instead. Lower is better.
- Memorized cards (average, all days): first, for each day, the number of cards memorized by that day is calculated as the sum of probabilities of recall of all cards on that day. Then, an average across all days is taken. This number cannot be greater than the deck size. Example: if you knew 10 cards by day 1, 15 cards by day 2, and 20 cards by day 3, the average is 15. Higher is better.
- Memorized/hours spent (average, all days): the value above divided by the average accumulated time spent on reviews. It's the ratio of average accumulated knowledge to average accumulated studying time. Think of it like this: "On average, by the time you have memorized X cards, you have spent Y hours on reviews, so the ratio is X/Y". Higher is better.

The reason why we look at average knowledge (number of memorized cards, aka sum of probabilities of recall of each card) across all days rather than only looking at knowledge at the end of the simulation is because the latter can be cheated. Imagine the following algorithm (duration of the simulation=5 years): first, it assigns an interval equal to 4 years and 11 months to every card. Then, during the last month, it assigns an interval equal to a few days to every card. The number of cards memorized *at the end* will be very high despite the fact that on *most* days knowledge was extremely low.

Deck size = 10000 cards, new cards per day = 10, max. reviews per day = 9999, max. studying time per day = 12 hours.

The best result is highlighted in **bold**. Arrows indicate whether lower (↓) or higher (↑) values are better.

### Duration of the simulation = 1825 days (5 years)

Results below are from user 5.

| Scheduling Policy | Reviews per day (average)↓ | Minutes of studying per day (average)↓ | Memorized cards (average, all days)↑ | Memorized/hours spent (average, all days)↑ |
| --- | --- | --- | --- | --- |
| SSP-MMC-FSRS (Maximum knowledge, 764da5a0) | 201.2 | 46.5 | 7080 | 10.0 |
| SSP-MMC-FSRS (15669e0e) | 156.2 | 38.3 | 7001 | 11.9 |
| SSP-MMC-FSRS (e2fed9af) | 200.1 | 46.1 | 7064 | 9.8 |
| SSP-MMC-FSRS (4f2a8915) | 212.6 | 48.3 | 7085 | 9.2 |
| SSP-MMC-FSRS (6bd51a08) | 305.2 | 65.2 | 7165 | 6.3 |
| SSP-MMC-FSRS (3f391790) | 325.1 | 70.3 | 7155 | 6.5 |
| SSP-MMC-FSRS (Balanced, 3c7fc4a3) | 223.2 | 50.0 | 7056 | 8.4 |
| SSP-MMC-FSRS (d5754591) | 220.8 | 49.9 | 7053 | 8.7 |
| SSP-MMC-FSRS (8ff0627f) | 189.5 | 43.5 | 6966 | 9.2 |
| SSP-MMC-FSRS (be6c4255) | 259.0 | 56.0 | 7025 | 6.7 |
| SSP-MMC-FSRS (cfe6fced) | 114.6 | 29.8 | 6807 | 13.3 |
| SSP-MMC-FSRS (4e3d11d1) | 159.9 | 38.6 | 6962 | 11.0 |
| SSP-MMC-FSRS (9344d791) | 131.9 | 32.7 | 6737 | 11.4 |
| SSP-MMC-FSRS (e99a6458) | 102.6 | 27.6 | 6807 | 14.2 |
| SSP-MMC-FSRS (23eecb8d) | 65.5 | 21.0 | 6477 | 17.9 |
| SSP-MMC-FSRS (561857c6) | 71.7 | 21.9 | 6522 | 16.8 |
| SSP-MMC-FSRS (117dafe3) | 61.3 | 20.2 | 6454 | 18.5 |
| SSP-MMC-FSRS (db46c93c) | 66.0 | 20.9 | 6516 | 17.8 |
| SSP-MMC-FSRS (bd48a41c) | 56.4 | 19.4 | 6339 | 19.0 |
| SSP-MMC-FSRS (6297b566) | 51.2 | 18.8 | 6148 | **19.4** |
| SSP-MMC-FSRS (a817ab70) | 53.3 | 19.1 | 6233 | 19.3 |
| SSP-MMC-FSRS (540514d6) | 50.0 | 18.8 | 6017 | 19.2 |
| SSP-MMC-FSRS (Maximum efficiency, 1814d95a) | **49.2** | **18.6** | 5953 | 19.2 |
| Memrise | 98.7 | 27.7 | 6753 | 15.6 |
| Anki-SM-2 | 93.7 | 26.0 | 6645 | 15.0 |
| DR=0.70 | 49.3 | 18.8 | 5946 | 18.9 |
| DR=0.71 | 50.7 | 19.0 | 6007 | 19.0 |
| DR=0.72 | 52.6 | 19.3 | 6063 | 18.9 |
| DR=0.73 | 54.5 | 19.7 | 6116 | 18.7 |
| DR=0.74 | 56.4 | 20.1 | 6170 | 18.5 |
| DR=0.75 | 58.3 | 20.4 | 6223 | 18.4 |
| DR=0.76 | 60.4 | 20.8 | 6271 | 18.2 |
| DR=0.77 | 62.7 | 21.2 | 6316 | 17.9 |
| DR=0.78 | 65.1 | 21.7 | 6360 | 17.7 |
| DR=0.79 | 67.8 | 22.1 | 6407 | 17.5 |
| DR=0.80 | 70.4 | 22.6 | 6454 | 17.3 |
| DR=0.81 | 73.3 | 23.1 | 6500 | 17.1 |
| DR=0.82 | 76.6 | 23.7 | 6547 | 16.8 |
| DR=0.83 | 79.8 | 24.2 | 6590 | 16.5 |
| DR=0.84 | 83.5 | 24.9 | 6633 | 16.2 |
| DR=0.85 | 87.6 | 25.6 | 6676 | 15.9 |
| DR=0.86 | 92.0 | 26.3 | 6718 | 15.5 |
| DR=0.87 | 97.0 | 27.2 | 6759 | 15.1 |
| DR=0.88 | 102.6 | 28.2 | 6801 | 14.7 |
| DR=0.89 | 109.0 | 29.3 | 6841 | 14.2 |
| DR=0.90 | 116.0 | 30.5 | 6881 | 13.7 |
| DR=0.91 | 124.2 | 31.9 | 6920 | 13.1 |
| DR=0.92 | 134.2 | 33.7 | 6959 | 12.5 |
| DR=0.93 | 145.7 | 35.8 | 6997 | 11.8 |
| DR=0.94 | 159.6 | 38.2 | 7035 | 11.0 |
| DR=0.95 | 176.7 | 41.3 | 7072 | 10.2 |
| DR=0.96 | 199.5 | 45.5 | 7109 | 9.2 |
| DR=0.97 | 231.4 | 51.3 | 7146 | 8.1 |
| DR=0.98 | 285.4 | 61.4 | 7182 | 6.7 |
| DR=0.99 | 425.1 | 88.1 | **7218** | 4.6 |
| Interval=7 | 1034.4 | 208.9 | 7200 | 2.9 |
| Interval=14 | 515.9 | 110.3 | 7098 | 5.3 |
| Interval=20 | 360.3 | 81.0 | 6994 | 7.0 |
| Interval=30 | 239.3 | 58.4 | 6810 | 9.2 |
| Interval=50 | 142.5 | 40.0 | 6437 | 12.4 |
| Interval=75 | 94.1 | 30.3 | 6011 | 15.0 |
| Interval=100 | 69.9 | 25.1 | 5644 | 16.9 |

It may be difficult to understand whether a certain scheduling policy is better than another based on this table.

"Balanced" roughly corresponds to "Number of memorized cards of DR=79% and efficiency of DR=70%".

While SSP-MMC outperforms low fixed desired retention, it struggles to outperform high fixed desired retention. Surprisingly, Memrise's simple algorithm performs remarkably well. We are unsure why.

## Caveats

Currently, the results are based on several assumptions and have several limitations:

1) FSRS is perfectly accurate at predicting the probability of recall
2) There are no same-day reviews, intervals must be at least one day long
3) There is no [fuzz](https://docs.ankiweb.net/studying.html#fuzz-factor)
4) This is based on default FSRS parameters and default review times, but of course parameters and review times vary across different users
