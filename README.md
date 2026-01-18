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

Run the hyperparameter optimizer. It will generate a DR baseline at
`outputs/checkpoints/dr_baseline.json` if missing, and write policy configs to
`outputs/checkpoints/policy_configs.json`.

```bash
uv run experiments/hyperparameter_optimizer.py
```

Generate SSP-MMC policies and surface plots (required before simulating SSP-MMC):

```bash
uv run experiments/generate_ssp_mmc_policies.py
```

SSP-MMC-FSRS policy titles include a short hash derived from the hyperparameters so
simulation results don't overwrite each other. If you change hyperparameters or
pull updates that affect policy serialization, regenerate policies before running
simulations.

Run the main simulation (generates plots and simulations under `outputs/`
and refreshes the DR baseline JSON when DR policies are simulated):

```bash
uv run experiments/simulate.py
```

The simulation summary is written to `outputs/checkpoints/simulation_results.json`.

Optional flags for the experiment runner:

```bash
uv run experiments/simulate.py --simulation-type lim_time_unlim_reviews
uv run experiments/simulate.py --policies all
uv run experiments/simulate.py --policies ssp-mmc,memrise,anki-sm-2
uv run experiments/simulate.py --policies dr,interval
uv run experiments/simulate.py --seed 123 --device cpu
```

Run the convergence checks (reads `outputs/checkpoints/policy_configs.json`; defaults assume sibling repos exist, override with flags):

```bash
uv run experiments/converge.py --help
uv run experiments/converge.py --button-usage <path> --parameters <path>
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
- Minutes per day (average): same as above, but minutes of studying are used instead. Lower is better.
- Memorized cards (average, all days): first, for each day, the number of cards memorized by that day is calculated as the sum of probabilities of recall of all cards on that day. Then, an average across all days is taken. This number cannot be greater than the deck size. Example: if you knew 10 cards by day 1, 15 cards by day 2, and 20 cards by day 3, the average is 15. Higher is better.
- Memorized/hours spent (average, all days): the value above divided by the average accumulated time spent on reviews. It's the ratio of average accumulated knowledge to average accumulated studying time. Think of it like this: "On average, by the time you have memorized X cards, you have spent Y hours on reviews, so the ratio is X/Y". Higher is better.

The reason why we look at average knowledge (number of memorized cards, aka sum of probabilities of recall of each card) across all days rather than only looking at knowledge at the end of the simulation is because the latter can be cheated. Imagine the following algorithm (duration of the simulation=5 years): first, it assigns an interval equal to 4 years and 11 months to every card. Then, during the last month, it assigns an interval equal to a few days to every card. The number of cards memorized *at the end* will be very high despite the fact that on *most* days knowledge was extremely low.

Deck size = 10000 cards, new cards per day = 10, max. reviews per day = 9999, max. studying time per day = 12 hours.

The best result is highlighted in **bold**. Arrows indicate whether lower (↓) or higher (↑) values are better.

### Duration of the simulation = 1825 days (5 years)

| Scheduling Policy | Reviews per day (average)↓ | Minutes per day (average)↓ | Memorized cards (average, all days)↑ | Memorized/hours spent (average, all days)↑ |
| --- | --- | --- | --- | --- |
| SSP-MMC-FSRS (Maximum knowledge, 2f9942e9) | 692.1 | 96.7 | **7217** | 4.6 |
| SSP-MMC-FSRS (e97e72a7) | 390.1 | 56.2 | 7169 | 8.2 |
| SSP-MMC-FSRS (a80e3a3a) | 347.8 | 51.2 | 7158 | 10.1 |
| SSP-MMC-FSRS (0b6d85f9) | 273.9 | 40.9 | 7124 | 11.7 |
| SSP-MMC-FSRS (82adc4ff) | 213.3 | 32.9 | 7098 | 14.3 |
| SSP-MMC-FSRS (8f9ad4f1) | 190.7 | 29.8 | 7034 | 14.9 |
| SSP-MMC-FSRS (a19b05ac) | 152.8 | 24.9 | 6991 | 17.7 |
| SSP-MMC-FSRS (9c5d7126) | 135.1 | 22.5 | 6908 | 18.3 |
| SSP-MMC-FSRS (2936917b) | 121.2 | 20.8 | 6894 | 20.2 |
| SSP-MMC-FSRS (e1baf345) | 112.2 | 19.7 | 6876 | 21.4 |
| SSP-MMC-FSRS (6bf66023) | 96.0 | 17.4 | 6686 | 22.3 |
| SSP-MMC-FSRS (6276af0c) | 77.1 | 15.2 | 6595 | 25.5 |
| SSP-MMC-FSRS (7146a1b8) | 69.0 | 14.5 | 6523 | 27.1 |
| SSP-MMC-FSRS (776057ab) | 68.9 | 14.3 | 6503 | 26.6 |
| SSP-MMC-FSRS (0b13bfd9) | 62.6 | 13.5 | 6403 | 27.6 |
| SSP-MMC-FSRS (1074977a) | 62.1 | 13.4 | 6392 | 27.6 |
| SSP-MMC-FSRS (Balanced, 566b6f5a) | 57.4 | 12.8 | 6280 | 28.3 |
| SSP-MMC-FSRS (bc7ec082) | 53.7 | 12.3 | 6168 | **28.6** |
| SSP-MMC-FSRS (d96feb3f) | 52.7 | 12.3 | 6141 | 28.5 |
| SSP-MMC-FSRS (44525cef) | 51.9 | 12.2 | 6105 | 28.5 |
| SSP-MMC-FSRS (Maximum efficiency, d538f86b) | 51.0 | **12.1** | 6059 | 28.5 |
| Memrise | 101.6 | 18.6 | 6710 | 22.5 |
| Anki-SM-2 | 122.2 | 20.7 | 6680 | 19.0 |
| DR=0.70 | **50.8** | 12.2 | 6052 | 28.3 |
| DR=0.71 | 52.0 | 12.3 | 6098 | 28.2 |
| DR=0.72 | 53.2 | 12.4 | 6143 | 28.2 |
| DR=0.73 | 54.6 | 12.6 | 6188 | 28.1 |
| DR=0.74 | 56.0 | 12.8 | 6233 | 28.0 |
| DR=0.75 | 57.6 | 12.9 | 6277 | 27.8 |
| DR=0.76 | 59.4 | 13.1 | 6320 | 27.6 |
| DR=0.77 | 61.2 | 13.3 | 6363 | 27.4 |
| DR=0.78 | 63.3 | 13.5 | 6406 | 27.1 |
| DR=0.79 | 65.4 | 13.8 | 6449 | 26.9 |
| DR=0.80 | 67.7 | 14.1 | 6490 | 26.5 |
| DR=0.81 | 70.1 | 14.3 | 6531 | 26.2 |
| DR=0.82 | 73.0 | 14.7 | 6572 | 25.8 |
| DR=0.83 | 75.9 | 15.0 | 6613 | 25.3 |
| DR=0.84 | 79.1 | 15.4 | 6653 | 24.9 |
| DR=0.85 | 82.9 | 15.9 | 6692 | 24.3 |
| DR=0.86 | 86.9 | 16.4 | 6732 | 23.7 |
| DR=0.87 | 91.7 | 17.0 | 6771 | 23.0 |
| DR=0.88 | 97.4 | 17.7 | 6809 | 22.2 |
| DR=0.89 | 103.7 | 18.4 | 6848 | 21.4 |
| DR=0.90 | 110.8 | 19.3 | 6886 | 20.5 |
| DR=0.91 | 119.5 | 20.4 | 6923 | 19.5 |
| DR=0.92 | 131.2 | 21.9 | 6961 | 18.3 |
| DR=0.93 | 144.4 | 23.6 | 6998 | 17.0 |
| DR=0.94 | 162.7 | 26.0 | 7035 | 15.6 |
| DR=0.95 | 187.2 | 29.2 | 7072 | 13.9 |
| DR=0.96 | 223.6 | 34.0 | 7109 | 12.1 |
| DR=0.97 | 282.5 | 41.9 | 7145 | 9.9 |
| DR=0.98 | 397.0 | 57.2 | 7181 | 7.3 |
| DR=0.99 | 728.0 | 101.5 | **7217** | 4.2 |
| Interval=7 | 1034.4 | 144.2 | 7190 | 4.2 |
| Interval=14 | 515.9 | 76.1 | 7062 | 7.6 |
| Interval=20 | 360.3 | 56.3 | 6905 | 9.8 |
| Interval=30 | 239.3 | 41.5 | 6575 | 12.3 |
| Interval=50 | 142.5 | 30.3 | 5788 | 14.6 |
| Interval=75 | 94.1 | 24.1 | 4857 | 15.6 |
| Interval=100 | 69.9 | 20.3 | 4126 | 15.9 |

It may be difficult to understand whether a certain scheduling policy is better than another based on this table. Below is a visualization:

<img width="1200" height="900" alt="Figure_1" src="https://github.com/user-attachments/assets/5ac8963e-f741-44b1-8789-b989620df096" />

"Balanced" roughly corresponds to "Number of memorized cards of DR=79% and efficiency of DR=70%".

While SSP-MMC outperforms low fixed desired retention, it struggles to outperform high fixed desired retention. Surprisingly, Memrise's simple algorithm performs remarkably well. We are unsure why.

## Caveats

Currently, the results are based on several assumptions and have several limitations:

1) FSRS is perfectly accurate at predicting the probability of recall
2) There are no same-day reviews, intervals must be at least one day long
3) There is no [fuzz](https://docs.ankiweb.net/studying.html#fuzz-factor)
4) This is based on default FSRS parameters and default review times, but of course parameters and review times vary across different users
