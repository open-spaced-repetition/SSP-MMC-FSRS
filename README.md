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

Run the main experiment (generates plots, simulations, and policies under `outputs/`):

```bash
uv run python script.py
```

Run the hyperparameter optimizer (saves checkpoints under `outputs/checkpoints/`):

```bash
uv run python hyperparameter_optimizer.py
```

Run the convergence checks (defaults assume sibling repos exist, override with flags):

```bash
uv run python converge.py --help
uv run python converge.py --button-usage <path> --parameters <path>
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
| SSP-MMC-FSRS (Maximum knowledge) | 512.3 | 72.9 | 7203 | 6.6 |
| SSP-MMC-FSRS | 340.5 | 49.9 | 7163 | 9.6 |
| SSP-MMC-FSRS | 336.0 | 49.3 | 7151 | 9.6 |
| SSP-MMC-FSRS | 235.3 | 36.0 | 7080 | 12.7 |
| SSP-MMC-FSRS | 225.8 | 34.5 | 7042 | 12.5 |
| SSP-MMC-FSRS | 182.7 | 29.0 | 7000 | 15.0 |
| SSP-MMC-FSRS | 164.6 | 26.7 | 6968 | 16.3 |
| SSP-MMC-FSRS | 146.9 | 24.4 | 6913 | 17.3 |
| SSP-MMC-FSRS | 137.5 | 23.2 | 6884 | 18.0 |
| SSP-MMC-FSRS | 128.5 | 22.1 | 6860 | 18.8 |
| SSP-MMC-FSRS | 121.7 | 21.2 | 6818 | 19.3 |
| SSP-MMC-FSRS | 120.4 | 21.5 | 6819 | 19.9 |
| SSP-MMC-FSRS | 104.2 | 19.1 | 6698 | 21.1 |
| SSP-MMC-FSRS | 94.1 | 18.2 | 6617 | 22.1 |
| SSP-MMC-FSRS | 92.7 | 18.1 | 6609 | 22.3 |
| SSP-MMC-FSRS | 108.1 | 19.8 | 6751 | 20.7 |
| SSP-MMC-FSRS | 90.2 | 17.7 | 6574 | 22.8 |
| SSP-MMC-FSRS | 83.4 | 16.9 | 6471 | 23.4 |
| SSP-MMC-FSRS | 80.9 | 16.5 | 6400 | 23.4 |
| SSP-MMC-FSRS | 72.8 | 15.6 | 6380 | 24.3 |
| SSP-MMC-FSRS | 67.6 | 14.9 | 6274 | 24.7 |
| SSP-MMC-FSRS | 64.6 | 14.5 | 6198 | **24.9** |
| SSP-MMC-FSRS | 65.4 | 14.7 | 6220 | **24.9** |
| SSP-MMC-FSRS | 62.8 | 14.3 | 6131 | **24.9** |
| SSP-MMC-FSRS | 62.5 | 14.4 | 6113 | 24.8 |
| SSP-MMC-FSRS (Maximum efficiency) | 61.5 | **14.2** | 6066 | 24.8 |
| Memrise | 109.1 | 20.2 | 6623 | 20.4 |
| Anki-SM-2 | 131.6 | 22.3 | 6658 | 17.6 |
| DR=0.70 | **61.2** | 14.4 | 6038 | 24.4 |
| DR=0.71 | 62.7 | 14.5 | 6084 | 24.3 |
| DR=0.72 | 63.9 | 14.6 | 6131 | 24.5 |
| DR=0.73 | 65.6 | 14.8 | 6177 | 24.4 |
| DR=0.74 | 67.3 | 15.0 | 6222 | 24.2 |
| DR=0.75 | 69.1 | 15.2 | 6267 | 24.1 |
| DR=0.76 | 71.1 | 15.4 | 6312 | 24.0 |
| DR=0.77 | 73.1 | 15.6 | 6356 | 23.8 |
| DR=0.78 | 75.3 | 15.8 | 6399 | 23.7 |
| DR=0.79 | 77.6 | 16.0 | 6442 | 23.5 |
| DR=0.80 | 80.1 | 16.3 | 6485 | 23.2 |
| DR=0.81 | 82.8 | 16.6 | 6526 | 22.9 |
| DR=0.82 | 85.7 | 16.9 | 6568 | 22.6 |
| DR=0.83 | 88.9 | 17.3 | 6609 | 22.3 |
| DR=0.84 | 92.3 | 17.7 | 6649 | 21.9 |
| DR=0.85 | 96.3 | 18.2 | 6689 | 21.5 |
| DR=0.86 | 100.6 | 18.7 | 6729 | 21.0 |
| DR=0.87 | 105.4 | 19.2 | 6769 | 20.5 |
| DR=0.88 | 111.3 | 19.9 | 6808 | 19.8 |
| DR=0.89 | 117.4 | 20.7 | 6846 | 19.2 |
| DR=0.90 | 124.7 | 21.5 | 6885 | 18.5 |
| DR=0.91 | 133.7 | 22.7 | 6922 | 17.7 |
| DR=0.92 | 145.2 | 24.1 | 6960 | 16.7 |
| DR=0.93 | 158.5 | 25.8 | 6998 | 15.7 |
| DR=0.94 | 176.8 | 28.1 | 7035 | 14.4 |
| DR=0.95 | 201.4 | 31.3 | 7072 | 13.0 |
| DR=0.96 | 237.8 | 36.1 | 7108 | 11.4 |
| DR=0.97 | 296.7 | 43.9 | 7145 | 9.4 |
| DR=0.98 | 410.8 | 59.1 | 7181 | 7.1 |
| DR=0.99 | 741.8 | 103.5 | **7217** | 4.1 |
| Interval=7 | 1034.4 | 144.7 | 7180 | 4.2 |
| Interval=14 | 515.9 | 77.0 | 7024 | 7.4 |
| Interval=20 | 360.3 | 57.6 | 6830 | 9.4 |
| Interval=30 | 239.3 | 43.3 | 6422 | 11.5 |
| Interval=50 | 142.5 | 32.1 | 5493 | 13.2 |
| Interval=75 | 94.1 | 25.6 | 4498 | 13.9 |
| Interval=100 | 69.9 | 21.2 | 3790 | 14.3 |

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
