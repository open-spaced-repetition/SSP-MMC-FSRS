# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, intervals are chosen in a way that minimizes the "cost" (in minutes of studying). When a fixed value of desired retention is used, intervals correspond to the desired probability of recall.
- Average number of reviews per day: the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

The duration of the simulation is 3650 days (10 years). SSP-MMC performs better as the number of days increases.
Scheduling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute
| --- | --- | --- | --- | --- |
| SSP-MMC | 41.4 | 11.4 | 9819 | 860 |
| DR=0.70 | 31.0 | 12.5 | 8706 | 696 |
| DR=0.73 | 32.7 | 12.4 | 8928 | 721 |
| DR=0.76 | 34.4 | 12.0 | 9103 | 755 |
| DR=0.79 | 36.8 | 12.0 | 9344 | 781 |
| DR=0.82 | 39.5 | 11.8 | 9522 | 808 |
| DR=0.85 | 43.6 | 11.9 | 9668 | 814 |
| DR=0.88 | 48.0 | 11.8 | 9788 | 830 |
| DR=0.91 | 57.1 | 12.4 | 9879 | 795 |
| DR=0.94 | 73.1 | 14.0 | 9941 | 711 |
| DR=0.97 | 115.8 | 19.0 | 9980 | 525 |
