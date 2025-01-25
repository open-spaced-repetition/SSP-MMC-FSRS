# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, the intervals are chosen so that the "cost" (in minutes of studying) is minimized. 
When a fixed value of desired retention is used, the intervals correspond to the desired probability of recall.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards. The best result is highlited in **bold**. New cards per day = 10, max. reviews per day = 9999.

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 53.8 | 15.9 | 3321 | **209** |
| DR=0.70 | **31.1** | 14.8 | 3031 | 205 |
| DR=0.73 | 33.6 | 14.9 | 3077 | 206 |
| DR=0.76 | 35.6 | **14.4** | 3141 | 218 |
| DR=0.79 | 40.0 | 15.0 | 3179 | 212 |
| DR=0.82 | 45.6 | 15.9 | 3228 | 203 |
| DR=0.85 | 48.7 | 15.2 | 3278 | 216 |
| DR=0.88 | 56.2 | 16.0 | 3315 | 208 |
| DR=0.91 | 72.0 | 18.0 | 3364 | 187 |
| DR=0.94 | 91.8 | 19.7 | 3393 | 172 |
| DR=0.97 | 160.5 | 28.7 | **3417** | 119 |


### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 41.0 | **11.3** | 9809 | **869** |
| DR=0.70 | **29.7** | 12.0 | 8688 | 722 |
| DR=0.73 | 31.2 | 11.8 | 8919 | 755 |
| DR=0.76 | 33.3 | 11.7 | 9023 | 769 |
| DR=0.79 | 35.8 | 11.7 | 9312 | 796 |
| DR=0.82 | 39.2 | 11.8 | 9491 | 804 |
| DR=0.85 | 42.5 | 11.6 | 9644 | 829 |
| DR=0.88 | 47.0 | 11.6 | 9774 | 843 |
| DR=0.91 | 56.0 | 12.2 | 9868 | 806 |
| DR=0.94 | 71.9 | 13.8 | 9939 | 722 |
| DR=0.97 | 118.4 | 19.3 | **9983** | 516 |

SSP-MMC performs better over longer periods of time.
