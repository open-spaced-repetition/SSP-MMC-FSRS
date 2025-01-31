# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, the intervals are chosen so that the "cost" (in minutes of studying) is minimized. 
When a fixed value of desired retention is used, the intervals correspond to the desired probability of recall.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards. New cards per day = 10, max. reviews per day = 9,999.

The best result is highlighted in **bold**. 

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 54.0 | 16.0 | 3362 | 210 |
| DR=0.70 | **31.1** | 14.8 | 3053 | 206 |
| DR=0.73 | 32.7 | **14.4** | 3106 | 216 |
| DR=0.76 | 36.1 | 14.7 | 3162 | 216 |
| DR=0.79 | 39.3 | 14.8 | 3204 | **217** |
| DR=0.82 | 44.6 | 15.4 | 3262 | 212 |
| DR=0.85 | 49.3 | 15.4 | 3307 | 214 |
| DR=0.88 | 57.2 | 16.3 | 3356 | 206 |
| DR=0.91 | 71.1 | 17.7 | 3406 | 193 |
| DR=0.94 | 95.0 | 20.6 | 3452 | 168 |
| DR=0.97 | 159.2 | 28.4 | **3501** | 123 |


### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 41.1 | **11.3** | 9809 | **867** |
| DR=0.70 | **29.6** | 12.0 | 8685 | 724 |
| DR=0.73 | 31.1 | 11.8 | 8918 | 756 |
| DR=0.76 | 33.5 | 11.8 | 9020 | 762 |
| DR=0.79 | 36.1 | 11.8 | 9301 | 788 |
| DR=0.82 | 39.4 | 11.9 | 9488 | 800 |
| DR=0.85 | 42.7 | 11.7 | 9641 | 824 |
| DR=0.88 | 47.1 | 11.6 | 9775 | 841 |
| DR=0.91 | 55.8 | 12.2 | 9870 | 809 |
| DR=0.94 | 71.8 | 13.8 | 9938 | 722 |
| DR=0.97 | 118.5 | 19.3 | **9983** | 516 |

SSP-MMC performs better over longer periods of time.
