# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, the intervals are chosen so that the "cost" (in minutes of studying) is minimized. 
When a fixed value of desired retention is used, the intervals correspond to the desired probability of recall.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards. The best result is highlited in **bold**.

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 173.4 | 43.2 | 9473 | 219 |
| DR=0.70 | **111.6** | 42.7 | 8493 | 199 |
| DR=0.73 | 119.2 | 42.6 | 8734 | 205 |
| DR=0.76 | 125.8 | 41.7 | 8850 | 212 |
| DR=0.79 | 132.9 | **40.8** | 8968 | **220** |
| DR=0.82 | 147.1 | 41.3 | 9217 | 223 |
| DR=0.85 | 164.2 | 42.4 | 9289 | 219 |
| DR=0.88 | 188.9 | 44.1 | 9440 | 214 |
| DR=0.91 | 229.5 | 48.0 | 9588 | 200 |
| DR=0.94 | 300.7 | 55.4 | 9702 | 175 |
| DR=0.97 | 523.6 | 83.8 | **9854** | 118 |

### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 39.0 | 10.1 | 9831 | **976** |
| DR=0.70 | **31.3** | 12.0 | 8748 | 731 |
| DR=0.73 | 32.6 | 11.6 | 8963 | 770 |
| DR=0.76 | 34.0 | 11.2 | 9198 | 818 |
| DR=0.79 | 35.9 | 11.0 | 9383 | 855 |
| DR=0.82 | 39.0 | 10.9 | 9558 | 874 |
| DR=0.85 | 42.5 | 10.9 | 9693 | 892 |
| DR=0.88 | 47.2 | **10.8** | 9806 | 905 |
| DR=0.91 | 55.3 | 11.3 | 9893 | 876 |
| DR=0.94 | 69.3 | 12.4 | 9955 | 802 |
| DR=0.97 | 115.4 | 18.0 | **9989** | 556 |

SSP-MMC performs better over longer periods of time.
