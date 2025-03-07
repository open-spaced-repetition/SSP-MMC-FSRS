# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, the intervals are chosen so that the "cost" (in minutes of studying) is minimized. 
When a fixed value of desired retention (DR) is used, the intervals correspond to the desired probability of recall. IVL means that constant interval lengths - in days - are used.
- Average number of reviews per day: the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards, new cards per day = 10, max. reviews per day = 9,999, max. studying time per day = 360 minutes.

The best result is highlighted in **bold**. The worst result is highlighted in $${\color{red}red}$$.

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 60.7 | 16.5 | 3381 | 205 |
| DR=0.70 | **30.9** | 14.6 | $${\color{red}3055}$$ | 209 |
| DR=0.73 | 32.8 | **14.5** | 3106 | 215 |
| DR=0.76 | 36.3 | 14.7 | 3161 | 215 |
| DR=0.79 | 39.7 | 14.9 | 3204 | 214 |
| DR=0.82 | 43.8 | 15.1 | 3260 | **216** |
| DR=0.85 | 48.8 | 15.3 | 3308 | **216** |
| DR=0.88 | 56.6 | 16.1 | 3356 | 209 |
| DR=0.91 | 70.2 | 17.5 | 3405 | 195 |
| DR=0.94 | 93.7 | 20.3 | 3452 | 170 |
| DR=0.97 | 162.3 | 29.1 | 3501 | 120 |
| IVL=3 | $${\color{red}603.3}$$ | $${\color{red}89.2}$$ | **3538** | $${\color{red}40}$$ |
| IVL=7 | 255.7 | 44.9 | 3503 | 78 |
| IVL=30 | 55.8 | 21.3 | 3119 | 146 |



### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 45.2 | **11.3** | 9818 | **870** |
| DR=0.70 | **29.6** | 12.0 | 8686 | 724 |
| DR=0.73 | 31.4 | 11.9 | 8904 | 748 |
| DR=0.76 | 33.4 | 11.8 | 9017 | 766 |
| DR=0.79 | 35.5 | 11.6 | 9318 | 806 |
| DR=0.82 | 38.5 | 11.5 | 9487 | 822 |
| DR=0.85 | 42.1 | 11.5 | 9642 | 838 |
| DR=0.88 | 47.2 | 11.6 | 9771 | 840 |
| DR=0.91 | 56.1 | 12.3 | 9869 | 804 |
| DR=0.94 | 71.5 | 13.7 | 9939 | 725 |
| DR=0.97 | 117.9 | 19.3 | 9984 | 518 |
| IVL=3 | $${\color{red}2383.5}$$ | $${\color{red}322.8}$$ | $${\color{red}8091}$$ | $${\color{red}25}$$ |
| IVL=7 | 1233.9 | 170.2 | **9999** | 59 |
| IVL=30 | 286.9 | 48.2 | 9970 | 207 |

SSP-MMC performs better over longer periods of time.

## Warning

Currently, the SSP-MMC matrix can be constructed for 7544 out of 9999 users (75.4%), and fails to converge for the remaining users. This indicates that the current implementation is not suitable for practical use and that further research is necessary.
