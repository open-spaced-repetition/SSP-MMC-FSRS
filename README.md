# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results

- Scheduling Policy: how the intervals are calculated. When SSP-MMC is used, the intervals are chosen so that the "cost" (in minutes of studying) is minimized. 
When a fixed value of desired retention (DR) is used, the intervals correspond to the desired probability of recall. IVL means that constant interval lengths - in days - are used. To account for the core hypothesis of SSP-MMC, that cards will be remembered forever once the stability exceeds a certain threshold, for both DR and IVL we allow the final interval to be shortened in order to reach the desired stability threshold on a "Good" review with the highest probability possible, and we allow cards that have reached the target stability threshold to never be scheduled again.
- Average number of reviews per day: the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards, new cards per day = 10, max. reviews per day = 9,999, max. studying time per day = 360 minutes.

The best result is highlighted in **bold**. The worst result is highlighted in $${\color{red}red}$$.

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 53.0 | 12.2 | 3292 | 271 |
| DR=0.70 | **29.6** | **9.2** | $${\color{red}2966}$$ | **324** |
| DR=0.73 | 33.3 | 9.7 | 3034 | 313 |
| DR=0.76 | 37.6 | 10.3 | 3108 | 302 |
| DR=0.79 | 42.4 | 10.9 | 3166 | 290 |
| DR=0.82 | 48.4 | 11.7 | 3225 | 277 |
| DR=0.85 | 56.2 | 12.7 | 3286 | 260 |
| DR=0.88 | 66.9 | 14.0 | 3342 | 238 |
| DR=0.91 | 83.0 | 16.1 | 3397 | 211 |
| DR=0.94 | 112.3 | 19.9 | 3449 | 173 |
| DR=0.97 | 184.7 | 29.5 | 3500 | 119 |
| IVL=3 | $${\color{red}603.3}$$ | $${\color{red}86.0}$$ | **3533** | $${\color{red}41}$$ |
| IVL=7 | 255.6 | 39.8 | 3497 | 88 |
| IVL=30 | 55.7 | 13.2 | 3200 | 242 |

### More Explanations

### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 42.1 | 8.3 | 9665 | 1159 |
| DR=0.70 | **30.4** | **7.1** | $${\color{red}8859}$$ | **1249** |
| DR=0.73 | 33.1 | 7.4 | 9075 | 1222 |
| DR=0.76 | 36.2 | 7.8 | 9255 | 1190 |
| DR=0.79 | 39.3 | 8.1 | 9421 | 1165 |
| DR=0.82 | 42.9 | 8.4 | 9572 | 1134 |
| DR=0.85 | 47.6 | 8.9 | 9704 | 1087 |
| DR=0.88 | 53.2 | 9.5 | 9814 | 1032 |
| DR=0.91 | 61.9 | 10.5 | 9902 | 944 |
| DR=0.94 | 76.9 | 12.3 | 9961 | 811 |
| DR=0.97 | 113.9 | 17.0 | 9993 | 587 |
| IVL=3 | $${\color{red}1031.3}$$ | $${\color{red}140.2}$$ | **10000** | $${\color{red}71}$$ |
| IVL=7 | 456.1 | 63.3 | 10000 | 158 |
| IVL=30 | 136.4 | 21.2 | 9983 | 470 |

SSP-MMC performs better over longer periods of time.

## Warning

Currently, the SSP-MMC matrix can be constructed for 7544 out of 9999 users (75.4%), and fails to converge for the remaining users. This indicates that the current implementation is not suitable for practical use and that further research is necessary.
