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
| SSP-MMC | 60.8 | 16.5 | 3381 | 205 |
| DR=0.70 | **30.7** | **14.5** | $${\color{red}3055}$$ | 211 |
| DR=0.73 | 33.1 | 14.6 | 3107 | 213 |
| DR=0.76 | 36.3 | 14.7 | 3163 | 215 |
| DR=0.79 | 39.5 | 14.8 | 3203 | **216** |
| DR=0.82 | 44.0 | 15.2 | 3261 | 215 |
| DR=0.85 | 49.6 | 15.6 | 3308 | 213 |
| DR=0.88 | 57.4 | 16.3 | 3357 | 206 |
| DR=0.91 | 70.8 | 17.6 | 3405 | 193 |
| DR=0.94 | 94.3 | 20.4 | 3452 | 169 |
| DR=0.97 | 161.8 | 29.0 | 3501 | 121 |
| IVL=3 | $${\color{red}603.3}$$ | $${\color{red}89.4}$$ | **3538** | $${\color{red}40}$$ |
| IVL=7 | 255.7 | 44.9 | 3504 | 78 |
| IVL=30 | 55.8 | 21.6 | 3105 | 144 |



### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 45.2 | **11.3** | 9817 | **868** |
| DR=0.70 | **31.0** | 12.3 | $${\color{red}9257}$$ | 753 |
| DR=0.73 | 32.6 | 12.1 | 9383 | 775 |
| DR=0.76 | 34.4 | 11.8 | 9505 | 803 |
| DR=0.79 | 36.3 | 11.6 | 9610 | 828 |
| DR=0.82 | 38.6 | 11.4 | 9702 | 852 |
| DR=0.85 | 42.0 | **11.3** | 9782 | 863 |
| DR=0.88 | 46.3 | 11.4 | 9852 | 865 |
| DR=0.91 | 53.6 | 11.8 | 9910 | 841 |
| DR=0.94 | 67.4 | 13.1 | 9955 | 762 |
| DR=0.97 | 105.6 | 17.6 | 9987 | 568 |
| IVL=3 | $${\color{red}1293.8}$$ | $${\color{red}177.3}$$ | **10000** | $${\color{red}56}$$ |
| IVL=7 | 550.5 | 78.5 | 9999 | 127 |
| IVL=30 | 152.2 | 30.3 | 9969 | 329 |

SSP-MMC performs better over longer periods of time.

## Warning

Currently, the SSP-MMC matrix can be constructed for 7544 out of 9999 users (75.4%), and fails to converge for the remaining users. This indicates that the current implementation is not suitable for practical use and that further research is necessary.
