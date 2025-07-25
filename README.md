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

The best result is highlighted in **bold**.

### Duration of the simulation = 365 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 64.2 | 13.9 | 3302 | 237 |
| DR=0.70 | **33.1** | **9.9** | 2898 | **293** |
| DR=0.73 | 37.9 | 10.6 | 2998 | 282 |
| DR=0.76 | 43.3 | 11.4 | 3076 | 270 |
| DR=0.79 | 49.1 | 12.1 | 3142 | 259 |
| DR=0.82 | 56.5 | 13.1 | 3203 | 244 |
| DR=0.85 | 66.0 | 14.4 | 3272 | 228 |
| DR=0.88 | 79.2 | 16.1 | 3330 | 207 |
| DR=0.91 | 100.5 | 18.9 | 3389 | 180 |
| DR=0.94 | 138.1 | 23.8 | 3443 | 144 |
| DR=0.97 | 238.8 | 37.3 | 3496 | 94 |
| IVL=3 | 603.3 | 86.8 | **3521** | 41 |
| IVL=7 | 255.7 | 40.6 | 3468 | 86 |
| IVL=30 | 55.8 | 13.4 | 3127 | 234 |

### Duration of the simulation = 3650 days

| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |
| --- | --- | --- | --- | --- |
| SSP-MMC | 60.2 | 11.5 | 9540 | 827 |
| DR=0.70 | **39.0** | **8.9** | 8605 | **968** |
| DR=0.73 | 43.3 | 9.5 | 8831 | 932 |
| DR=0.76 | 48.2 | 10.1 | 9043 | 895 |
| DR=0.79 | 53.8 | 10.8 | 9228 | 857 |
| DR=0.82 | 60.5 | 11.6 | 9394 | 811 |
| DR=0.85 | 69.1 | 12.6 | 9541 | 757 |
| DR=0.88 | 81.2 | 14.1 | 9665 | 686 |
| DR=0.91 | 100.2 | 16.5 | 9772 | 593 |
| DR=0.94 | 135.7 | 21.1 | 9866 | 468 |
| DR=0.97 | 236.7 | 34.5 | 9943 | 288 |
| IVL=3 | 2199.5 | 298.6 | **9991** | 33 |
| IVL=7 | 959.5 | 132.7 | 9981 | 75 |
| IVL=30 | 198.0 | 31.3 | 9850 | 315 |

SSP-MMC performs better over longer periods of time.

## Warning

Currently, the SSP-MMC matrix can be constructed for 7544 out of 9999 users (75.4%), and fails to converge for the remaining users. This indicates that the current implementation is not suitable for practical use and that further research is necessary.
