# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended version of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetition. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.

## Scheduling policies

Scheduling Policy: how the intervals are calculated.

- SSP-MMC: the intervals are chosen so that the "cost" (in minutes of studying) is minimized. The tradeoff between time efficiency and the total knowledge (number of cards memorized) can be controlled with hyperparameters.
- Fixed DR: fixed desired retention, meaning that the intervals correspond to the desired probability of recall, for example, 90%.
- Fixed intervals: intervals are always the same and do not depend on the rating (Again/Hard/Good/Easy) or anything else.
- Memrise: the following sequence of intervals, in days: 1 -> 6 -> 12 -> 48 -> 96 -> 180. If the user presses Again, the interval is reset to 1, and sequence starts all over again. If the user presses Hard, Good or Easy, the next interval in the sequence is used.

To account for the core hypothesis of SSP-MMC, that cards will be remembered forever once the stability exceeds a certain threshold, for both DR and fixed intervals we allow the final interval to be shortened in order to reach the desired stability threshold on a "Good" review with the highest probability possible, and we allow cards that have reached the target stability threshold to never be scheduled again.

## Results

- Average number of reviews per day: the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Average number of minutes per day: same as above, but minutes of studying are used instead. Lower is better.
- Total knowledge at the end: the sum of probabilities of recall of all reviewed cards by the end of the simulation. It cannot be greater than the deck size. Higher is better.
- Knowledge per minute: a measure of learning efficiency. Higher is better.

Deck size = 10,000 cards, new cards per day = 10, max. reviews per day = 9,999, max. studying time per day = 360 minutes.

The best result is highlighted in **bold**. Arrows indicate whether lower (↓) or higher (↑) values are better.

### Duration of the simulation = 1825 days ( 5 years)

| Scheduling Policy | Total number of reviews (thousands)↓ | Total number of hours↓ | Number of memorized cards↑ | Memorized per hour↑ |
| --- | --- | --- | --- | --- |
| SSP-MMC (Maximum knowledge) | 9832 | 23098 | **9946** | 0.43 |
| SSP-MMC | 5448 | 13449 | 9920 | 0.74 |
| SSP-MMC | 4922 | 12187 | 9891 | 0.81 |
| SSP-MMC | 3299 | 8652 | 9843 | 1.14 |
| SSP-MMC | 3230 | 8422 | 9791 | 1.16 |
| SSP-MMC | 2279 | 6422 | 9714 | 1.51 |
| SSP-MMC | 2122 | 6094 | 9668 | 1.59 |
| SSP-MMC | 1806 | 5358 | 9607 | 1.79 |
| SSP-MMC | 1498 | 4709 | 9537 | 2.03 |
| SSP-MMC | 1383 | 4441 | 9489 | 2.14 |
| SSP-MMC | 1095 | 3766 | 9317 | 2.47 |
| SSP-MMC | 972 | 3427 | 9140 | 2.67 |
| SSP-MMC (Balanced) | 908 | 3285 | 9044 | 2.75 |
| SSP-MMC | 863 | 3143 | 8892 | 2.83 |
| SSP-MMC | 843 | 3092 | 8844 | 2.86 |
| SSP-MMC | 831 | 3063 | 8797 | 2.87 |
| SSP-MMC | 806 | 3022 | 8744 | 2.89 |
| SSP-MMC | 800 | 2990 | 8702 | 2.91 |
| SSP-MMC | 787 | 2949 | 8650 | 2.93 |
| SSP-MMC | 777 | 2925 | 8621 | 2.95 |
| SSP-MMC (Maximum efficiency) | 771 | **2896** | 8584 | **2.96** |
| Memrise | 1446 | 4468 | 9458 | 2.12 |
| DR=0.70 | **702** | **2838** | 8081 | 2.85 |
| DR=0.71 | 738 | 2937 | 8167 | 2.78 |
| DR=0.72 | 758 | 2971 | 8243 | 2.77 |
| DR=0.73 | 787 | 3037 | 8299 | 2.73 |
| DR=0.74 | 822 | 3121 | 8381 | 2.68 |
| DR=0.75 | 854 | 3188 | 8482 | 2.66 |
| DR=0.76 | 887 | 3259 | 8567 | 2.63 |
| DR=0.77 | 930 | 3363 | 8641 | 2.57 |
| DR=0.78 | 971 | 3455 | 8729 | 2.53 |
| DR=0.79 | 1016 | 3556 | 8785 | 2.47 |
| DR=0.80 | 1054 | 3624 | 8834 | 2.44 |
| DR=0.81 | 1096 | 3707 | 8905 | 2.40 |
| DR=0.82 | 1148 | 3819 | 8965 | 2.35 |
| DR=0.83 | 1204 | 3937 | 9035 | 2.30 |
| DR=0.84 | 1271 | 4080 | 9109 | 2.23 |
| DR=0.85 | 1335 | 4213 | 9177 | 2.18 |
| DR=0.86 | 1408 | 4359 | 9238 | 2.12 |
| DR=0.87 | 1495 | 4546 | 9287 | 2.04 |
| DR=0.88 | 1604 | 4787 | 9349 | 1.95 |
| DR=0.89 | 1727 | 5053 | 9412 | 1.86 |
| DR=0.90 | 1857 | 5329 | 9465 | 1.78 |
| DR=0.91 | 2038 | 5727 | 9524 | 1.66 |
| DR=0.92 | 2208 | 6080 | 9579 | 1.58 |
| DR=0.93 | 2465 | 6645 | 9635 | 1.45 |
| DR=0.94 | 2794 | 7369 | 9686 | 1.31 |
| DR=0.95 | 3260 | 8406 | 9741 | 1.16 |
| DR=0.96 | 3923 | 9882 | 9793 | 0.99 |
| DR=0.97 | 4994 | 12274 | 9845 | 0.80 |
| DR=0.98 | 7045 | 16865 | 9896 | 0.59 |
| DR=0.99 | 13023 | 30284 | **9946** | 0.33 |
| Interval=5 | 21160 | 48740 | 9966 | 0.20 |
| Interval=10 | 10560 | 25149 | 9925 | 0.39 |
| Interval=20 | 5260 | 13421 | 9817 | 0.73 |
| Interval=50 | 2080 | 6311 | 9377 | 1.49 |
| Interval=100 | 1020 | 3748 | 8658 | 2.31 |

## Caveats

Currently, the results are based on several assumptions and have several limitations:

1) FSRS is perfectly accurate at predicting the probability of recall
2) There are no same-day reviews, interval lengths must be at least one day
3) There is no [fuzz](https://docs.ankiweb.net/studying.html#fuzz-factor)
4) This is based on default FSRS parameters and default review times
