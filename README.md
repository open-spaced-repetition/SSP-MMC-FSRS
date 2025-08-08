# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended version of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetition. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis and the memory state transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.

## Scheduling policies

Scheduling Policy: how the intervals are calculated.

- SSP-MMC: the intervals are chosen so that the "cost" (in minutes of studying) is minimized. The tradeoff between time efficiency and the total knowledge (number of cards memorized) can be controlled with hyperparameters. It uses values of memory stability and difficulty provided by FSRS. For every value of memory stability and difficulty, it finds the optimal next interval to minimize time spent on reviews.
- Fixed DR: fixed desired retention, meaning that the intervals correspond to the desired probability of recall, for example, 90%. This is how FSRS currently schedules reviews in Anki.
- Fixed intervals: intervals are always the same and do not depend on the rating (Again/Hard/Good/Easy) or anything else. The units are days.
- Memrise: the following sequence of intervals, in days: 1 -> 6 -> 12 -> 48 -> 96 -> 180. If the user presses Again, the interval is reset to 1, and the sequence starts all over again. If the user presses Hard, Good, or Easy, the next interval in the sequence is used.
- Anki-SM-2: Anki's version of SM-2. It's a relatively simple algorithm that keeps track of "ease" (which is based on ratings) and the length of the next interval is equal to the length of the last interval times ease; plus some bonuses and penalties related to Hard; plus some extra modifiers. Anki-specific settings (starting ease, interval modifier, etc.) are set to their default values as of Anki 24.11. 

To account for the core hypothesis of SSP-MMC, that cards will be remembered forever once the stability exceeds a certain threshold, for both DR and fixed intervals we allow the final interval to be shortened in order to reach the desired stability threshold on a "Good" review with the highest probability possible, and we allow cards that have reached the target stability threshold to never be scheduled again.

The probability that a card is recalled successfully is estimated by FSRS.

## Results

- Reviews per day (average): the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Minutes per day (average): same as above, but minutes of studying are used instead. Lower is better.
- Memorized cards (average, all days): first, for each day, the number of cards memorized by that day is calculated as the sum of probabilities of recall of all cards on that day. Then, an average across all days is taken. This number cannot be greater than the deck size. Higher is better.
- Memorized/hours spent (average, all days): the value above divided by the average accumulated time spent on reviews. It's the ratio of average accumulated knowledge to average accumulated studying time. Think of it like this: "On average, by the time you have memorized X cards, you have spent Y hours on reviews, so the ratio is X/Y". Higher is better.

The reason why we look at average knowledge (number of memorized cards, aka sum of probabilities of recall of each card) across all days rather than only looking at knowledge at the end of the simulation is because the latter can be cheated. Imagine the following algorithm (duration of the simulation=5 years): first, it assigns an interval equal to 4 years and 11 months to every card. Then, during the last month, it assigns an interval equal to a few days to every card. The number of cards memorized *at the end* will be very high despite the fact that on *most* days knowledge was extremely low.

Deck size = 10000 cards, new cards per day = 10, max. reviews per day = 9999, max. studying time per day = 12 hours.

The best result is highlighted in **bold**. Arrows indicate whether lower (↓) or higher (↑) values are better.

### Duration of the simulation = 1825 days (5 years)

| Scheduling Policy | Total number of reviews (thousands)↓ | Total number of hours↓ | Number of memorized cards↑ | Memorized per hour↑ |
| --- | --- | --- | --- | --- |
| SSP-MMC (Maximum knowledge) | 1224 | 2876 | 9946 | 3.46 |
| SSP-MMC | 682 | 1683 | 9920 | 5.90 |
| SSP-MMC | 615 | 1522 | 9891 | 6.50 |
| SSP-MMC | 410 | 1076 | 9842 | 9.15 |
| SSP-MMC | 403 | 1051 | 9790 | 9.32 |
| SSP-MMC | 284 | 801 | 9714 | 12.13 |
| SSP-MMC | 265 | 760 | 9670 | 12.72 |
| SSP-MMC | 225 | 668 | 9608 | 14.39 |
| SSP-MMC | 187 | 588 | 9537 | 16.23 |
| SSP-MMC | 174 | 557 | 9486 | 17.02 |
| SSP-MMC | 136 | 469 | 9318 | 19.88 |
| SSP-MMC | 121 | 428 | 9141 | 21.37 |
| SSP-MMC (Balanced) | 114 | 411 | 9049 | 22.04 |
| SSP-MMC | 108 | 392 | 8896 | 22.71 |
| SSP-MMC | 105 | 387 | 8843 | 22.85 |
| SSP-MMC | 104 | 383 | 8796 | 22.94 |
| SSP-MMC | 101 | 379 | 8740 | 23.06 |
| SSP-MMC | 100 | 374 | 8701 | 23.27 |
| SSP-MMC | 99 | 370 | 8649 | 23.39 |
| SSP-MMC | 98 | 367 | 8619 | 23.45 |
| SSP-MMC (Maximum efficiency) | 97 | 364 | 8579 | **23.59** |
| Memrise | 180 | 557 | 9463 | 16.99 |
| Anki-SM-2 | 223 | 639 | 9319 | 14.57 |
| DR=0.70 | **88** | **357** | 8083 | 22.65 |
| DR=0.71 | 92 | 364 | 8164 | 22.42 |
| DR=0.72 | 95 | 372 | 8244 | 22.15 |
| DR=0.73 | 99 | 380 | 8299 | 21.81 |
| DR=0.74 | 103 | 390 | 8381 | 21.51 |
| DR=0.75 | 107 | 399 | 8484 | 21.25 |
| DR=0.76 | 111 | 410 | 8567 | 20.91 |
| DR=0.77 | 116 | 419 | 8641 | 20.63 |
| DR=0.78 | 121 | 430 | 8729 | 20.30 |
| DR=0.79 | 126 | 441 | 8783 | 19.90 |
| DR=0.80 | 131 | 452 | 8835 | 19.54 |
| DR=0.81 | 137 | 464 | 8907 | 19.19 |
| DR=0.82 | 144 | 478 | 8965 | 18.75 |
| DR=0.83 | 151 | 492 | 9038 | 18.35 |
| DR=0.84 | 158 | 508 | 9110 | 17.95 |
| DR=0.85 | 167 | 527 | 9175 | 17.40 |
| DR=0.86 | 177 | 548 | 9236 | 16.86 |
| DR=0.87 | 188 | 570 | 9288 | 16.29 |
| DR=0.88 | 200 | 597 | 9349 | 15.66 |
| DR=0.89 | 214 | 627 | 9413 | 15.01 |
| DR=0.90 | 231 | 663 | 9466 | 14.28 |
| DR=0.91 | 252 | 708 | 9523 | 13.44 |
| DR=0.92 | 277 | 763 | 9580 | 12.56 |
| DR=0.93 | 308 | 832 | 9634 | 11.59 |
| DR=0.94 | 349 | 920 | 9688 | 10.53 |
| DR=0.95 | 406 | 1048 | 9741 | 9.29 |
| DR=0.96 | 488 | 1230 | 9793 | 7.96 |
| DR=0.97 | 623 | 1530 | 9845 | 6.43 |
| DR=0.98 | 883 | 2113 | 9896 | 4.68 |
| DR=0.99 | 1625 | 3778 | 9946 | 2.63 |
| Interval=5 | 2645 | 6093 | **9966** | 1.64 |
| Interval=10 | 1320 | 3144 | 9924 | 3.16 |
| Interval=20 | 658 | 1677 | 9816 | 5.85 |
| Interval=50 | 260 | 790 | 9381 | 11.88 |
| Interval=100 | 128 | 468 | 8665 | 18.50 |

It may be difficult to understand whether a certain scheduling policy is better than another based on this table. Below is a visualization:

<img width="1200" height="900" alt="Pareto frontier" src="https://github.com/user-attachments/assets/91488a62-3928-4678-a12b-8443daca94a4" />

"Balanced" roughly corresponds to "Number of memorized cards of DR=83% and efficiency of DR=72%".

Overall, it's clear that SSP-MMC outperforms fixed desired retention, which in turn outperforms fixed intervals. Surprisingly, Memrise's simple algorithm performs remarkably well. We are unsure why.

## Caveats

Currently, the results are based on several assumptions and have several limitations:

1) FSRS is perfectly accurate at predicting the probability of recall
2) There are no same-day reviews, intervals must be at least one day long
3) There is no [fuzz](https://docs.ankiweb.net/studying.html#fuzz-factor)
4) This is based on default FSRS parameters and default review times, but of course parameters and review times vary across different users
