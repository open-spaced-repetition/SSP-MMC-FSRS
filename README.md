# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended version of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetition. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis and the memory state transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.

## Scheduling policies

Scheduling Policy: how the intervals are calculated.

- SSP-MMC: the intervals are chosen so that the "cost" (in minutes of studying) is minimized. The tradeoff between time efficiency and knowledge (the number of cards memorized) can be controlled with hyperparameters. It uses values of memory stability and difficulty provided by FSRS. For every value of memory stability and difficulty, it finds the optimal next interval to minimize time spent on reviews.
- Fixed DR: fixed desired retention, meaning that the intervals correspond to the desired probability of recall, for example, 90%. This is how FSRS currently schedules reviews in Anki.
- Fixed intervals: intervals are always the same and do not depend on the rating (Again/Hard/Good/Easy) or anything else. The units are days.
- Memrise: the following sequence of intervals, in days: 1 -> 6 -> 12 -> 48 -> 96 -> 180. If the user presses Again, the interval is reset to 1, and the sequence starts all over again. If the user presses Hard, Good, or Easy, the next interval in the sequence is used.
- Anki-SM-2: Anki's version of SM-2. It's a relatively simple algorithm that keeps track of "ease" (which is based on ratings) and the length of the next interval is equal to the length of the last interval times ease; plus some bonuses and penalties related to Hard; plus some extra modifiers. Anki-specific settings (starting ease, interval modifier, etc.) are set to their default values as of Anki 24.11. 

To account for the core hypothesis of SSP-MMC, that cards will be remembered forever once the stability exceeds a certain threshold, for both DR and fixed intervals we allow the final interval to be shortened in order to reach the desired stability threshold on a "Good" review with the highest probability possible, and we allow cards that have reached the target stability threshold to never be scheduled again.

The probability that a card is recalled successfully is estimated by FSRS.

## Results

- Reviews per day (average): the average number of simulated reviews of flashcards per one day of the simulated review history. Lower is better.
- Minutes per day (average): same as above, but minutes of studying are used instead. Lower is better.
- Memorized cards (average, all days): first, for each day, the number of cards memorized by that day is calculated as the sum of probabilities of recall of all cards on that day. Then, an average across all days is taken. This number cannot be greater than the deck size. Example: if you knew 10 cards by day 1, 15 cards by day 2, and 20 cards by day 3, the average is 15. Higher is better.
- Memorized/hours spent (average, all days): the value above divided by the average accumulated time spent on reviews. It's the ratio of average accumulated knowledge to average accumulated studying time. Think of it like this: "On average, by the time you have memorized X cards, you have spent Y hours on reviews, so the ratio is X/Y". Higher is better.

The reason why we look at average knowledge (number of memorized cards, aka sum of probabilities of recall of each card) across all days rather than only looking at knowledge at the end of the simulation is because the latter can be cheated. Imagine the following algorithm (duration of the simulation=5 years): first, it assigns an interval equal to 4 years and 11 months to every card. Then, during the last month, it assigns an interval equal to a few days to every card. The number of cards memorized *at the end* will be very high despite the fact that on *most* days knowledge was extremely low.

Deck size = 10000 cards, new cards per day = 10, max. reviews per day = 9999, max. studying time per day = 12 hours.

The best result is highlighted in **bold**. Arrows indicate whether lower (↓) or higher (↑) values are better.

### Duration of the simulation = 1825 days (5 years)

| Scheduling Policy | Reviews per day (average)↓ | Minutes per day (average)↓ | Memorized cards (average, all days)↑ | Memorized/hours spent (average, all days)↑ |
| --- | --- | --- | --- | --- |
| SSP-MMC-FSRS (Maximum knowledge) | 429.8 | 62.6 | 7169 | 8.3 |
| SSP-MMC-FSRS | 351.3 | 52.0 | 7138 | 9.9 |
| SSP-MMC-FSRS | 306.5 | 46.3 | 7130 | 11.4 |
| SSP-MMC-FSRS | 288.8 | 43.7 | 7134 | 12.0 |
| SSP-MMC-FSRS | 281.6 | 43.0 | 7117 | 12.3 |
| SSP-MMC-FSRS | 254.6 | 39.3 | 7026 | 13.2 |
| SSP-MMC-FSRS | 196.1 | 31.5 | 7050 | 16.2 |
| SSP-MMC-FSRS | 181.3 | 29.6 | 6965 | 17.0 |
| SSP-MMC-FSRS | 134.2 | 23.4 | 6900 | 20.6 |
| SSP-MMC-FSRS | 125.1 | 22.3 | 6879 | 21.4 |
| SSP-MMC-FSRS | 122.1 | 22.0 | 6863 | 21.6 |
| SSP-MMC-FSRS | 100.9 | 19.1 | 6804 | 23.8 |
| SSP-MMC-FSRS | 92.7 | 18.0 | 6764 | 24.8 |
| SSP-MMC-FSRS | 78.0 | 15.7 | 6695 | 26.9 |
| SSP-MMC-FSRS | 67.4 | 13.8 | 6551 | 28.2 |
| SSP-MMC-FSRS | 62.5 | 13.2 | 6500 | 29.4 |
| SSP-MMC-FSRS | 59.8 | 12.7 | 6438 | 29.8 |
| SSP-MMC-FSRS | 61.2 | 13.0 | 6471 | 29.6 |
| SSP-MMC-FSRS | 61.2 | 12.9 | 6453 | 29.4 |
| SSP-MMC-FSRS (Balanced) | 56.1 | 12.3 | 6378 | 30.7 |
| SSP-MMC-FSRS | 53.3 | 12.0 | 6295 | 31.3 |
| SSP-MMC-FSRS | 53.4 | 12.0 | 6297 | 31.3 |
| SSP-MMC-FSRS | 52.8 | 11.9 | 6264 | 31.3 |
| SSP-MMC-FSRS (Maximum efficiency) | 50.2 | **11.7** | 6163 | **31.6** |
| Memrise | 99.1 | 18.4 | 6760 | 24.1 |
| Anki-SM-2 | 122.2 | 21.0 | 6821 | 19.8 |
| DR=0.70 | **48.1** | **11.7** | 5912 | 31.0 |
| DR=0.71 | 50.5 | 12.1 | 5966 | 30.3 |
| DR=0.72 | 51.9 | 12.2 | 6020 | 30.3 |
| DR=0.73 | 53.9 | 12.5 | 6073 | 29.7 |
| DR=0.74 | 56.3 | 12.8 | 6123 | 29.3 |
| DR=0.75 | 58.5 | 13.1 | 6175 | 29.0 |
| DR=0.76 | 60.8 | 13.4 | 6231 | 28.5 |
| DR=0.77 | 63.7 | 13.8 | 6283 | 28.0 |
| DR=0.78 | 66.5 | 14.2 | 6337 | 27.5 |
| DR=0.79 | 69.6 | 14.6 | 6388 | 27.0 |
| DR=0.80 | 72.2 | 14.9 | 6437 | 26.6 |
| DR=0.81 | 75.0 | 15.2 | 6483 | 26.2 |
| DR=0.82 | 78.6 | 15.7 | 6529 | 25.6 |
| DR=0.83 | 82.5 | 16.2 | 6573 | 25.0 |
| DR=0.84 | 87.1 | 16.8 | 6619 | 24.4 |
| DR=0.85 | 91.4 | 17.3 | 6665 | 23.8 |
| DR=0.86 | 96.5 | 17.9 | 6710 | 23.1 |
| DR=0.87 | 102.4 | 18.7 | 6752 | 22.3 |
| DR=0.88 | 109.8 | 19.7 | 6792 | 21.4 |
| DR=0.89 | 118.3 | 20.8 | 6834 | 20.4 |
| DR=0.90 | 127.2 | 21.9 | 6875 | 19.5 |
| DR=0.91 | 139.6 | 23.5 | 6915 | 18.3 |
| DR=0.92 | 151.2 | 25.0 | 6954 | 17.4 |
| DR=0.93 | 168.8 | 27.3 | 6993 | 16.0 |
| DR=0.94 | 191.4 | 30.3 | 7031 | 14.5 |
| DR=0.95 | 223.3 | 34.5 | 7069 | 12.9 |
| DR=0.96 | 268.7 | 40.6 | 7106 | 11.1 |
| DR=0.97 | 342.1 | 50.4 | 7143 | 9.0 |
| DR=0.98 | 482.5 | 69.3 | 7179 | 6.7 |
| DR=0.99 | 892.0 | 124.5 | **7214** | 3.8 |
| Interval=7 | 1034.4 | 144.9 | 7175 | 4.2 |
| Interval=14 | 515.9 | 75.8 | 7070 | 7.8 |
| Interval=20 | 360.3 | 55.2 | 6970 | 10.4 |
| Interval=30 | 239.3 | 39.1 | 6800 | 14.1 |
| Interval=50 | 142.5 | 25.9 | 6481 | 19.9 |
| Interval=75 | 94.1 | 19.1 | 6138 | 25.4 |
| Interval=100 | 69.9 | 15.4 | 5855 | 29.7 |

It may be difficult to understand whether a certain scheduling policy is better than another based on this table. Below is a visualization:

<img width="1200" height="900" alt="Figure_1" src="https://github.com/user-attachments/assets/5ac8963e-f741-44b1-8789-b989620df096" />

"Balanced" roughly corresponds to "Number of memorized cards of DR=79% and efficiency of DR=70%".

While SSP-MMC outperforms low fixed desired retention, it struggles to outperform high fixed desired retention. Surprisingly, Memrise's simple algorithm performs remarkably well. We are unsure why.

## Caveats

Currently, the results are based on several assumptions and have several limitations:

1) FSRS is perfectly accurate at predicting the probability of recall
2) There are no same-day reviews, intervals must be at least one day long
3) There is no [fuzz](https://docs.ankiweb.net/studying.html#fuzz-factor)
4) This is based on default FSRS parameters and default review times, but of course parameters and review times vary across different users
