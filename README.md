# SSP-MMC-FSRS

## Introduction

SSP-MMC-FSRS is an extended verson of [SSP-MMC](https://github.com/maimemo/SSP-MMC), which is an algorithm for minimizing the memorization cost in spaced repetiton. The core hypothesis of SSP-MMC is the learner will memorize a card forever when the stability exceeds a certain threshold. With this hypothesis, and the memory state-transition function (provided by [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)), we can formulate the problem as a special case of the Markov Decision Process (MDP), i.e., a stochastic shortest path problem.


## Results


| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute
| --- | --- | --- | --- | --- |
| SSP-MMC | 41.42 | 11.42 | 9818.93 | 859.79 |
| DR=0.70 | 30.96 | 12.51 | 8705.75 | 695.77 |
| DR=0.73 | 32.70 | 12.39 | 8928.16 | 720.68 |
| DR=0.76 | 34.39 | 12.05 | 9102.92 | 755.21 |
| DR=0.79 | 36.83 | 11.96 | 9343.61 | 781.14 |
| DR=0.82 | 39.54 | 11.79 | 9522.07 | 807.77 |
| DR=0.85 | 43.57 | 11.88 | 9668.09 | 814.10 |
| DR=0.88 | 48.04 | 11.79 | 9787.57 | 829.99 |
| DR=0.91 | 57.05 | 12.43 | 9878.64 | 794.77 |
| DR=0.94 | 73.07 | 13.99 | 9941.42 | 710.72 |
| DR=0.97 | 115.79 | 19.01 | 9979.66 | 525.09 |