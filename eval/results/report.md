# Evaluation results

Test set: 63 held-out queries. Robustness set: 20 typo/slang variants. Accept threshold: 0.63. LLM: Gemini (gemini-3.5-flash-lite). Tool latency: simulated (weather 150 ms, web search 400 ms).

## Routing quality (test set)

| router | accuracy [95% CI] | strict acc. | incorrect tool | missed tool | unnecessary tool | clarify | fallback | mean mistake cost | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| rules | 76.2% [65.1%, 85.7%] | 74.6% | 5.6% | 30.6% | 0.0% | 0.0% | 0.0% | 0.413 | 0.0 | 0.0 |
| embedding | 71.4% [60.3%, 82.5%] | 69.8% | 8.3% | 0.0% | 56.5% | 0.0% | 0.0% | 0.381 | 6.3 | 7.1 |
| classifier | 85.7% [77.8%, 93.7%] | 84.1% | 2.8% | 5.6% | 21.7% | 0.0% | 0.0% | 0.198 | 6.0 | 6.3 |
| hybrid | 92.1% [85.7%, 98.4%] | 90.5% | 0.0% | 8.3% | 4.3% | 0.0% | 15.9% | 0.111 | 6.0 | 7.9 |
| hybrid+rerank | 90.5% [82.5%, 96.8%] | 88.9% | 0.0% | 8.3% | 8.7% | 0.0% | 15.9% | 0.127 | 6.7 | 28.3 |

## Is the hybrid really better? (paired bootstrap, test set)

| comparison | accuracy difference | 95% CI | beyond luck? |
|---|---|---|---|
| hybrid vs rules | +15.9% | [+4.8%, +27.0%] | yes |
| hybrid vs embedding | +20.6% | [+9.5%, +31.7%] | yes |
| hybrid vs classifier | +6.3% | [+1.6%, +12.7%] | yes |
| hybrid vs hybrid+rerank | +1.6% | [+0.0%, +4.8%] | no |

## Robustness (typos and slang)

| router | test accuracy | robustness accuracy | change |
|---|---|---|---|
| rules | 76.2% | 55.0% | -21.2% |
| embedding | 71.4% | 90.0% | +18.6% |
| classifier | 85.7% | 95.0% | +9.3% |
| hybrid | 92.1% | 90.0% | -2.1% |
| hybrid+rerank | 90.5% | 90.0% | -0.5% |

## Accuracy by category (test set)

| category | rules | embedding | classifier | hybrid | hybrid+rerank |
|---|---|---|---|---|---|
| ambiguous | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| clear | 83.3% | 100.0% | 95.8% | 95.8% | 95.8% |
| multi_step | 20.0% | 60.0% | 80.0% | 80.0% | 80.0% |
| no_tool | 100.0% | 70.0% | 100.0% | 100.0% | 100.0% |
| out_of_scope | 100.0% | 20.0% | 80.0% | 100.0% | 100.0% |
| overlap | 69.2% | 30.8% | 53.8% | 76.9% | 69.2% |

Multi-step plans fully correct (hybrid): 80.0%


## Hybrid: which stage decided (test set)

| stage | queries | accuracy | p50 routing ms |
|---|---|---|---|
| classifier | 30 | 90.0% | 6.2 |
| rules | 19 | 100.0% | 0.0 |
| fallback | 6 | 66.7% | 6.5 |
| planner | 4 | 100.0% | 0.0 |
| out_of_scope | 4 | 100.0% | 0.0 |

## Hybrid: routing latency by chosen route (test set)

| route | p50 ms | p95 ms |
|---|---|---|
| calculator | 0.0 | 6.8 |
| llm | 6.1 | 7.3 |
| weather | 0.0 | 18.4 |
| web_search | 6.0 | 19.9 |

## End to end (hybrid + executor, test set)

Accuracy of the route that actually answered (after execution fallbacks): 92.1%. Total latency p50 415.5 ms, p95 1758.4 ms. Gemini rate-limit retries: 9.

| answered by | queries | total p50 ms | total p95 ms | mean routing ms | mean tools ms | mean LLM ms |
|---|---|---|---|---|---|---|
| calculator | 9 | 0.1 | 20.3 | 4.8 | 0.1 | 0.0 |
| llm | 27 | 1320.5 | 15496.1 | 14.2 | 0.0 | 2867.3 |
| weather | 12 | 155.5 | 306.8 | 2.1 | 180.1 | 0.0 |
| web_search | 15 | 412.8 | 428.8 | 7.5 | 404.4 | 0.0 |

Executor events: llm unavailable (2)


## Calibration on the test set (calibrators were fitted on validation only)

| router | ECE (lower = more honest) | AUROC (higher = better at spotting its own mistakes) |
|---|---|---|
| embedding | 0.041 | 0.817 |
| classifier | 0.027 | 0.714 |

## Hybrid confusion matrix (rows = labelled route, columns = predicted)

| labelled \ predicted | calculator | llm | weather | web_search |
|---|---|---|---|---|
| calculator | 9 | 1 | 0 | 0 |
| llm | 0 | 24 | 1 | 2 |
| weather | 0 | 0 | 11 | 0 |
| web_search | 0 | 2 | 0 | 13 |
