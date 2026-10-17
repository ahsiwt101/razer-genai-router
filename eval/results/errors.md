# Misrouted queries (hybrid)


## test (5 errors)

| query | category | expected | got | stage | reason |
|---|---|---|---|---|---|
| what are today's top stories in tech | clear | web_search | llm | classifier | Classifier chose llm (probability 0.61, calibrated confidence 0.76) |
| what was Apple's first product | overlap | llm | web_search | fallback | Unsure (confidence 0.57 < 0.63); lowest expected cost: web_search (0.54); ruled out calculator (nothing to extract) |
| what's 30 degrees celsius in fahrenheit | overlap | calculator | llm | classifier | Classifier chose llm (probability 0.49, calibrated confidence 0.71) |
| how much does it usually rain in London in November | overlap | llm/web_search | weather | classifier | Classifier chose weather (probability 0.90, calibrated confidence 0.97) |
| what's the price of gold divided by 3 | multi_step | web_search | llm | fallback | Unsure (top route calculator can't extract an argument); lowest expected cost: llm (0.36); ruled out calculator (nothing to extract) |

## robustness (2 errors)

| query | category | expected | got | stage | reason |
|---|---|---|---|---|---|
| wat is 832 divded by 13 | clear | calculator | llm | fallback | Unsure (top route calculator can't extract an argument); lowest expected cost: llm (0.07); ruled out calculator (nothing to extract) |
| whats 3 to the powr of 7 | clear | calculator | llm | fallback | Unsure (top route calculator can't extract an argument); lowest expected cost: llm (0.06); ruled out calculator (nothing to extract) |
