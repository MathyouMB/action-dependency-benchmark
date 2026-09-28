# Evaluation

This directory contains the code and data used in the results section of the "*Can AI Agents Identify What State Their
Actions Depend On?*" paper.

Running:

```python
uv run python evaluation/models_table.py
```

Will output the table used in **RQ1** and **RQ2** of the paper.

Running


```python
uv run python evaluation/annotations_table.py
```

Will output the table used in **RQ3** of the paper.

## Example output

### `models_table.py`

Every model was run over the same 50 instances. Attempts beyond 50 are repeat
runs; errored attempts are counted but never scored.

| Model | Attempts | Duration | Tok/s | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| **openrouter-claude-opus-5.5**<br>`anthropic/claude-opus-5.5` | 50 | 7.3s\* | — | **0.99** | 0.98 | 1.00 |
| **openrouter-gpt-6-sol**<br>`openai/gpt-6-sol` | 50 | 4.8s\* | — | 0.97 | 0.97 | 0.99 |
| **openrouter-gemini-3.7-flash**<br>`google/gemini-3.7-flash` | 50 | 8.8s\* | — | 0.97 | 0.98 | 0.97 |
| **openrouter-claude-sonnet-5**<br>`anthropic/claude-sonnet-5` | 50 | 6.0s\* | — | 0.97 | 0.97 | 0.97 |
| **openrouter-gpt-4.1**<br>`openai/gpt-4.1` | 50 | 2.1s\* | — | 0.97 | 0.97 | 0.97 |
| **qwen3-14b-thinking**<br>`qwen3:14b · think=True` | 55 (8 errored) | 117.0s | 15.2 | 0.97 | 1.00 | 0.94 |
| **openrouter-gpt-6-luna**<br>`openai/gpt-6-luna` | 55 | 4.7s\* | — | 0.96 | 0.96 | 0.97 |
| **openrouter-gpt-oss-120b**<br>`openai/gpt-oss-120b` | 51 | 34.8s\* | — | 0.95 | 0.97 | 0.95 |
| **openrouter-qwen3-30b-a3b**<br>`qwen/qwen3-30b-a3b` | 50 | 19.4s\* | — | 0.95 | 1.00 | 0.92 |
| **openrouter-gpt-4.1-mini**<br>`openai/gpt-4.1-mini` | 50 | 2.7s\* | — | 0.94 | 0.99 | 0.91 |
| **typesafe-jev-1.13**<br>`jev-1.13.0` | 50 | 0.2s\* | — | 0.93 | 0.93 | 0.95 |
| **gpt-oss-20b-medium**<br>`gpt-oss:20b · think=medium` | 60 (3 errored) | 59.5s | 30.3 | 0.93 | 0.98 | 0.90 |
| **openrouter-llama-4-maverick**<br>`meta-llama/llama-4-maverick` | 50 | 10.5s\* | — | 0.93 | 0.95 | 0.92 |
| **qwen3-8b-thinking**<br>`qwen3:8b · think=True` | 55 (8 errored) | 115.7s | 23.7 | 0.92 | 1.00 | 0.87 |
| **openrouter-gpt-oss-20b-medium**<br>`openai/gpt-oss-20b` | 50 | 22.7s\* | — | 0.92 | 0.96 | 0.90 |
| **gpt-oss-20b-low**<br>`gpt-oss:20b · think=low` | 66 | 16.1s | 30.4 | 0.90 | 0.95 | 0.87 |
| **deepseek-r1-14b**<br>`deepseek-r1:14b · think=True` | 55 (2 errored) | 74.3s | 15.6 | 0.90 | 0.92 | 0.90 |
| **qwen3-14b-no-think**<br>`qwen3:14b · think=False` | 60 | 21.6s | 16.0 | 0.88 | 0.94 | 0.85 |
| **qwen3-8b-no-think**<br>`qwen3:8b · think=False` | 70 | 13.3s | 26.3 | 0.87 | 0.91 | 0.85 |
| **qwen2.5-14b**<br>`qwen2.5:14b` | 60 | 18.9s | 15.5 | 0.85 | 0.94 | 0.80 |
| **openrouter-gpt-4.1-nano**<br>`openai/gpt-4.1-nano` | 50 | 2.3s\* | — | 0.52 | 0.68 | 0.50 |

\* wall-clock duration including the network; the other models report model-side time.

### `annotations_table.py`

Pooled over the bottom 70% of models by F1 — 14 models, 764 scored attempts
over 50 instances:

> openrouter-gpt-oss-120b, openrouter-qwen3-30b-a3b, openrouter-gpt-4.1-mini,
> typesafe-jev-1.13, gpt-oss-20b-medium, openrouter-llama-4-maverick,
> qwen3-8b-thinking, openrouter-gpt-oss-20b-medium, gpt-oss-20b-low,
> deepseek-r1-14b, qwen3-14b-no-think, qwen3-8b-no-think, qwen2.5-14b,
> openrouter-gpt-4.1-nano

**domain**

| Value | Required | Miss rate | Not required | Over-read rate |
| --- | ---: | ---: | ---: | ---: |
| categorical | 3770 | **20%** (751 / 3770) | 591 | **53%** (311 / 591) |
| numeric | 3368 | 10% (330 / 3368) | 957 | 11% (107 / 957) |
| temporal | 766 | 13% (98 / 766) | 528 | 9% (48 / 528) |

**polarity**

| Value | Required | Miss rate | Not required | Over-read rate |
| --- | ---: | ---: | ---: | ---: |
| positive | 7904 | 15% (1179 / 7904) | 2076 | 22% (466 / 2076) |

**structure**

| Value | Required | Miss rate | Not required | Over-read rate |
| --- | ---: | ---: | ---: | ---: |
| attribute | 6792 | 14% (973 / 6792) | 2076 | 22% (466 / 2076) |
| relation | 1112 | **19%** (206 / 1112) | 0 | — |

A miss is a required fact the model left out; an over-read is a fact that was
not required and was named anyway.
