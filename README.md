# Action Dependency Identification — Research Artifact

Code and model configurations for "*Can AI Agents Identify What State Their
Actions Depend On?*"

A long-running agent often acts on state it observed earlier, which may have
changed by the time the action runs. It's costly and time consuming to revalidate everything when only some of its state matters on whether the action is still justified.
**Action dependency identification** is the task of picking out that subset:
given a user task, the observed state as a list of atomic facts, and a proposed
action, name the facts that should be rechecked before executing.
    
Each benchmark instance gives a model exactly that, and is scored against a
hand-annotated question set with precision, recall, and F1.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

API keys go in a `.env` file at the repo root (loaded by
[env.py](src/benchmark/env.py)); only the keys for the models you intend to run
are needed:

```
OPENROUTER_API_KEY=...
TYPESAFE_API_KEY=...
```

Local models are served by [Ollama](https://ollama.com) on
`http://localhost:11434`, and need no key.

## Running the benchmark

```
uv run python src/benchmark/models/runner.py --config src/benchmark/models/configurations/openrouter-gpt-4.1.json
```

`--tasks` takes a single task file, one scenario folder, or a directory to
search recursively; it defaults to `src/benchmark/data/tasks/`.

Each run writes two views of the same records: one JSONL file per run under
`src/benchmark/results/runs/<config>/<timestamp>.jsonl`, and one JSON file per
instance under `src/benchmark/results/tasks/<scenario>/<instance>/<config>/`.
Results accumulate, so running the same model again adds a timestamped attempt
rather than replacing the last one.

To see the exact prompt a model is sent for one instance, with the answer key
left out:

```
uv run python src/scripts/render_prompt.py src/benchmark/data/tasks/<scenario>/<instance>.json --stdout
```

## Model configurations

A configuration is a small JSON file naming a runner, a prompt template, and
the runner's arguments:

```json
{
  "name": "openrouter-gpt-4.1",
  "runner": "openrouter",
  "prompt": "src/benchmark/prompts/demo.md",
  "args": {
    "model": "openai/gpt-4.1",
    "timeout_s": 900,
    "params": { "temperature": 0, "seed": 0 }
  }
}
```

The runners live in [src/benchmark/models/runners/](src/benchmark/models/runners/)
and are registered in [runner.py](src/benchmark/models/runner.py#L26):

| Runner | Serves |
| --- | --- |
| `openrouter` | Any chat model on OpenRouter |
| `ollama` | Any local Ollama chat model |
| `gpt_oss_ollama` | GPT-OSS locally, with a reasoning-effort setting |
| `qwen3_ollama` | Qwen3 locally, with thinking on or off |
| `deepseek_r1_ollama` | DeepSeek-R1 locally |
| `typesafe_jev` | JEV, via the TypeSafe API |

Adding a model is usually a new file in
[configurations/](src/benchmark/models/configurations/); adding a new *kind* of
model is a `Runner` subclass in `runners/` plus an entry in `RUNNERS`.

## Layout

```
src/benchmark/data/          WebArena-Verified source data and the task instances
src/benchmark/prompts/       Prompt templates ({{placeholders}} filled per instance)
src/benchmark/models/        Runners, configurations, and the run loop
src/benchmark/results/       Run output
src/scripts/                 One-off scripts
tests/                       pytest suite
```

Task instances, rendered prompts, and results are generated or working data and
are not tracked in git.

## Development

```
uv run ruff format .
uv run ruff check .
uv run pytest .
```

## Data

Scenarios derive from
[WebArena-Verified](https://github.com/ServiceNow/webarena-verified); see
[src/benchmark/data/README.md](src/benchmark/data/README.md).
