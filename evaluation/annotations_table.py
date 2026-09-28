"""Print the Fact annotations tables for the results under evaluation/results/tasks.

Pools every fact of every selected attempt by one annotation value and reports
errors out of opportunities: a required fact the model left out is a miss, a
fact that was not required and was named anyway is an over-read. Rates, not
counts, are what let a fact kind seen 3770 times be compared against one seen
591 times.

Only the bottom `--bottom-pct` percent of models by F1 are pooled, because the
models at the top make almost no errors and washing them into the denominators
hides the pattern. The ranking is the one models_table.py prints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from models_table import DASH, build_rows, read_attempts

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_RESULTS = HERE / "results" / "tasks"
DEFAULT_TASKS = REPO_ROOT / "src" / "benchmark" / "data" / "tasks"
DEFAULT_CONFIGS = REPO_ROOT / "src" / "benchmark" / "models" / "configurations"
DEFAULT_BOTTOM_PCT = 70

NO_VALUE = DASH


@dataclass(frozen=True)
class Fact:
    id: str
    annotations: dict[str, str]


@dataclass(frozen=True)
class Task:
    instance_id: str
    facts: tuple[Fact, ...]
    required_ids: frozenset[str]


@dataclass(frozen=True)
class Prediction:
    """One scored attempt, reduced to what a tally needs."""

    instance_id: str
    config: str
    predicted_ids: frozenset[str]


@dataclass(frozen=True)
class Tally:
    """Errors out of opportunities, for one annotation value."""

    value: str
    required: int
    missed: int
    not_required: int
    over_read: int

    @property
    def miss_rate(self) -> float | None:
        return self.missed / self.required if self.required else None

    @property
    def over_read_rate(self) -> float | None:
        return self.over_read / self.not_required if self.not_required else None


def read_tasks(tasks_root: Path) -> dict[str, Task]:
    """Read every {scenario}/{instance}.json under tasks_root."""
    tasks = {}
    for path in sorted(tasks_root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        facts = tuple(
            Fact(id=fact["id"], annotations=dict(fact.get("annotations") or {}))
            for fact in data.get("state", [])
        )
        instance_id = data.get("instance_id") or path.stem
        tasks[instance_id] = Task(
            instance_id=instance_id,
            facts=facts,
            required_ids=frozenset(data.get("supporting_fact_ids") or ()),
        )
    return tasks


def read_predictions(results_root: Path, configs: set[str]) -> list[Prediction]:
    """The scored attempts of the selected configs. An errored attempt named no
    facts because the harness failed, not because the model chose none."""
    predictions = []
    for path in sorted(results_root.glob("*/*/*/*.json")):
        if path.parts[-2] not in configs:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        if data.get("error"):
            continue
        predictions.append(
            Prediction(
                instance_id=path.parts[-3],
                config=path.parts[-2],
                predicted_ids=frozenset(data.get("predicted_ids") or ()),
            )
        )
    return predictions


def bottom_by_f1(results_root: Path, configs_root: Path, pct: int) -> tuple[str, ...]:
    """The worst `pct` percent of the models, worst last.

    The count rounds down but never to nothing: 5% over four models still
    selects something rather than emptying the table.
    """
    rows = build_rows(read_attempts(results_root), {})
    ranked = [row.config for row in rows]  # best F1 first; unscored models sort last
    if not ranked:
        return ()
    count = max(1, min(len(ranked), int(len(ranked) * pct / 100)))
    return tuple(ranked[-count:])


def axes_of(tasks: dict[str, Task]) -> list[str]:
    return sorted(
        {axis for task in tasks.values() for fact in task.facts for axis in fact.annotations}
    )


def tally(
    tasks: dict[str, Task], predictions: list[Prediction], axis: str, config: str | None = None
) -> list[Tally]:
    """Pool every fact of every prediction by its value on one axis.

    Within one instance a fact is required or not for every attempt alike, fixed
    by the task file, so each fact contributes to exactly one of the two
    denominators. Instances with no task file are skipped: their annotations are
    unavailable, and guessing would be worse than omitting them.
    """
    by_instance: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if config is None or prediction.config == config:
            by_instance[prediction.instance_id].append(prediction)

    totals: dict[str, list[int]] = {}  # value -> [required, missed, not_required, over_read]
    for instance_id, attempts in by_instance.items():
        task = tasks.get(instance_id)
        if task is None:
            continue
        for fact in task.facts:
            value = fact.annotations.get(axis) or NO_VALUE
            bucket = totals.setdefault(value, [0, 0, 0, 0])
            named = sum(1 for a in attempts if fact.id in a.predicted_ids)
            if fact.id in task.required_ids:
                bucket[0] += len(attempts)
                bucket[1] += len(attempts) - named
            else:
                bucket[2] += len(attempts)
                bucket[3] += named
    return [Tally(value, *counts) for value, counts in sorted(totals.items())]


def rate_text(rate: float | None, numerator: int, denominator: int) -> str:
    if rate is None:
        return DASH
    return f"{rate * 100:.0f}%  {numerator} / {denominator}"


COLUMNS = (
    ("VALUE", "<"),
    ("REQUIRED", ">"),
    ("MISS RATE", ">"),
    ("NOT REQUIRED", ">"),
    ("OVER-READ RATE", ">"),
)


def cells_of(label: str, row: Tally) -> list[str]:
    return [
        label,
        str(row.required),
        rate_text(row.miss_rate, row.missed, row.required),
        str(row.not_required),
        rate_text(row.over_read_rate, row.over_read, row.not_required),
    ]


def print_axis(axis: str, rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(cells[index]) for cells in rows))
        for index, (header, _) in enumerate(COLUMNS)
    ]

    def line(values: list[str]) -> str:
        return "  ".join(
            f"{value:{align}{width}}"
            for value, width, (_, align) in zip(values, widths, COLUMNS, strict=True)
        ).rstrip()

    print(f"\n{axis}")
    print(line([header for header, _ in COLUMNS]))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for cells in rows:
        print(line(cells))


def print_tables(
    tasks: dict[str, Task],
    predictions: list[Prediction],
    configs: tuple[str, ...],
    per_model: bool,
) -> None:
    print("Fact annotations")
    for axis in axes_of(tasks):
        rows: list[list[str]] = []
        for row in tally(tasks, predictions, axis):
            rows.append(cells_of(row.value, row))
            if not per_model:
                continue
            for config in configs:
                for sub in tally(tasks, predictions, axis, config):
                    if sub.value == row.value:
                        rows.append(cells_of(f"  {config}", sub))
        if rows:
            print_axis(axis, rows)


def write_csv(
    tasks: dict[str, Task],
    predictions: list[Prediction],
    configs: tuple[str, ...],
    per_model: bool,
) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "axis",
            "value",
            "model",
            "required",
            "missed",
            "miss_rate",
            "not_required",
            "over_read",
            "over_read_rate",
        ]
    )
    for axis in axes_of(tasks):
        scopes = [None, *configs] if per_model else [None]
        for config in scopes:
            for row in tally(tasks, predictions, axis, config):
                writer.writerow(
                    [
                        axis,
                        row.value,
                        config or "",
                        row.required,
                        row.missed,
                        row.miss_rate,
                        row.not_required,
                        row.over_read,
                        row.over_read_rate,
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--bottom-pct",
        type=int,
        default=DEFAULT_BOTTOM_PCT,
        help="pool the worst this many percent of models by F1 (default: %(default)s)",
    )
    parser.add_argument("--per-model", action="store_true", help="break each value down by model")
    parser.add_argument("--csv", action="store_true", help="write machine-readable rows instead")
    args = parser.parse_args()

    if not 1 <= args.bottom_pct <= 100:
        parser.error("--bottom-pct must be between 1 and 100")

    tasks = read_tasks(args.tasks)
    if not tasks:
        print(f"no task files found under {args.tasks}", file=sys.stderr)
        return 1

    configs = bottom_by_f1(args.results, args.configs, args.bottom_pct)
    if not configs:
        print(f"no results found under {args.results}", file=sys.stderr)
        return 1

    predictions = read_predictions(args.results, set(configs))
    if not predictions:
        print(f"no scored attempts for {', '.join(configs)}", file=sys.stderr)
        return 1

    if args.csv:
        write_csv(tasks, predictions, configs, args.per_model)
        return 0

    instances = len({p.instance_id for p in predictions})
    print(
        f"bottom {args.bottom_pct}% by F1: {len(configs)} models, "
        f"{len(predictions)} scored attempts over {instances} instances"
    )
    print("  " + ", ".join(configs))
    print_tables(tasks, predictions, configs, args.per_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
