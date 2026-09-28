"""Print the Models leaderboard for the results under evaluation/results/tasks.

Reads every {scenario}/{instance}/{config}/{stamp}.json and aggregates the same
way the results viewer does: an errored attempt is counted but never averaged,
each instance is averaged over its own attempts first, and the per-model figure
is the mean across instances, so a model that ran one instance twice does not
weigh double.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_RESULTS = HERE / "results" / "tasks"
DEFAULT_CONFIGS = REPO_ROOT / "src" / "benchmark" / "models" / "configurations"

DASH = "—"
WALL_CLOCK_MARK = "*"
WALL_CLOCK_NOTE = (
    f"{WALL_CLOCK_MARK} wall-clock duration including the network; "
    "the other models report model-side time."
)


@dataclass(frozen=True)
class Attempt:
    instance_id: str
    config: str
    scored: bool
    f1: float | None
    precision: float | None
    recall: float | None
    total_duration_s: float | None
    eval_count: int | None
    eval_duration_s: float | None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.eval_count or not self.eval_duration_s:
            return None
        return self.eval_count / self.eval_duration_s


@dataclass(frozen=True)
class ModelRow:
    config: str
    sub_label: str
    attempts: int
    errored: int
    instances: int
    f1: float | None
    precision: float | None
    recall: float | None
    duration_s: float | None
    tokens_per_second: float | None
    wall_clock_only: bool


def mean(values: list[float]) -> float | None:
    """The mean, or None over an empty list — never 0.0 standing in for absent."""
    return sum(values) / len(values) if values else None


def present(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def read_attempts(results_root: Path) -> list[Attempt]:
    attempts = []
    for path in sorted(results_root.glob("*/*/*/*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        error = data.get("error")
        attempts.append(
            Attempt(
                instance_id=path.parts[-3],
                config=path.parts[-2],
                scored=error is None,
                # The runner writes 0.0 metrics on an errored attempt. Read them as absent.
                f1=None if error else data.get("f1"),
                precision=None if error else data.get("precision"),
                recall=None if error else data.get("recall"),
                total_duration_s=data.get("total_duration_s"),
                eval_count=data.get("eval_count"),
                eval_duration_s=data.get("eval_duration_s"),
            )
        )
    return attempts


def read_sub_labels(configs_root: Path) -> dict[str, str]:
    """The model id and think setting shown under each model name."""
    labels = {}
    for path in sorted(configs_root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        args = data.get("args") or {}
        parts = [str(args["model"])] if args.get("model") else []
        if args.get("think") is not None:
            parts.append(f"think={args['think']}")
        labels[data.get("name") or path.stem] = " · ".join(parts)
    return labels


def summarize(config: str, attempts: list[Attempt], sub_label: str) -> ModelRow:
    by_instance: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_instance[attempt.instance_id].append(attempt)

    scored_per_instance = [[a for a in group if a.scored] for group in by_instance.values()]
    scored_per_instance = [group for group in scored_per_instance if group]

    def macro(field: str) -> float | None:
        per_instance = [
            mean(present([getattr(a, field) for a in group])) for group in scored_per_instance
        ]
        return mean(present(per_instance))

    return ModelRow(
        config=config,
        sub_label=sub_label,
        attempts=len(attempts),
        errored=sum(1 for a in attempts if not a.scored),
        instances=len(by_instance),
        f1=macro("f1"),
        precision=macro("precision"),
        recall=macro("recall"),
        duration_s=macro("total_duration_s"),
        tokens_per_second=macro("tokens_per_second"),
        wall_clock_only=bool(scored_per_instance)
        and all(a.eval_duration_s is None for group in scored_per_instance for a in group),
    )


def build_rows(attempts: list[Attempt], sub_labels: dict[str, str]) -> list[ModelRow]:
    by_config: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_config[attempt.config].append(attempt)
    rows = [
        summarize(config, group, sub_labels.get(config, "no configuration file"))
        for config, group in by_config.items()
    ]
    # Best F1 first; a model with nothing scored has no performance to rank on.
    return sorted(rows, key=lambda row: (row.f1 is None, -(row.f1 or 0.0), row.config))


def num(value: float | None, places: int = 2) -> str:
    return DASH if value is None else f"{value:.{places}f}"


def duration_text(row: ModelRow) -> str:
    if row.duration_s is None:
        return DASH
    return f"{row.duration_s:.1f}s{WALL_CLOCK_MARK if row.wall_clock_only else ''}"


def attempts_text(row: ModelRow) -> str:
    errored = f" {row.errored} errored" if row.errored else ""
    return f"{row.attempts}{errored} · {row.instances} instances"


COLUMNS = (
    ("MODEL", lambda row: row.config, "<"),
    ("ATTEMPTS", attempts_text, ">"),
    ("DURATION", duration_text, ">"),
    ("TOK/S", lambda row: num(row.tokens_per_second, 1), ">"),
    ("F1", lambda row: num(row.f1), ">"),
    ("PRECISION", lambda row: num(row.precision), ">"),
    ("RECALL", lambda row: num(row.recall), ">"),
)


def print_table(rows: list[ModelRow]) -> None:
    cells = [[render(row) for _, render, _ in COLUMNS] for row in rows]
    sub_labels = [row.sub_label for row in rows]
    widths = [
        max(len(header), *(len(line[index]) for line in cells)) if cells else len(header)
        for index, (header, _, _) in enumerate(COLUMNS)
    ]
    widths[0] = max(widths[0], *(len(label) for label in sub_labels)) if sub_labels else widths[0]

    def line(values: list[str]) -> str:
        return "  ".join(
            f"{value:{align}{width}}"
            for value, width, (_, _, align) in zip(values, widths, COLUMNS, strict=True)
        ).rstrip()

    print(line([header for header, _, _ in COLUMNS]))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for values, sub_label in zip(cells, sub_labels, strict=True):
        print(line(values))
        print(f"  {sub_label}")
    if any(row.wall_clock_only and row.duration_s is not None for row in rows):
        print(f"\n{WALL_CLOCK_NOTE}")


def write_csv(rows: list[ModelRow]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "model",
            "model_id",
            "attempts",
            "errored",
            "instances",
            "duration_s",
            "tokens_per_second",
            "f1",
            "precision",
            "recall",
            "wall_clock_only",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.config,
                row.sub_label,
                row.attempts,
                row.errored,
                row.instances,
                row.duration_s,
                row.tokens_per_second,
                row.f1,
                row.precision,
                row.recall,
                row.wall_clock_only,
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--csv", action="store_true", help="write machine-readable rows instead")
    args = parser.parse_args()

    attempts = read_attempts(args.results)
    if not attempts:
        print(f"no results found under {args.results}", file=sys.stderr)
        return 1
    rows = build_rows(attempts, read_sub_labels(args.configs))
    write_csv(rows) if args.csv else print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
