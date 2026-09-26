"""Render one scenario instance into the prompt, as a .md file to hand an agent.

The rendering is the runner's own, so the file you copy out of here is the same
text a model is sent for that instance - nothing added, and the answer key in
`supporting_fact_ids` left behind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.models.runners.base import Instance, Runner
from benchmark.models.runners.typesafe_jev import build_request
from benchmark.models.view import RED, RESET

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMPT = REPO_ROOT / "src" / "benchmark" / "prompts" / "demo.md"
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "benchmark" / "prompts" / "rendered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="path to one scenario instance JSON")
    parser.add_argument(
        "--prompt",
        default=str(DEFAULT_PROMPT),
        help="the prompt template to fill (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        help=f"where to write the .md (default: {DEFAULT_OUT_DIR}/<instance_id>.md)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="also print the prompt itself, for piping to pbcopy",
    )
    return parser.parse_args()


def render(instance_path: str | Path, prompt_path: str | Path) -> str:
    """The instance's task, state and proposed action filled into the template.

    A `.json` template is Jev's: it renders to the request body that would be
    posted, questions and all, so the payload can be read without a billed call.
    """
    instance = Instance.from_file(instance_path)
    prompt_path = Path(prompt_path)
    if prompt_path.suffix == ".json":
        body, _ = build_request(json.loads(prompt_path.read_text()), instance)
        return json.dumps(body, indent=2) + "\n"
    return Runner({}).render_prompt(prompt_path.read_text(), instance)


def resolve_out_path(out: str | None, instance_path: str | Path, suffix: str = ".md") -> Path:
    """An explicit `--out`, or the rendered directory named for the instance."""
    if out:
        return Path(out)
    return DEFAULT_OUT_DIR / f"{Path(instance_path).stem}{suffix}"


def main() -> None:
    args = parse_args()

    try:
        rendered = render(args.instance, args.prompt)
    except Exception as exception:
        print(f"{RED}Error: {type(exception).__name__}: {exception}{RESET}")
        raise SystemExit(1) from None

    out_path = resolve_out_path(args.out, args.instance, Path(args.prompt).suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)

    if args.stdout:
        print(rendered)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
