import argparse
import csv
import os
import re
from datetime import datetime, timezone

RUN_NAME_RE = re.compile(
    r"^(?P<backbone>[^_]+)_(?P<policy>[^_]+)_(?P<game>.+)_seed(?P<seed>\d+)_(?P<mode>O|X|Both)$",
    re.IGNORECASE,
)

MANUAL_EVAL_RETURNS = {
    "Mamba2_AC_Assault_seed6010_O": 630,
    "Mamba2_AC_Assault_seed6010_X": 336,
    "Mamba2_AC_Pong_seed2010_O": 20,
    "Mamba2_AC_Pong_seed2010_X": 20,
}


def get_config_value(config, path, default="N/A"):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def parse_run_name(name):
    match = RUN_NAME_RE.match(str(name))
    if not match:
        return None
    values = match.groupdict()
    values["mode"] = values["mode"].upper()
    values["seed"] = int(values["seed"])
    return values


def as_text(value):
    return "N/A" if value is None else value


def is_missing(value):
    return value is None or str(value).strip().lower() in {"", "n/a", "na", "nan", "none"}


def main():
    import wandb

    parser = argparse.ArgumentParser(description="Export Drama WandB runs to CSV.")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "choemj-kaist"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "Mamba_dreamer"))
    parser.add_argument("--output", default="drama_wandb_runs.csv")
    args = parser.parse_args()

    api_key_path = os.path.join(os.path.dirname(__file__), ".wandb_api_key")
    if os.path.exists(api_key_path):
        with open(api_key_path, encoding="utf-8") as key_file:
            wandb.login(key=key_file.read().strip())

    path = f"{args.entity}/{args.project}"
    print(f"Reading runs from {path} ...")
    runs = wandb.Api().runs(path)
    existing_scores = {}
    if os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                existing_scores[row.get("Run ID", "")] = row

    rows = []

    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"Skipping unrecognized run name: {run.name}")
            continue
        if parsed["mode"] == "BOTH" and run.state != "running":
            continue
        if run.state == "killed":
            continue

        config = run.config
        summary = run.summary
        eval_return = as_text(summary.get("evaluate/score", "N/A"))
        normalized_return = as_text(summary.get("evaluate/normalised_score", "N/A"))
        if is_missing(eval_return) and run.name in MANUAL_EVAL_RETURNS:
            eval_return = MANUAL_EVAL_RETURNS[run.name]
        previous_row = existing_scores.get(run.id, {})
        if is_missing(eval_return) and not is_missing(previous_row.get("Eval Return")):
            eval_return = previous_row["Eval Return"]
        if is_missing(normalized_return) and not is_missing(previous_row.get("Eval Normalized Return")):
            normalized_return = previous_row["Eval Normalized Return"]

        rows.append({
            "Run Name": run.name,
            "Run ID": run.id,
            "State": run.state,
            "Backbone": parsed["backbone"],
            "Policy": parsed["policy"],
            "Game": parsed["game"],
            "Seed": parsed["seed"],
            "Retrieval": parsed["mode"],
            "Eval Return": eval_return,
            "Eval Normalized Return": normalized_return,
            "Warmup Steps": as_text(get_config_value(config, "JointTrainAgent.Retrieval.warmup_steps")),
            "Batch Size Reduction": as_text(get_config_value(config, "JointTrainAgent.Retrieval.batch_size_reduction")),
            "Hash Bits": as_text(get_config_value(config, "JointTrainAgent.Retrieval.hash_bits")),
            "Retrieval Target": as_text(get_config_value(config, "JointTrainAgent.Retrieval.target")),
            "Anchor Weight": as_text(get_config_value(config, "JointTrainAgent.Retrieval.anchor_weight")),
            "Created At": run.created_at,
        })

    fieldnames = list(rows[0]) if rows else [
        "Run Name", "Run ID", "State", "Backbone", "Policy", "Game", "Seed",
        "Retrieval", "Eval Return", "Eval Normalized Return", "Warmup Steps",
        "Batch Size Reduction", "Hash Bits", "Retrieval Target", "Anchor Weight", "Created At",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} O/X runs to {args.output}")


if __name__ == "__main__":
    main()
