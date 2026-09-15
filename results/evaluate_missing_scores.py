import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import yaml


def is_missing(value):
    return pd.isna(value) or str(value).strip().lower() in {"", "n/a", "na", "nan", "none"}


class EvaluationLogger:
    enabled = True

    def __init__(self):
        self.values = {}

    def log(self, tag, value, global_step=None):
        self.values[tag] = value


def find_checkpoint(saved_models, mode, game, run_id):
    checkpoint_dir = saved_models / f"standard_{mode}" / "ALE" / f"{game}-v5" / run_id / "ckpt"
    world_model_path = checkpoint_dir / "world_model.pth"
    agent_path = checkpoint_dir / "agent.pth"
    if world_model_path.is_file() and agent_path.is_file():
        return checkpoint_dir
    return None


def convert_config_dtypes(config):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    for section in (config["Models"]["WorldModel"], config["Models"]["Agent"]):
        if isinstance(section.get("dtype"), str):
            section["dtype"] = dtype_map[section["dtype"]]


def load_model_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)


def resolve_gpu5_device():
    """Resolve physical GPU 5 to its logical CUDA index after visibility remapping."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_ids = [device_id.strip() for device_id in visible_devices.split(",")]
        if "5" not in visible_ids:
            raise RuntimeError(
                "Physical GPU 5 is not visible. "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}."
            )
        device = torch.device(f"cuda:{visible_ids.index('5')}")
    else:
        device = torch.device("cuda:5")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Resolved GPU 5 to {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )
    return device


def evaluate_checkpoint(checkpoint_dir, game, seed, device, episodes):
    # Imports are delayed so CSV-only conversion remains usable without torch startup cost.
    import sys

    device = resolve_gpu5_device()
    torch.cuda.set_device(device)
    drama_root = Path(__file__).resolve().parents[1]
    if str(drama_root) not in sys.path:
        sys.path.insert(0, str(drama_root))
    from envs.my_atari import Atari
    from eval import eval_episodes
    from train import DotDict, build_agent, build_world_model

    with open(drama_root / "config_files" / "configure.yaml", encoding="utf-8") as config_file:
        config = DotDict(yaml.safe_load(config_file))

    config["BasicSettings"]["Env_name"] = f"ALE/{game}-v5"
    config["BasicSettings"]["Seed"] = int(seed)
    config["BasicSettings"]["Device"] = str(device)
    config["BasicSettings"]["Compile"] = False
    config["Evaluate"]["EpisodeNum"] = int(episodes)
    config["Evaluate"]["NumEnvs"] = min(int(config["Evaluate"]["NumEnvs"]), int(episodes))
    convert_config_dtypes(config)

    dummy_env = Atari(config.BasicSettings.Env_name, size=config.BasicSettings.ImageSize, seed=int(seed))
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    world_model = build_world_model(config, action_dim, device=device, is_discrete=True)
    agent = build_agent(config, action_dim, device=device, is_discrete=True)
    load_model_checkpoint(world_model, checkpoint_dir / "world_model.pth", device)
    load_model_checkpoint(agent, checkpoint_dir / "agent.pth", device)

    logger = EvaluationLogger()
    original_cwd = Path.cwd()
    try:
        # eval.py reads atari_performance.csv relative to the Drama root.
        os.chdir(drama_root)
        eval_episodes(config, world_model, agent, logger)
    finally:
        os.chdir(original_cwd)

    score = logger.values.get("evaluate/score")
    normalized_score = logger.values.get("evaluate/normalised_score")
    return score, normalized_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate missing Drama WandB scores from checkpoints.")
    parser.add_argument("--input", default="drama_wandb_runs.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--saved-models", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent
    drama_root = results_dir.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = results_dir / input_path
    output_path = Path(args.output) if args.output else input_path
    if not output_path.is_absolute():
        output_path = results_dir / output_path
    saved_models = Path(args.saved_models) if args.saved_models else drama_root / "saved_models"

    runs = pd.read_csv(input_path)
    required_columns = {"Run ID", "Game", "Seed", "Retrieval", "Eval Return"}
    missing_columns = required_columns - set(runs.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {sorted(missing_columns)}")

    if args.device != "cuda:5":
        raise ValueError("This evaluator must use GPU 5. Run with --device cuda:5.")
    device = resolve_gpu5_device()
    torch.cuda.set_device(device)
    missing_scores = runs["Eval Return"].map(is_missing)
    running_runs = (
        runs["State"].astype(str).str.lower() == "running"
        if "State" in runs
        else pd.Series(False, index=runs.index)
    )
    pending = runs[missing_scores & ~running_runs]
    print(f"Missing scores: {len(pending)}; running runs skipped: {int(running_runs.sum())}; evaluation device: {device}")

    for index, row in pending.iterrows():
        mode = str(row["Retrieval"]).upper()
        if mode not in {"O", "X"}:
            continue
        checkpoint_dir = find_checkpoint(
            saved_models, mode, str(row["Game"]), str(row["Run ID"])
        )
        if checkpoint_dir is None:
            print(f"Skipping {row['Run Name']}: checkpoint not found")
            continue

        try:
            score, normalized_score = evaluate_checkpoint(
                checkpoint_dir, str(row["Game"]), int(row["Seed"]), device, args.episodes
            )
        except Exception as error:
            print(f"Failed to evaluate {row['Run Name']}: {error}")
            continue

        if score is not None:
            runs.at[index, "Eval Return"] = score
        if "Eval Normalized Return" in runs.columns and normalized_score is not None:
            runs.at[index, "Eval Normalized Return"] = normalized_score
        print(f"Evaluated {row['Run Name']}: score={score}")

    runs.to_csv(output_path, index=False)
    print(f"Saved updated CSV to {output_path}")


if __name__ == "__main__":
    main()
