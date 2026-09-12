"""Full warmup checkpoints used by the two independent retrieval branches."""
from pathlib import Path

import numpy as np
import torch

from training_branches import capture_rng_state


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def model_training_state(model):
    model = getattr(model, "_orig_mod", model)
    state = {"parameters": model.state_dict()}
    for name in ("optimizer", "scaler", "lr_scheduler", "warmup_scheduler"):
        if hasattr(model, name):
            state[name] = getattr(model, name).state_dict()
    state["ema"] = {name: getattr(model, name).scalar
                    for name in ("lowerbound_ema", "upperbound_ema") if hasattr(model, name)}
    # Drama's VecNormalize statistics are not registered torch buffers.
    state["normalizers"] = {
        name: {key: getattr(module.ob_rms, key) for key in ("mean", "var", "count")}
        for name, module in model.named_modules() if hasattr(module, "ob_rms")}
    return to_cpu(state)


def restore_model_training_state(model, state):
    model = getattr(model, "_orig_mod", model)
    model.load_state_dict(state["parameters"])
    for name in ("optimizer", "scaler", "lr_scheduler", "warmup_scheduler"):
        if name in state:
            getattr(model, name).load_state_dict(state[name])
    device = next(model.parameters()).device
    for name, scalar in state["ema"].items():
        getattr(model, name).scalar = scalar.to(device) if isinstance(scalar, torch.Tensor) else scalar
    modules = dict(model.named_modules())
    for name, stats in state["normalizers"].items():
        for key, value in stats.items():
            setattr(modules[name].ob_rms, key, value.to(device))


def save_branch_checkpoint(directory, config, world_model, agent, replay_buffer, loop_state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    length = replay_buffer.length
    buffers = {name: to_cpu(getattr(replay_buffer, name)[:length]) for name in (
        "obs_buffer", "action_buffer", "reward_buffer", "termination_buffer",
        "sampled_counter", "imagined_counter", "episode_end_buffer")}
    state = {"config": to_cpu(config), "world_model": model_training_state(world_model),
             "agent": model_training_state(agent), "buffers": buffers,
             "length": length, "last_pointer": replay_buffer.last_pointer,
             "loop": to_cpu(loop_state), "rng": capture_rng_state()}
    # Children can read the configuration without loading the large replay twice.
    torch.save(state["config"], directory / "config.pt")
    torch.save(state, directory / "training_state.pt")


def load_branch_checkpoint(directory, world_model, agent, replay_buffer):
    state = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=False)
    restore_model_training_state(world_model, state["world_model"])
    restore_model_training_state(agent, state["agent"])
    length = state["length"]
    if length > replay_buffer.max_length:
        raise ValueError("Warmup replay buffer exceeds the configured capacity")
    for name, value in state["buffers"].items():
        target = getattr(replay_buffer, name)
        if isinstance(target, torch.Tensor):
            target[:length].copy_(torch.as_tensor(value, device=target.device))
        else:
            target[:length] = np.asarray(value)
    replay_buffer.length = length
    replay_buffer.last_pointer = state["last_pointer"]
    return state["loop"], state["rng"]
