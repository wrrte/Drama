import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama

import pandas as pd

from utils import seed_np_torch, WandbLogger, is_logging_enabled
from replay_buffer import ReplayBuffer
import agents
from sub_models.world_models import WorldModel
from line_profiler import profile
import yaml
from envs.my_memory_maze import MemoryMaze
from envs.my_atari import Atari
from eval import eval_episodes
import warnings
import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval import RetrievalContextManager
from training_branches import (parse_retrieval_mode, capture_rng_state,
                               restore_rng_state, launch_training_branches, save_final_models)
from training_checkpoint import save_branch_checkpoint, load_branch_checkpoint


def retrieval_warmup(config, step, episode_rewards, state):
    warmup_steps = int(config.get("warmup_steps", 50000))
    if warmup_steps >= 0:
        return step < warmup_steps
    if warmup_steps != -1:
        raise ValueError("Retrieval.warmup_steps must be nonnegative or -1 for dynamic warmup")
    if state.get("warmup_finished", False):
        return False
    if step >= config.get("max_warmup_steps", 90000):
        state["warmup_finished"] = True
    elif step >= config.get("min_warmup_steps", 5000):
        met_step = state.get("dynamic_warmup_met_step", -1)
        if met_step < 0 and len(episode_rewards) >= 25:
            smoothed = pd.Series(list(episode_rewards)).rolling(5, min_periods=1).mean()
            band = smoothed.rolling(20, min_periods=5)
            if smoothed.iloc[-1] > (band.mean() + 2.6 * band.std()).iloc[-1]:
                met_step = state["dynamic_warmup_met_step"] = step
        if met_step >= 0 and step >= max(met_step, config.get("dynamic_warmup_target_steps", 20000)):
            state["warmup_finished"] = True
    return not state.get("warmup_finished", False)


@profile
def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, batch_length, logger, epoch, global_step,
                           agent=None, retrieval_manager=None, imagine_context_length=8, is_warmup=False):
    log_metrics = is_logging_enabled(logger)
    epoch_reconstruction_loss_list = []
    epoch_reward_loss_list = []
    epoch_termination_loss_list = []
    epoch_dynamics_loss_list = []
    epoch_dynamics_real_kl_div_list = []
    epoch_representation_loss_list = []
    epoch_representation_real_kl_div_list = []
    epoch_total_loss_list = []
    use_retrieval = retrieval_manager is not None and retrieval_manager.enabled
    num_triggered = 0
    for e in range(epoch):
        sample = replay_buffer.sample(batch_size, batch_length, imagine=False, return_indices=use_retrieval)
        obs, action, reward, termination = sample[:4]
        metrics = world_model.update(
            obs, action, reward, termination, global_step=global_step,
            epoch_step=e, logger=logger, return_metrics=log_metrics,
            return_latent=use_retrieval,
        )
        if use_retrieval:
            metrics, latent = metrics
            with torch.no_grad():
                with torch.autocast(device_type=torch.device(world_model.device).type,
                                    dtype=torch.bfloat16, enabled=getattr(agent, "use_amp", False)):
                    values = agent.value(latent.float()).squeeze(-1)
                num_triggered += retrieval_manager.add_batch_transitions(
                    values, reward, termination, agent.gamma, sample[4], sample[5],
                    replay_buffer.max_length, skip_len=imagine_context_length, is_warmup=is_warmup)
        if not log_metrics:
            continue
        reconstruction_loss, reward_loss, termination_loss, \
        dynamics_loss, dynamics_real_kl_div, representation_loss, \
        representation_real_kl_div, total_loss = metrics

        epoch_reconstruction_loss_list.append(reconstruction_loss)
        epoch_reward_loss_list.append(reward_loss)
        epoch_termination_loss_list.append(termination_loss)
        epoch_dynamics_loss_list.append(dynamics_loss)
        epoch_dynamics_real_kl_div_list.append(dynamics_real_kl_div)
        epoch_representation_loss_list.append(representation_loss)
        epoch_representation_real_kl_div_list.append(representation_real_kl_div)
        epoch_total_loss_list.append(total_loss)
    if log_metrics:
        logger.log("WorldModel/reconstruction_loss", np.mean(epoch_reconstruction_loss_list), global_step=global_step)
        # logger.log("WorldModel/augmented_reconstruction_loss", augmented_reconstruction_loss.item(), global_step=global_step)
        logger.log("WorldModel/reward_loss",np.mean(epoch_reward_loss_list), global_step=global_step)
        logger.log("WorldModel/termination_loss", np.mean(epoch_termination_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_loss", np.mean(epoch_dynamics_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_real_kl_div", np.mean(epoch_dynamics_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/representation_loss", np.mean(epoch_representation_loss_list), global_step=global_step)
        logger.log("WorldModel/representation_real_kl_div", np.mean(epoch_representation_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/total_loss", np.mean(epoch_total_loss_list), global_step=global_step)    
    if use_retrieval:
        logger.log("Retrieval/triggered_anchors_step", num_triggered, global_step=global_step)
        for name in ("ema_mean", "ema_var", "ema_vd_mean", "ema_vd_var"):
            logger.log(f"Retrieval/{name}", getattr(retrieval_manager, name).mean(), global_step=global_step)

@profile
@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger, global_step, retrieval_manager=None, is_warmup=False):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    log_video = log_video and is_logging_enabled(logger)
    ret_obs = None
    lazy_hit_rate = 0.0
    batch_weights = None
    random_batch_size = imagine_batch_size
    if retrieval_manager is not None and retrieval_manager.enabled and not is_warmup:
        cfg = retrieval_manager.config
        ret_obs, ret_action, candidates, lazy_hit_rate, weights, anchors, indices = retrieval_manager.retrieve_contexts(
            replay_buffer, world_model, max_anchors=cfg.get("max_anchors", 16),
            multiplier=cfg.get("multiplier", 16), target=cfg.get("target", 16),
            max_contexts=cfg.get("max_contexts", 256), return_indices=True)
        logger.log("Retrieval/candidates_before_max", candidates, global_step=global_step)
        logger.log("Retrieval/lazy_rebuild_hit_rate", lazy_hit_rate, global_step=global_step)
        if ret_obs is not None:
            reduction = cfg.get("batch_size_reduction", "retrieved")
            count = ret_obs.shape[0]
            reduction_count = count if reduction == "retrieved" else ((count + anchors) // 2 if reduction == "half" else anchors)
            random_batch_size = max(0, imagine_batch_size - reduction_count)
            # Retrieval indices point to the LAST frame of each context.
            ends = np.asarray([p for p, _ in indices])
            starts = (ends - imagine_context_length + 1) % replay_buffer.max_length
            rows = (starts[:, None] + np.arange(imagine_context_length)) % replay_buffer.max_length
            storage_rows = torch.as_tensor(rows, device=replay_buffer.device) if replay_buffer.store_on_gpu else rows
            ret_reward = torch.as_tensor(replay_buffer.reward_buffer[storage_rows], device=world_model.device)
            ret_term = torch.as_tensor(replay_buffer.termination_buffer[storage_rows], device=world_model.device)
            counter_rows = torch.as_tensor(starts, device=replay_buffer.device) if replay_buffer.store_on_gpu else starts
            if replay_buffer.store_on_gpu:
                replay_buffer.imagined_counter.index_add_(0, counter_rows, torch.ones_like(counter_rows, dtype=replay_buffer.imagined_counter.dtype))
            else:
                np.add.at(replay_buffer.imagined_counter, counter_rows, 1)
            logger.log("Retrieval/retrieved_contexts", count, global_step=global_step)
            logger.log("Retrieval/active_anchors_queue", len(retrieval_manager.active_anchors), global_step=global_step)
    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        random_batch_size, imagine_context_length, imagine=True)
    if ret_obs is not None:
        sample_obs = torch.cat((sample_obs, ret_obs))
        sample_action = torch.cat((sample_action, ret_action))
        sample_reward = torch.cat((sample_reward, ret_reward))
        sample_termination = torch.cat((sample_termination, ret_term))
        batch_weights = torch.cat((torch.ones(random_batch_size, device=world_model.device),
                                   torch.tensor(weights, device=world_model.device)))
    imagine_batch_size = sample_obs.shape[0]
    if world_model.model == 'Transformer':
        latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
         latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data2(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    return latent, action, old_logits, context_latent, sample_reward, sample_termination, reward_hat, termination_hat, batch_weights, lazy_hit_rate

@profile
def joint_train_world_model_agent(config, logdir,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  logger, resume_state=None, resume_rng=None):
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)


    if config.BasicSettings.Env_name.startswith('ALE'):
        env = Atari(config.BasicSettings.Env_name, size=config.BasicSettings.ImageSize, seed=config.BasicSettings.Seed)
    elif config.BasicSettings.Env_name.startswith('memory'):
        env = MemoryMaze(config.BasicSettings.Env_name, size=config.BasicSettings.ImageSize, seed=config.BasicSettings.Seed)
    elif config.BasicSettings.Env_name.startswith('dm_'):
        from envs.my_dmc import DMControl
        # Parse dm_control environment name: dm_domain_task
        # Example: dm_cheetah_run, dm_walker_walk, dm_humanoid_stand
        parts = config.BasicSettings.Env_name.split('_')
        domain_name = parts[1]
        task_name = '_'.join(parts[2:])  # Handle multi-word tasks
        env = DMControl(
            domain_name=domain_name,
            task_name=task_name,
            action_repeat=config.BasicSettings.ActionRepeat if hasattr(config.BasicSettings, 'ActionRepeat') else 2,
            size=config.BasicSettings.ImageSize,
            camera_id=config.BasicSettings.CameraId if hasattr(config.BasicSettings, 'CameraId') else 0,
            seed=config.BasicSettings.Seed,
            length=config.BasicSettings.MaxEpisodeSteps if hasattr(config.BasicSettings, 'MaxEpisodeSteps') else 1000
        )
        is_discrete = False
    else:
        assert ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')
    is_discrete = hasattr(env.action_space, 'n')
    env.action_space.seed(config.BasicSettings.Seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)

    # Benchmark handling (only for Atari)
    if config.BasicSettings.Env_name.startswith('ALE'):
        atari_benchmark_df = pd.read_csv(Path(__file__).resolve().parent / "atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
        atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        game_benchmark_df = atari_benchmark_df.get(atari_pure_name)
    else:
        game_benchmark_df = None
    
    sum_reward = 0
    current_ob, info = env.reset()
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)

    state = dict(resume_state or {})
    episode_rewards = deque(state.get("episode_rewards", []), maxlen=200)
    retrieval_config = dict(config.JointTrainAgent.get("Retrieval", {"enable": False}))
    retrieval_config["context_length"] = config.JointTrainAgent.ImagineContextLength
    retrieval_mode = parse_retrieval_mode(retrieval_config.get("enable", False))
    disabled_rng = capture_rng_state() if retrieval_mode is False else None
    retrieval_manager = RetrievalContextManager(
        num_envs=1, config=retrieval_config,
        latent_dim=config.Models.WorldModel.CategoricalDim * config.Models.WorldModel.ClassDim,
        device=world_model.device)
    if disabled_rng is not None:
        restore_rng_state(disabled_rng)
    if "retrieval" in state:
        retrieval_manager.load_state_dict(state["retrieval"])
    last_rebuild_step = state.get("last_rebuild_step", -retrieval_config.get("global_rebuild_cooldown", 2000))
    hash_built = state.get("hash_built", False)
    if resume_rng is not None:
        restore_rng_state(resume_rng)

    # sample and train
    for total_steps in tqdm(range(state.get("next_step", 0), config.JointTrainAgent.SampleMaxSteps), desc='Training'):
        is_retrieval_warmup = retrieval_warmup(retrieval_config, total_steps, episode_rewards, state)
        if retrieval_manager.enabled and not is_retrieval_warmup and not hash_built and replay_buffer.ready():
            world_model.eval()
            retrieval_manager.rebuild_all_hash_buckets(replay_buffer, world_model, chunk_size=1024)
            hash_built = True
            last_rebuild_step = total_steps
            logger.log("Retrieval/warmup_ended_at_step", total_steps, global_step=total_steps)
        if retrieval_mode == "Both" and not is_retrieval_warmup:
            # Both branches start a fresh episode; stop contexts crossing that boundary.
            if replay_buffer.length:
                replay_buffer.episode_end_buffer[replay_buffer.last_pointer] = True
            state.update(next_step=total_steps, episode_rewards=list(episode_rewards),
                         retrieval=retrieval_manager.state_dict(), last_rebuild_step=last_rebuild_step,
                         hash_built=hash_built)
            checkpoint_dir = str(Path(logdir, "shared_warmup").resolve())
            save_branch_checkpoint(checkpoint_dir, config, world_model, agent, replay_buffer, state)
            env.close()
            return checkpoint_dir
        # sample part >>>
        device_obs = None
        device_action = None
        if replay_buffer.ready('world_model'):
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = env.action_space.sample()
                    device_action = torch.as_tensor(action, device=world_model.device)
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = torch.stack(list(context_action))

                    # FIXED: Handle both discrete and continuous actions
                    if is_discrete:
                        # Discrete: shape is (L,) -> reshape to (1, L)
                        model_context_action = rearrange(model_context_action, "L -> 1 L")
                    else:
                        # Continuous: shape is (L, A) -> reshape to (1, L, A)
                        model_context_action = rearrange(model_context_action, "L A -> 1 L A")
                    
                    if world_model.model == 'Transformer':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    env_actions, device_actions = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False, return_device_action=True,
                    )
                    action = env_actions[0]
                    device_action = device_actions[0]

            # Upload the uint8 image once; keep normalization and context timing unchanged.
            device_obs = torch.from_numpy(current_ob).to(world_model.device)
            context_obs.append(rearrange(device_obs.float(), "H W C -> 1 1 C H W")/255)
            context_action.append(device_action.to(dtype=torch.float32))
        else:
            action = env.action_space.sample()

        ob, reward, is_last, info = env.step(action)
        replay_buffer.append(
            device_obs if replay_buffer.store_on_gpu and device_obs is not None else current_ob,
            device_action if replay_buffer.store_on_gpu and device_action is not None else action,
            reward, info['is_terminal'],
            episode_end=is_last,
        )
        if retrieval_manager.enabled and replay_buffer.ready():
            world_model.eval()
            with torch.no_grad():
                hash_obs = torch.as_tensor(current_ob, device=world_model.device)
                hash_obs = rearrange(hash_obs.float(), "H W C -> 1 1 C H W") / 255
                hash_latent = world_model.encode_obs(hash_obs, sample_mode=retrieval_manager.hash_sample_mode).squeeze(1)
                retrieval_manager.add_transition(replay_buffer.last_pointer, 0, hash_latent)

        sum_reward += reward
        current_ob = ob

        if is_last:
            episode_rewards.append(sum_reward)
            logger.log(f"episode/score", sum_reward, global_step=total_steps)
            logger.log(f"episode/length", info["episode_frame_number"], global_step=total_steps)  # framskip=4
            if config.BasicSettings.Env_name.startswith('ALE'):
                logger.log(f"episode/normalised score", (sum_reward - game_benchmark_df['Random'])/(game_benchmark_df['Human'] - game_benchmark_df['Random']), global_step=total_steps)
                for algorithm in game_benchmark_df.index[2:]:
                    denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                    if denominator != 0:
                        normalized_score = (sum_reward - game_benchmark_df['Random']) / denominator
                        logger.log(f"benchmark/normalised {algorithm} score", normalized_score, global_step=total_steps)
            
            sum_reward = 0
            current_ob, info = env.reset()
            context_obs.clear()
            context_action.clear()



        if replay_buffer.ready('world_model') and total_steps % (config.JointTrainAgent.TrainDynamicsEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeWorldModelAfterSteps:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=config.JointTrainAgent.BatchSize,
                batch_length=config.JointTrainAgent.BatchLength,
                logger=logger,
                epoch=config.JointTrainAgent.TrainDynamicsEpoch,
                global_step=total_steps,
                agent=agent, retrieval_manager=retrieval_manager,
                imagine_context_length=config.JointTrainAgent.ImagineContextLength,
                is_warmup=is_retrieval_warmup,
            )


        if replay_buffer.ready('behaviour') and total_steps % (config.JointTrainAgent.TrainAgentEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeBehaviourAfterSteps:
            log_video = total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0

            imagine_latent, agent_action, old_logits, context_latent, context_reward, context_termination, imagine_reward, imagine_termination, batch_weights, lazy_hit_rate = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=config.JointTrainAgent.ImagineBatchSize,
                imagine_context_length=config.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=config.JointTrainAgent.ImagineBatchLength,
                log_video=log_video,
                logger=logger,
                global_step=total_steps, retrieval_manager=retrieval_manager, is_warmup=is_retrieval_warmup,
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logits=old_logits,
                context_latent=context_latent,
                context_reward=context_reward,
                context_termination=context_termination,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
                global_step=total_steps, weights=batch_weights,
            )
            if retrieval_manager.enabled and not is_retrieval_warmup:
                rebuild = (retrieval_config.get("global_rebuild_enable", True)
                           and lazy_hit_rate < retrieval_config.get("global_rebuild_threshold", 0.06)
                           and total_steps - last_rebuild_step >= retrieval_config.get("global_rebuild_cooldown", 2000))
                if rebuild:
                    retrieval_manager.rebuild_all_hash_buckets(replay_buffer, world_model, chunk_size=1024)
                    last_rebuild_step = total_steps
                logger.log("Retrieval/global_rebuild_triggered", float(rebuild), global_step=total_steps)

        if config.Evaluate.DuringTraining and total_steps % (config.Evaluate.EverySteps // config.JointTrainAgent.NumEnvs) == 0:
            _ = eval_episodes(config, world_model, agent, logger, total_steps)
        if config.JointTrainAgent.SaveModels and total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"{logdir}/ckpt/world_model.pth")
            torch.save(agent.state_dict(), f"{logdir}/ckpt/agent.pth")

    env.close()
    if retrieval_mode == "Both":
        raise ValueError("Retrieval warmup did not finish before SampleMaxSteps; no branches could run")
    if config.JointTrainAgent.SaveModels or resume_state is not None:
        # Both children must keep their final result even when periodic saves are disabled.
        save_final_models(Path(logdir) / "ckpt", world_model, agent,
                          config.JointTrainAgent.SampleMaxSteps,
                          filenames=("world_model.pth", "agent.pth"))



def build_world_model(conf, action_dim, device, is_discrete=True):
    return WorldModel(
        action_dim = action_dim,
        config = conf, 
        device = device,
        is_discrete=is_discrete,
    ).cuda(device)


def build_agent(conf, action_dim, device, is_discrete=True):
    if conf.Models.Agent.Policy == 'AC':
        return agents.ActorCriticAgent(
            conf = conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)
    elif conf.Models.Agent.Policy == 'PPO':
        return agents.PPOAgent(
            conf=conf,
            action_dim=action_dim,
            device = device,
            is_discrete=is_discrete,
        ).cuda(device)        


class DotDict(dict):
    """Dictionary with dot notation access."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'")
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_or_create(self, key_path, value):
        keys = key_path.split('.')
        d = self
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = DotDict()
            d = d[key]
        d[keys[-1]] = value

# Function to parse and update config from arguments
def parse_args_and_update_config(config, prefix='', argv=None):
    parser = argparse.ArgumentParser()

    # Map string dtype to torch dtype
    def dtype_mapper(dtype_str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        return dtype_map[dtype_str]

    def add_arguments(config, prefix=''):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(value, prefix + key + '.')
            elif prefix + key == 'JointTrainAgent.Retrieval.enable':
                parser.add_argument(f'--{prefix}{key}', type=parse_retrieval_mode, default=value)
            elif isinstance(value, bool):
                # Special handling for boolean arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: x.lower() in ['true', '1', 'yes'], default=value)
            elif key == 'dtype':
                # Special handling for dtype arguments
                parser.add_argument(f'--{prefix}{key}', type=dtype_mapper, default=value)
            elif isinstance(value, (list, dict)):
                # Use a custom converter for list/dict-like arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: ast.literal_eval(x), default=value)
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), default=value)

    def update_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    add_arguments(config, prefix)
    
    args = parser.parse_args(argv)
    args_dict = vars(args)
    
    for arg_key, arg_value in args_dict.items():
        if arg_value is not None:
            keys = arg_key.split('.')
            update_dict(config, keys, arg_value)
    
    return config

def update_model_parameters(config, world_model, agent):
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    print(f'World model total parameters: {sum([p.numel() for p in world_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    print(f'Dynamic model parameters: {sum([p.numel() for p in world_model.sequence_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.EncoderParamNum', sum([p.numel() for p in world_model.encoder.parameters()]))
    print(f'Encoder parameters: {sum([p.numel() for p in world_model.encoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DecoderParamNum', sum([p.numel() for p in world_model.image_decoder.parameters()]))
    print(f'Decoder parameters: {sum([p.numel() for p in world_model.image_decoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DiscretisationLayerParamNum', sum([p.numel() for p in world_model.dist_head.parameters()]))
    print(f'Discretisation layer parameters: {sum([p.numel() for p in world_model.dist_head.parameters()]):,}')
    
    actor = agent.actor if hasattr(agent, 'actor') else agent.actor_mean
    config.update_or_create('Models.Agent.ActorParamNum', sum(p.numel() for p in actor.parameters()))
    print(f'Actor parameters: {sum(p.numel() for p in actor.parameters()):,}')
    
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    print(f'Critic parameters: {sum([p.numel() for p in agent.critic.parameters()]):,}')

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    internal_parser = argparse.ArgumentParser(add_help=False)
    internal_parser.add_argument('--branch_checkpoint', default=None)
    internal_parser.add_argument('--branch_mode', type=parse_retrieval_mode, default=None)
    branch_args, config_args = internal_parser.parse_known_args()

    with open(Path(__file__).resolve().parent / 'config_files/configure.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config = parse_args_and_update_config(config, argv=config_args)
    if branch_args.branch_checkpoint:
        # The exact effective parent config is authoritative for both children.
        config = torch.load(Path(branch_args.branch_checkpoint) / 'config.pt', map_location='cpu', weights_only=False)
        if branch_args.branch_mode not in (True, False):
            raise ValueError('--branch_checkpoint requires --branch_mode True or False')
        config['JointTrainAgent']['Retrieval']['enable'] = branch_args.branch_mode
        config['n'] = config['n'].removesuffix('_Both')
    elif branch_args.branch_mode is not None:
        raise ValueError('--branch_mode requires --branch_checkpoint')
    mode = parse_retrieval_mode(config['JointTrainAgent'].get('Retrieval', {}).get('enable', False))
    config['JointTrainAgent'].setdefault('Retrieval', {})['enable'] = mode
    config['n'] += '_Both' if mode == 'Both' else ('_O' if mode else '_X')
    config = DotDict(config)
    if config.JointTrainAgent.NumEnvs != 1:
        raise ValueError('Drama training uses one environment; JointTrainAgent.NumEnvs must be 1')
    warmup_steps = config.JointTrainAgent.Retrieval.get('warmup_steps', 50000)
    if warmup_steps < -1:
        raise ValueError('Retrieval.warmup_steps must be nonnegative or -1')
    if mode == 'Both' and warmup_steps >= config.JointTrainAgent.SampleMaxSteps:
        raise ValueError('Both requires Retrieval.warmup_steps < JointTrainAgent.SampleMaxSteps')
    reduction = config.JointTrainAgent.Retrieval.get('batch_size_reduction', 'retrieved')
    if reduction not in ('anchors', 'retrieved', 'half'):
        raise ValueError('Retrieval.batch_size_reduction must be anchors, retrieved, or half')

    device = torch.device(config.BasicSettings.Device)
    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    
    # getting action_dim with dummy env
    if config.BasicSettings.Env_name.startswith('ALE'):
        dummy_env = Atari(config.BasicSettings.Env_name)
    elif config.BasicSettings.Env_name.startswith('memory'):
        dummy_env = MemoryMaze(config.BasicSettings.Env_name)
    elif config.BasicSettings.Env_name.startswith('dm_'):
        from envs.my_dmc import DMControl
        parts = config.BasicSettings.Env_name.split('_')
        domain_name = parts[1]
        task_name = '_'.join(parts[2:])
        dummy_env = DMControl(domain_name=domain_name, task_name=task_name)
    else:
        raise ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')

    action_dim = dummy_env.action_space.n if hasattr(dummy_env.action_space, 'n') else dummy_env.action_space.shape[0]
    is_discrete = hasattr(dummy_env.action_space, 'n')
    dummy_env.close()

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device, is_discrete=is_discrete)
    agent = build_agent(config, action_dim, device=device, is_discrete=is_discrete)
    update_model_parameters(config, world_model, agent)
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    if config.BasicSettings.SavePath != 'None' and not branch_args.branch_checkpoint:
        print('Loading models')
        world_model.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/world_model.pth"))
        agent.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/agent.pth"))
   
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = f"./saved_models/{config.n}/{config.BasicSettings.Env_name}/{logger.run.id}"

    # build replay buffer
    replay_buffer = ReplayBuffer(
        config,
        device=device,
        action_dim=action_dim,
        is_discrete=is_discrete
    )
    resume_state = resume_rng = None
    if branch_args.branch_checkpoint:
        resume_state, resume_rng = load_branch_checkpoint(branch_args.branch_checkpoint, world_model, agent, replay_buffer)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        import json
        Path(logdir, 'shared_warmup.json').write_text(json.dumps({
            'checkpoint': str(Path(branch_args.branch_checkpoint).resolve()),
            'next_step': resume_state['next_step'], 'retrieval_enabled': mode,
            'environment_reset': True,
        }, indent=2))

    # train
    checkpoint_dir = joint_train_world_model_agent(config, logdir, replay_buffer, world_model, agent, logger,
                                                   resume_state=resume_state, resume_rng=resume_rng)

    logger.close()
    if checkpoint_dir:
        command = [sys.executable, str(Path(__file__).resolve()), *config_args,
                   '--branch_checkpoint', checkpoint_dir]
        launch_training_branches(checkpoint_dir, command + ['--branch_mode', 'True'],
                                 command + ['--branch_mode', 'False'])
