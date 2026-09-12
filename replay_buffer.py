import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, config, device="cuda", action_dim=1, is_discrete=True) -> None:
        self.store_on_gpu = config.BasicSettings.ReplayBufferOnGPU
        max_length = config.JointTrainAgent.BufferMaxLength
        obs_shape = (*config.BasicSettings.ImageSize, config.BasicSettings.ImageChannel)
        self.device = device
        self.num_envs = 1
        self.is_discrete = is_discrete  # NEW
        self.action_dim = action_dim  # NEW

        # Determine action buffer shape
        if is_discrete:
            action_shape = (max_length,)  # Scalar for discrete
        else:
            action_shape = (max_length, action_dim)  # Vector for continuous

        if self.store_on_gpu:
            self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty(action_shape, dtype=torch.float32, device=device, requires_grad=False)  # CHANGED
            self.reward_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.episode_end_buffer = torch.empty((max_length), dtype=torch.bool, device=device, requires_grad=False)
            self.sampled_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            self.imagined_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty(action_shape, dtype=np.float32)
            self.reward_buffer = np.empty((max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_length), dtype=np.float32)
            self.episode_end_buffer = np.empty((max_length), dtype=np.bool_)
            self.sampled_counter = np.zeros((max_length), dtype=np.int32)
            self.imagined_counter = np.zeros((max_length), dtype=np.int32)

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.world_model_warmup_length = config.JointTrainAgent.WorldModelWarmUp
        self.behaviour_warmup_length = config.JointTrainAgent.BehaviourWarmUp
        self.tau = config.JointTrainAgent.Tau
        self.imagination_tau = config.JointTrainAgent.ImaginationTau
        self.alpha = config.JointTrainAgent.Alpha
        self.beta = config.JointTrainAgent.Beta
        self.batch_scale_factor = config.JointTrainAgent.ImagineBatchSize / config.JointTrainAgent.BatchSize

    def ready(self, model_name='world_model'):
        return self.length  > self.world_model_warmup_length if model_name == 'world_model' else self.length  > self.behaviour_warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length, imagine=False, return_indices=False):
        """Sample chronological sequences, optionally returning their physical starts."""
        if batch_size < 0 or batch_length <= 0:
            raise ValueError("batch_size must be non-negative and batch_length positive")
        available = self.length + 1 - batch_length
        if batch_size and available <= 0:
            raise ValueError("Replay buffer does not contain a complete sequence")
        oldest = (self.last_pointer + 1) % self.max_length if self.length == self.max_length else 0
        if self.store_on_gpu:
            candidates = (torch.arange(max(0, available), device=self.device) + oldest) % self.max_length
            counts = self.sampled_counter[candidates]
            imagine_counts = self.imagined_counter[candidates] / self.batch_scale_factor
            if not batch_size:
                start_indexes = torch.empty(0, dtype=torch.long, device=self.device)
            elif imagine:
                linear_penalty = torch.maximum(torch.zeros_like(counts), counts - imagine_counts)
                score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                score = score / self.imagination_tau
                probabilities = torch.softmax(score, dim=0)
                start_indexes = candidates[torch.multinomial(probabilities, batch_size, replacement=batch_size > available)]
            else:
                logits = -counts / self.tau
                probabilities = torch.softmax(logits, dim=0)
                start_indexes = candidates[torch.multinomial(probabilities, batch_size, replacement=batch_size > available)]
            counter = self.imagined_counter if imagine else self.sampled_counter
            counter.index_add_(0, start_indexes, torch.ones_like(start_indexes, dtype=counter.dtype))
            indexes = (start_indexes.unsqueeze(-1) + torch.arange(batch_length, device=self.device)) % self.max_length
            obs = self.obs_buffer[indexes].float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = self.action_buffer[indexes]
            reward = self.reward_buffer[indexes]
            termination = self.termination_buffer[indexes]
        else:
            candidates = (np.arange(max(0, available)) + oldest) % self.max_length
            start_indexes = np.empty(0, dtype=np.int64)
            if batch_size > 0:
                counts = self.sampled_counter[candidates]
                imagine_counts = self.imagined_counter[candidates] / self.batch_scale_factor
                if imagine:
                    linear_penalty = np.maximum(np.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score /= self.imagination_tau
                else:
                    score = -counts / self.tau

                exp_score = np.exp(score - np.max(score))
                probabilities = exp_score / np.sum(exp_score)

                start_indexes = np.random.choice(candidates, size=batch_size, replace=batch_size > available, p=probabilities)
            counter = self.imagined_counter if imagine else self.sampled_counter
            np.add.at(counter, start_indexes, 1)
            indexes = (start_indexes[:, np.newaxis] + np.arange(batch_length)) % self.max_length
            obs = torch.from_numpy(self.obs_buffer[indexes]).to(self.device).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(self.action_buffer[indexes]).to(self.device)
            reward = torch.from_numpy(self.reward_buffer[indexes]).to(self.device)
            termination = torch.from_numpy(self.termination_buffer[indexes]).to(self.device)

        result = (obs, action, reward, termination)
        if return_indices:
            base_indexes = start_indexes.detach().cpu().numpy() if torch.is_tensor(start_indexes) else start_indexes
            return (*result, base_indexes, np.zeros(batch_size, dtype=np.int64))
        return result

    def retrieval_view(self):
        """Expose an environment axis without copying replay storage."""
        return _RetrievalReplayView(self)

    def is_valid_context(self, pointer, env_idx, context_length):
        if env_idx != 0 or context_length <= 0 or not 0 <= pointer < self.length:
            return False
        age = (self.last_pointer - pointer) % self.max_length
        if age + context_length > self.length:
            return False
        indexes = (np.arange(pointer - context_length + 1, pointer)) % self.max_length
        if self.store_on_gpu:
            indexes = torch.as_tensor(indexes, device=self.device)
            ends = (self.termination_buffer[indexes] > 0.5) | self.episode_end_buffer[indexes]
            return not bool(ends.any())
        ends = (self.termination_buffer[indexes] > 0.5) | self.episode_end_buffer[indexes]
        return not bool(np.any(ends))

    def append(self, obs, action, reward, termination, episode_end=None):
        # Time-limit resets break retrieval contexts but retain critic bootstrapping.
        if episode_end is None:
            episode_end = termination > 0.5
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)
        self.sampled_counter[self.last_pointer] = 0
        self.imagined_counter[self.last_pointer] = 0
        if self.store_on_gpu:
            # Reuse tensors already uploaded for the policy context.
            obs_tensor = obs.detach() if torch.is_tensor(obs) else torch.from_numpy(obs)
            self.obs_buffer[self.last_pointer] = obs_tensor
            action_tensor = action.detach() if torch.is_tensor(action) else torch.tensor(action, device=self.device)
            if self.is_discrete:
                self.action_buffer[self.last_pointer] = action_tensor
            else:
                # Ensure action is a vector
                if action_tensor.dim() == 0:
                    action_tensor = action_tensor.unsqueeze(0)
                self.action_buffer[self.last_pointer] = action_tensor
            self.reward_buffer[self.last_pointer] = torch.tensor(reward, device=self.device)
            self.termination_buffer[self.last_pointer] = torch.tensor(termination, device=self.device)
            self.episode_end_buffer[self.last_pointer] = torch.as_tensor(episode_end, device=self.device, dtype=torch.bool)
        else:
            self.obs_buffer[self.last_pointer] = obs
            if self.is_discrete:
                self.action_buffer[self.last_pointer] = action
            else:
                # Ensure action is stored as vector
                if isinstance(action, (int, float)):
                    action = np.array([action])
                self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            self.episode_end_buffer[self.last_pointer] = episode_end

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length


class _RetrievalReplayView:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __getattr__(self, name):
        value = getattr(self.replay_buffer, name)
        if name in {"obs_buffer", "action_buffer", "reward_buffer", "termination_buffer", "episode_end_buffer"}:
            return value[:, None]
        return value
