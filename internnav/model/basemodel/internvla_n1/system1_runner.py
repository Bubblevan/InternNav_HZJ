import json
from collections import defaultdict
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

from .internvla_n1_arch import (
    LatentEmbSize,
    MemoryEncoder,
    QFormer,
    SinusoidalPositionalEncoding,
    build_depthanythingv2,
    build_traj_dit,
)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def _load_config(model_path: str) -> SimpleNamespace:
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(**config)


def _load_subset_state_dict(model_path: str, tensor_names: list[str]) -> dict[str, torch.Tensor]:
    index_path = Path(model_path) / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data["weight_map"]

    shard_to_tensor_names: dict[str, list[str]] = defaultdict(list)
    for tensor_name in tensor_names:
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            continue
        shard_to_tensor_names[shard_name].append(tensor_name)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, shard_tensor_names in shard_to_tensor_names.items():
        shard_state = load_file(str(Path(model_path) / shard_name), device="cpu")
        for tensor_name in shard_tensor_names:
            local_name = tensor_name.removeprefix("model.")
            state_dict[local_name] = shard_state[tensor_name]

    return state_dict


class InternVLAN1System1Runner(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

        if "nextdit" not in config.system1:
            raise NotImplementedError(f"Unsupported system1 type: {config.system1}")

        self.traj_dit, self.noise_scheduler = build_traj_dit(config)
        self.action_encoder = nn.Linear(3, 384, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(384)
        self.action_decoder = nn.Linear(384, 3, bias=True)
        self.cond_projector = nn.Sequential(
            nn.Linear(config.hidden_size, LatentEmbSize),
            nn.GELU(approximate="tanh"),
            nn.Linear(LatentEmbSize, LatentEmbSize),
        )

        if "async" in config.system1:
            self.rgb_model = build_depthanythingv2(config)
            self.memory_encoder = MemoryEncoder()
            self.rgb_resampler = QFormer()

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "InternVLAN1System1Runner":
        config = _load_config(model_path)
        runner = cls(config)

        required_tensor_names = []
        for key in runner.state_dict().keys():
            if key.startswith("_resnet_"):
                continue
            required_tensor_names.append(f"model.{key}")

        state_dict = _load_subset_state_dict(model_path, required_tensor_names)
        missing_keys, unexpected_keys = runner.load_state_dict(state_dict, strict=False)
        missing_keys = [key for key in missing_keys if not key.startswith("_resnet_")]
        if missing_keys:
            raise RuntimeError(f"Missing S1 weights when loading {model_path}: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected S1 weights when loading {model_path}: {unexpected_keys}")

        if device is not None:
            runner.to(device=device, dtype=torch_dtype)
        else:
            runner.to(dtype=torch_dtype)
        runner.eval()
        return runner

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
        generator: Optional[torch.Generator] = None,
    ):
        if "nextdit" not in self.config.system1:
            raise NotImplementedError(f"Unsupported system1 type: {self.config.system1}")

        total_start = time.perf_counter()
        scheduler = FlowMatchEulerDiscreteScheduler()
        device = traj_latents.device
        dtype = traj_latents.dtype

        cond_project_start = time.perf_counter()
        traj_latents = self.cond_projector(traj_latents)
        cond_project_ms = (time.perf_counter() - cond_project_start) * 1000.0
        rgb_encode_ms = 0.0
        memory_encode_ms = 0.0
        if "async" in self.config.system1:
            with torch.no_grad():
                images_dp = images_dp.permute(0, 1, 4, 2, 3)
                images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                self.rgb_model.to(dtype)
                rgb_encode_start = time.perf_counter()
                images_dp_feat = (
                    self.rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                    .unflatten(dim=0, sizes=(1, -1))
                )
                rgb_encode_ms = (time.perf_counter() - rgb_encode_start) * 1000.0
                memory_encode_start = time.perf_counter()
                memory_feat = self.memory_encoder(images_dp_feat.flatten(1, 2))
                memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                memory_tokens = self.rgb_resampler(memory_feat)
                memory_encode_ms = (time.perf_counter() - memory_encode_start) * 1000.0
            hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
        else:
            hidden_states = traj_latents

        hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
        hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
        batch_size = traj_latents.shape[0]
        latent_size = predict_step_nums
        latent_channels = 3

        latents = randn_tensor(
            shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        dit_loop_start = time.perf_counter()
        action_decode_ms = 0.0
        for t in scheduler.timesteps:
            latent_features = self.action_encoder(latents)
            pos_ids = (
                torch.arange(latent_features.shape[1]).reshape(1, -1).repeat(batch_size, 1).to(latent_features.device)
            )
            pos_embed = self.pos_encoding(pos_ids)
            latent_features += pos_embed
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                z_latents=hidden_states_input,
            )

            action_decode_start = time.perf_counter()
            noise_pred = self.action_decoder(noise_pred)
            action_decode_ms += (time.perf_counter() - action_decode_start) * 1000.0

            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        dit_loop_ms = (time.perf_counter() - dit_loop_start) * 1000.0

        generator_seed = None
        if generator is not None:
            try:
                generator_seed = int(generator.initial_seed())
            except Exception:
                generator_seed = None
        diffusion_steps_total = int(len(scheduler.timesteps))
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self._last_generate_traj_metrics = {
            "s1_generate_traj_ms_total": total_ms,
            "s1_memory_encode_ms": memory_encode_ms,
            "s1_rgb_encode_ms": rgb_encode_ms,
            "s1_cond_project_ms": cond_project_ms,
            "s1_dit_loop_ms": dit_loop_ms,
            "s1_action_decode_ms": action_decode_ms,
            "dit_cache_enabled": False,
            "dit_cache_hit_rate": 0.0,
            "dit_cache_hits": 0,
            "dit_cache_misses": 0,
            "dit_cache_saved_ms_total": 0.0,
            "dit_cache_saved_ms_per_call": 0.0,
            "diffusion_steps_total": diffusion_steps_total,
            "diffusion_steps_reused": 0,
            "diffusion_steps_executed": diffusion_steps_total,
            "s1_generator_seed": generator_seed,
            "s1_deterministic_mode": generator is not None,
        }

        # Match InternVLAN1ForCausalLM.generate_traj(): [batch_size * num_sample_trajs, T, 3]
        return latents
