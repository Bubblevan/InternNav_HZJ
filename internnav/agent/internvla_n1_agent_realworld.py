import copy
import itertools
import os
import re
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"


class InternVLAN1AsyncAgent:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"args.model_path{args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=getattr(args, "attn_backend", "flash_attention_2"),
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        processor_use_fast = getattr(args, "processor_use_fast", "auto")
        processor_kwargs = {}
        if processor_use_fast != "auto":
            processor_kwargs["use_fast"] = processor_use_fast == "true"
        self.processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap
        self.max_new_tokens = getattr(args, "max_new_tokens", 128)
        self.kv_cache_mode = getattr(args, "kv_cache_mode", "disabled")
        self.kv_cache_debug = bool(getattr(args, "kv_cache_debug", False))
        self.attn_backend = getattr(args, "attn_backend", "flash_attention_2")
        self.processor_use_fast = processor_use_fast

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_output_ids = None
        self.last_s2_idx = -100
        self.kv_cache_stats = defaultdict(int)
        self.kv_cache_exception_types = defaultdict(int)
        self.kv_cache_error_samples = []
        self.last_kv_cache_event = {}

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_output_ids = None
        self.last_kv_cache_event = {}

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def get_kv_cache_stats(self):
        stats = dict(self.kv_cache_stats)
        if self.kv_cache_exception_types:
            stats["exception_type_counts"] = dict(self.kv_cache_exception_types)
        if self.kv_cache_error_samples:
            stats["exception_samples"] = list(self.kv_cache_error_samples)
        return stats

    def get_last_kv_cache_event(self):
        return dict(self.last_kv_cache_event)

    def _count_common_prefix(self, lhs, rhs):
        prefix = 0
        limit = min(lhs.shape[0], rhs.shape[0])
        while prefix < limit and lhs[prefix].item() == rhs[prefix].item():
            prefix += 1
        return prefix

    def _prepare_image_grid_thw(self, image_grid_thw):
        if isinstance(image_grid_thw, torch.Tensor):
            if image_grid_thw.ndim == 1:
                return image_grid_thw.unsqueeze(0)
            return image_grid_thw
        return torch.stack([thw if isinstance(thw, torch.Tensor) else torch.as_tensor(thw) for thw in image_grid_thw], dim=0)

    def _slice_last_image_pixel_values(self, pixel_values, image_grid_thw):
        grid = self._prepare_image_grid_thw(image_grid_thw)
        patch_counts = torch.prod(grid.to(torch.int64), dim=-1)
        last_patch_count = int(patch_counts[-1].item())
        if pixel_values.shape[0] < last_patch_count:
            raise ValueError("Not enough pixel_values rows for the last image")
        return pixel_values[-last_patch_count:], grid[-1:].to(pixel_values.device)

    def _decode_generation_output(self, output_ids, prompt_token_count):
        generated_ids = output_ids[0][prompt_token_count:]
        output_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text, generated_ids

    def _run_generate(self, generation_inputs, past_key_values=None):
        with torch.no_grad():
            outputs = self.model.generate(
                **generation_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
            )
        return outputs

    def _get_input_shape(self, inputs_like, key):
        if isinstance(inputs_like, dict):
            return inputs_like[key].shape
        return getattr(inputs_like, key).shape

    def _get_image_grid(self, inputs_like, key="image_grid_thw"):
        if isinstance(inputs_like, dict):
            return inputs_like[key]
        return getattr(inputs_like, key)

    def _record_kv_exception(self, exc, full_inputs, generation_inputs):
        exception_type = type(exc).__name__
        self.kv_cache_exception_types[exception_type] += 1
        sample = {
            "exception_type": exception_type,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "full_input_tokens": int(full_inputs.input_ids.shape[1]),
            "delta_input_tokens": int(self._get_input_shape(generation_inputs, "input_ids")[1]),
            "full_image_tokens": int(self._prepare_image_grid_thw(full_inputs.image_grid_thw).prod(dim=-1).sum().item()),
            "delta_image_tokens": int(
                self._prepare_image_grid_thw(self._get_image_grid(generation_inputs)).prod(dim=-1).sum().item()
            ),
            "episode_idx": int(self.episode_idx),
        }
        if len(self.kv_cache_error_samples) < 5:
            self.kv_cache_error_samples.append(sample)
        self.last_kv_cache_event = {
            "attempted": True,
            "used_cache": False,
            "exception_type": exception_type,
            "exception_message": str(exc),
            "full_input_tokens": sample["full_input_tokens"],
            "delta_input_tokens": sample["delta_input_tokens"],
            "full_image_tokens": sample["full_image_tokens"],
            "delta_image_tokens": sample["delta_image_tokens"],
        }

    def _build_cached_lookdown_inputs(self, full_inputs):
        if self.past_key_values is None or self.last_output_ids is None:
            self.kv_cache_stats["lookdown_no_cache_state"] += 1
            return None

        full_ids = full_inputs.input_ids[0].detach().cpu()
        cached_ids = self.last_output_ids.detach().cpu()
        prefix_len = self._count_common_prefix(full_ids, cached_ids)
        self.kv_cache_stats["lookdown_prefix_len_total"] += int(prefix_len)

        if prefix_len != cached_ids.shape[0]:
            self.kv_cache_stats["lookdown_prefix_mismatch"] += 1
            return None

        delta_input_ids = full_inputs.input_ids[:, prefix_len:]
        if delta_input_ids.shape[1] == 0:
            self.kv_cache_stats["lookdown_empty_delta"] += 1
            return None

        try:
            delta_pixel_values, delta_image_grid_thw = self._slice_last_image_pixel_values(
                full_inputs.pixel_values,
                full_inputs.image_grid_thw,
            )
        except Exception:
            self.kv_cache_stats["lookdown_image_slice_error"] += 1
            return None

        generation_inputs = {
            "input_ids": delta_input_ids.to(self.device),
            "attention_mask": torch.ones_like(delta_input_ids, device=self.device),
            "pixel_values": delta_pixel_values.to(self.device),
            "image_grid_thw": delta_image_grid_thw.to(self.device),
            # HF generate derives an empty cache_position when past_length > input_len.
            # Provide the explicit positions for the newly appended delta tokens.
            "cache_position": torch.arange(
                prefix_len,
                prefix_len + delta_input_ids.shape[1],
                dtype=torch.long,
                device=self.device,
            ),
        }
        self.kv_cache_stats["lookdown_cache_ready"] += 1
        self.kv_cache_stats["lookdown_delta_tokens_total"] += int(delta_input_ids.shape[1])
        return generation_inputs

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        dual_sys_output = S2Output()
        self.last_kv_cache_event = {}
        no_output_flag = self.output_action is None and self.output_latent is None
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)

            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}_look_down.jpg")
        if not look_down:
            self.conversation_history = []
            self.past_key_values = None
            self.last_output_ids = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
        full_inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        generation_inputs = full_inputs
        generation_past = None
        used_kv_cache = False
        if look_down:
            self.kv_cache_stats["lookdown_attempts"] += 1
            self.last_kv_cache_event = {
                "attempted": True,
                "used_cache": False,
                "mode": self.kv_cache_mode,
                "full_input_tokens": int(full_inputs.input_ids.shape[1]),
                "full_image_tokens": int(
                    self._prepare_image_grid_thw(full_inputs.image_grid_thw).prod(dim=-1).sum().item()
                ),
            }
            if self.kv_cache_mode == "lookdown_experimental":
                cached_inputs = self._build_cached_lookdown_inputs(full_inputs)
                if cached_inputs is not None:
                    generation_inputs = cached_inputs
                    generation_past = self.past_key_values
                    used_kv_cache = True
                    self.kv_cache_stats["lookdown_cache_hits"] += 1
                    self.last_kv_cache_event.update(
                        {
                            "cache_ready": True,
                            "delta_input_tokens": int(generation_inputs["input_ids"].shape[1]),
                            "delta_image_tokens": int(
                                self._prepare_image_grid_thw(generation_inputs["image_grid_thw"])
                                .prod(dim=-1)
                                .sum()
                                .item()
                            ),
                        }
                    )
                else:
                    self.kv_cache_stats["lookdown_cache_fallbacks"] += 1
                    self.last_kv_cache_event["fallback_reason"] = "cache_not_ready"
            else:
                self.kv_cache_stats["lookdown_cache_disabled"] += 1
                self.last_kv_cache_event["fallback_reason"] = "cache_disabled"

        t0 = time.time()
        try:
            outputs = self._run_generate(generation_inputs, past_key_values=generation_past)
        except Exception as exc:
            if used_kv_cache:
                self.kv_cache_stats["lookdown_cache_exceptions"] += 1
                self._record_kv_exception(exc, full_inputs, generation_inputs)
                outputs = self._run_generate(full_inputs, past_key_values=None)
                generation_inputs = full_inputs
                generation_past = None
                used_kv_cache = False
                self.last_kv_cache_event["fallback_reason"] = "cache_exception"
            else:
                raise
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output, generated_ids = self._decode_generation_output(output_ids, generation_inputs.input_ids.shape[1])
        with open(f"{self.save_dir}/llm_output_{self.episode_idx: 04d}.txt", 'w') as f:
            f.write(self.llm_output)
        if used_kv_cache:
            self.last_output_ids = torch.cat(
                [self.last_output_ids.to(output_ids[0].device), generation_inputs.input_ids[0], generated_ids], dim=0
            ).detach().cpu()
        else:
            self.last_output_ids = copy.deepcopy(output_ids[0]).detach().cpu()
        full_output_ids = self.last_output_ids.unsqueeze(0).to(self.device)
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        self.kv_cache_stats["s2_calls_total"] += 1
        if look_down:
            self.kv_cache_stats["lookdown_used_cache_total"] += int(used_kv_cache)
            self.kv_cache_stats["lookdown_generate_seconds_ms_total"] += int((t1 - t0) * 1000)
            self.last_kv_cache_event["used_cache"] = bool(used_kv_cache)
            self.last_kv_cache_event["generation_seconds"] = float(t1 - t0)
        if self.kv_cache_debug:
            print(
                f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s "
                f"kv_mode={self.kv_cache_mode} look_down={look_down} used_cache={used_kv_cache}"
            )
        else:
            print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = self._prepare_image_grid_thw(full_inputs.image_grid_thw).to(full_inputs.pixel_values.device)
            pixel_values = full_inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(full_output_ids, pixel_values, image_grid_thw)
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs
