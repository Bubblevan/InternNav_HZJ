import argparse
import base64
import io
import json
import os
import sys
from enum import IntEnum

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import habitat
import numpy as np
import quaternion
import requests as http_requests
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.save_video_failures = getattr(args, 'save_video_failures', False)
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        self._apply_dataset_overrides(cfg.eval_settings)

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path
        cfg.env.env_settings['allowed_scene_ids'] = cfg.eval_settings.get("allowed_scene_ids")
        cfg.env.env_settings['allowed_episode_ids'] = cfg.eval_settings.get("allowed_episode_ids")
        cfg.env.env_settings['max_eval_episodes'] = cfg.eval_settings.get("max_eval_episodes")

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self._init_env_capabilities()

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
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
            }
        )
        if self.has_pitch_actions:
            self.actions2idx["↓"] = [5]

        self.num_history = self.model_args.num_history

        self.s2_vllm_url = getattr(self.model_args, 's2_vllm_url', None)
        self.s2_vllm_model = getattr(self.model_args, 's2_vllm_model', None)
        if self.s2_vllm_url:
            print(f"[HabitatVLNEvaluator] S2 vLLM backend enabled: {self.s2_vllm_url}")
            if self.s2_vllm_model is None:
                try:
                    resp = http_requests.get(f"{self.s2_vllm_url}/v1/models", timeout=5)
                    models = resp.json().get("data", [])
                    if models:
                        self.s2_vllm_model = models[0]["id"]
                except Exception as e:
                    print(f"[HabitatVLNEvaluator] Failed to detect vLLM model: {e}")
                    self.s2_vllm_model = "default"
            print(f"[HabitatVLNEvaluator] vLLM model: {self.s2_vllm_model}")

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def _apply_dataset_overrides(self, eval_settings):
        dataset_cfg = self.config.habitat.dataset
        dataset_path_override = eval_settings.get("dataset_path_override")
        scenes_dir_override = eval_settings.get("scenes_dir_override")
        dataset_split_override = eval_settings.get("dataset_split_override")
        allowed_scene_ids = eval_settings.get("allowed_scene_ids")
        allowed_episode_ids = eval_settings.get("allowed_episode_ids")
        max_eval_episodes = eval_settings.get("max_eval_episodes")

        with habitat.config.read_write(self.config):
            if dataset_path_override:
                dataset_cfg.data_path = dataset_path_override
            if scenes_dir_override:
                dataset_cfg.scenes_dir = scenes_dir_override
            if dataset_split_override:
                dataset_cfg.split = dataset_split_override
            if allowed_scene_ids is not None and "allowed_scene_ids" in dataset_cfg:
                dataset_cfg.allowed_scene_ids = list(allowed_scene_ids)
            if allowed_episode_ids is not None and "allowed_episode_ids" in dataset_cfg:
                dataset_cfg.allowed_episode_ids = [int(ep_id) for ep_id in allowed_episode_ids]
            if max_eval_episodes is not None and "max_eval_episodes" in dataset_cfg:
                dataset_cfg.max_eval_episodes = int(max_eval_episodes)

    @staticmethod
    def _pil_to_data_url(image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _conversation_to_openai(self, messages):
        out = []
        for msg in messages:
            items = []
            for p in msg["content"]:
                if p["type"] == "text":
                    items.append({"type": "text", "text": p["text"]})
                elif p["type"] == "image":
                    items.append({"type": "image_url", "image_url": {"url": self._pil_to_data_url(p["image"])}})
            out.append({"role": msg["role"], "content": items})
        return out

    def _vllm_generate(self, messages, max_new_tokens=128):
        openai_msgs = self._conversation_to_openai(messages)
        payload = {
            "model": self.s2_vllm_model,
            "messages": openai_msgs,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        resp = http_requests.post(f"{self.s2_vllm_url}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _init_env_capabilities(self):
        lab_sensors = getattr(self.config.habitat.task, "lab_sensors", None)
        lab_sensor_names = set(lab_sensors.keys()) if lab_sensors is not None else set()
        action_names = set(self.config.habitat.task.actions.keys())

        self.has_pose_sensors = "gps_sensor" in lab_sensor_names and "compass_sensor" in lab_sensor_names
        self.has_pitch_actions = "look_up" in action_names and "look_down" in action_names
        self.use_system1_local_policy = self.model_args.mode == 'dual_system' and self.has_pitch_actions

        print(
            "env_capabilities:",
            {
                "has_pose_sensors": self.has_pose_sensors,
                "has_pitch_actions": self.has_pitch_actions,
                "use_system1_local_policy": self.use_system1_local_policy,
            },
            flush=True,
        )

    def _pixel_goal_to_discrete_action(self, pixel_goal, image_width: int) -> int:
        left_threshold = int(image_width * 0.4)
        right_threshold = int(image_width * 0.6)
        if pixel_goal[0] < left_threshold:
            return action_code.LEFT
        if pixel_goal[0] > right_threshold:
            return action_code.RIGHT
        return action_code.FORWARD

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws, collision_counts, psi_rates = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws, collision_counts, psi_rates = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
            "collision_counts": collision_counts,
            "psi_rates": psi_rates,
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        if "collision_counts" in global_metrics:
            ccs = global_metrics["collision_counts"]
            prs = global_metrics["psi_rates"]
            result_all["avg_collision_count"] = float(ccs.mean().item()) if denom > 0 else 0.0
            result_all["psi_rate_all"] = float(prs.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        done_episodes = set()
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw, done_episodes

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
                    done_episodes.add((res.get('scene_id', ''), res.get('episode_id', -1)))
            if done_episodes:
                print(f"[Resume] Loaded {len(done_episodes)} completed episodes from {self.output_path}/progress.json")
        return sucs, spls, oss, nes, ndtw, done_episodes

    def _run_eval_dual_system(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw, done_episodes = self.resume_from_output_path()
        collision_counts: list = []
        psi_rates: list = []

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")
        _n_skipped = 0

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text

            if (scene_id, episode_id) in done_episodes:
                self.env.step(0)
                _n_skipped += 1
                process_bar.update(1)
                if _n_skipped <= 3 or _n_skipped % 100 == 0:
                    process_bar.write(f"[Resume] Skipping {scene_id}_{episode_id:04d} ({_n_skipped}/{len(done_episodes)} done)")
                continue

            if _n_skipped > 0 and _n_skipped == len(done_episodes):
                process_bar.write(f"[Resume] Skipped {_n_skipped} completed episodes, resuming from {scene_id}_{episode_id:04d}")

            print("episode start", episode_instruction)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0

            if self.save_video or self.save_video_failures:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []

            done = False
            flag = False
            pixel_goal = None

            # social metric accumulators (reset per episode)
            _psi_steps = 0          # steps where any human was within 1.2 m
            _sum_min_dist_h = 0.0
            _dist_h_count = 0

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if self.has_pitch_actions and action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    if self.has_pitch_actions:
                        down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                        down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                        look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                        depth_for_local = down_observations["depth"]
                        depth_for_local = filter_depth(depth_for_local.reshape(depth_for_local.shape[:2]), blur_type=None)
                        depth_for_local = depth_for_local * (self._max_depth - self._min_depth) + self._min_depth
                        depth_for_local = depth_for_local * 1000
                    else:
                        look_down_image = image.copy()
                        depth_for_local = depth

                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth_for_local.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    if self.has_pitch_actions:
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                if len(action_seq) == 0 and pixel_goal is None:
                    if self.has_pitch_actions and action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    if self.s2_vllm_url:
                        llm_outputs = self._vllm_generate(messages, max_new_tokens=128)
                        generated_ids = self.processor.tokenizer.encode(llm_outputs, add_special_tokens=False)
                        output_ids = torch.cat([
                            inputs.input_ids,
                            torch.tensor([generated_ids], device=inputs.input_ids.device),
                        ], dim=1)
                    else:
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=128,
                                do_sample=False,
                                use_cache=True,
                                past_key_values=None,
                                return_dict_in_generate=True,
                            ).sequences
                        llm_outputs = self.processor.tokenizer.decode(
                            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )

                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]

                        if not self.use_system1_local_policy:
                            action_seq = [self._pixel_goal_to_discrete_action(pixel_goal, rgb.shape[1])]
                            print('pixel_goal_fallback_actions', action_seq, flush=True)
                            pixel_goal = None
                            output_ids = None
                        else:
                            # look down --> horizontal
                            self.env.step(action_code.LOOKUP)
                            self.env.step(action_code.LOOKUP)

                            local_actions = []
                            pixel_values = inputs.pixel_values
                            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                            with torch.no_grad():
                                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                            # prepocess align with navdp
                            image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                            pix_goal_image = copy.copy(image_dp)
                            images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                            depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                            pix_goal_depth = copy.copy(depth_dp)
                            depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                            action_list = traj_to_actions(dp_actions)
                            if len(action_list) < MAX_STEPS:
                                action_list += [0] * (MAX_STEPS - len(action_list))

                            local_actions = action_list
                            if len(local_actions) >= MAX_LOCAL_STEPS:
                                local_actions = local_actions[:MAX_LOCAL_STEPS]

                            action = local_actions[0]
                            if action == action_code.STOP:
                                pixel_goal = None
                                output_ids = None
                                action = action_code.LEFT
                                observations, _, done, _ = self.env.step(action)
                                step_id += 1
                                messages = []
                                continue
                            print('predicted goal', pixel_goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if self.save_video or self.save_video_failures:
                    if info.get('top_down_map') is not None:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        frame = np.array(save_raw_image)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if self.has_pitch_actions and action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _step_info = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False
                    # track social metrics each non-pitch step
                    _dth = (_step_info or {}).get('distance_to_human')
                    if _dth:
                        _min_d = min(v[0] for v in _dth.values())
                        _sum_min_dist_h += _min_d
                        _dist_h_count += 1
                        if _min_d < 1.2:
                            _psi_steps += 1

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            # --- social metrics ---
            _collision_count = 0
            _cd = metrics.get('collisions_detail')
            if isinstance(_cd, dict):
                _collision_count = int(_cd.get('count', 0))
            _psi_rate = _psi_steps / max(step_id, 1)
            _avg_min_dist_h = _sum_min_dist_h / _dist_h_count if _dist_h_count > 0 else -1.0
            collision_counts.append(float(_collision_count))
            psi_rates.append(_psi_rate)

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}, "
                f"collisions: {_collision_count}, psi_rate: {_psi_rate:.3f}, "
                f"avg_min_dist_h: {_avg_min_dist_h:.2f}m"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                "collision_count": _collision_count,
                "psi_steps": _psi_steps,
                "psi_rate": round(_psi_rate, 4),
                "avg_min_dist_to_human": round(_avg_min_dist_h, 4),
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video: always if save_video=True; only failures if save_video_failures=True
            _is_failure = metrics['success'] == 0.0
            _should_save_video = (self.save_video or (self.save_video_failures and _is_failure))
            if _should_save_video and len(vis_frames) > 0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
            torch.tensor(collision_counts).to(self.device),
            torch.tensor(psi_rates).to(self.device),
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw, done_episodes = self.resume_from_output_path()
        collision_counts: list = []
        psi_rates: list = []

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")
        _n_skipped = 0

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text

            if (scene_id, episode_id) in done_episodes:
                self.env.step(0)
                _n_skipped += 1
                process_bar.update(1)
                if _n_skipped <= 3 or _n_skipped % 100 == 0:
                    process_bar.write(f"[Resume] Skipping {scene_id}_{episode_id:04d} ({_n_skipped}/{len(done_episodes)} done)")
                continue

            if _n_skipped > 0 and _n_skipped == len(done_episodes):
                process_bar.write(f"[Resume] Skipped {_n_skipped} completed episodes, resuming from {scene_id}_{episode_id:04d}")

            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False) if self.has_pose_sensors else None

            intrinsic_matrix = (
                get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
                if self.has_pose_sensors
                else None
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0

            if self.save_video or self.save_video_failures:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # social metric accumulators (reset per episode)
            _psi_steps = 0
            _sum_min_dist_h = 0.0
            _dist_h_count = 0

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                if self.has_pose_sensors:
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    agent_state = self.env._env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    tf_camera_to_episodic = (
                        xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                        @ get_axis_align_matrix()
                    )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if self.has_pitch_actions and action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if self.has_pitch_actions and action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    if self.s2_vllm_url:
                        llm_outputs = self._vllm_generate(messages, max_new_tokens=128)
                        generated_ids = self.processor.tokenizer.encode(llm_outputs, add_special_tokens=False)
                        output_ids = torch.cat([
                            inputs.input_ids,
                            torch.tensor([generated_ids], device=inputs.input_ids.device),
                        ], dim=1)
                    else:
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=128,
                                do_sample=False,
                                use_cache=True,
                                past_key_values=None,
                                return_dict_in_generate=True,
                            ).sequences
                        llm_outputs = self.processor.tokenizer.decode(
                            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )

                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]

                        if not (self.has_pose_sensors and self.has_pitch_actions):
                            action_seq = [self._pixel_goal_to_discrete_action(pixel_goal, rgb.shape[1])]
                            print('pixel_goal_fallback_actions', action_seq, flush=True)
                        else:
                            # look down --> horizontal
                            self.env.step(action_code.LOOKUP)
                            self.env.step(action_code.LOOKUP)

                            goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                            goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                            if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                                goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                            action = agent.get_next_action(goal)
                            if action == action_code.STOP:
                                goal = None
                                output_ids = None
                                action = action_code.LEFT  # random action to avoid deadlock
                                observations, _, done, _ = self.env.step(action)
                                step_id += 1
                                messages = []
                                continue
                            print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if self.save_video or self.save_video_failures:
                    if info.get('top_down_map') is not None:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        frame = np.array(save_raw_image)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if self.has_pitch_actions and action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _step_info = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False
                    # track social metrics each non-pitch step
                    _dth = (_step_info or {}).get('distance_to_human')
                    if _dth:
                        _min_d = min(v[0] for v in _dth.values())
                        _sum_min_dist_h += _min_d
                        _dist_h_count += 1
                        if _min_d < 1.2:
                            _psi_steps += 1

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            # --- social metrics ---
            _collision_count = 0
            _cd = metrics.get('collisions_detail')
            if isinstance(_cd, dict):
                _collision_count = int(_cd.get('count', 0))
            _psi_rate = _psi_steps / max(step_id, 1)
            _avg_min_dist_h = _sum_min_dist_h / _dist_h_count if _dist_h_count > 0 else -1.0
            collision_counts.append(float(_collision_count))
            psi_rates.append(_psi_rate)

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}, "
                f"collisions: {_collision_count}, psi_rate: {_psi_rate:.3f}, "
                f"avg_min_dist_h: {_avg_min_dist_h:.2f}m"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                "collision_count": _collision_count,
                "psi_steps": _psi_steps,
                "psi_rate": round(_psi_rate, 4),
                "avg_min_dist_to_human": round(_avg_min_dist_h, 4),
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video: always if save_video=True; only failures if save_video_failures=True
            _is_failure = metrics['success'] == 0.0
            _should_save_video = (self.save_video or (self.save_video_failures and _is_failure))
            if _should_save_video and len(vis_frames) > 0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
            torch.tensor(collision_counts).to(self.device),
            torch.tensor(psi_rates).to(self.device),
        )
