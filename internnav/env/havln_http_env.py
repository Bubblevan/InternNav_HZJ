import base64
import io
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base


@base.Env.register('havln_http')
class HAVLNHttpEnv(base.Env):
    def __init__(self, env_config: EnvCfg, task_config: TaskCfg = None):
        super().__init__(env_config, task_config)
        self.server_url = env_config.env_settings["server_url"].rstrip("/")
        self.output_path = env_config.env_settings.get("output_path", "./output")
        self.is_running = True
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_metrics: Dict[str, Any] = {"top_down_map": None}
        self._current_episode = None
        self._capabilities: Dict[str, Any] = {"has_external_lookdown_views": False}
        self.session = requests.Session()

        meta = self.session.get(f"{self.server_url}/metadata", timeout=30).json()
        self._capabilities.update(meta.get("capabilities", {}))
        total_episodes = int(meta.get("max_episodes") or meta.get("total_episodes") or 0)
        client_cap = env_config.env_settings.get("max_eval_episodes")
        if client_cap is not None:
            client_cap = int(client_cap)
            if client_cap > 0:
                total_episodes = min(total_episodes, client_cap) if total_episodes > 0 else client_cap
        self.episodes = [None] * total_episodes

    def _decode_obs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["rgb_png_b64"]))).convert("RGB"))
        depth = np.load(io.BytesIO(base64.b64decode(payload["depth_npy_b64"])), allow_pickle=False)
        obs = {
            "rgb": rgb,
            "depth": depth,
        }
        if "lookdown_rgb_png_b64" in payload:
            obs["lookdown_rgb"] = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(payload["lookdown_rgb_png_b64"]))).convert("RGB")
            )
        if "lookdown_depth_npy_b64" in payload:
            obs["lookdown_depth"] = np.load(
                io.BytesIO(base64.b64decode(payload["lookdown_depth_npy_b64"])),
                allow_pickle=False,
            )
        if "gps" in payload:
            obs["gps"] = np.array(payload["gps"], dtype=np.float32)
        if "compass" in payload:
            obs["compass"] = np.array([payload["compass"]], dtype=np.float32)
        return obs

    def _set_episode(self, payload: Dict[str, Any]):
        episode_id = payload.get("episode_id")
        scene_id = payload.get("scene_id")
        instruction_text = payload.get("instruction_text", "")
        if episode_id is None or scene_id is None:
            raise KeyError(
                f"HA-VLN server payload missing episode_id/scene_id: {list(payload.keys())}. "
                f"Check server is running and env initialized."
            )
        self._current_episode = SimpleNamespace(
            episode_id=str(episode_id),
            scene_id=scene_id,
            instruction=SimpleNamespace(instruction_text=instruction_text),
        )

    def reset(self):
        resp = self.session.post(f"{self.server_url}/reset", json={}, timeout=120)
        payload = resp.json()
        if payload.get("finished"):
            self.is_running = False
            return None
        if payload.get("error"):
            raise RuntimeError(f"HA-VLN server error: {payload['error']}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HA-VLN server reset failed: {resp.status_code} {payload}")

        self._set_episode(payload)
        self._last_obs = self._decode_obs(payload)
        self._capabilities.update(payload.get("capabilities", {}))
        self._last_metrics = dict(payload.get("metrics", {}))
        self._last_metrics.setdefault("top_down_map", None)
        return self._last_obs

    def step(self, action: List[Any]):
        action_int = int(action)
        resp = self.session.post(
            f"{self.server_url}/step",
            json={"action": action_int},
            timeout=120,
        )
        payload = resp.json()
        if payload.get("error"):
            raise RuntimeError(f"HA-VLN server step failed for action {action_int}: {payload['error']}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HA-VLN server step failed: {resp.status_code} {payload}")
        self._set_episode(payload)
        self._last_obs = self._decode_obs(payload)
        self._capabilities.update(payload.get("capabilities", {}))
        self._last_metrics = dict(payload.get("metrics", {}))
        self._last_metrics.setdefault("top_down_map", None)
        return self._last_obs, float(payload.get("reward", 0.0)), bool(payload["done"]), self._last_metrics

    def close(self):
        try:
            self.session.post(f"{self.server_url}/close", json={}, timeout=10)
        except Exception:
            pass
        self.session.close()

    def render(self):
        return None

    def get_observation(self) -> Dict[str, Any]:
        return self._last_obs

    def get_metrics(self) -> Dict[str, Any]:
        return self._last_metrics

    def get_current_episode(self):
        return self._current_episode

    def get_capabilities(self) -> Dict[str, Any]:
        return dict(self._capabilities)
