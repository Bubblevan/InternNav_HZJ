import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional

from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env import base


@base.Env.register('habitat')
class HabitatEnv(base.Env):
    """A lightweight wrapper around `habitat.Env` that adapts Habitat to the project's `base.Env` interface.

    Args:
        env_config (EnvCfg): Environment configuration.
        task_config (TaskCfg): Optional task configuration passed to the base environment.
    """

    def __init__(self, env_config: EnvCfg, task_config: TaskCfg = None):
        super().__init__(env_config, task_config)
        self.config = env_config.env_settings['habitat_config']
        self._legacy_rl_env = False

        legacy_config_path = env_config.env_settings.get('legacy_config_path')
        if legacy_config_path:
            self._env = self._init_legacy_havln_env(env_config)
            self._legacy_rl_env = True
        else:
            try:
                from habitat import Env
            except ImportError as e:
                raise RuntimeError(
                    "Habitat modules could not be imported. "
                    "Make sure both repositories are installed and on PYTHONPATH."
                ) from e
            self._env = Env(self.config)

        self.rank = env_config.env_settings.get('rank', 0)
        self.world_size = env_config.env_settings.get('world_size', 1)
        self._current_episode_index: int = 0
        self._last_obs: Optional[Dict[str, Any]] = None

        self.is_running = True
        self.output_path = env_config.env_settings.get('output_path', './output')
        self.max_eval_episodes = env_config.env_settings.get('max_eval_episodes')

        # generate episodes
        self.episodes = self.generate_episodes()

    def _init_legacy_havln_env(self, env_config: EnvCfg):
        # 先插入 HA-VLN 的 habitat-lab，再 import bootstrap，这样 habitat 会解析到旧版（yacs Config）
        for path in env_config.env_settings.get('extra_python_paths', []):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        ha_habitat_lab = None
        for path in env_config.env_settings.get('extra_python_paths', []):
            if path:
                candidate = os.path.join(path, "habitat-lab")
                if os.path.isdir(candidate) and candidate not in sys.path:
                    sys.path.insert(0, candidate)
                    ha_habitat_lab = candidate
                    break

        if ha_habitat_lab:
            legacy_hb_default = os.path.join(
                ha_habitat_lab, "habitat_baselines", "config", "default.py"
            )
            if os.path.isfile(legacy_hb_default):
                self._load_module_from_file(
                    "habitat_baselines.config.default",
                    legacy_hb_default,
                    force_reload=True,
                )

        for module_name in env_config.env_settings.get('bootstrap_modules', []):
            importlib.import_module(module_name)

        legacy_workdir = env_config.env_settings.get('legacy_workdir')
        prev_cwd = os.getcwd()
        try:
            if legacy_workdir:
                os.chdir(legacy_workdir)

            config_file = env_config.env_settings.get('legacy_config_file')
            if config_file:
                config_module = self._load_module_from_file("_havln_legacy_config", config_file)
            else:
                config_module = importlib.import_module(
                    env_config.env_settings.get('legacy_config_module', 'vlnce_baselines.config.default')
                )
            get_config = getattr(
                config_module,
                env_config.env_settings.get('legacy_config_fn', 'get_config'),
            )
            legacy_config = get_config(
                env_config.env_settings['legacy_config_path'],
                env_config.env_settings.get('legacy_config_opts'),
            )

            env_file = env_config.env_settings.get('legacy_env_file')
            if env_file:
                env_module = self._load_module_from_file("_havln_legacy_envs", env_file)
            else:
                env_module = importlib.import_module(
                    env_config.env_settings.get('legacy_env_module', 'vlnce_baselines.common.environments')
                )
            env_class = getattr(
                env_module,
                env_config.env_settings.get('legacy_env_class', 'HAVLNCEDaggerEnv'),
            )

            self.legacy_config = legacy_config
            return env_class(config=legacy_config)
        finally:
            if legacy_workdir:
                os.chdir(prev_cwd)

    def _load_module_from_file(self, module_name: str, file_path: str, force_reload: bool = False):
        if module_name in sys.modules and not force_reload:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        parent_name, _, child_name = module_name.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, module)
        return module

    def generate_episodes(self) -> List[Any]:
        """
        Generate list of episodes for the current split.

        Returns:
            List[Any]: A list of episode objects for the current split.
        """
        all_episodes = []
        allowed_scene_ids = self.config.habitat.dataset.get("allowed_scene_ids", None)
        if allowed_scene_ids:
            allowed_scene_ids = set(allowed_scene_ids)

        allowed_episode_ids = self.config.habitat.dataset.get("allowed_episode_ids", None)
        if allowed_episode_ids:
            allowed_episode_ids = {str(ep_id) for ep_id in allowed_episode_ids}

        max_eval_episodes = self.max_eval_episodes
        if max_eval_episodes is None:
            max_eval_episodes = self.config.habitat.dataset.get("max_eval_episodes", None)

        # group episodes by scene
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in self._env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)

        # load done_res
        done_res = set()
        result_path = os.path.join(self.output_path, 'progress.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                for line in f:
                    res = json.loads(line)
                    # only skip if current format has scene_id
                    if "scene_id" in res:
                        done_res.add((res["scene_id"], res["episode_id"]))

        # iterate scenes in order, collect all episodes
        for scene in sorted(scene_episode_dict.keys()):
            per_scene_eps = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            # allow filtering to a tiny subset without modifying Habitat internals
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue

            # shard by rank index / world_size
            for episode in per_scene_eps[self.rank :: self.world_size]:
                episode_id = int(episode.episode_id)
                if allowed_episode_ids and str(episode_id) not in allowed_episode_ids:
                    continue
                if (scene_id, episode_id) in done_res:
                    continue
                all_episodes.append(episode)
                if max_eval_episodes is not None and len(all_episodes) >= int(max_eval_episodes):
                    return all_episodes

        return all_episodes

    def reset(self):
        # no more episodes
        if not (0 <= self._current_episode_index < len(self.episodes)):
            self.is_running = False
            return

        # Manually set to next episode in habitat
        self._env.current_episode = self.episodes[self._current_episode_index]
        self._current_episode_index += 1

        # Habitat reset
        self._last_obs = self._env.reset()
        return self._last_obs

    def step(self, action: List[Any]):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 4:
            return result

        obs = result
        done = self._env.episode_over
        info = self.get_metrics()
        reward = info.get('reward', 0.0)
        return obs, reward, done, info

    def close(self):
        print('Habitat Env close')
        self._env.close()

    def render(self):
        self._env.render()

    def get_observation(self) -> Dict[str, Any]:
        if hasattr(self._env, 'get_observations'):
            return self._env.get_observations()
        return self._env.habitat_env.get_observations()

    def get_metrics(self) -> Dict[str, Any]:
        if hasattr(self._env, 'get_metrics'):
            return self._env.get_metrics()
        return self._env.habitat_env.get_metrics()

    def get_current_episode(self):
        return self._env.current_episode
