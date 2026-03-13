from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat/vis_debug_ha_http",
        },
    ),
    env=EnvCfg(
        env_type='havln_http',
        env_settings={
            'server_url': 'http://127.0.0.1:8899',
            'config_path': 'scripts/eval/configs/vln_ha_r2r.yaml',
            'havln_root': '/root/backup/HA-VLN',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system_ha_http",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "dataset_path_override": None,
        "scenes_dir_override": None,
        "dataset_split_override": None,
        "allowed_scene_ids": [],
        "allowed_episode_ids": [],
        "max_eval_episodes": 1,
        "profile_runtime": True,
        "profile_modules": True,
        "export_replay_subset": False,
        "replay_num_episodes": 0,
        "replay_seed": 0,
        "port": "2333",
        "dist_url": "env://",
    },
)
