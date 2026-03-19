"""A/B test — vLLM S2 backend. Same fixed episode subset from R2R val_unseen."""
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
            "vis_debug_path": "./logs/habitat/vis_debug",
            "s2_vllm_url": "http://127.0.0.1:8001",
            "s2_vllm_model": None,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
            'max_eval_episodes': 128,
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/ab_test_vllm",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "dataset_path_override": None,
        "scenes_dir_override": "data/scene_data",
        "dataset_split_override": None,
        "allowed_scene_ids": [],
        "allowed_episode_ids": [10, 11, 12, 16, 17, 18, 43, 44],
        "max_eval_episodes": 128,
        "profile_runtime": True,
        "profile_modules": True,
        "export_replay_subset": True,
        "replay_num_episodes": 128,
        "replay_seed": 0,
        "port": "2333",
        "dist_url": "env://",
    },
)
