from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8000,
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",

            # 本地 checkpoint 仍然给 system1 runner 用
            "model_path": "checkpoints/InternVLA-N1-DualVLN",

            # 关键：真正启用 single-vLLM HTTP backend
            "dualvln_single_vllm_url": "http://127.0.0.1:8000",
            "force_single_vllm_http": True,
            "dualvln_single_vllm_timeout": 600.0,
            "s2_vllm_url": None,
            "s2_vllm_model": None,

            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 256,
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat/vis_debug_vllm_server",
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={
            "config_path": "scripts/eval/configs/vln_r2r.yaml",
        },
    ),
    eval_type="habitat_vln",
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system_vllm_server_smoke",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,

        "dataset_path_override": None,
        "scenes_dir_override": "data/scene_data",
        "dataset_split_override": None,
        "allowed_scene_ids": [],
        "allowed_episode_ids": [],
        "max_eval_episodes": 8,

        "profile_runtime": True,
        "profile_modules": True,
        "export_replay_subset": True,
        "replay_num_episodes": 8,
        "replay_seed": 0,

        "port": "2333",
        "dist_url": "env://",
    },
)
