from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-DualVLN",  # path to model checkpoint
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # mini-subset / local data overrides
        "dataset_path_override": None,  # e.g. data/vln_ce/raw_data/r2r/val_unseen_mini/val_unseen_mini.json.gz
        "scenes_dir_override": "data/scene_data",
        "dataset_split_override": None,
        "allowed_scene_ids": [],  # e.g. ["zsNo4HB9uLZ"]
        "allowed_episode_ids": [],  # e.g. [1, 2, 3]
        "max_eval_episodes": None,  # e.g. 8
        # runtime profiling
        "profile_runtime": True,
        "profile_modules": True,
        # replay subset export
        "export_replay_subset": True,
        "replay_num_episodes": 20,
        "replay_seed": 0,
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
