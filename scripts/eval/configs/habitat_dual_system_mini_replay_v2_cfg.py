from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_mini_replay_v2"
eval_cfg.eval_settings["dataset_path_override"] = (
    "data/vln_ce/raw_data/r2r/val_unseen_mini/val_unseen_mini.json.gz"
)
eval_cfg.eval_settings["allowed_scene_ids"] = []
eval_cfg.eval_settings["allowed_episode_ids"] = []
eval_cfg.eval_settings["max_eval_episodes"] = 8
eval_cfg.eval_settings["replay_num_episodes"] = 8
