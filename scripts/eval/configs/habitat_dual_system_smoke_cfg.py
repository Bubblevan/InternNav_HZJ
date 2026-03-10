from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_smoke"
eval_cfg.eval_settings["dataset_path_override"] = None
eval_cfg.eval_settings["allowed_scene_ids"] = []
eval_cfg.eval_settings["allowed_episode_ids"] = []
eval_cfg.eval_settings["max_eval_episodes"] = 8
eval_cfg.eval_settings["replay_num_episodes"] = 8
