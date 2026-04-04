from scripts.eval.configs.habitat_dual_system_32ep_base_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [10]

eval_cfg.eval_settings["output_path"] = "./logs/habitat/metrics_validation_hf_1ep"
eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = 1
eval_cfg.eval_settings["replay_num_episodes"] = 1
