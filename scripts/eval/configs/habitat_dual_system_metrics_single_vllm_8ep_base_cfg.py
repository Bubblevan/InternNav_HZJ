from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [10, 11, 12, 16, 17, 18, 43, 44]

eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["replay_num_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.agent.model_settings["dit_cond_cache_enabled"] = True
eval_cfg.agent.model_settings["dit_crossattn_kv_cache_enabled"] = True
