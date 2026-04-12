from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [
    10, 11, 12, 16, 17, 18, 43, 44,
    45, 61, 62, 63, 70, 71, 72, 76,
    77, 78, 79, 80, 81, 88, 89, 90,
    115, 116, 117, 121, 122, 123, 139, 140,
    141, 157, 158, 159, 160, 161, 162, 166,
    167, 168, 178, 179, 180, 193, 194, 195,
    214, 215, 216, 262, 263, 264, 265, 266,
    267, 271, 272, 273, 277, 278, 279, 286,
]

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_64ep_single_vllm_history_probe_a5_gateway_followup_keep_full64"
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["replay_num_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.agent.model_settings["enable_history_probe"] = True
eval_cfg.agent.model_settings["history_probe_mode"] = "keep_one"
eval_cfg.agent.model_settings["history_probe_target"] = "history_conditioned_gateway_only"
eval_cfg.agent.model_settings["history_probe_max_steps"] = 8
eval_cfg.agent.model_settings["history_probe_run_followup_replay"] = True
eval_cfg.agent.model_settings["history_probe_interventions"] = [
    "replace_with_other_history",
    "light_blur",
    "downsample_then_upsample",
]
eval_cfg.agent.model_settings["history_probe_blur_radius"] = 2.0
eval_cfg.agent.model_settings["history_probe_downsample_factor"] = 4
eval_cfg.agent.model_settings["vis_debug"]=False