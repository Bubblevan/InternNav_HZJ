from scripts.eval.configs.habitat_dual_system_vllm_server_smoke_cfg import eval_cfg


TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [10, 16, 17, 43, 44]


eval_cfg.agent.model_settings.update(
    {
        "deterministic_seed": 0,
        "deterministic_conjunction_index": 0,
        "shadow_diff_enabled": True,
        "shadow_diff_reference": "hf",
        "shadow_diff_hf_model_path": "checkpoints/InternVLA-N1-DualVLN",
        "shadow_diff_max_new_tokens": 128,
    }
)

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_vllm_shadow_diff_5ep"
eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["export_replay_subset"] = False
eval_cfg.eval_settings["replay_num_episodes"] = 0
