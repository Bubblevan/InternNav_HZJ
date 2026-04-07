from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [
    10, 11, 12, 16, 17, 18, 43, 44,
    45, 61, 62, 63, 70, 71, 72, 76,
    77, 78, 79, 80, 81, 88, 89, 90,
    115, 116, 117, 121, 122, 123, 139, 140,
]

# Opt-in single-engine backend:
# S2 generate + generate_latents both come from one custom vLLM server.
eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8000"
eval_cfg.agent.model_settings["dualvln_single_vllm_timeout"] = 300.0

# Fixed evaluation set
eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_32ep_single_vllm"
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["replay_num_episodes"] = len(TARGET_EPISODE_IDS)

# DiT Cache
eval_cfg.agent.model_settings["dit_cond_cache_enabled"] = True
eval_cfg.agent.model_settings["dit_crossattn_kv_cache_enabled"] = True


