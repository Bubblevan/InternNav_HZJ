from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [45]

eval_cfg.eval_settings["output_path"] = "./logs/habitat/shadowdiff_ep45_patched_port8003"
eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["replay_num_episodes"] = len(TARGET_EPISODE_IDS)

eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8003"
eval_cfg.agent.model_settings["deterministic_seed"] = 0
eval_cfg.agent.model_settings["shadow_diff_enabled"] = True
eval_cfg.agent.model_settings["shadow_diff_reference"] = "hf"
eval_cfg.agent.model_settings["shadow_diff_max_new_tokens"] = 128
eval_cfg.agent.model_settings["shadow_diff_dump_images"] = True
eval_cfg.agent.model_settings["shadow_diff_hf_model_path"] = (
    "checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view"
)
