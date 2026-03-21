from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

TARGET_SCENE_IDS = ["2azQ1b91cZZ"]
TARGET_EPISODE_IDS = [
    10, 11, 12, 16, 17, 18, 43, 44,
    45, 61, 62, 63, 70, 71, 72, 76,
    77, 78, 79, 80, 81, 88, 89, 90,
    115, 116, 117, 121, 122, 123, 139, 140,
]

# S2 -> vLLM OpenAI server
eval_cfg.agent.model_settings["s2_vllm_url"] = "http://127.0.0.1:8000"
eval_cfg.agent.model_settings["s2_vllm_model"] = "dualvln-s2"

# generate_latents -> patched vLLM hidden-state service
eval_cfg.agent.model_settings["generate_latents_backend"] = "vllm_hidden"
eval_cfg.agent.model_settings["generate_latents_vllm_url"] = "http://127.0.0.1:8011"
eval_cfg.agent.model_settings["generate_latents_vllm_dump_dir"] = (
    "./logs/habitat/vllm_generate_latents_runtime_dump_32ep_both"
)

# Fixed evaluation set
eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_32ep_vllm_both"
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = len(TARGET_EPISODE_IDS)
eval_cfg.eval_settings["replay_num_episodes"] = len(TARGET_EPISODE_IDS)