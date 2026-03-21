from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

eval_cfg.agent.model_settings["generate_latents_backend"] = "vllm_hidden"
eval_cfg.agent.model_settings["generate_latents_vllm_url"] = "http://127.0.0.1:8011"
eval_cfg.agent.model_settings["generate_latents_vllm_dump_dir"] = (
    "./logs/habitat/vllm_generate_latents_runtime_dump_ab_32"
)

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_ab_vllm_hidden_32"
eval_cfg.eval_settings["dataset_path_override"] = None
eval_cfg.eval_settings["allowed_scene_ids"] = []
eval_cfg.eval_settings["allowed_episode_ids"] = []
eval_cfg.eval_settings["max_eval_episodes"] = 32
eval_cfg.eval_settings["replay_num_episodes"] = 32