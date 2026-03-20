from scripts.eval.configs.habitat_dual_system_mini_cfg import eval_cfg

eval_cfg.agent.model_settings["generate_latents_backend"] = "vllm_hidden"
eval_cfg.agent.model_settings["generate_latents_vllm_url"] = "http://127.0.0.1:8011"
eval_cfg.agent.model_settings["generate_latents_vllm_dump_dir"] = (
    "./logs/habitat/vllm_generate_latents_runtime_dump_mini_http"
)

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_mini_vllm_hidden_latents_http"
