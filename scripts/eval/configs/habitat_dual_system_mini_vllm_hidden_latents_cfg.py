from scripts.eval.configs.habitat_dual_system_mini_cfg import eval_cfg

eval_cfg.agent.model_settings["generate_latents_backend"] = "vllm_hidden"
eval_cfg.agent.model_settings["generate_latents_vllm_model_path"] = (
    "checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view"
)
eval_cfg.agent.model_settings["generate_latents_vllm_dump_dir"] = (
    "./logs/habitat/vllm_generate_latents_runtime_dump_mini"
)
eval_cfg.agent.model_settings["generate_latents_vllm_gpu_memory_utilization"] = 0.45
eval_cfg.agent.model_settings["generate_latents_vllm_limit_mm_per_prompt_image"] = 16
eval_cfg.agent.model_settings["generate_latents_vllm_enforce_eager"] = True

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_mini_vllm_hidden_latents"
