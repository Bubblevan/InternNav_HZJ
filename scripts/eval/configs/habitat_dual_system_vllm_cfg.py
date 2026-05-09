from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg
# Opt-in single-engine backend:
# S2 generate + generate_latents both come from one custom vLLM server.
eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8000"
eval_cfg.agent.model_settings["dualvln_single_vllm_timeout"] = 300.0

# Fixed evaluation set
eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_full_single_vllm"
eval_cfg.eval_settings["max_eval_episodes"] = None
eval_cfg.agent.model_settings["dit_cond_cache_enabled"] = True
eval_cfg.agent.model_settings["dit_crossattn_kv_cache_enabled"] = True

eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["vis_debug"] = False