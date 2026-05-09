from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/metrics_validation_single_vllm_32ep_native_validated"
eval_cfg.eval_settings["save_video"] = False

eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8002"
eval_cfg.agent.model_settings["deterministic_seed"] = 0
eval_cfg.agent.model_settings["shadow_diff_enabled"] = False
