from scripts.eval.configs.habitat_dual_system_shadowdiff_ep12_native_fa2_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/shadowdiff_ep12_native_fa2_port8002"
eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8002"
