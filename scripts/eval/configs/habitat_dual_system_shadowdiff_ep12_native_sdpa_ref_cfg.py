from scripts.eval.configs.habitat_dual_system_shadowdiff_ep12_native_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/shadowdiff_ep12_native_sdpa_ref"
eval_cfg.agent.model_settings["shadow_diff_hf_attn_backend"] = "sdpa"
