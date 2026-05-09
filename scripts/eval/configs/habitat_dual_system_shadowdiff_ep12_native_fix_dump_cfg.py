from scripts.eval.configs.habitat_dual_system_shadowdiff_ep12_native_fix_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/shadowdiff_ep12_native_fix_dump"
eval_cfg.agent.model_settings["shadow_diff_dump_images"] = True
