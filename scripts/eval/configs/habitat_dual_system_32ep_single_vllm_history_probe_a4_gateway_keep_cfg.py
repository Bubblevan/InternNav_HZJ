from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_32ep_single_vllm_history_probe_a4_gateway_keep"
eval_cfg.agent.model_settings["enable_history_probe"] = True
eval_cfg.agent.model_settings["history_probe_mode"] = "keep_one"
eval_cfg.agent.model_settings["history_probe_target"] = "history_conditioned_gateway_only"
eval_cfg.agent.model_settings["history_probe_max_steps"] = 2
eval_cfg.agent.model_settings["history_probe_interventions"] = [
    "replace_with_other_history",
    "light_blur",
    "downsample_then_upsample",
]
eval_cfg.agent.model_settings["history_probe_blur_radius"] = 2.0
eval_cfg.agent.model_settings["history_probe_downsample_factor"] = 4
