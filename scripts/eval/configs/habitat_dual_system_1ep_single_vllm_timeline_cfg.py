from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

# Restrict eval to one sample so we can dump a clean per-episode control timeline.
eval_cfg.eval_settings["allowed_scene_ids"] = ["2azQ1b91cZZ"]
eval_cfg.eval_settings["allowed_episode_ids"] = [10]
eval_cfg.eval_settings["max_eval_episodes"] = 1
eval_cfg.eval_settings["replay_num_episodes"] = 1
eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_1ep_single_vllm_timeline"

# Fine-grained timeline trace for exact gantt reconstruction.
eval_cfg.eval_settings["timeline_trace_enabled"] = True
eval_cfg.eval_settings["timeline_trace_scene_id"] = "2azQ1b91cZZ"
eval_cfg.eval_settings["timeline_trace_episode_id"] = 10

# Plot defaults.
eval_cfg.eval_settings["timeline_plot_start_s"] = 0.0
eval_cfg.eval_settings["timeline_plot_window_s"] = 20.0
eval_cfg.eval_settings["timeline_plot_title"] = "Dual-System Closed-Loop Timeline"

eval_cfg.eval_settings["save_video"] = False
eval_cfg.eval_settings["vis_debug"] = False
