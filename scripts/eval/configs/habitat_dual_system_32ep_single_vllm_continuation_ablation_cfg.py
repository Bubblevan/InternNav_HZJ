"""
32ep快速验证continuation影响（用于快速迭代）
"""
from scripts.eval.configs.habitat_dual_system_32ep_single_vllm_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_32ep_single_vllm_continuation_ablation"
eval_cfg.eval_settings["save_video"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
# continuation通过server环境变量控制（开 vs 关）
