"""
E3: "continuation-only" single-vLLM 全量消融实验 (1024 episodes)
配置：
- shared-memory：开
- compile / CUDA Graph：开
- DiT Cond Cache + CrossAttn KV Cache：开
- same-request continuation：开
- latent prefill reuse：关（continuation的latent复用归到continuation本身）

对比E2可以看出continuation本身的影响。
"""
from scripts.eval.configs.habitat_dual_system_1024ep_single_vllm_semantic_conservative_cfg import eval_cfg

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_1024ep_single_vllm_continuation"
eval_cfg.eval_settings["save_video"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
# 其他配置与E2相同，continuation通过server环境变量控制
