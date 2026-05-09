"""
E2: "语义保守版" single-vLLM 全量消融实验 (1024 episodes)
配置：
- shared-memory：开
- compile / CUDA Graph：开
- DiT Cond Cache + CrossAttn KV Cache：开
- same-request continuation：关
- latent prefill reuse：关（continuation关了自然就没有reuse）
"""
from scripts.eval.configs.habitat_dual_system_cfg import eval_cfg

# R2R val_unseen 1024 episode subset (deterministic selection)
# Scenes from val_unseen: 2azQ1b91cZZ, 8194nk5LbLH, etc.
TARGET_SCENE_IDS = [
    "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "oLBMNvg9in8",
    "QUCTc6BB5sX", "TbHJrupSAjP", "XcA2TqTSSAj", "i5noydFURQK",
    "mJXqzFtmKg4", "p5wJjkQkbXX", "s8pcmisQ38h", "wcojb4TFT35",
    "29hnd4uzFmX", "5LpN3gDmAk7", "5q7pvUzZiYz", "759xd9YjKW5",
    "Pm6F8kyY3z2", "S9hNv5qa7GM", "V2XKFQLXJAQ", "VzqfbhrpDEA",
    "YFuZgdQ5vWj", "b8cTxDM8gDG", "cV4RVeZvu5T", "e9zR4mvMWw7",
    "h8cxS4PZ9L9", "oRSSPeZ9uIZ", "pLe4wQe7qrG", "sT4fr6TAbpF",
    "ur6pFq6Qu1A", "vtj4Do2WSyo", "X7HyMhZNoso", "YVbv4upA8JN",
    "Z6MFQCViBuw", "ZMojNkEp431", "fzynW3qQVFf", "gxdoqLR6rwA",
    "gYvKGZ5eR3Z", "jtcxE69GiFV", "pa4otMbVnkk", "q9vSo1VnCiC",
    "rqfALeAoiTq", "uwVdk7zrZft", "wc2Jmh7sJ8z", "xxYm3xymUr6",
    "yqstnuAEVhm", "zsNo4HB9uLZ", "D7G3Y4RWsrU", "EDJbREhghzL",
    "JeFG25nYj2p", "SN83YJsR3w2", "WYY7iVv5hZw", "YmJkqBEsHnH",
    "arnzJ2nCWDP", "cvZr5TUy5RP", "duWWjT2t1jV", "h1zeeAwLh9Z",
    "jh4fc5c5qoQ", "nB8DNYgQhAc", "pRbA3pwrgk9", "sKLMLpTHeUy",
    "u9Xw2MFgHjS", "uoNH8PwgtBZ", "vx2X9z2B5rJ", "x8F5xyUWy9e",
]

# Empty episode IDs means all episodes from these scenes
TARGET_EPISODE_IDS = []

eval_cfg.eval_settings["output_path"] = "./logs/habitat/test_dual_system_1024ep_single_vllm_semantic_conservative"
eval_cfg.eval_settings["allowed_scene_ids"] = TARGET_SCENE_IDS
eval_cfg.eval_settings["allowed_episode_ids"] = TARGET_EPISODE_IDS
eval_cfg.eval_settings["max_eval_episodes"] = 1024
eval_cfg.eval_settings["replay_num_episodes"] = 1024

# DiT Cache 保持开启（纯工程优化）
eval_cfg.agent.model_settings["dit_cond_cache_enabled"] = True
eval_cfg.agent.model_settings["dit_crossattn_kv_cache_enabled"] = True

# Single-vLLM backend settings
eval_cfg.agent.model_settings["dualvln_single_vllm_url"] = "http://127.0.0.1:8000"
eval_cfg.agent.model_settings["dualvln_single_vllm_timeout"] = 300.0

# 关闭视频保存和可视化调试以加速评估
eval_cfg.eval_settings["save_video"] = False
eval_cfg.agent.model_settings["vis_debug"] = False
