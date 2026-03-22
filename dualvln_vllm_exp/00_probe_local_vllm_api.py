import inspect
import json
import os
import sys

def safe_sig(obj):
    try:
        return str(inspect.signature(obj))
    except Exception as e:
        return f"<sig unavailable: {e!r}>"

out = {}
try:
    import vllm
    out["vllm_module"] = getattr(vllm, "__file__", None)
    out["vllm_version"] = getattr(vllm, "__version__", None)
    from vllm import LLM
    out["LLM_signature"] = safe_sig(LLM)
    out["LLM_has_encode"] = hasattr(LLM, "encode")
    out["LLM_has_embed"] = hasattr(LLM, "embed")
    out["LLM_has_reward"] = hasattr(LLM, "reward")
    if hasattr(LLM, "encode"):
        out["encode_signature"] = safe_sig(LLM.encode)
    if hasattr(LLM, "embed"):
        out["embed_signature"] = safe_sig(LLM.embed)
    if hasattr(LLM, "reward"):
        out["reward_signature"] = safe_sig(LLM.reward)
except Exception as e:
    out["import_error"] = repr(e)

print(json.dumps(out, ensure_ascii=False, indent=2))
