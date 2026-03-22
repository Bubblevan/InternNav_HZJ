import argparse
import json
from collections import Counter
from pathlib import Path

def short_reason(row):
    txt = " | ".join([
        str(row.get("export_stderr_tail", "")),
        str(row.get("compare_stderr_tail", "")),
        str(row.get("compare_stdout_tail", "")),
    ]).lower()

    if row.get("export_returncode", 0) != 0:
        if "out of memory" in txt or "cuda oom" in txt:
            return "export_oom"
        if "timeout" in txt:
            return "export_timeout"
        return "export_failed"

    if row.get("compare_returncode", 0) != 0:
        if "out of memory" in txt or "cuda oom" in txt:
            return "compare_oom"
        if "unsupported" in txt:
            return "compare_unsupported"
        if "forward context" in txt:
            return "compare_forward_context"
        if "shape" in txt or "size mismatch" in txt:
            return "compare_shape_mismatch"
        return "compare_failed"

    if row.get("missing_token_embed_tensor"):
        return "missing_token_embed_tensor"

    if not row.get("token_embed_tail_vs_ref"):
        return "no_tail_vs_ref"

    return "valid"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.dataset_json).read_text())
    rows = data["rows"]

    summary = []
    counts = Counter()
    for row in rows:
        reason = short_reason(row)
        counts[reason] += 1
        summary.append({
            "sample": row.get("sample"),
            "reason": reason,
            "export_returncode": row.get("export_returncode"),
            "compare_returncode": row.get("compare_returncode"),
            "token_embed_tensor_shape": row.get("token_embed_tensor_shape"),
            "token_embed_tail_vs_ref": row.get("token_embed_tail_vs_ref"),
            "export_stderr_tail": row.get("export_stderr_tail"),
            "compare_stderr_tail": row.get("compare_stderr_tail"),
        })

    out = {
        "counts": dict(counts),
        "rows": summary,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out["counts"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
