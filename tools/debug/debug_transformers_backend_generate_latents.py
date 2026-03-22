import argparse
import functools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internnav.model.utils.dualvln_single_vllm import (  # noqa: E402
    _generate_latents_via_transformers_backend_apply_model,
)
from scripts.eval.tools.test_vllm_s2_equivalence import (  # noqa: E402
    build_messages,
    load_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare standalone HF generate_latents vs vLLM transformers-backend apply_model on one sample."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "hf_ref"),
        default="all",
        help="Internal helper mode. Use all for end-to-end compare; hf_ref is for the habitat subprocess export.",
    )
    parser.add_argument(
        "--sample-pt",
        default="logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt",
    )
    parser.add_argument(
        "--manifest",
        default="logs/habitat/test_dual_system_mini/replay_subset/manifest_rank0.jsonl",
    )
    parser.add_argument("--hf-model-path", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--vllm-model-path", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument(
        "--llm-output",
        default=None,
        help="If omitted, use sample['baseline_output']['llm_output'].",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--limit-mm-per-prompt-image",
        type=int,
        default=1,
        help="Only affects vLLM startup profiling for this debug script.",
    )
    parser.add_argument(
        "--mm-processor-min-pixels",
        type=int,
        default=3136,
        help="Only affects vLLM startup profiling for this debug script.",
    )
    parser.add_argument(
        "--mm-processor-max-pixels",
        type=int,
        default=200704,
        help="Only affects vLLM startup profiling for this debug script.",
    )
    parser.add_argument(
        "--output-json",
        default="logs/habitat/transformers_backend_generate_latents_compare_sample_0000.json",
    )
    parser.add_argument(
        "--hf-ref-output-pt",
        default="logs/habitat/transformers_backend_generate_latents_hf_ref_sample_0000.pt",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def _tensor_stats(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    diff = (a_cpu - b_cpu).abs()
    cosine = None
    if a_cpu.numel() > 0 and b_cpu.numel() > 0:
        cosine = float(F.cosine_similarity(a_cpu.reshape(1, -1), b_cpu.reshape(1, -1)).item())
    return {
        "a": _tensor_stats(a_cpu),
        "b": _tensor_stats(b_cpu),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
        "allclose": bool(torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)),
    }


def _resolve_llm_output(sample: dict, override: Optional[str]) -> str:
    if override is not None:
        return override
    baseline_output = sample.get("baseline_output", {})
    llm_output = baseline_output.get("llm_output")
    if not llm_output:
        raise RuntimeError("Could not resolve llm_output from sample; pass --llm-output explicitly.")
    return llm_output


def _build_full_output_ids(sample: dict, manifest_path: Path, processor, llm_output: str) -> dict:
    replay = load_manifest(str(manifest_path), base_path="logs")
    steps = replay[(sample["scene_id"], int(sample["episode_id"]))]
    target_item = None
    prev_llm = None
    for step in steps:
        if (
            int(step["step_id"]) == int(sample["step_id"])
            and step["baseline_output"]["output_kind"] == sample["baseline_output"]["output_kind"]
        ):
            target_item = step
            break
        prev_llm = step["baseline_output"]["llm_output"]
    if target_item is None:
        raise RuntimeError("Could not locate matching replay item for sample.")

    messages, input_images = build_messages(
        sample["instruction"],
        target_item,
        steps,
        num_history=8,
        is_lookdown=bool(sample["is_inferred_lookdown_followup"]),
        prev_llm_output=prev_llm,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    hf_inputs = processor(text=[text], images=input_images, return_tensors="pt")
    generated_ids = processor.tokenizer.encode(llm_output, add_special_tokens=False)
    full_output_ids = torch.cat(
        [hf_inputs.input_ids, torch.tensor([generated_ids], dtype=torch.long)],
        dim=1,
    )
    return {
        "messages": messages,
        "input_ids": hf_inputs.input_ids.detach().cpu(),
        "generated_ids": torch.tensor(generated_ids, dtype=torch.long),
        "full_output_ids": full_output_ids.detach().cpu(),
    }


def _export_hf_reference(args):
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    sample = torch.load(args.sample_pt, map_location="cpu")
    llm_output = _resolve_llm_output(sample, args.llm_output)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, use_fast=False)
    processor.tokenizer.padding_side = "left"
    rebuilt = _build_full_output_ids(sample, Path(args.manifest), processor, llm_output)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_cuda else "sdpa",
        device_map={"": device} if use_cuda else None,
    )
    model.eval()
    with torch.no_grad():
        traj_latents = model.generate_latents(
            rebuilt["full_output_ids"].to(device),
            sample["pixel_values"].to(device),
            sample["image_grid_thw"].to(device),
        ).detach().cpu()

    payload = {
        "llm_output": llm_output,
        "input_ids": rebuilt["input_ids"],
        "generated_ids": rebuilt["generated_ids"],
        "full_output_ids": rebuilt["full_output_ids"],
        "pixel_values": sample["pixel_values"].detach().cpu(),
        "image_grid_thw": sample["image_grid_thw"].detach().cpu(),
        "traj_latents": traj_latents,
    }
    out_path = Path(args.hf_ref_output_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[HF_REF] wrote {out_path}")


def _run_hf_reference_subprocess(args):
    script_path = Path(__file__).resolve()
    hf_ref_output_pt = Path(args.hf_ref_output_pt).resolve()
    hf_ref_output_pt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "source /root/miniforge3/etc/profile.d/conda.sh",
        "conda run -n habitat python",
        shlex.quote(str(script_path)),
        "--stage hf_ref",
        "--sample-pt",
        shlex.quote(str(Path(args.sample_pt).resolve())),
        "--manifest",
        shlex.quote(str(Path(args.manifest).resolve())),
        "--hf-model-path",
        shlex.quote(str(Path(args.hf_model_path).resolve())),
        "--hf-ref-output-pt",
        shlex.quote(str(hf_ref_output_pt)),
    ]
    if args.llm_output is not None:
        cmd.extend(["--llm-output", shlex.quote(args.llm_output)])

    subprocess.run(["bash", "-lc", " ".join(cmd)], check=True, cwd=str(PROJECT_ROOT))
    return torch.load(hf_ref_output_pt, map_location="cpu")


def _load_vllm_and_generate(args, full_output_ids_cpu, pixel_values_cpu, image_grid_thw_cpu):
    from vllm import LLM

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    mm_processor_kwargs = {
        "min_pixels": args.mm_processor_min_pixels,
        "max_pixels": args.mm_processor_max_pixels,
    }
    llm = LLM(
        model=args.vllm_model_path,
        trust_remote_code=args.trust_remote_code,
        model_impl="transformers",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        mm_processor_kwargs=mm_processor_kwargs,
        disable_log_stats=True,
    )
    fn = functools.partial(
        _generate_latents_via_transformers_backend_apply_model,
        full_output_ids_cpu=full_output_ids_cpu,
        pixel_values_cpu=pixel_values_cpu,
        image_grid_thw_cpu=image_grid_thw_cpu,
    )
    latents = llm.apply_model(fn)[0].detach().cpu()
    model_tree = llm.apply_model(
        lambda model: {
            "worker_model_type": str(type(model)),
            "wrapped_type": str(type(model.model)) if hasattr(model, "model") else None,
            "wrapped_has_language_model": hasattr(model.model, "language_model") if hasattr(model, "model") else False,
            "wrapped_has_latent_queries": hasattr(model.model, "latent_queries") if hasattr(model, "model") else False,
        }
    )[0]
    return {
        "traj_latents": latents,
        "model_tree": model_tree,
        "using_transformers_backend": True,
        "mm_processor_kwargs": mm_processor_kwargs,
    }


def main():
    args = parse_args()
    if args.stage == "hf_ref":
        _export_hf_reference(args)
        return

    sample = torch.load(args.sample_pt, map_location="cpu")
    llm_output = _resolve_llm_output(sample, args.llm_output)

    processor = AutoProcessor.from_pretrained(args.hf_model_path, use_fast=False)
    processor.tokenizer.padding_side = "left"
    rebuilt_local = _build_full_output_ids(sample, Path(args.manifest), processor, llm_output)

    hf_ref = _run_hf_reference_subprocess(args)
    vllm_ref = None
    vllm_error = None
    try:
        vllm_ref = _load_vllm_and_generate(
            args,
            rebuilt_local["full_output_ids"],
            sample["pixel_values"].detach().cpu(),
            sample["image_grid_thw"].detach().cpu(),
        )
    except Exception as exc:
        vllm_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    results = {
        "sample_pt": str(Path(args.sample_pt).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "hf_model_path": str(Path(args.hf_model_path).resolve()),
        "vllm_model_path": str(Path(args.vllm_model_path).resolve()),
        "llm_output": llm_output,
        "local_vs_hf_ref.full_output_ids": _compare_tensors(
            rebuilt_local["full_output_ids"],
            hf_ref["full_output_ids"],
            args.atol,
            args.rtol,
        ),
        "local_vs_hf_ref.input_ids": _compare_tensors(
            rebuilt_local["input_ids"],
            hf_ref["input_ids"],
            args.atol,
            args.rtol,
        ),
        "sample_vs_hf_ref.image_grid_thw": _compare_tensors(
            sample["image_grid_thw"],
            hf_ref["image_grid_thw"],
            args.atol,
            args.rtol,
        ),
        "hf_ref_stats": {
            "traj_latents": _tensor_stats(hf_ref["traj_latents"]),
        },
        "vllm_error": vllm_error,
    }
    if vllm_ref is not None:
        results["hf_ref_vs_vllm_ref.traj_latents"] = _compare_tensors(
            hf_ref["traj_latents"],
            vllm_ref["traj_latents"],
            args.atol,
            args.rtol,
        )
        results["vllm_ref_stats"] = {
            "traj_latents": _tensor_stats(vllm_ref["traj_latents"]),
            "model_tree": vllm_ref["model_tree"],
            "using_transformers_backend": vllm_ref["using_transformers_backend"],
            "mm_processor_kwargs": vllm_ref["mm_processor_kwargs"],
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[COMPARE] wrote {out_path}")
    if "hf_ref_vs_vllm_ref.traj_latents" in results:
        print(json.dumps(results["hf_ref_vs_vllm_ref.traj_latents"], indent=2))
    else:
        print(json.dumps({"vllm_error": vllm_error}, indent=2))


if __name__ == "__main__":
    main()
