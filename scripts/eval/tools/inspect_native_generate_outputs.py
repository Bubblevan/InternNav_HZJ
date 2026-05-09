import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name in {"float16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(name)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else None,
    ).to(device=device, dtype=dtype)
    model.eval()

    image = Image.open(Path(args.image)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.instruction},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **{k: v for k, v in model_inputs.items() if k != "mm_token_type_ids"},
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=False,
        )
    payload = {
        "output_type": type(outputs).__name__,
        "attrs": sorted([name for name in dir(outputs) if not name.startswith("_")]),
        "has_hidden_states": getattr(outputs, "hidden_states", None) is not None,
        "hidden_states_len": (
            len(outputs.hidden_states) if getattr(outputs, "hidden_states", None) is not None else None
        ),
        "hidden_states_last_shapes": (
            [list(t.shape) for t in outputs.hidden_states[-1]]
            if getattr(outputs, "hidden_states", None) is not None and outputs.hidden_states
            else None
        ),
        "has_past_key_values": getattr(outputs, "past_key_values", None) is not None,
        "past_key_values_type": (
            type(getattr(outputs, "past_key_values", None)).__name__
            if getattr(outputs, "past_key_values", None) is not None
            else None
        ),
        "sequence_shape": list(outputs.sequences.shape),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
