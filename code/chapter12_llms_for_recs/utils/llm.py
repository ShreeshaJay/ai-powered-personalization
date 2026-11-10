"""
Loading and batched inference for open-weights causal LLMs (Hugging Face transformers).

No remote code is executed (``trust_remote_code`` stays False): the Qwen2.5 family is
supported natively by transformers. 4-bit loading needs a CUDA GPU and bitsandbytes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def pick_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, device: str, quantize_4bit: bool = False):
    """Return (model, tokenizer) in eval mode on ``device``."""
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_name)
    dtype = pick_dtype(device)
    kwargs: Dict = {"dtype": dtype}
    if quantize_4bit:
        if not device.startswith("cuda"):
            raise ValueError("4-bit loading needs a CUDA GPU; drop --quantize-4bit on CPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not quantize_4bit:
        model.to(device)
    model.eval()
    return model, tokenizer


def chat_prompt(tokenizer, user_text: str) -> str:
    """Wrap a user message in the model's chat template, ending at the assistant turn."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
        )
    return user_text + "\nAnswer: "


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def generate(model, tokenizer, prompts: Sequence[str], *, max_new_tokens: int,
             batch_size: int, device: str) -> List[str]:
    """Greedy decoding for a list of already-templated prompts."""
    outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def answer_token_ids(tokenizer, yes: str = "Yes", no: str = "No") -> Tuple[int, int]:
    """First token of each answer word as it appears at the start of the assistant turn."""
    yes_ids = tokenizer.encode(yes, add_special_tokens=False)
    no_ids = tokenizer.encode(no, add_special_tokens=False)
    if yes_ids[0] == no_ids[0]:
        raise ValueError("Yes and No share a first token; choose different answer words")
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def yes_probability(model, tokenizer, prompts: Sequence[str], *, batch_size: int,
                    device: str, yes_id: int, no_id: int) -> np.ndarray:
    """P(Yes) / (P(Yes) + P(No)) at the first answer position, for templated prompts."""
    scores: List[float] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       position_ids=_position_ids(enc["attention_mask"])).logits[:, -1, :]
        pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1).float()
        scores.extend(torch.softmax(pair, dim=-1)[:, 0].cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


# ----------------------------------------------------------------------------
# Resumable JSONL output for long generation jobs
# ----------------------------------------------------------------------------

def read_jsonl(path: Path, key: str) -> Dict:
    path = Path(path)
    done: Dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping a truncated line in %s", path)
                    continue
                done[record[key]] = record
    return done


def append_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
