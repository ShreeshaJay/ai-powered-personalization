"""
Section 12.5: TALLRec-style LoRA tuning of an open-weights LLM for a Yes/No engagement prompt.

Training examples come from *train-split slates of the same mode as the evaluation slates*
(build_slates.py --splits train), so the negatives the model learns to reject look like
the candidates it is tested on. Each training session gives one positive (the true next
item, answer "Yes") and --negatives candidates from its slate (answer "No").

Only the answer tokens are trained: the prompt's labels are -100, so the loss is the
log-likelihood of "Yes" or "No" (plus the end-of-turn token) given the prompt. If a
prompt is too long, history items are dropped from the front until the full answer fits;
an example is never truncated through its answer.

Evaluation scores every candidate of the shared test slates (the same file the zero-shot
script reads) by P(Yes) / (P(Yes) + P(No)) at the first answer token, twice: with the
adapter disabled (the untuned base model, pointwise) and enabled (tuned).

Outputs:
    <work_dir>/models/tallrec_<model>/          LoRA adapter
    <work_dir>/metrics/tallrec_<model>_<split>_<mode>_n<size>.json

Examples:
    python train_tallrec.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct \\
        --max-train-sessions 8 --max-eval-sessions 2 --slate-size 20
    python train_tallrec.py --work-dir outputs/full --quantize-4bit        # GPU, see CLOUD_RUN.md
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TALLRecConfig, work_paths
from data.amazon_session_loader import PreparedData
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.llm import answer_token_ids, chat_prompt, load_causal_lm, yes_probability
from utils.prompt_templates import NO, YES, engagement_prompt
from utils.slates import baseline_rankings, evaluate_slate_rankings, read_slates, slate_summary

logger = logging.getLogger(__name__)
K_VALUES = [1, 5, 10]
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    d = TALLRecConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--model", default=d.model_name)
    p.add_argument("--slate-mode", default=d.slate_mode)
    p.add_argument("--slate-size", type=int, default=d.slate_size)
    p.add_argument("--eval-split", default="test", choices=["val", "test"])
    p.add_argument("--max-train-sessions", type=int, default=d.max_train_sessions)
    p.add_argument("--max-eval-sessions", type=int, default=d.max_eval_sessions)
    p.add_argument("--negatives", type=int, default=d.num_negatives_per_positive)
    p.add_argument("--max-history-items", type=int, default=d.max_history_items)
    p.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    p.add_argument("--lora-rank", type=int, default=d.lora_rank)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--grad-accum", type=int, default=d.gradient_accumulation_steps)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--eval-batch-size", type=int, default=d.eval_batch_size)
    p.add_argument("--quantize-4bit", action="store_true", help="QLoRA; CUDA only, needs bitsandbytes")
    p.add_argument("--skip-base-eval", action="store_true", help="Do not score the untuned base model")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def build_example(tokenizer, prompt_builder, history_titles: Sequence[str], answer: str,
                  max_len: int, template) -> Optional[Dict[str, List[int]]]:
    """Tokenise prompt + answer with prompt labels masked; drop old history to fit.

    ``template(user_text)`` returns the templated prompt that ends at the assistant turn.
    Returns None when even a single history item does not fit.
    """
    answer_ids = tokenizer.encode(answer + (tokenizer.eos_token or ""), add_special_tokens=False)
    for keep in range(len(history_titles), 0, -1):
        prompt_ids = tokenizer.encode(template(prompt_builder(history_titles[-keep:])), add_special_tokens=False)
        if len(prompt_ids) + len(answer_ids) <= max_len:
            return {"input_ids": prompt_ids + answer_ids,
                    "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids}
    return None


def pad_right(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def training_examples(slates, titles, tokenizer, args, rng) -> tuple:
    examples, skipped = [], 0
    template = lambda text: chat_prompt(tokenizer, text)  # noqa: E731
    for s in slates[: args.max_train_sessions]:
        history = [titles[h] for h in s["history"][-args.max_history_items:]]
        negatives = [c for c in s["candidates"] if c != s["target"]]
        picks = rng.choice(len(negatives), size=min(args.negatives, len(negatives)), replace=False)
        pairs = [(s["target"], YES)] + [(negatives[i], NO) for i in picks.tolist()]
        for item, answer in pairs:
            ex = build_example(tokenizer, lambda h, c=titles[item]: engagement_prompt(h, c),
                               history, answer, args.max_seq_length, template)
            if ex is None:
                skipped += 1
            else:
                examples.append(ex)
    order = rng.permutation(len(examples))
    return [examples[i] for i in order], skipped


def score_slates(model, tokenizer, slates, titles, args, device, yes_id, no_id) -> List[List[str]]:
    rankings = []
    for s in slates:
        history = [titles[h] for h in s["history"][-args.max_history_items:]]
        prompts = [chat_prompt(tokenizer, engagement_prompt(history, titles[c])) for c in s["candidates"]]
        scores = yes_probability(model, tokenizer, prompts, batch_size=args.eval_batch_size,
                                 device=device, yes_id=yes_id, no_id=no_id)
        order = np.argsort(-scores, kind="stable")
        rankings.append([s["candidates"][i] for i in order])
    return rankings


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    titles = dict(zip(data.item_ids, data.titles()))
    train_slates = read_slates(paths.slate_file("train", args.slate_mode, args.slate_size))
    eval_slates = read_slates(paths.slate_file(args.eval_split, args.slate_mode, args.slate_size))[: args.max_eval_sessions]

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, tokenizer = load_causal_lm(args.model, device, args.quantize_4bit)
    yes_id, no_id = answer_token_ids(tokenizer, YES, NO)
    examples, skipped = training_examples(train_slates, titles, tokenizer, args, rng)
    if not examples:
        raise RuntimeError("no training examples fit in --max-seq-length")
    logger.info("%d training examples from %d train slates (%d skipped as too long); %d answer tokens in the first",
                len(examples), min(len(train_slates), args.max_train_sessions), skipped,
                sum(t != IGNORE_INDEX for t in examples[0]["labels"]))

    if args.quantize_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=TALLRecConfig().lora_dropout,
                      target_modules=TALLRecConfig().lora_target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable LoRA parameters: %s", f"{trainable:,}")

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: pad_right(b, tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  weight_decay=TALLRecConfig().weight_decay)
    total = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = max(1, int(TALLRecConfig().warmup_ratio * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup)))
    use_autocast = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_autocast and torch.cuda.is_bf16_supported() else torch.float16
    # fp16 (GPUs without bf16) needs loss scaling or small gradients underflow to zero
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast and amp_dtype == torch.float16)

    history, start = [], time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running, steps = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            steps += 1
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               TALLRecConfig().max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            if step % 50 == 0:
                logger.info("epoch %d step %d / %d loss %.4f", epoch, step, len(loader), running / steps)
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1)})
        logger.info("epoch %d mean loss %.4f", epoch, running / max(steps, 1))
    train_seconds = time.time() - start

    slug = args.model.rstrip("/").split("/")[-1]
    adapter_dir = paths.models_dir / f"tallrec_{slug}"
    model.save_pretrained(adapter_dir)

    model.eval()
    start = time.time()
    results = {}
    if not args.skip_base_eval:
        with model.disable_adapter():
            results["base_pointwise"] = evaluate_slate_rankings(
                eval_slates, score_slates(model, tokenizer, eval_slates, titles, args, device, yes_id, no_id), K_VALUES)
    results["tallrec_lora"] = evaluate_slate_rankings(
        eval_slates, score_slates(model, tokenizer, eval_slates, titles, args, device, yes_id, no_id), K_VALUES)
    eval_seconds = time.time() - start
    popularity = dict(zip(data.item_ids, data.train_item_counts().tolist()))
    baselines = {name: evaluate_slate_rankings(eval_slates, r, K_VALUES)
                 for name, r in baseline_rankings(eval_slates, popularity, args.seed).items()}
    for name, m in {**results, **baselines}.items():
        logger.info("%-20s recall@10 (target in slate) %.4f  mrr %.4f", name,
                    m["target_in_slate"].get("recall@10", float("nan")), m["target_in_slate"].get("mrr", float("nan")))
    tag = f"{slug}_{args.eval_split}_{args.slate_mode}_n{args.slate_size}"
    write_json(paths.metrics_dir / f"tallrec_{tag}.json", run_record(
        "train_tallrec.py", args, device=device, trainable_parameters=trainable,
        training_examples=len(examples), skipped_too_long=skipped, history=history,
        train_seconds=round(train_seconds, 1), eval_seconds=round(eval_seconds, 1),
        slates=slate_summary(eval_slates, K_VALUES), results=results, baselines=baselines,
        adapter=adapter_dir.name))


if __name__ == "__main__":
    main()
