"""
Section 12.5: zero-shot listwise ranking with an open-weights instruction-tuned LLM.

Reads one shared slate file from build_slates.py (default: test, retrieved, 20
candidates), asks the model to order the lettered candidates, and scores the parsed
ranking. The target is never inserted into a slate here; see utils/slates.py for what
each slate mode means.

Reported, in <work_dir>/metrics/zeroshot_<model>_<split>_<mode>_n<size>.json:
    ranking metrics       over all sessions and over sessions whose slate holds the target
    output validity       answers with a 'Ranking:' line, complete permutations, duplicate
                          and out-of-range letters; omitted candidates are appended in a
                          seeded random order
    position bias         how often the first choice is the first displayed candidate,
                          against the 1/N a position-blind ranker would show, and the
                          display positions (1 = candidate A) of first choices
    baselines             random, popularity and history-similarity orders on the same slates

Raw generations are appended to <work_dir>/llm/ as JSONL, so an interrupted run resumes.

Examples:
    python evaluate_zeroshot_llm.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct --max-sessions 20
    python evaluate_zeroshot_llm.py --work-dir outputs/full --quantize-4bit        # GPU
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

import numpy as np

from config import ZeroShotLLMConfig, work_paths
from data.amazon_session_loader import PreparedData
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.llm import append_jsonl, chat_prompt, generate, load_causal_lm, read_jsonl
from utils.prompt_templates import listwise_ranking_prompt, parse_listwise_ranking
from utils.slates import baseline_rankings, evaluate_slate_rankings, read_slates, slate_summary

logger = logging.getLogger(__name__)
K_VALUES = [1, 5, 10]


def parse_args() -> argparse.Namespace:
    d = ZeroShotLLMConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--model", default=d.model_name)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--slate-mode", default=d.slate_mode)
    p.add_argument("--slate-size", type=int, default=d.slate_size)
    p.add_argument("--max-sessions", type=int, default=d.max_sessions)
    p.add_argument("--max-history-items", type=int, default=d.max_history_items)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--quantize-4bit", action="store_true", help="CUDA only; needs bitsandbytes")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def model_slug(name: str) -> str:
    return name.rstrip("/").split("/")[-1]


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    titles = dict(zip(data.item_ids, data.titles()))

    slates = read_slates(paths.slate_file(args.split, args.slate_mode, args.slate_size))[: args.max_sessions]
    tag = f"{model_slug(args.model)}_{args.split}_{args.slate_mode}_n{args.slate_size}"
    raw_path = paths.llm_dir / f"zeroshot_{tag}.jsonl"
    done = read_jsonl(raw_path, "session_id")
    todo = [s for s in slates if s["session_id"] not in done]
    logger.info("%d slates, %d already generated, %d to go", len(slates), len(slates) - len(todo), len(todo))

    gen_seconds = 0.0
    if todo:
        model, tokenizer = load_causal_lm(args.model, device, args.quantize_4bit)
        start = time.time()
        step = max(args.batch_size * 4, 1)
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            prompts = [chat_prompt(tokenizer, listwise_ranking_prompt(
                [titles[h] for h in s["history"][-args.max_history_items:]],
                [titles[c] for c in s["candidates"]])) for s in chunk]
            outputs = generate(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens,
                               batch_size=args.batch_size, device=device)
            append_jsonl(raw_path, [{"session_id": s["session_id"], "output": o} for s, o in zip(chunk, outputs)])
            logger.info("generated %d / %d", min(i + step, len(todo)), len(todo))
        gen_seconds = time.time() - start
        done = read_jsonl(raw_path, "session_id")

    rng = np.random.default_rng(args.seed)
    rankings, parsed = [], []
    for s in slates:
        result = parse_listwise_ranking(done[s["session_id"]]["output"], len(s["candidates"]), rng)
        parsed.append(result)
        rankings.append([s["candidates"][i] for i in result.order])

    n = len(slates)
    size = len(slates[0]["candidates"]) if slates else 0
    with_answer = [p for p in parsed if p.parsed]
    first_positions = Counter(p.order[0] for p in with_answer)
    validity = {
        "sessions": n,
        "found_ranking_line_rate": float(np.mean([p.found_ranking_line for p in parsed])) if n else 0.0,
        "parsed_any_label_rate": len(with_answer) / n if n else 0.0,
        "complete_permutation_rate": float(np.mean([p.complete(size) for p in parsed])) if n else 0.0,
        "mean_valid_labels": float(np.mean([p.valid_labels for p in parsed])) if n else 0.0,
        "sessions_with_duplicate_labels": float(np.mean([p.duplicate_labels > 0 for p in parsed])) if n else 0.0,
        "sessions_with_out_of_range_labels": float(np.mean([p.invalid_labels > 0 for p in parsed])) if n else 0.0,
    }
    position_bias = {
        "first_choice_is_first_displayed_rate": first_positions.get(0, 0) / len(with_answer) if with_answer else None,
        "position_blind_expectation": 1.0 / size if size else None,
        "first_choice_display_position_counts": {int(k) + 1: v for k, v in sorted(first_positions.items())},
    }
    popularity = dict(zip(data.item_ids, data.train_item_counts().tolist()))
    baselines = {name: evaluate_slate_rankings(slates, r, K_VALUES)
                 for name, r in baseline_rankings(slates, popularity, args.seed).items()}
    llm_metrics = evaluate_slate_rankings(slates, rankings, K_VALUES)
    logger.info("LLM recall@10 (target in slate): %.4f | history-similarity: %.4f | random expectation: %.4f",
                llm_metrics["target_in_slate"].get("recall@10", float("nan")),
                baselines["history_similarity"]["target_in_slate"].get("recall@10", float("nan")),
                min(10, size) / size if size else float("nan"))
    logger.info("Validity: %s", validity)
    logger.info("Position bias: %s", position_bias)
    write_json(paths.metrics_dir / f"zeroshot_{tag}.json", run_record(
        "evaluate_zeroshot_llm.py", args, device=device, generation_seconds=round(gen_seconds, 1),
        slates=slate_summary(slates, K_VALUES), llm=llm_metrics, validity=validity,
        position_bias=position_bias, baselines=baselines, raw_outputs=raw_path.name))


if __name__ == "__main__":
    main()
