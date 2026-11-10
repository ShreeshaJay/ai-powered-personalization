"""
Section 12.4: generate LLM features offline and embed them for the augmented ranker.

Two KAR-style generations (Xi et al., 2024), both from an open-weights LLM:

    item knowledge   a two-sentence description of every item in the work directory
    session intent   one sentence on what the shopper is after, from the session
                     *history only* (never the target), for every session

Both are embedded with the same text encoder that produced item_embeddings.npz, so the
ranker can compare LLM text with item content in one vector space; the script refuses to
mix encoders. Coverage is complete by construction unless --max-items or --max-sessions
is set, in which case the ranker stops unless it is told to accept partial coverage.

Generations are appended to <work_dir>/llm/*.jsonl and an interrupted run resumes.
Outputs:
    <work_dir>/embeddings/llm_item_knowledge_embeddings.npz      item_ids, embeddings, encoder
    <work_dir>/embeddings/llm_session_intent_embeddings.npz      session_ids, embeddings, encoder

--backend copy-input skips the LLM and embeds the prompt's own product text instead. It
is a pipeline check and a control (does the LLM add anything beyond the input text?), not
an LLM feature.

Examples:
    python generate_llm_features.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct --max-items 40 --max-sessions 40
    python generate_llm_features.py --work-dir outputs/ranker                     # GPU, see CLOUD_RUN.md
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Sequence

import numpy as np

from config import LLMFeatureConfig, work_paths
from data.amazon_session_loader import PreparedData
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.encoders import load_encoder, save_embeddings, stored_encoder
from utils.llm import append_jsonl, chat_prompt, generate, load_causal_lm, read_jsonl
from utils.prompt_templates import clean_generation, item_knowledge_prompt, session_intent_prompt, short_title

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    d = LLMFeatureConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--model", default=d.model_name)
    p.add_argument("--backend", choices=["hf", "copy-input"], default="hf")
    p.add_argument("--stages", nargs="+", default=["items", "intents"], choices=["items", "intents"])
    p.add_argument("--max-items", type=int, default=d.max_items)
    p.add_argument("--max-sessions", type=int, default=d.max_sessions)
    p.add_argument("--item-max-new-tokens", type=int, default=d.item_max_new_tokens)
    p.add_argument("--intent-max-new-tokens", type=int, default=d.intent_max_new_tokens)
    p.add_argument("--max-history-items", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


class Generator:
    """Runs prompts through the LLM, or returns the input text for the copy-input control."""

    def __init__(self, args, device: str):
        self.args, self.device = args, device
        self.model = self.tokenizer = None

    def __call__(self, prompts: Sequence[str], fallbacks: Sequence[str], max_new_tokens: int) -> List[str]:
        if self.args.backend == "copy-input":
            return list(fallbacks)
        if self.model is None:
            self.model, self.tokenizer = load_causal_lm(self.args.model, self.device, self.args.quantize_4bit)
        templated = [chat_prompt(self.tokenizer, p) for p in prompts]
        return generate(self.model, self.tokenizer, templated, max_new_tokens=max_new_tokens,
                        batch_size=self.args.batch_size, device=self.device)


def run_stage(name: str, keys: List, prompts: List[str], fallbacks: List[str], raw_path, gen: Generator,
              max_new_tokens: int, chunk: int) -> Dict:
    done = read_jsonl(raw_path, "key")
    todo = [i for i, k in enumerate(keys) if k not in done]
    logger.info("%s: %d to generate (%d already done)", name, len(todo), len(keys) - len(todo))
    start = time.time()
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        outs = gen([prompts[i] for i in idx], [fallbacks[i] for i in idx], max_new_tokens)
        append_jsonl(raw_path, [{"key": keys[i], "text": clean_generation(o)} for i, o in zip(idx, outs)])
        if (s // chunk) % 10 == 0:
            logger.info("%s: %d / %d", name, min(s + chunk, len(todo)), len(todo))
    seconds = time.time() - start
    done = read_jsonl(raw_path, "key")
    empty = sum(1 for k in keys if not done[k]["text"])
    return {"generated_now": len(todo), "seconds": round(seconds, 1), "empty_outputs": empty,
            "texts": [done[k]["text"] for k in keys]}


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    encoder_name = stored_encoder(paths.item_embeddings)
    encoder = load_encoder(encoder_name, paths.embeddings_dir, device)
    logger.info("Embedding LLM text with the item encoder: %s", encoder_name)
    gen = Generator(args, device)
    source = args.model if args.backend == "hf" else "copy-input"
    slug = source.rstrip("/").split("/")[-1]
    chunk = max(args.batch_size * 8, 8)
    summary: Dict = {"llm": source, "encoder": encoder_name}

    if "items" in args.stages:
        items = data.items if args.max_items is None else data.items.head(args.max_items)
        keys = items["item_id"].to_list()
        prompts = [item_knowledge_prompt(t, b, d) for t, b, d in
                   zip(items["title"].to_list(), items["brand"].to_list(), items["desc"].to_list())]
        stage = run_stage("item knowledge", keys, prompts, items["text"].to_list(),
                          paths.llm_dir / f"item_knowledge_{slug}.jsonl", gen, args.item_max_new_tokens, chunk)
        vectors = encoder.encode(stage.pop("texts"))
        save_embeddings(paths.llm_item_embeddings, keys, vectors, encoder_name,
                        extra={"llm": np.asarray(source)})
        summary["items"] = {**stage, "covered": len(keys), "universe": data.num_items,
                            "coverage": len(keys) / data.num_items}

    if "intents" in args.stages:
        sessions = data.sessions if args.max_sessions is None else data.sessions.head(args.max_sessions)
        keys = sessions["session_id"].to_list()
        title_of = dict(zip(data.item_ids, data.titles()))
        histories = [[title_of[i] for i in h[-args.max_history_items:]] for h in sessions["history"].to_list()]
        prompts = [session_intent_prompt(h) for h in histories]
        fallbacks = ["; ".join(short_title(t, 60) for t in h) for h in histories]
        stage = run_stage("session intent", keys, prompts, fallbacks,
                          paths.llm_dir / f"session_intent_{slug}.jsonl", gen, args.intent_max_new_tokens, chunk)
        vectors = encoder.encode(stage.pop("texts"))
        save_embeddings(paths.llm_intent_embeddings, keys, vectors, encoder_name, key_name="session_ids",
                        extra={"llm": np.asarray(source)})
        summary["intents"] = {**stage, "covered": len(keys), "universe": data.sessions.height,
                              "coverage": len(keys) / data.sessions.height}

    for stage in ("items", "intents"):
        if stage in summary:
            logger.info("%s coverage %.4f (%d of %d)", stage, summary[stage]["coverage"],
                        summary[stage]["covered"], summary[stage]["universe"])
    write_json(paths.metrics_dir / f"llm_features_{slug}.json",
               run_record("generate_llm_features.py", args, device=device, summary=summary))


if __name__ == "__main__":
    main()
