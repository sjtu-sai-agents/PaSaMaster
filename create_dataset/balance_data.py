#!/usr/bin/env python3
# coding: utf-8
"""
Balance sampling for pointwise scores JSONL.

Input: a JSONL where each line has at least:
  - "query": str
  - "score": int in {1,2,3,4,5}

For each query:
  - collect items by score 1..5
  - let n = min(count(score=1..5))
  - randomly sample n items from each score bucket
  - write all sampled items to output JSONL

Finally:
  - print per-query output count (should be 5*n)
  - print total output count
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "result" / "dataset_refcluster" / "deepseek_pointwise_scores.jsonl"),
        help="Path to input JSONL.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "result" / "dataset_refcluster" / "deepseek_pointwise_scores_balanced.jsonl"),
        help="Path to output JSONL.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20250103,
        help="Random seed for reproducible sampling.",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Skip queries whose min bucket size is < min-n (default 1).",
    )
    ap.add_argument(
        "--print-topk",
        type=int,
        default=-1,
        help="Print at most topk queries by output size; -1 prints all.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # query -> score -> items
    buckets: DefaultDict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: {1: [], 2: [], 3: [], 4: [], 5: []}
    )

    bad_lines = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            q = str(obj.get("query", "") or "").strip()
            if not q:
                continue
            s = _safe_int(obj.get("score", None))
            if s not in (1, 2, 3, 4, 5):
                continue
            buckets[q][s].append(obj)

    if bad_lines:
        print(f"[WARN] skipped bad json lines: {bad_lines}")

    per_query_out: Dict[str, int] = {}
    total_out = 0
    kept_queries = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for q, by_score in buckets.items():
            counts = {s: len(by_score[s]) for s in (1, 2, 3, 4, 5)}
            n = min(counts.values()) if counts else 0
            if n < int(args.min_n):
                continue

            sampled: List[Dict[str, Any]] = []
            for s in (1, 2, 3, 4, 5):
                # n <= len(bucket) by definition of min
                sampled.extend(rng.sample(by_score[s], n))

            # write
            for obj in sampled:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            out_cnt = len(sampled)
            per_query_out[q] = out_cnt
            total_out += out_cnt
            kept_queries += 1

    # print per-query counts (optionally limited)
    items = sorted(per_query_out.items(), key=lambda kv: kv[1], reverse=True)
    if int(args.print_topk) >= 0:
        items = items[: int(args.print_topk)]

    for q, cnt in items:
        # keep printing compact (avoid huge query spam)
        q_short = q.replace("\n", " ").strip()
        if len(q_short) > 140:
            q_short = q_short[:140] + "..."
        print(f"[QUERY] out={cnt}  query={q_short!r}")

    print(f"[SUMMARY] kept_queries={kept_queries} total_out={total_out} output={args.output}")


if __name__ == "__main__":
    main()


