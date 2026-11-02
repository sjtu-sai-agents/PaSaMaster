#!/usr/bin/env python3
# coding: utf-8
"""
Re-write dataset scoring pipeline for refcluster results.

For each folder under a root directory, read `results.jsonl` and `topics.json`.
Each line in results.jsonl contains:
  - question: [q1, q2, q3]
  - paper_list: [{title, abstract, arxiv_id, select_score, ...}, ...]

For each query:
  1) Call model (deepseek-v3.2-our) to generate checklist using prompt:
     prompt/papermaster_prompt.json -> keyminder.extract_keywords_and_queries
  2) Build final_paper_list:
     - Include papers referenced by the query's topic paper_ids (topics.json)
     - For each score interval bucket: >=77, [61,77), [46,61), [31,46), [15,31), [0,15):
       take top-K (by select_score desc) where K defaults to 50 (configurable)
     - De-duplicate by paper_id (normalized as "arxiv:{arxiv_id}")
  3) For each selected paper, call model to score relevance (1-5) using prompt:
     prompt/papermaster_prompt.json -> ranker.point_wise_rank_english
  4) Persist per (query, paper) result as JSONL with keys:
     - instruction: full prompt sent to model
     - output: raw model output
     - score: parsed 1-5 (or null)
     - paper: paper title
     - query: query string
  5) Write per-query metric JSONL line:
     counts + ratios across score 1..5 for that query (based on parsed scores).

Outputs are written into each processed folder by default:
  - deepseek_checklist_outputs.jsonl
  - deepseek_pointwise_scores.jsonl
  - deepseek_metrics.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_float(x: Any, default: float = -1.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_paper_id(p: Dict[str, Any]) -> str:
    """
    Normalize a paper identifier for de-duplication.

    - Prefer arXiv id when present.
    - Otherwise fall back to a normalized title key:
      remove all whitespace (spaces/newlines/tabs), lowercased.

    This prevents papers without arxiv_id from being silently dropped.
    """

    # topics.json uses "arxiv:XXXX.YYYYY" while paper_list has arxiv_id "XXXX.YYYYY"
    aid = p.get("arxiv_id") or p.get("paper_id") or p.get("id")
    if aid is not None:
        s = str(aid).strip()
        if s:
            return s if s.startswith("arxiv:") else f"arxiv:{s}"

    # Fallback: title-based id (whitespace stripped + lowercased)
    title = str(p.get("title") or "").strip()
    if not title:
        return ""
    tkey = re.sub(r"\s+", "", title).lower()
    return f"title:{tkey}" if tkey else ""

def _seed_arxiv_id_from_folder(seed_folder: Path) -> str:
    """
    Infer seed arXiv id from seed folder name like: seed_arxiv_2306.00890
    """
    name = seed_folder.name
    if name.startswith("seed_arxiv_"):
        return name[len("seed_arxiv_") :].strip()
    # fallback: try to find an arxiv-ish token
    m = re.search(r"(\d{4}\.\d{4,5})", name)
    return m.group(1) if m else name


def _extract_domain_topic(root: Path, seed_folder: Path) -> Tuple[str, str]:
    """
    For structure: root/<domain>/<topic>/<seed_folder>/results.jsonl
    """
    try:
        rel = seed_folder.resolve().relative_to(root.resolve())
    except Exception:
        rel = seed_folder
    parts = list(rel.parts)
    domain = parts[0] if len(parts) >= 1 else "unknown"
    topic = parts[1] if len(parts) >= 2 else "unknown"
    return domain, topic


def _progress_bar(cur: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    cur = max(0, min(int(cur), total))
    filled = int(width * cur / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _fmt_secs(sec: float) -> str:
    try:
        sec = float(sec)
    except Exception:
        sec = 0.0
    if sec < 60:
        return f"{sec:.2f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}m{s:.1f}s"


TAG_RE_CACHE: Dict[str, re.Pattern[str]] = {}


def _extract_tag(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    pat = TAG_RE_CACHE.get(tag)
    if pat is None:
        pat = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
        TAG_RE_CACHE[tag] = pat
    m = pat.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _parse_checkpoints(text: str) -> List[str]:
    raw = _extract_tag(text, "checkpoints")
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def _parse_score_1_to_5(text: str) -> Optional[int]:
    """
    Parse the score, following the logic in quick_thinker.py:
    1. Extract the <relevant>, <reason>, <constraint>, <irrelevant> tags
    2. Extract all Score:number patterns from <constraint>
    3. Compute the score using L2-norm: sqrt((a^2 + b^2) / (5 * len(constraint_scores))) * 100
    4. If prob is present, then (score+prob*10)/110*100
    5. Map the 0-100 score back to a 1-5 integer
    """
    import math
    
    if not text:
        return None
    
    try:
        # Extract the scores of all constraints in <constraint>
        constraint_begin = text.find("<constraint>")
        constraint_end = text.find("</constraint>")
        constraint_text = ""
        if constraint_begin >= 0 and constraint_end > constraint_begin:
            constraint_text = text[constraint_begin + len("<constraint>"):constraint_end].strip()
    
        # Parse each constraint's score Score:<1-5>
        constraint_scores = []
        if constraint_text:
            # Match all Score:number patterns
            score_matches = re.findall(r'Score:\s*(\d+)', constraint_text)
            constraint_scores = [int(s) for s in score_matches if 1 <= int(s) <= 5]
        
        # If no constraint scores were found, try the traditional <score> tag
        if not constraint_scores:
            raw = _extract_tag(text, "score")
            if raw:
                m = re.search(r"([1-5])", raw)
                if m:
                    return int(m.group(1))
            # fallback: search anywhere
            m = re.search(r"<score>\s*([1-5])\s*</score>", text or "", re.IGNORECASE)
            if m:
                return int(m.group(1))
            return None
        
        # Compute the score: use L2-norm (square-sum normalized then square-rooted)
        num_constraints = len(constraint_scores)
        if num_constraints == 0:
            return None
        
        # Weighting scheme: L2-norm (square-sum normalized then square-rooted)
        # Assuming constraint_scores = [a, b], then score = sqrt((a^2 + b^2) / (5 * len(constraint_scores))) * 100
        mean_sq = sum(s * s for s in constraint_scores) / (25 * num_constraints)
        score = math.sqrt(mean_sq) * 100
        
        # Note: quick_thinker.py has a prob parameter, but there is none here, so the prob adjustment is skipped
        # if prob != 0:
        #     score = (score+prob*10)/110*100
        
        # Map the 0-100 score back to a 1-5 integer
        # Mapping rule: 0-20 -> 1, 20-40 -> 2, 40-60 -> 3, 60-80 -> 4, 80-100 -> 5
        if score < 20:
            return 1
        elif score < 40:
            return 2
        elif score < 60:
            return 3
        elif score < 80:
            return 4
        else:
            return 5
            
    except Exception as e:
        # If parsing fails, try the traditional <score> tag as a fallback
        raw = _extract_tag(text, "score")
        if raw:
            m = re.search(r"([1-5])", raw)
            if m:
                return int(m.group(1))
        # fallback: search anywhere
        m = re.search(r"<score>\s*([1-5])\s*</score>", text or "", re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None


def _load_prompts(prompts_path: Path) -> Tuple[str, str]:
    prompts = _read_json(prompts_path)
    checklist_tpl = prompts["keyminder"]["extract_keywords_and_queries"]
    rank_tpl = prompts["ranker"]["point_wise_rank_english"]
    return checklist_tpl, rank_tpl


def _topic_paper_ids_for_query(topics_obj: Dict[str, Any], query: str) -> List[str]:
    # topics.json is a dict: { "<topic>(cXXXX)": {paper_ids: [...], questions: [{query: ...}, ...]} }
    for _, topic in (topics_obj or {}).items():
        if not isinstance(topic, dict):
            continue
        questions = topic.get("questions")
        if not isinstance(questions, list):
            continue
        for q in questions:
            if isinstance(q, dict) and q.get("query") == query:
                paper_ids = topic.get("paper_ids") or []
                if isinstance(paper_ids, list):
                    return [str(x).strip() for x in paper_ids if str(x).strip()]
    return []


@dataclass(frozen=True)
class Bin:
    name: str
    lo: float  # inclusive
    hi: float  # exclusive

    def contains(self, x: float) -> bool:
        return self.lo <= x < self.hi


BINS: List[Bin] = [
    Bin("[61,77)", 61.0, 77.0),
    Bin("[46,61)", 46.0, 61.0),
    Bin("[31,46)", 31.0, 46.0),
    Bin("[15,31)", 15.0, 31.0),
    Bin("[0,15)", 0.0, 15.0),
]


def _select_final_papers(
    paper_list: Sequence[Dict[str, Any]],
    *,
    topic_paper_ids: Sequence[str],
    topk_per_bin: int = 50,
) -> List[Dict[str, Any]]:
    """
    Build the final scoring list for a query.

    New behavior (per request):
    - Do NOT filter by select_score buckets / topk.
    - Score ALL papers in `paper_list` (deduplicated by normalized paper_id).
    - Also include the topic construction papers (`topic_paper_ids`) *if present in paper_list*,
      and place them first (keep topic order).

    NOTE:
    - `topics.json` typically only provides paper_ids; if a topic paper_id is not present in
      `paper_list`, we cannot score it because we don't have title/abstract. We skip those ids.
    - `topk_per_bin` is kept for backward compatibility but is unused in this mode.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for p in paper_list:
        if not isinstance(p, dict):
            continue
        pid = _normalize_paper_id(p)
        if pid:
            by_id[pid] = p

    selected_ids: List[str] = []
    selected_set: Set[str] = set()

    def add_pid(pid: str) -> None:
        if not pid or pid not in by_id:
            return
        if pid in selected_set:
            return
        selected_set.add(pid)
        selected_ids.append(pid)

    # 1) topic paper_ids first (keep topic order; only if present in paper_list)
    for pid in topic_paper_ids:
        add_pid(str(pid).strip())

    # 2) then ALL papers in paper_list order (deduplicated)
    for p in paper_list:
        if not isinstance(p, dict):
            continue
        pid = _normalize_paper_id(p)
        add_pid(pid)
    return [by_id[pid] for pid in selected_ids if pid in by_id]


def _load_existing_pairs(scores_jsonl: Path) -> Set[Tuple[str, str, str]]:
    """
    Returns a set of (seed_arxiv_id, query, paper_id) that already have a scoring record.
    """
    done: Set[Tuple[str, str, str]] = set()
    if not scores_jsonl.exists():
        return done
    with scores_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seed = obj.get("seed_arxiv_id")
            q = obj.get("query")
            pid = obj.get("paper_id") or obj.get("arxiv_id") or obj.get("id")
            if isinstance(seed, str) and isinstance(q, str) and isinstance(pid, str):
                pid_s = pid.strip()
                if pid_s and not pid_s.startswith("arxiv:"):
                    # normalize old-format arxiv_id into "arxiv:XXXX.YYYYY"
                    pid_s = f"arxiv:{pid_s}"
                done.add((seed.strip(), q, pid_s))
    return done


def _ck_key(seed_arxiv_id: str, query: str) -> str:
    return f"{seed_arxiv_id}\n{query}"


def _load_checklist_cache(checklist_jsonl: Path) -> Dict[str, List[str]]:
    """
    Returns {(seed_arxiv_id, query): checkpoints} as a single string key for existing
    checklist generations (best-effort).
    """
    cache: Dict[str, List[str]] = {}
    if not checklist_jsonl.exists():
        return cache
    with checklist_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seed = obj.get("seed_arxiv_id")
            q = obj.get("query")
            cps = obj.get("checkpoints")
            if (
                isinstance(seed, str)
                and isinstance(q, str)
                and isinstance(cps, list)
                and all(isinstance(x, str) for x in cps)
            ):
                cache[_ck_key(seed.strip(), q)] = [x.strip() for x in cps if x and x.strip()]
    return cache


def _load_metrics_cache(metrics_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """
    Keep only the latest metric record per query (best-effort).
    """
    cache: Dict[str, Dict[str, Any]] = {}
    if not metrics_jsonl.exists():
        return cache
    with metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = obj.get("query")
            if isinstance(q, str) and isinstance(obj, dict):
                cache[q] = obj
    return cache


def _rewrite_metrics_jsonl(metrics_jsonl: Path, metrics_by_query: Dict[str, Dict[str, Any]]) -> None:
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # stable output order for diffs/debugging
    items = [metrics_by_query[k] for k in sorted(metrics_by_query.keys())]
    with metrics_jsonl.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _call_llm(prompt: str, model_name: str, timeout: int) -> Optional[str]:
    """
    Call LLM with retries.

    Requirements (per user request):
    - timeout: 120s per attempt
    - retry: 3 times
    - on failure: return None (caller should skip and NOT write output)
    """
    _ensure_repo_on_path()
    from tools.llm_call import llm_call  # type: ignore

    attempt_timeout = 120  # enforce 120s per attempt
    max_retries = 3
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            # NOTE: ignore passed `timeout` and enforce `attempt_timeout` as requested
            return str(llm_call(prompt, model_name=model_name, timeout=attempt_timeout))
        except BaseException as e:
            last_err = e
            # best-effort logging; never interrupt the whole pipeline
            msg = str(e).replace("\n", " ").strip()
            print(
                f"[LLM] call failed (attempt {attempt}/{max_retries}) model={model_name} timeout={attempt_timeout}s err={msg}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                # small backoff to avoid hammering the endpoint
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
    # final failure: skip
    if last_err is not None:
        msg = str(last_err).replace("\n", " ").strip()
        print(f"[LLM] giving up after {max_retries} attempts; skip. err={msg}", file=sys.stderr)
    return None


def _build_rank_prompt(
    rank_tpl: str,
    *,
    title: str,
    abstract: str,
    user_query: str,
    constraints: List[str],
    meta_info: Dict[str, Any],
) -> str:
    return rank_tpl.format(
        title=title or "",
        abstract=abstract or "",
        user_query=user_query or "",
        constraints=json.dumps(constraints, ensure_ascii=False),
        meta_info=json.dumps(meta_info, ensure_ascii=False),
    )


def _metric_from_scores(scores: List[Optional[int]]) -> Dict[str, Any]:
    total = len(scores)
    counts = {str(i): 0 for i in range(1, 6)}
    for s in scores:
        if isinstance(s, int) and 1 <= s <= 5:
            counts[str(s)] += 1
    ratios = {k: (counts[k] / total if total else 0.0) for k in counts}
    return {"total": total, "counts": counts, "ratios": ratios}


def _collect_scores_for_seed_query(scores_jsonl: Path, *, seed_arxiv_id: str, query: str) -> List[Optional[int]]:
    """
    Collect parsed scores (1-5 or None) for a (seed_arxiv_id, query) from the persisted JSONL file.
    """
    out: List[Optional[int]] = []
    if not scores_jsonl.exists():
        return out
    with scores_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("query") != query or obj.get("seed_arxiv_id") != seed_arxiv_id:
                continue
            s = obj.get("score")
            if isinstance(s, int):
                out.append(s if 1 <= s <= 5 else None)
            else:
                out.append(None)
    return out


def process_folder(
    seed_folder: Path,
    *,
    root: Path,
    domain: str,
    topic: str,
    seed_arxiv_id: str,
    checklist_out: Path,
    scores_out: Path,
    metrics_out: Path,
    prompts_path: Path,
    model_name: str,
    timeout: int,
    concurrency: int,
    verbose_paper_logs: bool,
    dry_run: bool,
    limit_queries: int,
    limit_papers_per_query: int,
    n_if_no_77: int,
    topk_per_bin: int,
) -> None:
    results_path = seed_folder / "results.jsonl"
    topics_path = seed_folder / "topics.json"
    if not results_path.exists() or not topics_path.exists():
        return

    t_folder0 = time.monotonic()
    print(f"[SEED] start: domain={domain} topic={topic} seed={seed_arxiv_id} path={seed_folder}")

    checklist_tpl, rank_tpl = _load_prompts(prompts_path)
    topics_obj = _read_json(topics_path)

    done_pairs = _load_existing_pairs(scores_out)
    checklist_cache = _load_checklist_cache(checklist_out)
    metrics_cache = _load_metrics_cache(metrics_out)

    folder_queries = 0
    folder_new_checklists = 0
    folder_score_skipped = 0
    folder_score_done = 0
    folder_checklist_time = 0.0
    folder_scoring_time = 0.0

    for item in _iter_jsonl(results_path):
        questions = item.get("question") or []
        paper_list = item.get("paper_list") or []
        if not isinstance(questions, list) or not isinstance(paper_list, list):
            continue

        for qi, query in enumerate(questions):
            if limit_queries != -1 and qi >= limit_queries:
                break
            if not isinstance(query, str) or not query.strip():
                continue
            query = query.strip()
            folder_queries += 1

            # 1) checklist generation
            checklist_prompt = checklist_tpl.format(user_query=query)
            checklist_output = ""
            checkpoints: List[str] = checklist_cache.get(_ck_key(seed_arxiv_id, query), [])
            generated_checklist = False
            t0 = time.monotonic()
            if not checkpoints and not dry_run:
                checklist_output_opt = _call_llm(checklist_prompt, model_name=model_name, timeout=timeout)
                if checklist_output_opt is None:
                    # fail closed: do not write checklist_out, do not interrupt
                    print(f"[CHECKLIST] failed; skip persist | query={query[:120]}", file=sys.stderr)
                else:
                    checklist_output = checklist_output_opt
                    checkpoints = _parse_checkpoints(checklist_output)
                    generated_checklist = True
            t_check = time.monotonic() - t0
            if generated_checklist:
                folder_new_checklists += 1
                folder_checklist_time += t_check
                print(f"[CHECKLIST] generated in {_fmt_secs(t_check)} | checkpoints={len(checkpoints)} | query={query[:120]}")
            else:
                # cache-hit or dry-run: no model call
                print(f"[CHECKLIST] cache-hit | checkpoints=[{checkpoints}] | query={query}")

            # Incremental behavior:
            # - If checklist already exists (cached checkpoints), do not call LLM and do not append duplicate records.
            # - Only persist when newly generated (or dry_run explicitly requested).
            if generated_checklist or dry_run:
                _append_jsonl(
                    checklist_out,
                    {
                        "instruction": checklist_prompt,
                        "output": checklist_output,
                        "checkpoints": checkpoints,
                        "query": query,
                        "seed_arxiv_id": seed_arxiv_id,
                        "domain": domain,
                        "topic": topic,
                    },
                )
                print(f"[CHECKLIST] save checklist_out: {checklist_out}")
            ck = _ck_key(seed_arxiv_id, query)
            if checkpoints and ck not in checklist_cache:
                checklist_cache[ck] = checkpoints

            # 2) select papers
            topic_paper_ids = _topic_paper_ids_for_query(topics_obj, query)
            selected = _select_final_papers(
                paper_list,
                topic_paper_ids=topic_paper_ids,
                topk_per_bin=topk_per_bin,
            )
            if limit_papers_per_query != -1:
                selected = selected[:limit_papers_per_query]

            # 3) per-paper scoring
            work: List[Tuple[str, Dict[str, Any]]] = []
            for p in selected:
                if not isinstance(p, dict):
                    continue
                pid_full = _normalize_paper_id(p)
                pid = pid_full or ""
                if (seed_arxiv_id, query, pid) in done_pairs:
                    folder_score_skipped += 1
                    continue
                work.append((pid, p))

            def _score_one(pid: str, p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                title = str(p.get("title") or "")
                abstract = str(p.get("abstract") or "")
                meta_info = {
                    "year": p.get("year"),
                    "authors": p.get("authors"),
                }
                rank_prompt = _build_rank_prompt(
                    rank_tpl,
                    title=title,
                    abstract=abstract,
                    user_query=query,
                    constraints=checkpoints,
                    meta_info=meta_info,
                )

                raw = ""
                score = None
                if not dry_run:
                    raw_opt = _call_llm(rank_prompt, model_name=model_name, timeout=timeout)
                    if raw_opt is None:
                        # skip this paper; do not write output
                        if verbose_paper_logs:
                            print(f"[SCORING] failed; skip | title={title[:120]}", file=sys.stderr)
                        return None
                    raw = raw_opt
                    score = _parse_score_1_to_5(raw)
                    if verbose_paper_logs:
                        print(f"[SCORING] done | score={score} | title={title[:120]}")

                return {
                    "instruction": rank_prompt,
                    "output": raw,
                    "score": score,
                    "paper": title,
                    "query": query,
                    "paper_id": pid,
                    "arxiv_id": p.get("arxiv_id"),
                    "select_score": p.get("select_score"),
                    "seed_arxiv_id": seed_arxiv_id,
                    "domain": domain,
                    "topic": topic,
                }

            # Run scoring concurrently, then write results sequentially to avoid file corruption.
            t1 = time.monotonic()
            total_to_score = len(work)
            print(f"\033[92m[SCORING] Begin scoring for {total_to_score} papers...\033[0m")
            folder_score_failed = 0
            if work:
                max_workers = max(1, int(concurrency))
                if dry_run or max_workers == 1:
                    done = 0
                    for pid, p in work:
                        out_obj = _score_one(pid, p)
                        if out_obj is None:
                            folder_score_failed += 1
                            done += 1
                            if total_to_score > 0 and (done == total_to_score or done % 1 == 0):
                                bar = _progress_bar(done, total_to_score)
                                sys.stdout.write(f"\r[SCORING] {done}/{total_to_score} {bar}")
                                sys.stdout.flush()
                            continue
                        _append_jsonl(scores_out, out_obj)
                        qv = out_obj.get("query")
                        pidv = out_obj.get("paper_id")
                        seedv = out_obj.get("seed_arxiv_id")
                        if isinstance(seedv, str) and isinstance(qv, str) and isinstance(pidv, str):
                            done_pairs.add((seedv.strip(), qv, pidv))
                        folder_score_done += 1
                        done += 1
                        if total_to_score > 0 and (done == total_to_score or done % 1 == 0):
                            bar = _progress_bar(done, total_to_score)
                            sys.stdout.write(f"\r[SCORING] {done}/{total_to_score} {bar}")
                            sys.stdout.flush()
                    if total_to_score > 0:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = [ex.submit(_score_one, pid, p) for pid, p in work]
                        done = 0
                        last_print = time.monotonic()
                        for fut in concurrent.futures.as_completed(futs):
                            try:
                                out_obj = fut.result()
                            except BaseException as e:
                                msg = str(e).replace("\n", " ").strip()
                                print(f"[SCORING] worker exception; skip | err={msg}", file=sys.stderr)
                                out_obj = None
                            if out_obj is None:
                                folder_score_failed += 1
                                done += 1
                                now = time.monotonic()
                                should_print = (
                                    done == total_to_score
                                    or done % max(1, total_to_score // 20) == 0
                                    or (now - last_print) >= 5.0
                                )
                                if should_print and total_to_score > 0:
                                    bar = _progress_bar(done, total_to_score)
                                    sys.stdout.write(f"\r[SCORING] {done}/{total_to_score} {bar}")
                                    sys.stdout.flush()
                                    last_print = now
                                continue
                            _append_jsonl(scores_out, out_obj)
                            qv = out_obj.get("query")
                            pidv = out_obj.get("paper_id")
                            seedv = out_obj.get("seed_arxiv_id")
                            if isinstance(seedv, str) and isinstance(qv, str) and isinstance(pidv, str):
                                done_pairs.add((seedv.strip(), qv, pidv))
                            folder_score_done += 1
                            done += 1
                            now = time.monotonic()
                            # throttle progress output
                            should_print = (
                                done == total_to_score
                                or done % max(1, total_to_score // 20) == 0
                                or (now - last_print) >= 5.0
                            )
                            if should_print and total_to_score > 0:
                                bar = _progress_bar(done, total_to_score)
                                sys.stdout.write(f"\r[SCORING] {done}/{total_to_score} {bar}")
                                sys.stdout.flush()
                                last_print = now
                        if total_to_score > 0:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
            t_score = time.monotonic() - t1
            if work and not dry_run:
                folder_scoring_time += t_score
            print(
                f"[SCORING] papers_total_selected={len(selected)} to_score={len(work)} "
                f"concurrency={max(1,int(concurrency))} time={_fmt_secs(t_score)} failed={folder_score_failed}"
            )
            # 4) metrics for this query (based on all persisted scores so far)
            all_scores = _collect_scores_for_seed_query(scores_out, seed_arxiv_id=seed_arxiv_id, query=query)
            metric = _metric_from_scores(all_scores)
            metrics_cache[_ck_key(seed_arxiv_id, query)] = {
                "query": query,
                "seed_arxiv_id": seed_arxiv_id,
                "domain": domain,
                "topic": topic,
                "seed_folder": str(seed_folder),
                "root": str(root),
                "topk_per_bin": int(topk_per_bin),
                "n_if_no_77": n_if_no_77,
                "topic_paper_ids_count": len(topic_paper_ids),
                "selected_papers": len(selected),
                "scored_papers_total": metric["total"],
                "score_counts": metric["counts"],
                "score_ratios": metric["ratios"],
            }

    # Rewrite metrics as de-duplicated (one line per query).
    _rewrite_metrics_jsonl(metrics_out, metrics_cache)

    t_folder = time.monotonic() - t_folder0
    print(
        f"[SEED] done: domain={domain} topic={topic} seed={seed_arxiv_id} | queries={folder_queries} "
        f"new_checklists={folder_new_checklists} checklist_time={_fmt_secs(folder_checklist_time)} "
        f"scored_new={folder_score_done} skipped_scored={folder_score_skipped} scoring_time={_fmt_secs(folder_scoring_time)} "
        f"total_time={_fmt_secs(t_folder)}"
    )


def _find_seed_folders(root: Path) -> List[Path]:
    """
    Find seed folders containing results.jsonl. Expected structure:
      root/<domain>/<topic>/<seed_folder>/results.jsonl
    """
    folders: List[Path] = []
    for p in root.rglob("results.jsonl"):
        if not p.is_file():
            continue
        seed_dir = p.parent
        # Avoid accidentally picking up a root-level results.jsonl (if any)
        try:
            rel = seed_dir.resolve().relative_to(root.resolve())
        except Exception:
            rel = seed_dir
        if len(rel.parts) < 3:
            continue
        folders.append(seed_dir)
    folders.sort()
    return folders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(_repo_root() / "result" / "dataset_refcluster"),
        help="Root folder containing many subfolders with results.jsonl + topics.json (domain/topic/seed_*/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_repo_root() / "result" / "dataset_refcluster" / "new_ranker"),
        help="Where to write aggregated JSONL outputs (default: result/dataset_refcluster/new_ranker)",
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=str(_repo_root() / "prompt" / "papermaster_prompt.json"),
    )
    parser.add_argument("--model-name", type=str, default="deepseek-v3.2-our")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=30, help="Max concurrent per-paper scoring calls per query")
    parser.add_argument("--verbose-paper-logs", action="store_true", help="Print per-paper scoring logs (will make progress bar noisy)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; still write selection + empty outputs")
    parser.add_argument("--limit-folders", type=int, default=-1)
    parser.add_argument("--limit-queries", type=int, default=-1, help="Per results.jsonl line, limit questions processed")
    parser.add_argument("--limit-papers-per-query", type=int, default=-1)
    parser.add_argument("--n-if-no-77", type=int, default=10, help="Fallback n when no paper has select_score>=77")
    parser.add_argument(
        "--topk-per-bin",
        type=int,
        default=50,
        help="For each score bucket (>=77, [61,77), [46,61), [31,46), [15,31), [0,15)), take top-K papers for scoring (default 50).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else root
    prompts_path = Path(args.prompts_path)
    seed_folders = _find_seed_folders(root)
    if args.limit_folders != -1:
        seed_folders = seed_folders[: args.limit_folders]

    # Aggregate outputs at root-level (as requested)
    checklist_out = out_dir / "deepseek_checklist_outputs.jsonl"
    scores_out = out_dir / "deepseek_pointwise_scores.jsonl"
    metrics_out = out_dir / "deepseek_metrics.jsonl"

    total_folders = len(seed_folders)
    t_all0 = time.monotonic()
    for i, seed_folder in enumerate(seed_folders, start=1):
        domain, topic = _extract_domain_topic(root, seed_folder)
        seed_arxiv_id = _seed_arxiv_id_from_folder(seed_folder)
        bar = _progress_bar(i - 1, total_folders)
        print(f"[PROGRESS] seeds {i}/{total_folders} {bar} -> {domain}/{topic}/{seed_folder.name}")
        process_folder(
            seed_folder,
            root=root,
            domain=domain,
            topic=topic,
            seed_arxiv_id=seed_arxiv_id,
            checklist_out=checklist_out,
            scores_out=scores_out,
            metrics_out=metrics_out,
            prompts_path=prompts_path,
            model_name=args.model_name,
            timeout=args.timeout,
            concurrency=args.concurrency,
            verbose_paper_logs=bool(args.verbose_paper_logs),
            dry_run=args.dry_run,
            limit_queries=args.limit_queries,
            limit_papers_per_query=args.limit_papers_per_query,
            n_if_no_77=args.n_if_no_77,
            topk_per_bin=args.topk_per_bin,
        )
        bar2 = _progress_bar(i, total_folders)
        print(f"[PROGRESS] seeds {i}/{total_folders} {bar2} done")
    print(f"[ALL] done seeds={total_folders} total_time={_fmt_secs(time.monotonic() - t_all0)} out_dir={out_dir}")


if __name__ == "__main__":
    main()


