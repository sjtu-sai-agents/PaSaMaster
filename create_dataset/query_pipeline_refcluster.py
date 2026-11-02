#!/usr/bin/env python3
# coding: utf-8
"""
Ref+Cluster Query Pipeline (new file; does NOT modify query_pipeline.py)

Goal:
1) direction -> retrieve seed papers (unchanged: reuse query_pipeline.find_papers_for_direction)
2) expand references n layers (default 1) to build a larger paper pool
3) per paper: extract core keywords with qwen-72b -> cluster by keyword overlap -> per cluster generate topic
4) per topic: generate a general complex query (A/B/C) with kimi2 based on abstracts
5) use the complex query to retrieve papers, save results

All stages are resumable via cache files under:
  {dataset_dir}/{run_id}/

Notes:
- We intentionally avoid touching existing query_pipeline.py per user request.
- We prefer SN APIs for reference expansion: tools/sn_server.py (sn_get_ref + sn_get_paper_info).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.json"

# Models (fixed per requirement)
KEYWORDS_MODEL = "qwen-72b"
QUESTION_MODEL = "deepseek-v3.2-our"

DEFAULT_DATASET_DIR = REPO_ROOT / "result" / "dataset_refcluster"
TOPIC_EN_PATH = REPO_ROOT / "create_dataset" / "data" / "topic_en.json"

# Module-level log file handles (None when not logging to file)
_log_file_handle: Optional[Any] = None  # master run log
_topic_log_file_handle: Optional[Any] = None  # current topic log


def _log_print(*args, **kwargs) -> None:
    """
    Custom print function that outputs to both console and log file.
    This ensures our pipeline logs are saved while PasaMaster's internal prints remain console-only.
    """
    # Always print to console
    print(*args, **kwargs)
    # Also write to master log file if it's open
    if _log_file_handle is not None:
        try:
            # Convert args to string and write to file
            message = " ".join(str(arg) for arg in args)
            if kwargs.get("end", "\n") != "\n":
                message += kwargs.get("end", "")
            else:
                message += "\n"
            _log_file_handle.write(message)
            _log_file_handle.flush()
            try:
                os.fsync(_log_file_handle.fileno())
            except Exception:
                pass
        except Exception:
            pass  # Silently fail if log file write fails

    # Also write to per-topic log file if it's open
    if _topic_log_file_handle is not None:
        try:
            message = " ".join(str(arg) for arg in args)
            if kwargs.get("end", "\n") != "\n":
                message += kwargs.get("end", "")
            else:
                message += "\n"
            _topic_log_file_handle.write(message)
            _topic_log_file_handle.flush()
            try:
                os.fsync(_topic_log_file_handle.fileno())
            except Exception:
                pass
        except Exception:
            pass


def _progress_bar(cur: int, total: int, width: int = 30) -> str:
    """
    Deterministic ASCII progress bar for logging (no carriage returns),
    so it renders nicely in both console output and pipeline.log.
    """
    total = max(1, int(total))
    cur = max(0, min(int(cur), total))
    filled = int(width * cur / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _log_seed_progress(idx: int, total: int, seed_paper: Dict[str, Any]) -> None:
    seed_title = str((seed_paper or {}).get("title", "")).strip()
    seed_arxiv = str((seed_paper or {}).get("arxiv_id", "")).strip()
    bar = _progress_bar(idx, total)
    pct = (100.0 * idx / max(1, total))
    _log_print(f"[PROGRESS] seeds {idx}/{total} {pct:6.2f}% {bar} | arxiv_id={seed_arxiv} | title={seed_title[:120]}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    if not path.exists():
        return []
    results = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        results.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return results


def _read_rejected_seed_ids(rejected_jsonl: Path) -> set[str]:
    """
    Load rejected seed ids from rejected_seeds.jsonl so we can skip them on next runs.
    """
    out: set[str] = set()
    for obj in _read_jsonl(rejected_jsonl):
        if not isinstance(obj, dict):
            continue
        sid = obj.get("seed_paper_id")
        if isinstance(sid, str) and sid.strip():
            out.add(sid.strip())
    return out


def _safe_rmtree_seed_dir(seed_dir: Path, run_root: Path) -> bool:
    """
    Safety guard: only delete directories that look like per-seed output dirs under run_root.
    """
    try:
        seed_dir = seed_dir.resolve()
        run_root = run_root.resolve()
    except Exception:
        return False
    if not seed_dir.exists() or not seed_dir.is_dir():
        return False
    if run_root not in seed_dir.parents:
        return False
    if not seed_dir.name.startswith("seed_"):
        return False
    try:
        shutil.rmtree(seed_dir)
        return True
    except Exception:
        return False


def _read_lines_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                out.add(s)
    except Exception:
        return set()
    return out


def _append_line(path: Path, line: str) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(line).rstrip("\n") + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _safe_json_extract(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _paper_id(p: Dict[str, Any]) -> str:
    """Stable id for de-dup: prefer arxiv_id -> doi -> normalized title."""
    arxiv_id = str(p.get("arxiv_id", "") or "").strip()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    doi = str(p.get("doi", "") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = str(p.get("title", "") or "").strip()
    return f"title:{_norm_title(title)}"


def _norm_key(s: str) -> str:
    """Normalize a key for fuzzy matching direction/category names."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def _resolve_direction_subtopics(direction: str) -> Tuple[str, Dict[str, str], List[Tuple[str, str]]]:
    """
    Resolve `direction` to a list of (topic_label, topic_desc) using `topic_en.json`.
    Returns (matched_key, subtopics_map, retrieval_queries).
    If no match is found, fallback to [(direction, "")].
    """
    topics_obj = _read_json(TOPIC_EN_PATH, {})
    if not isinstance(topics_obj, dict):
        topics_obj = {}

    dir_norm = _norm_key(direction)
    matched_key = None
    for k in topics_obj.keys():
        if _norm_key(str(k)) == dir_norm:
            matched_key = str(k)
            break
    if matched_key is None:
        for k in topics_obj.keys():
            kn = _norm_key(str(k))
            if dir_norm and (dir_norm in kn or kn in dir_norm):
                matched_key = str(k)
                break

    if matched_key is None:
        matched_key = ""

    subtopics: Dict[str, str] = {}
    if matched_key and isinstance(topics_obj.get(matched_key), dict):
        subtopics = {str(k): str(v) for k, v in topics_obj[matched_key].items()}

    retrieval_queries: List[Tuple[str, str]] = []
    if subtopics:
        for sub, desc in subtopics.items():
            retrieval_queries.append((sub, desc))
    else:
        retrieval_queries.append((direction, ""))

    return matched_key, subtopics, retrieval_queries


async def _step_seed_papers_one_topic(
    *,
    direction: str,
    topic_label: str,
    topic_desc: str,
    out_dir: Path,
    max_rounds: int,
    use_sn: bool = True,
    retrieval_concurrency: int = 10,
    keyword_concurrency: int = 50,
) -> List[Dict[str, Any]]:
    """
    Per-topic seed selection:
    retrieve -> keyword extraction -> cluster -> randomly pick 1 seed per cluster.
    All intermediate outputs are saved under `out_dir` (topic folder).
    """
    find_papers_for_direction = _import_query_pipeline_find()

    sem = asyncio.Semaphore(max(1, int(retrieval_concurrency)))

    async def _retrieve_one() -> List[Dict[str, Any]]:
        q = topic_label if not topic_desc else f"{topic_label}"
        async with sem:
            res = await find_papers_for_direction(
                q,
                search_text=None,
                max_rounds=max_rounds,
                topk_return=-1,
                output_log_dir=out_dir / "seed_retrieval" / _slug(topic_label),
                expand_layers=2,
                filter_score=0,
            )
        extra = res.get("extra", {}) if isinstance(res, dict) else {}
        out: List[Dict[str, Any]] = []
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            for p in extra["all_papers"]:
                if not isinstance(p, dict):
                    continue
                out.append(
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "description": p.get("description", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "doi": p.get("doi", ""),
                        "year": p.get("year", ""),
                        "authors": p.get("authors", ""),
                        "journal": p.get("journal", ""),
                        "source": p.get("source", ""),
                        "seed_query": q,
                        "seed_topic": topic_label,
                    }
                )
        return out

    retrieved = await _retrieve_one()

    # If SN is disabled, filter out any candidate papers that are DOI-based or from SN.
    # Requirement: filter BEFORE keyword extraction + clustering.
    if not use_sn:
        before = len(retrieved)
        filtered: List[Dict[str, Any]] = []
        removed = 0
        for p in retrieved:
            if not isinstance(p, dict):
                continue
            doi = str(p.get("doi", "") or "").strip()
            source = str(p.get("source", "") or "").strip()
            pid = _paper_id(p)
            if doi or pid.startswith("doi:") or source == "SearchFrom:sn" or "sn" in source.lower():
                removed += 1
                continue
            filtered.append(p)
        retrieved = filtered
        _log_print(
            f"[TOPIC] --no-sn enabled: filtered retrieved papers with DOI or source==SN: {before} -> {len(retrieved)} (removed={removed})"
        )
    pool: Dict[str, Dict[str, Any]] = {}
    for p in retrieved:
        if not isinstance(p, dict):
            continue
        pid = _paper_id(p)
        if pid not in pool:
            pool[pid] = dict(p)
            pool[pid]["paper_id"] = pid

    _log_print(f"[TOPIC] Retrieved papers total={len(retrieved)} dedup_pool={len(pool)} topic='{topic_label}'")
    if not pool:
        return []

    seed_paths = RunPaths(
        root=out_dir,
        seed_papers_json=out_dir / "seed_papers.json",
        ref_state_json=out_dir / "ref_expand_state.json",
        paper_pool_json=out_dir / "seed_selection_pool.json",
        paper_keywords_json=out_dir / "seed_selection_keywords.json",
        paper_keywords_done=out_dir / "seed_selection_keywords.done.txt",
        clusters_json=out_dir / "seed_selection_clusters.json",
        topics_json=out_dir / "seed_selection_topics.json",
        topic_questions_json=out_dir / "seed_selection_questions.json",
        topic_questions_done=out_dir / "seed_selection_questions.done.txt",
        topic_retrieval_jsonl=out_dir / "seed_selection_retrieval.jsonl",
        topic_retrieval_done=out_dir / "seed_selection_retrieval.done.txt",
    )

    _write_json(seed_paths.paper_pool_json, pool)
    paper_keywords = await step_keywords(
        pool, paths=seed_paths, resume=True, max_concurrent=max(1, int(keyword_concurrency))
    )
    Agent = _import_agent()
    keyword_agent = Agent(KEYWORDS_MODEL, config_path=str(CONFIG_PATH))
    clusters, topics = step_cluster(pool, paper_keywords, sim_threshold=0.15,merge_threshold=0.1, seed_paper_id="", agent=keyword_agent)

    _write_json(seed_paths.clusters_json, clusters)
    _write_json(seed_paths.topics_json, topics)

    random.seed(66)
    selected_seeds: List[Dict[str, Any]] = []
    selection_records: List[Dict[str, Any]] = []
    cluster_topic: Dict[str, str] = {}
    for _, t in (topics or {}).items():
        if isinstance(t, dict):
            cid = str(t.get("cluster_id", "") or t.get("topic_id", "")).strip()
            tp = str(t.get("topic", "")).strip()
            if cid:
                cluster_topic[cid] = tp

    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("cluster_id", "")).strip()
        pids = c.get("paper_ids", [])
        if not cid or not isinstance(pids, list) or not pids:
            continue
        chosen_pid = random.choice([str(x) for x in pids if str(x).strip()])
        chosen = pool.get(chosen_pid) or {}
        seed_rec = {
            "title": chosen.get("title", ""),
            "abstract": chosen.get("abstract", ""),
            "description": chosen.get("description", ""),
            "arxiv_id": chosen.get("arxiv_id", ""),
            "doi": chosen.get("doi", ""),
            "year": chosen.get("year", ""),
            "authors": chosen.get("authors", ""),
            "journal": chosen.get("journal", ""),
            "source": chosen.get("source", ""),
        }
        selected_seeds.append(seed_rec)
        selection_records.append(
            {
                "cluster_id": cid,
                "cluster_topic": cluster_topic.get(cid, ""),
                "cluster_size": len(pids),
                "seed_paper_id": chosen_pid,
                "seed_title": chosen.get("title", ""),
                "paper_ids": pids,
            }
        )

    selection_path = out_dir / "seed_selection.json"
    _write_json(
        selection_path,
        {
            "direction": direction,
            "topic_label": topic_label,
            "topic_desc": topic_desc,
            "pool_size": len(pool),
            "num_clusters": len(clusters),
            "selected_seeds": selection_records,
        },
    )
    _log_print(f"[TOPIC] Selected {len(selected_seeds)} seed papers from {len(clusters)} clusters -> {selection_path}")
    return selected_seeds


def _safe_dir_name(paper_id: str) -> str:
    """Convert paper_id to a safe directory name by replacing problematic characters."""
    # Replace : and / with _ to make it filesystem-safe
    safe = str(paper_id).replace(":", "_").replace("/", "_").replace("\\", "_")
    # Remove any other problematic characters
    safe = re.sub(r"[<>\"|?*]", "_", safe)
    return safe


@dataclass
class RunPaths:
    root: Path
    seed_papers_json: Path
    ref_state_json: Path
    paper_pool_json: Path
    paper_keywords_json: Path
    paper_keywords_done: Path
    clusters_json: Path
    topics_json: Path
    topic_questions_json: Path
    topic_questions_done: Path
    topic_retrieval_jsonl: Path
    topic_retrieval_done: Path


def build_run_paths(dataset_dir: Path, run_id: str) -> RunPaths:
    root = dataset_dir / run_id
    return RunPaths(
        root=root,
        seed_papers_json=root / "seed_papers.json",
        ref_state_json=root / "ref_expand_state.json",
        paper_pool_json=root / "paper_pool.json",
        paper_keywords_json=root / "paper_keywords.json",
        paper_keywords_done=root / "paper_keywords.done.txt",
        clusters_json=root / "clusters.json",
        topics_json=root / "topics.json",
        topic_questions_json=root / "topic_to_question.json",
        topic_questions_done=root / "topic_to_question.done.txt",
        topic_retrieval_jsonl=root / "topic_query_retrieval.jsonl",
        topic_retrieval_done=root / "topic_query_retrieval.done.txt",
    )


def _import_agent():
    sys.path.insert(0, str(REPO_ROOT))
    from model_vllm import Agent  # type: ignore
    return Agent


def _import_sn_tools():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from sn_server import sn_get_ref, sn_get_paper_info, sn_paper_search  # type: ignore
    return sn_get_ref, sn_get_paper_info, sn_paper_search




def _import_arxiv_meta_tools():
    """Import arxiv metadata extraction tools (title -> paper dict with abstract)."""
    sys.path.insert(0, str(REPO_ROOT))
    from utils import search_paper_by_title  # type: ignore
    return search_paper_by_title


def _import_query_pipeline_find():
    # Reuse existing retrieval logic without modifying it.
    sys.path.insert(0, str(REPO_ROOT))
    from create_dataset.query_pipeline import find_papers_for_direction  # type: ignore
    return find_papers_for_direction


async def step_seed_papers(
    direction: str,
    *,
    out_dir: Path,
    max_rounds: int,
) -> List[Dict[str, Any]]:
    """
    New seed collection:
    - Read `create_dataset/data/topic_en.json`
    - Find the category matching `direction` (fuzzy normalized)
    - Use its subtopics (e.g. AI/ML/CV/...) to retrieve papers
    - Build a combined pool -> extract keywords -> cluster
    - Randomly pick ONE paper per cluster as seed paper
    - Persist clustering+seed selection record into `out_dir/seed_selection.json`
    """
    find_papers_for_direction = _import_query_pipeline_find()

    # 1) load topics and resolve direction -> subtopics
    topics_obj = _read_json(TOPIC_EN_PATH, {})
    if not isinstance(topics_obj, dict):
        topics_obj = {}

    dir_norm = _norm_key(direction)
    matched_key = None
    for k in topics_obj.keys():
        if _norm_key(str(k)) == dir_norm:
            matched_key = str(k)
            break
    if matched_key is None:
        # fallback: contains match
        for k in topics_obj.keys():
            kn = _norm_key(str(k))
            if dir_norm and (dir_norm in kn or kn in dir_norm):
                matched_key = str(k)
                break

    if matched_key is None:
        _log_print(f"[SEEDS] No matching direction in topic_en.json for direction='{direction}'. Falling back to direction retrieval only.")
        matched_key = ""

    subtopics: Dict[str, str] = {}
    if matched_key and isinstance(topics_obj.get(matched_key), dict):
        subtopics = {str(k): str(v) for k, v in topics_obj[matched_key].items()}

    # if no subtopics matched, fallback to a single retrieval by direction
    retrieval_queries: List[Tuple[str, str]] = []
    if subtopics:
        for sub, desc in subtopics.items():
            retrieval_queries.append((sub, desc))
    else:
        retrieval_queries.append((direction, ""))

    _log_print(f"[SEEDS] direction='{direction}' matched='{matched_key}' subtopics={len(subtopics)} retrieval_queries={len(retrieval_queries)}")

    # 2) retrieve papers for each topic/subtopic (bounded concurrency)
    # Requirement: retrieval concurrency = 10
    sem = asyncio.Semaphore(10)

    async def _retrieve_one(label: str, desc: str) -> List[Dict[str, Any]]:
        q = label if not desc else f"{label}"
        async with sem:
            res = await find_papers_for_direction(
                q,
                search_text=None,
                max_rounds=max_rounds,
                topk_return=-1,
                output_log_dir=out_dir / "seed_retrieval" / _slug(label),
                expand_layers=1,
                filter_score=0,
            )
        extra = res.get("extra", {}) if isinstance(res, dict) else {}
        out: List[Dict[str, Any]] = []
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            for p in extra["all_papers"]:
                if not isinstance(p, dict):
                    continue
                out.append(
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "description": p.get("description", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "doi": p.get("doi", ""),
                        "year": p.get("year", ""),
                        "authors": p.get("authors", ""),
                        "journal": p.get("journal", ""),
                        "source": p.get("source", ""),
                        "seed_query": q,
                        "seed_topic": label,
                    }
                )
        return out

    tasks = [_retrieve_one(label, desc) for (label, desc) in retrieval_queries]
    retrieved_lists = await asyncio.gather(*tasks, return_exceptions=True)

    retrieved: List[Dict[str, Any]] = []
    for r in retrieved_lists:
        if isinstance(r, Exception):
            _log_print(f"[WARNING] seed retrieval failed: {r}")
            continue
        retrieved.extend(r)

    # 3) build pool dict keyed by stable paper_id and keep the "best" record
    pool: Dict[str, Dict[str, Any]] = {}
    for p in retrieved:
        if not isinstance(p, dict):
            continue
        pid = _paper_id(p)
        if pid not in pool:
            pool[pid] = dict(p)
            pool[pid]["paper_id"] = pid

    _log_print(f"[SEEDS] Retrieved papers total={len(retrieved)} dedup_pool={len(pool)}")
    if not pool:
        return []

    # 4) keywords + clustering (reuse existing logic)
    # Use a lightweight RunPaths rooted at out_dir for caching this stage
    seed_paths = RunPaths(
        root=out_dir,
        seed_papers_json=out_dir / "seed_papers.json",
        ref_state_json=out_dir / "ref_expand_state.json",
        paper_pool_json=out_dir / "seed_selection_pool.json",
        paper_keywords_json=out_dir / "seed_selection_keywords.json",
        paper_keywords_done=out_dir / "seed_selection_keywords.done.txt",
        clusters_json=out_dir / "seed_selection_clusters.json",
        topics_json=out_dir / "seed_selection_topics.json",
        topic_questions_json=out_dir / "seed_selection_questions.json",
        topic_questions_done=out_dir / "seed_selection_questions.done.txt",
        topic_retrieval_jsonl=out_dir / "seed_selection_retrieval.jsonl",
        topic_retrieval_done=out_dir / "seed_selection_retrieval.done.txt",
    )

    # Requirement: keyword extraction concurrency = 50
    _write_json(seed_paths.paper_pool_json, pool)
    paper_keywords = await step_keywords(pool, paths=seed_paths, resume=True, max_concurrent=100)
    Agent = _import_agent()
    keyword_agent = Agent(KEYWORDS_MODEL, config_path=str(CONFIG_PATH))
    clusters, topics = step_cluster(pool, paper_keywords, sim_threshold=0.25, seed_paper_id="", agent=keyword_agent)

    _write_json(seed_paths.clusters_json, clusters)
    _write_json(seed_paths.topics_json, topics)

    # 5) pick one seed per cluster randomly
    random.seed(66)
    selected_seeds: List[Dict[str, Any]] = []
    selection_records: List[Dict[str, Any]] = []
    # map cluster_id -> topic string (from topics map)
    cluster_topic: Dict[str, str] = {}
    for _, t in (topics or {}).items():
        if isinstance(t, dict):
            cid = str(t.get("cluster_id", "") or t.get("topic_id", "")).strip()
            topic = str(t.get("topic", "")).strip()
            if cid:
                cluster_topic[cid] = topic

    for c in clusters:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("cluster_id", "")).strip()
        pids = c.get("paper_ids", [])
        if not cid or not isinstance(pids, list) or not pids:
            continue
        chosen_pid = random.choice([str(x) for x in pids if str(x).strip()])
        chosen = pool.get(chosen_pid) or {}
        seed_rec = {
            "title": chosen.get("title", ""),
            "abstract": chosen.get("abstract", ""),
            "description": chosen.get("description", ""),
            "arxiv_id": chosen.get("arxiv_id", ""),
            "doi": chosen.get("doi", ""),
            "year": chosen.get("year", ""),
            "authors": chosen.get("authors", ""),
            "journal": chosen.get("journal", ""),
            "source": chosen.get("source", ""),
        }
        selected_seeds.append(seed_rec)
        selection_records.append(
            {
                "cluster_id": cid,
                "cluster_topic": cluster_topic.get(cid, ""),
                "cluster_size": len(pids),
                "seed_paper_id": chosen_pid,
                "seed_title": chosen.get("title", ""),
                "paper_ids": pids,
            }
        )

    selection_path = out_dir / "seed_selection.json"
    _write_json(
        selection_path,
        {
            "direction": direction,
            "matched_key": matched_key,
            "subtopics": subtopics,
            "pool_size": len(pool),
            "num_clusters": len(clusters),
            "selected_seeds": selection_records,
        },
    )
    _log_print(f"[SEEDS] Selected {len(selected_seeds)} seed papers from {len(clusters)} clusters -> {selection_path}")
    return selected_seeds


def step_expand_refs_single_seed(
    seed_paper: Dict[str, Any],
    *,
    n_layers: int,
    seed_paper_id: str,
    state_file: Path,
    pool_file: Path = None,
    sn_batch: int = 20,
    sn_fuzzy_threshold: int = 80,
    resume: bool = True,
    use_sn: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a paper pool for a single seed through SN reference expansion.
    For each layer, fetch references through SN, then resolve their titles
    against the arXiv metadata index with SN metadata as a fallback.
    Papers without abstracts are excluded from the pool.
    """
    sn_get_ref = sn_get_paper_info = sn_paper_search = None
    if use_sn:
        sn_get_ref, sn_get_paper_info, sn_paper_search = _import_sn_tools()
    search_paper_by_title = _import_arxiv_meta_tools()

    # Load existing pool if resuming
    pool: Dict[str, Dict[str, Any]] = {}
    if resume and pool_file and pool_file.exists():
        pool = _read_json(pool_file, {})
        if not isinstance(pool, dict):
            pool = {}

    # ----------------------------
    # State / frontier management
    # ----------------------------
    state = _read_json(state_file, {})
    if not isinstance(state, dict) or not resume:
        state = {"layer": 0, "frontier_pids": [], "seen_titles_norm": []}

    # init seed paper in pool if not already present
    seed_pid = ""
    if seed_paper and isinstance(seed_paper, dict):
        seed_pid = _paper_id(seed_paper)
        if seed_pid not in pool:
            rec = dict(seed_paper)
            rec["paper_id"] = seed_pid
            rec["depth"] = 0
            rec["seed_paper_id"] = seed_paper_id
            pool[seed_pid] = rec

    # Back-compat: older state might store frontier_titles
    if "frontier_pids" not in state and isinstance(state.get("frontier_titles", None), list):
        title_list = [str(x).strip() for x in (state.get("frontier_titles") or []) if str(x).strip()]
        mapped: List[str] = []
        for t in title_list:
            for pid, p in pool.items():
                if str(p.get("title", "")).strip() == t:
                    mapped.append(pid)
                    break
        state["frontier_pids"] = mapped

    # Initialize frontier for a fresh run
    if state.get("layer", 0) == 0 and not state.get("frontier_pids"):
        if seed_pid:
            state["frontier_pids"] = [seed_pid]
        seed_title = str(seed_paper.get("title", "")).strip() if seed_paper else ""
        if seed_title:
            state["seen_titles_norm"] = sorted({_norm_title(seed_title)})

    cur_layer = int(state.get("layer", 0) or 0)
    frontier_pids: List[str] = list(state.get("frontier_pids", []) or [])
    seen_norm: set[str] = set(state.get("seen_titles_norm", []) or [])

    # Track if we found any ref titles (only create state file if we did)
    found_ref_titles = False

    for layer in range(cur_layer + 1, n_layers + 1):
        if not frontier_pids:
            # Only write state file if we previously found ref titles
            if found_ref_titles:
                state["layer"] = layer
                _write_json(state_file, state)
            break

        _log_print(
            f"[EXPAND] Layer {layer}: frontier={len(frontier_pids)} pool_size={len(pool)}"
        )
        raw_ref_titles: List[str] = []
        to_sn_titles = [
            str(pool[fpid].get("title", "")).strip()
            for fpid in frontier_pids
            if str(pool.get(fpid, {}).get("title", "")).strip()
        ]

        if use_sn and to_sn_titles and sn_get_ref is not None:
            sn_refs_list = sn_get_ref(to_sn_titles, batch_size=sn_batch, threshold=sn_fuzzy_threshold)
            for refs in sn_refs_list:
                if not isinstance(refs, dict):
                    continue
                for _, t in refs.items():
                    tt = str(t or "").replace("\n", " ").strip()
                    if tt:
                        raw_ref_titles.append(tt)
        # Dedup titles globally
        new_titles: List[str] = []
        for t in raw_ref_titles:
            nt = _norm_title(t)
            if not nt or nt in seen_norm:
                continue
            seen_norm.add(nt)
            new_titles.append(t)

        # Mark that we found ref titles
        if new_titles:
            found_ref_titles = True

        _log_print(
            f"[EXPAND] Layer {layer}: sn_reference_papers={(len(to_sn_titles) if use_sn else 0)}, new_ref_titles={len(new_titles)}"
        )
        # if new_titles:
        #     # Print ref titles directly (truncate if too many)
        #     show_n = 50
        #     _log_print(f"[EXPAND] Layer {layer}: ref titles (showing first {min(show_n, len(new_titles))}):")
        #     for t in new_titles[:show_n]:
        #         _log_print(f"  - {t}")
        #     if len(new_titles) > show_n:
        #         _log_print(f"  ... and {len(new_titles) - show_n} more (see layer_{layer}_new_titles.txt)")

        # Persist titles for debugging (full list)
        if new_titles:
            titles_txt = state_file.parent / f"layer_{layer}_new_titles.txt"
            _ensure_dir(titles_txt.parent)
            titles_txt.write_text("\n".join(new_titles) + "\n", encoding="utf-8")
        # ---------------------------------------------
        # Build next layer pool:
        # - Prefer arxiv (title -> arxiv paper dict with abstract)
        # - Fallback to SN (title -> paper dict with abstract)
        # - IMPORTANT: papers WITHOUT abstract are NOT added to pool
        # ---------------------------------------------
        next_frontier_pids: List[str] = []
        added_titles: List[str] = []
        added_count = 0
        arxiv_added = 0
        sn_added = 0
        no_abs_skipped = 0
        remaining_titles: List[str] = []
        arxiv_lookup_hit = 0
        arxiv_lookup_miss = 0
        for t in new_titles:
            try:
                arxiv_p = search_paper_by_title(t)
            except Exception:
                arxiv_p = None

            abs_text = str((arxiv_p or {}).get("abstract", "")).strip() if isinstance(arxiv_p, dict) else ""
            if isinstance(arxiv_p, dict) and abs_text:
                arxiv_lookup_hit += 1
                rec = {
                    "title": str(arxiv_p.get("title", "")).strip() or t,
                    "abstract": abs_text,
                    "arxiv_id": str(arxiv_p.get("arxiv_id", "")).strip(),
                    "doi": "",
                    "year": str(arxiv_p.get("year", "")).strip(),
                    "authors": arxiv_p.get("authors", ""),
                    "journal": str(arxiv_p.get("journal", "")).strip(),
                    "source": str(arxiv_p.get("source", "SearchFrom:arxivdb")).strip(),
                }
                pid = _paper_id(rec)
                if pid not in pool:
                    rec["paper_id"] = pid
                    rec["depth"] = layer
                    rec["seed_paper_id"] = seed_paper_id
                    pool[pid] = rec
                    next_frontier_pids.append(pid)
                    added_titles.append(rec["title"])
                    added_count += 1
                    arxiv_added += 1
            else:
                arxiv_lookup_miss += 1
                remaining_titles.append(t)

        if remaining_titles and use_sn and sn_get_paper_info is not None:
            _log_print(
                f"[EXPAND] Layer {layer}: abstract lookup via arxiv done: hit={arxiv_lookup_hit}, miss={arxiv_lookup_miss}. Fallback SN for {len(remaining_titles)} titles"
            )
            try:
                sn_infos = sn_get_paper_info(remaining_titles, batch_size=sn_batch)
            except Exception:
                sn_infos = []
            if not isinstance(sn_infos, list):
                sn_infos = []
            _log_print(f"[EXPAND] Layer {layer}: SN get_paper_info returned {len(sn_infos)} records")

            # If SN returns nothing, fallback to SN keyword search for a few titles (best-effort)
            if len(sn_infos) == 0 and len(remaining_titles) > 0 and sn_paper_search is not None:
                _log_print(f"[WARNING] Layer {layer}: SN get_paper_info returned 0; trying sn_paper_search for up to 20 titles")
                for t in remaining_titles[:20]:
                    try:
                        cand = sn_paper_search(t, num=1)
                    except Exception:
                        cand = []
                    if isinstance(cand, list) and cand:
                        sn_infos.append(cand[0] if isinstance(cand[0], dict) else {})

            for info in sn_infos:
                if not isinstance(info, dict):
                    continue
                title = str(info.get("title", "")).strip()
                abs_text = str(info.get("abstract", "")).strip()
                if not title or not abs_text:
                    no_abs_skipped += 1
                    continue
                rec = {
                    "title": title,
                    "abstract": abs_text,
                    "doi": str(info.get("doi", "")).strip(),
                    "year": str(info.get("year", "")).strip(),
                    "authors": info.get("authors", ""),
                    "journal": str(info.get("journal", "")).strip(),
                    "source": str(info.get("source", "SearchFrom:sn")).strip(),
                }
                pid = _paper_id(rec)
                if pid in pool:
                    continue
                rec["paper_id"] = pid
                rec["depth"] = layer
                rec["seed_paper_id"] = seed_paper_id
                pool[pid] = rec
                next_frontier_pids.append(pid)
                added_titles.append(rec["title"])
                added_count += 1
                sn_added += 1

        # NOTE: titles that neither arxiv nor SN returned an abstract for are skipped by design.
        if use_sn:
            no_abs_skipped += max(0, len(remaining_titles) - sn_added)
        else:
            # When SN disabled, all arxiv misses are effectively skipped (expected).
            no_abs_skipped += len(remaining_titles)

        # Print per-layer stats and titles (as requested)
        _log_print(
            f"[EXPAND] Layer {layer}: added_papers={added_count} (arxiv={arxiv_added}, sn={sn_added}, no_abs_skipped={no_abs_skipped})"
        )
        _log_print(f"[EXPAND] Layer {layer}: pool_size={len(pool)} next_frontier={len(next_frontier_pids)}")

        frontier_pids = next_frontier_pids
        state = {"layer": layer, "frontier_pids": frontier_pids, "seen_titles_norm": sorted(seen_norm)}
        # Only write state file if we found ref titles in this layer or previous layers
        if found_ref_titles:
            _write_json(state_file, state)

        if pool_file:
            _write_json(pool_file, pool)

    return pool


def _extract_keywords_one(p: Dict[str, Any], agent) -> Optional[List[str]]:
    title = str(p.get("title", "")).strip()
    abstract = str(p.get("abstract", "")).strip()
    if not title and not abstract:
        return None
    prompt = (
        "You are extracting a concise research direction description for a paper.\n"
        "Return STRICT JSON only:\n"
        "{\"keywords\": [\"...\"]}\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON.\n"
        "- Generate ONLY ONE concise phrase or short sentence (3-10 words) that describes the research direction.\n"
        "- Be as concise and precise as possible.\n"
        "- Use standardized academic terms that are likely shared across related papers.\n"
        "- Focus on the core research theme/direction, not specific methods or results.\n"
        "- Avoid named datasets/models/systems, author names, venues, years, locations, and other metadata.\n"
        "- Avoid generic terms like \"research\", \"study\", \"paper\", \"method\", \"approach\".\n"
        "- Examples: \"foundation language model training\", \"medical image analysis\", \"reinforcement learning for robotics\", \"neural architecture search\"\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        "JSON:\n"
    )
    out = agent.infer(prompt, use_cot=False)
    obj = _safe_json_extract(out or "")
    if not isinstance(obj, dict):
        return None
    kws = obj.get("keywords", None)
    if not isinstance(kws, list):
        return None
    # Extract the first (and should be only) keyword/phrase
    if not kws:
        return None
    cleaned = []
    for k in kws:
        s = str(k or "").strip().strip(",;.")
        if not s:
            continue
        cleaned.append(s)
        # Only take the first valid keyword/phrase
        break
    return cleaned[:1] if cleaned else None


async def step_keywords(
    pool: Dict[str, Dict[str, Any]],
    *,
    paths: RunPaths,
    resume: bool = True,
    max_concurrent: int = 100,
) -> Dict[str, Any]:
    """
    Extract keywords for all papers in pool concurrently.
    Uses semaphore to limit concurrent LLM calls to max_concurrent (default 100).
    """
    Agent = _import_agent()
    agent = Agent(KEYWORDS_MODEL, config_path=str(CONFIG_PATH))

    out_map = _read_json(paths.paper_keywords_json, {})
    if not isinstance(out_map, dict):
        out_map = {}
    done = _read_lines_set(paths.paper_keywords_done) if resume else set()
    _log_print(f"[KEYWORDS] Done: {len(done)} papers, left: {len(pool) - len(done)} papers")
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))
    
    # Shared state for incremental saving
    save_lock = asyncio.Lock()
    save_batch_size = 10  # Save every N completed tasks
    completed_since_last_save = [0]  # Use list to allow modification in nested function

    async def _extract_keywords_one_async(pid: str, p: Dict[str, Any]) -> Tuple[str, Optional[List[str]]]:
        """Async wrapper for keyword extraction with semaphore and incremental saving."""
        async with semaphore:
            # Run synchronous agent.infer in thread pool
            loop = asyncio.get_event_loop()
            kws = await loop.run_in_executor(None, _extract_keywords_one, p, agent)
            
            # Incremental save: save immediately after each successful extraction
            if kws is not None:
                async with save_lock:
                    out_map[pid] = {
                        "paper_id": pid,
                        "title": pool[pid].get("title", ""),
                        "keywords": kws,
                        "model": KEYWORDS_MODEL,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    _append_line(paths.paper_keywords_done, pid)
                    completed_since_last_save[0] += 1
                    
                    # Incremental save: save every N tasks to support resume
                    if completed_since_last_save[0] >= save_batch_size:
                        _write_json(paths.paper_keywords_json, out_map)
                        # _log_print(f"[KEYWORDS] Progress: {completed_since_last_save[0]} papers completed (saved)")
                        completed_since_last_save[0] = 0
            
            return (pid, kws)

    # Collect tasks for papers not yet done
    tasks = []
    pending_pids = []
    for pid, p in pool.items():
        if pid in done:
            continue
        tasks.append(_extract_keywords_one_async(pid, p))
        pending_pids.append(pid)

    if not tasks:
        return out_map

    _log_print(f"[KEYWORDS] Extracting keywords for {len(tasks)} papers (concurrent={max_concurrent}, resume supported)...")
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process any remaining results (handle exceptions)
    updated_count = 0
    for i, result in enumerate(results):
        pid = pending_pids[i]
        if isinstance(result, Exception):
            _log_print(f"[WARNING] Failed to extract keywords for {pid}: {result}")
            continue
        
        pid_result, kws = result
        if kws is None:
            # do not mark done; allow retry (already handled in async function)
            continue
        updated_count += 1
    
    # Final save for any remaining updates
    async with save_lock:
        if completed_since_last_save[0] > 0:
            _write_json(paths.paper_keywords_json, out_map)
            _log_print(f"[KEYWORDS] Final save: {completed_since_last_save[0]} remaining updates")
    
    _log_print(f"[KEYWORDS] Completed: {updated_count} papers processed")
    
    return out_map


def _kw_set(kws: List[str]) -> set[str]:
    """
    Convert keyword phrases into a normalized token set for clustering.
    We include:
    - the normalized full phrase
    - unigram tokens
    - bigram tokens
    This is more robust than phrase-level exact matching (helps clustering across paraphrases).
    """
    STOP = {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "via", "from", "by",
        "based", "using", "use", "study", "paper", "research", "analysis", "approach", "method",
        "proceedings", "conference", "workshop", "symposium", "volume", "issue", "table", "contents",
        "review", "introduction", "overview", "editorial",
    }
    out: set[str] = set()
    for k in kws:
        phrase = re.sub(r"[^a-z0-9]+", " ", (k or "").lower()).strip()
        if not phrase:
            continue
        # keep phrase token (helps when exact phrase matches)
        out.add(phrase)
        toks = [t for t in phrase.split() if len(t) >= 3 and t not in STOP]
        for t in toks:
            out.add(t)
        # bigrams
        for i in range(len(toks) - 1):
            out.add(f"{toks[i]}_{toks[i+1]}")
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def step_cluster(
    pool: Dict[str, Dict[str, Any]],
    paper_keywords: Dict[str, Any],
    *,
    sim_threshold: float = 0.25,
    merge_threshold: Optional[float] = None,
    seed_paper_id: str = "",
    agent=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Greedy clustering by Jaccard similarity on keyword phrase sets.
    Generate core topic (one sentence) for each cluster.
    Returns (clusters, topics_map).
    """
    items: List[Tuple[str, set[str], List[str]]] = []
    for pid, kwobj in paper_keywords.items():
        if not isinstance(kwobj, dict):
            continue
        kws = kwobj.get("keywords", [])
        if not isinstance(kws, list) or not kws:
            continue
        items.append((pid, _kw_set([str(x) for x in kws]), [str(x) for x in kws]))

    # Greedy clustering (more robust than union-rep):
    # assign by max similarity to ANY member in cluster (helps avoid union dilution).
    clusters: List[Dict[str, Any]] = []
    member_sets: List[List[set[str]]] = []
    for pid, sset, kws in items:
        assigned = -1
        best = 0.0
        for i, ms in enumerate(member_sets):
            # max-to-member
            sc = 0.0
            for mset in ms:
                sc = max(sc, _jaccard(sset, mset))
                if sc >= sim_threshold:
                    break
            if sc >= sim_threshold and sc > best:
                best = sc
                assigned = i
        if assigned < 0:
            clusters.append({"cluster_id": f"c{len(clusters):04d}", "paper_ids": [pid]})
            member_sets.append([set(sset)])
        else:
            clusters[assigned]["paper_ids"].append(pid)
            member_sets[assigned].append(set(sset))

    # Optional merge pass: merge clusters whose cores are similar (reduces singleton fragmentation).
    if merge_threshold is None:
        # slightly lower than sim_threshold to allow bridging
        merge_threshold = max(0.05, min(sim_threshold * 0.8, sim_threshold - 0.02))

    def _cluster_core(ms: List[set[str]], topk: int = 40) -> set[str]:
        freq: Dict[str, int] = {}
        for s in ms:
            for tok in s:
                freq[tok] = freq.get(tok, 0) + 1
        # core = top tokens by frequency (ignores long tail noise)
        return set([t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:topk]])

    merged = True
    while merged and len(clusters) > 1:
        merged = False
        i = 0
        while i < len(clusters):
            j = i + 1
            core_i = _cluster_core(member_sets[i])
            while j < len(clusters):
                core_j = _cluster_core(member_sets[j])
                if _jaccard(core_i, core_j) >= (merge_threshold or 0.0):
                    # merge j into i
                    clusters[i]["paper_ids"].extend(clusters[j]["paper_ids"])
                    member_sets[i].extend(member_sets[j])
                    del clusters[j]
                    del member_sets[j]
                    merged = True
                    continue
                j += 1
            i += 1
    
    # Enrich clusters with paper titles
    for c in clusters:
        pids = c.get("paper_ids", [])
        paper_titles = []
        for pid in pids:
            p = pool.get(pid, {}) if isinstance(pool, dict) else {}
            title = str(p.get("title", "")).strip()
            if title:
                paper_titles.append(title)
        c["paper_titles"] = paper_titles

    # Generate core topic (one sentence) for each cluster
    topics: Dict[str, Any] = {}
    for c in clusters:
        pids = c.get("paper_ids", [])
        freq: Dict[str, int] = {}
        cluster_abstracts = []
        cluster_keywords_list = []
        
        for pid in pids:
            kwobj = paper_keywords.get(pid, {}) if isinstance(paper_keywords, dict) else {}
            kws = kwobj.get("keywords", []) if isinstance(kwobj, dict) else []
            if isinstance(kws, list):
                cluster_keywords_list.extend([str(k).strip() for k in kws if str(k).strip()])
                for k in kws:
                    kk = str(k).strip()
                    if kk:
                        freq[kk] = freq.get(kk, 0) + 1
                
            p = pool.get(pid, {}) if isinstance(pool, dict) else {}
            abstract = str(p.get("abstract", "")).strip()
            if abstract:
                cluster_abstracts.append(abstract[:500])  # Limit abstract length
        
        top_kws = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:6]
        top_kw_list = [k for k, _ in top_kws]
        
        # Generate core topic (one sentence) using LLM if agent provided
        core_topic = " / ".join(top_kw_list[:3]) if top_kw_list else c["cluster_id"]
        if agent and cluster_abstracts and cluster_keywords_list:
            topic_prompt = (
                "Generate a concise one-sentence core topic that summarizes the research theme of these papers.\n"
                "The topic should be broad enough to cover all papers in the cluster.\n"
                "Output ONLY the topic sentence, no extra text.\n\n"
                f"Top keywords: {', '.join(top_kw_list[:10])}\n"
                f"Sample abstracts:\n" + "\n".join(cluster_abstracts[:3]) + "\n\n"
                "Core topic:"
            )
            try:
                topic_out = agent.infer(topic_prompt, use_cot=False)
                if topic_out and isinstance(topic_out, str):
                    core_topic = topic_out.strip().strip('"').strip("'")
                    if not core_topic or len(core_topic) > 200:
                        core_topic = " / ".join(top_kw_list[:3]) if top_kw_list else c["cluster_id"]
            except Exception:
                pass
        
        topic_key = f"{core_topic} ({c['cluster_id']})"
        topics[topic_key] = {
            "topic": core_topic,
            "topic_id": c["cluster_id"],
            "cluster_id": c["cluster_id"],
            "top_keywords": top_kw_list,
            "paper_ids": pids,
            "seed_paper_id": seed_paper_id,
        }

    return clusters, topics


def _pick_support_papers(
    topic_obj: Dict[str, Any],
    pool: Dict[str, Dict[str, Any]],
    paper_keywords: Dict[str, Any],
    *,
    max_papers: int = 8,
) -> List[Dict[str, Any]]:
    pids = topic_obj.get("paper_ids", []) if isinstance(topic_obj, dict) else []
    if not isinstance(pids, list):
        return []
    out: List[Dict[str, Any]] = []
    for pid in pids[:max_papers]:
        p = pool.get(pid, {}) if isinstance(pool, dict) else {}
        kwobj = paper_keywords.get(pid, {}) if isinstance(paper_keywords, dict) else {}
        out.append(
            {
                "paper_id": pid,
                "title": p.get("title", ""),
                "keywords": kwobj.get("keywords", []) if isinstance(kwobj, dict) else [],
                "abstract": p.get("abstract", ""),
            }
        )
    return out


def _generate_abc_questions(
    *,
    direction: str,
    topic: str,
    support: List[Dict[str, Any]],
    agent,
    all_papers_in_cluster: List[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    # build a compact abstract context from ALL papers in cluster
    if all_papers_in_cluster is None:
        all_papers_in_cluster = support
    
    abstract_lines: List[str] = []
    keywords_all = []
    max_paper = 10
    for i, p in enumerate(all_papers_in_cluster[:max_paper]):
        t = str(p.get("title", "")).strip()
        a = str(p.get("abstract", "")).strip()
        kws = p.get("keywords", [])
        if not isinstance(kws, list):
            kws = []
        kw_text = ", ".join([str(x) for x in kws[:8] if str(x).strip()])
        keywords_all.extend([str(x).strip() for x in kws if str(x).strip()])
        abstract_lines.append(f"{i+1}. Title: {t}\n   Keywords: {kw_text}\n   Abstract: {a[:300]}")  # Limit abstract length

    ctx = "\n".join(abstract_lines)
    unique_keywords = list(set(keywords_all))[:20]  # Get unique keywords
    _log_print(f"unique_keywords: {unique_keywords}")
    prompt = (
        "You are an expert literature searcher.\n"
        "Your task: Given a research direction, a core TOPIC, and ALL papers in a cluster, generate THREE retrieval queries (A/B/C) plus checklists, so that EVERY SINGLE paper in the cluster is HIGHLY relevant (score = 5) to EACH query.\n\n"
        "CRITICAL HARD CONSTRAINT:\n"
        "- For every generated query, EVERY paper in the cluster must fully satisfy ALL checklist items (relevance score = 5).\n"
        "- If any paper fails any checklist item, you MUST revise the query and/or the checklist until all papers score 5.\n\n"

        "OPTIMIZATION GOAL (under the hard constraint above):\n"
        "- Pack in as many SHARED, MEANINGFUL requirements as possible into each query, but ONLY if they are truly satisfied by ALL papers.\n"
        "- Do NOT add any condition that is not clearly supported by EVERY paper.\n"
        "- Do NOT omit important shared aspects just to keep the query overly broad.\n\n"
        "- The checklist must be a faithful, non-redundant decomposition of the query: together, all checklist items must cover all substantive constraints expressed in the query, and no checklist item may introduce a stricter or looser condition than the query.\n"
        "- Preserve logical relationships from the query (such as 'and', 'or', 'must', 'rather than', 'for practical applications') exactly in the checklist; do not change 'and' to 'or', do not drop qualifiers, and do not broaden or weaken constraints.\n"
        "- If you decide that some constraint should be generalized or specialized, you must modify BOTH the query and the checklist so that they remain semantically equivalent.\n"

        "Query Design Principles:\n"
        "- Make queries BROAD and COMPREHENSIVE at the cluster level: they should capture the common research direction shared by ALL papers.\n"
        "- Use GENERAL academic terms and concepts, not very specific datasets, models, or system names, unless all papers explicitly share them.\n"
        "- Each query must contain only SHARED requirements that are supported by EVERY paper in the cluster.\n"
        "- Avoid any constraint (e.g., specific benchmark, dataset, model name, architecture, domain, language, year, venue) unless ALL papers clearly satisfy it.\n"
        "- Focus on ESSENTIAL shared research themes/methods/tasks/questions across ALL papers.\n"
        "- The \"query\" field must be plain English, 1–3 sentences, no JSON/code, and must NOT contain curly braces { } or square brackets [ ].\n\n"

        "Three variants (you MUST produce all three):\n"
        "- Variant A (style = \"A\"): Multi-constraint, explicit, clear query covering the core shared research theme; as detailed as possible while still satisfied by all papers.\n"
        "- Variant B (style = \"B\"): Scenario / application-driven query that describes a practical use-case or pain point, but still exactly matches the same shared theme.\n"
        "- Variant C (style = \"C\"): Alternative framing or broader context query (e.g., methodological, theoretical, or comparative perspective), but still covering the SAME shared core theme.\n\n"

        "Reference examples (for style ONLY; do not copy entities unless supported by the abstract/direction):\n"
        "Example A (multi-constraint, explicit, unambiguous):\n"
        "Please help me find papers on context management for LLM Agent when they are using tools. The methods must test on browsecomp benchmark.\n"
        "Example B (scenario / pain-point driven):\n"
        "I'm building a single agent system with LLMs to use tools, but I find that the tool response is too long and it's hard to be used by the LLM Agent. Is there any paper that can help me to solve this problem?\n"
        "Example C (exclusion / negative constraint):\n"
        "Could you help check whether there are any papers that do context management for LLM Agent when they are using tools? It must be context management that solves the problem of long contexts caused by LLM agent calling tools, rather than simply reading long text and managing the context.\n\n"

        "OUTPUT FORMAT:\n"
        "- Output ONLY valid JSON, no markdown, no code blocks (no ```json or ```).\n"
        "- JSON structure must be:\n"
        "{\n"
        '  \"variants\": [\n'
        '    {\"style\":\"A\",\"query\":\"...\",\"checklist\":[\"...\",\"...\"]},\n'
        '    {\"style\":\"B\",\"query\":\"...\",\"checklist\":[\"...\",\"...\"]},\n'
        '    {\"style\":\"C\",\"query\":\"...\",\"checklist\":[\"...\",\"...\"]}\n'
        "  ],\n"
        '  \"reason\": \"The reason why each paper is highly relevant to the query, give the relevant score and reason for each paper according to the checklist and query, the score between 1-5(integer)\"\n'
        "}\n"
        "- The \"query\" text must NOT contain curly braces { } or square brackets [ ].\n\n"

        "Checklist construction and use:\n"
        "- From each query, decompose it into 2–5 concise 'checkpoints' (constraints) in the \"checklist\" field.\n"
        "- Each checkpoint should capture exactly ONE specific requirement that a paper must satisfy.\n"
        "- Do NOT duplicate or overlap requirements across checklist items; each item should describe a different aspect.\n"
        "- Combine concepts only when they jointly define a single, indivisible requirement.\n"
        "- Keep the MAIN TASK or PROBLEM TYPE as its own separate checkpoint when appropriate.\n"
        "- The checklist should be detailed enough to judge relevance precisely, but only include aspects that ALL papers share.\n"
        "- You will use this checklist to internally check whether each paper satisfies ALL constraints (see self-check step below).\n\n"

        "Step-by-step for checklist (illustrative example):\n"
        "- From the question, decompose it into 2–5 'checkpoints' (constraints) that will be used later to judge whether a paper is relevant. Each checkpoint should capture exactly one specific requirement that a paper must satisfy. Step:1. Keyword extraction 2.nChecklist construction\n"
        "- Combine concepts when they jointly define a single requirement.\n"
        "- Keep task requirements as a separate checkpoint. Note: Each keyword should be highlighted in the checklist. For example: My query is \"I'm looking for papers on using multimodal LLM agents to diagnose medical images, rather than traditional CNN-based medical image classifiers.\n\n"
        "Keyword extraction:\n"
        "1. multimodal LLM agents\n"
        "2. diagnose medical images\n"
        "3. CNN-based medical image classifiers\n\n"
        "Scoring rule (strict): 5 = The status of all critical checkpoints are Satisfied; 4 = The status of majority (>50%) of critical checkpoints Satisfied; 3 = The status of some critical checkpoints are Satisfied, but not a majority; 2 = no critical checkpoints Satisfied, but the paper is in the same domain; 1 = completely unrelated (domain mismatch).\n\n"

        "Self-check procedure (INTERNAL reasoning, do NOT output these steps explicitly):\n"
        "1. For Variant A:\n"
        "   - Draft a candidate query and its 2–5-item checklist.\n"
        "   - For EACH paper in the cluster, internally check each checklist item and assign a 1–5 relevance score using the scoring rule.\n"
        "   - If ANY paper scores less than 5, revise the query and/or checklist (usually by relaxing or generalizing requirements, or by removing non-shared details) and re-check.\n"
        "   - Repeat until EVERY paper scores 5 on this query.\n"
        "2. Do the same process independently for Variant B and Variant C.\n"
        "3. While revising, try to MAXIMIZE the number of distinct, non-overlapping checklist items that are still satisfied by ALL papers.\n"
        "4. Only once all three variants pass (all papers score 5 for each), output the final JSON.\n"
        "5. Do NOT output intermediate drafts or the self-check process; only output the final JSON.\n\n"

        "Use shared keywords and concepts when possible: "
        + ", ".join(unique_keywords[:15]) + "\n\n"

        f"Research direction: {direction}\n"
        f"Core topic: {topic}\n\n"
        f"ALL papers in cluster (total: {len(all_papers_in_cluster)}):\n"
        f"{ctx}\n\n"

        "Your final task:\n"
        "- Based on the research direction, core topic, and ALL papers above, generate THREE queries (A/B/C) that are as informative and constrained as possible WHILE remaining fully satisfied by EVERY paper.\n"
        "- Ensure each query is general and inclusive at the cluster level, captures the core shared research theme, and avoids any constraint not supported by all papers.\n"
        "- For each query, also output a 2–5 item checklist as described.\n"
        "- Then, in the \"reason\" field, explain briefly why every paper in the cluster scores 5 for EACH query, following the scoring rule.\n"
        "JSON:\n"
    )
    out = agent.infer(prompt, use_cot=False)

    if not out or not isinstance(out, str):
        _log_print(f"[WARNING] LLM returned empty or invalid output for topic: {topic}")
        return None
    
    obj = _safe_json_extract(out or "")

    if not isinstance(obj, dict):
        _log_print(f"[WARNING] Failed to parse JSON from LLM output for topic: {topic}")
        _log_print(f"[DEBUG] LLM output (first 500 chars): {str(out)[:500]}")
        return None
    
    variants = obj.get("variants", [])
    if not isinstance(variants, list) or len(variants) == 0:
        _log_print(f"[WARNING] No valid variants found in LLM output for topic: {topic}")
        _log_print(f"[DEBUG] Parsed obj keys: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}")
        _log_print(f"[DEBUG] Variants type: {type(variants)}, value: {variants}")
        return None

    # Validate and normalize variants
    normalized_variants = []
    invalid_variants = []
    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            invalid_variants.append(f"variant[{i}]: not a dict (type={type(v).__name__})")
            continue
        style = str(v.get("style", "")).strip().upper()
        query = str(v.get("query", "")).strip()
        checklist = v.get("checklist", [])
        
        # Check why variant is invalid
        if style not in ["A", "B", "C"]:
            invalid_variants.append(f"variant[{i}]: invalid style='{style}' (expected A/B/C)")
        elif not query:
            invalid_variants.append(f"variant[{i}]: empty query")
        elif not isinstance(checklist, list):
            invalid_variants.append(f"variant[{i}]: checklist is not a list (type={type(checklist).__name__})")
        else:
            normalized_variants.append({"style": style, "query": query, "checklist": checklist})
    
    if len(normalized_variants) == 0:
        _log_print(f"[WARNING] No valid variants after normalization for topic: {topic}")
        _log_print(f"[DEBUG] Total variants received: {len(variants)}")
        if invalid_variants:
            _log_print(f"[DEBUG] Invalid variant reasons:")
            for reason in invalid_variants:
                _log_print(f"  - {reason}")
        return None
    out = {"variants": normalized_variants,"reason": obj.get("reason", "")}
    return out


def step_generate_questions(
    *,
    direction: str,
    topics: Dict[str, Any],
    pool: Dict[str, Dict[str, Any]],
    paper_keywords: Dict[str, Any],
    paths: RunPaths,
    resume: bool = True,
) -> Dict[str, Any]:
    Agent = _import_agent()
    agent = Agent(QUESTION_MODEL, config_path=str(CONFIG_PATH))

    out_map = _read_json(paths.topic_questions_json, {})
    if not isinstance(out_map, dict):
        out_map = {}
    done = _read_lines_set(paths.topic_questions_done) if resume else set()

    for topic_key, topic_obj in topics.items():
        topic_id = str(topic_obj.get("topic_id", "")).strip() if isinstance(topic_obj, dict) else ""
        if not topic_id:
            continue
        if topic_id in done:
            continue
        support = _pick_support_papers(topic_obj, pool, paper_keywords, max_papers=8)
        topic_label = str(topic_obj.get("topic", topic_key)).strip()
        gen = _generate_abc_questions(direction=direction, topic=topic_label, support=support, agent=agent)
        if gen is None:
            # do not mark done; allow retry
            continue
        out_map[topic_key] = {
            "topic": topic_label,
            "topic_id": topic_id,
            "questions": gen.get("variants", []),
            "reason": gen.get("reason", ""),
            "support_papers": support,
            "model": QUESTION_MODEL,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(paths.topic_questions_json, out_map)
        _append_line(paths.topic_questions_done, topic_id)
        done.add(topic_id)
    return out_map


async def step_retrieve_by_questions(
    *,
    topic_questions: Dict[str, Any],
    out_dir: Path,
    max_rounds: int,
    paths: RunPaths,
    resume: bool = True,
    max_concurrent: int = 10,
) -> None:
    find_papers_for_direction = _import_query_pipeline_find()
    done = _read_lines_set(paths.topic_retrieval_done) if resume else set()
    sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def _retrieve_one(q: str) -> List[Dict[str, Any]]:
        if not q.strip():
            return []
        async with sem:
            res = await find_papers_for_direction(
                q,
                search_text=None,
                max_rounds=max_rounds,
                topk_return=-1,
                output_log_dir=out_dir,
                expand_layers=2,
                filter_score=0,
            )
        extra = res.get("extra", {}) if isinstance(res, dict) else {}
        papers: List[Dict[str, Any]] = []
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            for p in extra["all_papers"]:
                if not isinstance(p, dict):
                    continue
                papers.append(
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "doi": p.get("doi", ""),
                        "select_score": p.get("select_score", None),
                        "source": p.get("source", ""),
                        "year": p.get("year", ""),
                    }
                )
        return papers

    async def _run_one_topic_retrieval(topic_key: str, item: Dict[str, Any]) -> None:
        topic_id = str(item.get("topic_id", "")).strip()
        if not topic_id or topic_id in done:
            return
        variants = item.get("questions", [])
        if not isinstance(variants, list):
            return
        by_style = {str(v.get("style", "")).strip().upper(): v for v in variants if isinstance(v, dict)}
        qA = str((by_style.get("A") or {}).get("query", "")).strip()
        qB = str((by_style.get("B") or {}).get("query", "")).strip()
        qC = str((by_style.get("C") or {}).get("query", "")).strip()
        A, B, C = await asyncio.gather(_retrieve_one(qA), _retrieve_one(qB), _retrieve_one(qC))
        record = {
            "topic": item.get("topic", topic_key),
            "topic_id": topic_id,
            "question": [qA, qB, qC],
            "retrieved": {"A": A, "B": B, "C": C},
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _append_jsonl(paths.topic_retrieval_jsonl, record)
        _append_line(paths.topic_retrieval_done, topic_id)
        done.add(topic_id)

    tasks: List[asyncio.Task] = []
    for topic_key, item in topic_questions.items():
        if not isinstance(item, dict):
            continue
        tasks.append(asyncio.create_task(_run_one_topic_retrieval(topic_key, item)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_full(args) -> None:
    global _log_file_handle, _topic_log_file_handle

    dataset_dir = Path(args.dataset_dir)
    master_run_id = args.run_id or _slug(args.direction)
    master_paths = build_run_paths(dataset_dir, master_run_id)
    _ensure_dir(master_paths.root)

    # Open log file for this run
    log_file_path = master_paths.root / "pipeline.log"
    try:
        _log_file_handle = log_file_path.open("a", encoding="utf-8")
        print(f"[LOG] Pipeline log started at {datetime.now().isoformat()}")
        print(f"[LOG] Log file: {log_file_path}")
        _log_file_handle.write(f"[LOG] Pipeline log started at {datetime.now().isoformat()}\n")
        _log_file_handle.write(f"[LOG] Log file: {log_file_path}\n")
        _log_file_handle.flush()
    except Exception as e:
        _log_file_handle = None
        print(f"[WARNING] Failed to open log file {log_file_path}: {e}")

    async def _run_one_topic(paths: RunPaths, seed_papers: List[Dict[str, Any]]) -> None:
        log_dir = paths.root / "logs"
        _ensure_dir(log_dir)

        all_questions: Dict[str, Any] = {}
        processed_seeds = _read_lines_set(paths.root / "processed_seeds.txt") if args.resume else set()
        rejected_log = paths.root / "rejected_seeds.jsonl"
        rejected_seeds = _read_rejected_seed_ids(rejected_log) if args.resume else set()

        # Tools for seed abstract backfill
        search_paper_by_title = _import_arxiv_meta_tools()
        use_sn = not bool(getattr(args, "no_sn", False))
        _, sn_get_paper_info, _ = _import_sn_tools() if use_sn else (None, None, None)

        # When SN is disabled, pre-filter seeds (before any processing)
        if not use_sn and isinstance(seed_papers, list):
            before = len(seed_papers)
            filtered: List[Dict[str, Any]] = []
            removed = 0
            for p in seed_papers:
                if not isinstance(p, dict):
                    continue
                doi = str(p.get("doi", "") or "").strip()
                source = str(p.get("source", "") or "").strip()
                pid = _paper_id(p)
                if doi or pid.startswith("doi:") or source == "SearchFrom:sn" or "sn" in source.lower():
                    removed += 1
                    continue
                filtered.append(p)
            seed_papers = filtered
            _log_print(
                f"[SEEDS] --no-sn enabled: filtered seeds with DOI or source==SN: {before} -> {len(seed_papers)} (removed={removed})"
            )
            try:
                _write_json(paths.seed_papers_json, seed_papers)
            except Exception:
                pass

        # Seed-level processing (bounded concurrency)
        shared_lock = asyncio.Lock()
        Agent = _import_agent()
        retrieve_concurrency = max(1, int(getattr(args, "retrieve_concurrency", 10) or 10))
        retrieve_sem = asyncio.Semaphore(retrieve_concurrency)
        keyword_concurrency = max(1, int(getattr(args, "keyword_concurrency", 100) or 100))

        async def _process_one_seed(idx: int, seed_paper: Dict[str, Any]) -> None:
            seed_pid = _paper_id(seed_paper)
            seed_title = str(seed_paper.get("title", "")).strip()

            async with shared_lock:
                if seed_pid in processed_seeds:
                    _log_print(f"[SKIP] Seed {idx+1}/{len(seed_papers)} already processed: {seed_title[:80]}")
                    return
                if seed_pid in rejected_seeds:
                    _log_print(f"[SKIP] Rejected seed {idx+1}/{len(seed_papers)}: {seed_title[:80]} (seed_paper_id={seed_pid})")
                    return

            _log_print(f"\n{'='*80}")
            _log_print(f"[PROCESSING] Seed {idx+1}/{len(seed_papers)}: {seed_title} | arxiv_id={seed_paper.get('arxiv_id')}")
            _log_print(f"{'='*80}")

            # When SN is disabled, reject seeds with DOI or SN source (double guard)
            if not use_sn:
                seed_doi = str(seed_paper.get("doi", "") or "").strip()
                seed_source = str(seed_paper.get("source", "") or "").strip()
                if seed_pid.startswith("doi:") or seed_doi:
                    async with shared_lock:
                        _append_jsonl(
                            rejected_log,
                            {
                                "seed_paper_id": seed_pid,
                                "seed_paper_title": seed_title,
                                "reason": "seed_has_doi_when_no_sn",
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        rejected_seeds.add(seed_pid)
                    _log_print(f"[REJECT] Seed has DOI when --no-sn is enabled -> {seed_title[:120]}")
                    return
                if "sn" in seed_source.lower() or seed_source == "SearchFrom:sn":
                    async with shared_lock:
                        _append_jsonl(
                            rejected_log,
                            {
                                "seed_paper_id": seed_pid,
                                "seed_paper_title": seed_title,
                                "seed_source": seed_source,
                                "reason": "seed_from_sn_when_no_sn",
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        rejected_seeds.add(seed_pid)
                    _log_print(f"[REJECT] Seed from SN when --no-sn is enabled -> {seed_title[:120]} (source={seed_source})")
                    return

            # Per-seed agents (avoid cross-task shared state)
            question_agent = Agent(QUESTION_MODEL, config_path=str(CONFIG_PATH))
            keyword_agent = Agent(KEYWORDS_MODEL, config_path=str(CONFIG_PATH))

            safe_seed_id = _safe_dir_name(seed_pid)
            seed_dir = paths.root / f"seed_{safe_seed_id}"
            seed_state_file = seed_dir / "ref_expand_state.json"
            seed_pool_file = seed_dir / "paper_pool.json"
            seed_pool_build_file = seed_dir / "paper_pool.build.json"
            seed_keywords_file = seed_dir / "paper_keywords.json"
            seed_keywords_done = seed_dir / "paper_keywords.done.txt"
            seed_clusters_file = seed_dir / "clusters.json"
            seed_topics_file = seed_dir / "topics.json"

            try:
                # Seed must have abstract; try backfill if missing
                seed_abs = str(seed_paper.get("abstract", "")).strip()
                if not seed_abs:
                    try:
                        arxiv_p = search_paper_by_title(seed_title) if seed_title else None
                    except Exception:
                        arxiv_p = None
                    if isinstance(arxiv_p, dict) and str(arxiv_p.get("abstract", "")).strip():
                        seed_paper = dict(seed_paper)
                        seed_paper["abstract"] = str(arxiv_p.get("abstract", "")).strip()
                        seed_paper["arxiv_id"] = str(arxiv_p.get("arxiv_id", "")).strip() or seed_paper.get("arxiv_id", "")
                        seed_abs = str(seed_paper.get("abstract", "")).strip()
                    elif use_sn and sn_get_paper_info is not None:
                        try:
                            sn_infos = sn_get_paper_info([seed_title], batch_size=1) if seed_title else []
                        except Exception:
                            sn_infos = []
                        if isinstance(sn_infos, list) and sn_infos and isinstance(sn_infos[0], dict):
                            sn0 = sn_infos[0]
                            if str(sn0.get("abstract", "")).strip():
                                seed_paper = dict(seed_paper)
                                seed_paper["abstract"] = str(sn0.get("abstract", "")).strip()
                                seed_paper["doi"] = str(sn0.get("doi", "")).strip() or seed_paper.get("doi", "")
                                seed_abs = str(seed_paper.get("abstract", "")).strip()

                if not seed_abs:
                    async with shared_lock:
                        _append_jsonl(
                            rejected_log,
                            {
                                "seed_paper_id": seed_pid,
                                "seed_paper_title": seed_title,
                                "reason": "seed_missing_abstract",
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        rejected_seeds.add(seed_pid)
                    _log_print(f"[REJECT] seed missing abstract -> {seed_title[:120]}")
                    return

                # Step 2: reference expansion (avoid creating seed_dir until pool is valid)
                _log_print("[STEP 2] Expanding references for seed paper...")
                temp_dir = paths.root / ".temp_seeds"
                _ensure_dir(temp_dir)
                temp_state_file = temp_dir / f"{safe_seed_id}_ref_expand_state.json"
                temp_pool_build_file = temp_dir / f"{safe_seed_id}_paper_pool.build.json"

                if seed_pool_file.exists():
                    pool = _read_json(seed_pool_file, {})
                    if not isinstance(pool, dict):
                        pool = {}
                else:
                    pool = step_expand_refs_single_seed(
                        seed_paper,
                        n_layers=args.ref_layers,
                        seed_paper_id=seed_pid,
                        state_file=temp_state_file,
                        pool_file=temp_pool_build_file,
                        resume=bool(args.resume),
                        use_sn=use_sn,
                    )
                    if not isinstance(pool, dict):
                        pool = {}

                    # Enforce: pool papers must have abstract
                    pool = {pid: p for pid, p in pool.items() if str((p or {}).get("abstract", "")).strip()}
                    min_pool = int(getattr(args, "min_pool_size", 10) or 10)
                    _log_print(f"[STEP 2] Expanded+filtered pool to {len(pool)} papers (must have abstract)")

                    if len(pool) < min_pool:
                        # cleanup temp files
                        try:
                            if temp_pool_build_file.exists():
                                temp_pool_build_file.unlink()
                            if temp_state_file.exists():
                                temp_state_file.unlink()
                        except Exception:
                            pass
                        async with shared_lock:
                            _append_jsonl(
                                rejected_log,
                                {
                                    "seed_paper_id": seed_pid,
                                    "seed_paper_title": seed_title,
                                    "pool_size": len(pool),
                                    "min_pool_size": min_pool,
                                    "reason": "pool_too_small",
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                },
                            )
                            rejected_seeds.add(seed_pid)
                        _log_print(f"[REJECT] pool too small ({len(pool)} < {min_pool}) -> {seed_title[:120]}")
                        return

                    # commit
                    _ensure_dir(seed_dir)
                    try:
                        if temp_state_file.exists():
                            temp_state_file.replace(seed_state_file)
                        if temp_pool_build_file.exists():
                            temp_pool_build_file.replace(seed_pool_build_file)
                    except Exception:
                        pass
                    _write_json(seed_pool_file, pool)
                    try:
                        if seed_pool_build_file.exists():
                            seed_pool_build_file.unlink()
                    except Exception:
                        pass

                # Step 3: keywords
                _log_print(f"[STEP 3] Extracting keywords (concurrency={keyword_concurrency})...")
                paper_keywords = await step_keywords(
                    pool,
                    paths=RunPaths(
                        root=seed_dir,
                        seed_papers_json=seed_dir / "seed.json",
                        ref_state_json=seed_state_file,
                        paper_pool_json=seed_pool_file,
                        paper_keywords_json=seed_keywords_file,
                        paper_keywords_done=seed_keywords_done,
                        clusters_json=seed_clusters_file,
                        topics_json=seed_topics_file,
                        topic_questions_json=seed_dir / "questions.json",
                        topic_questions_done=seed_dir / "questions.done.txt",
                        topic_retrieval_jsonl=seed_dir / "retrieval.jsonl",
                        topic_retrieval_done=seed_dir / "retrieval.done.txt",
                    ),
                    resume=bool(args.resume),
                    max_concurrent=keyword_concurrency,
                )

                # Step 4: cluster + topics
                if seed_clusters_file.exists() and seed_topics_file.exists():
                    filtered_clusters = _read_json(seed_clusters_file, [])
                    filtered_topics = _read_json(seed_topics_file, {})
                else:
                    _log_print("[STEP 4] Clustering papers and generating topics...")
                    clusters, topics = step_cluster(
                        pool,
                        paper_keywords,
                        sim_threshold=args.cluster_sim,
                        merge_threshold=getattr(args, "cluster_merge_sim", None),
                        seed_paper_id=seed_pid,
                        agent=keyword_agent,
                    )
                    filtered_clusters: List[Dict[str, Any]] = []
                    filtered_topics: Dict[str, Any] = {}
                    for c in clusters:
                        pids = c.get("paper_ids", []) if isinstance(c, dict) else []
                        if isinstance(pids, list) and len(pids) >= 3:
                            filtered_clusters.append(c)
                            cluster_id = str(c.get("cluster_id", "")).strip()
                            for topic_key, topic_obj in (topics or {}).items():
                                if isinstance(topic_obj, dict) and str(topic_obj.get("topic_id", "")).strip() == cluster_id:
                                    filtered_topics[topic_key] = topic_obj
                                    break
                    _log_print(
                        f"[STEP 4] Generated {len(clusters)} clusters, filtered to {len(filtered_clusters)} clusters (>=3 papers)"
                    )
                    _write_json(seed_clusters_file, filtered_clusters)
                    _write_json(seed_topics_file, filtered_topics)

                # Requirement: if clustering empty -> reject + delete seed folder
                if not filtered_clusters or not isinstance(filtered_clusters, list):
                    async with shared_lock:
                        _append_jsonl(
                            rejected_log,
                            {
                                "seed_paper_id": seed_pid,
                                "seed_paper_title": seed_title,
                                "reason": "empty_clusters",
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        rejected_seeds.add(seed_pid)
                    deleted = _safe_rmtree_seed_dir(seed_dir, paths.root)
                    _log_print(f"[REJECT] empty_clusters -> seed_paper_id={seed_pid} deleted_dir={deleted} path={seed_dir}")
                    return

                # Step 5: generate questions + retrieve (per cluster)
                _log_print(f"[STEP 5] Generating questions + retrieving papers for clusters (concurrency={retrieve_concurrency})...")
                find_papers_for_direction = _import_query_pipeline_find()

                seed_results: List[Dict[str, Any]] = []
                seed_results_file = seed_dir / "results.jsonl"
                processed_clusters_file = seed_dir / "processed_clusters.txt"
                processed_clusters = _read_lines_set(processed_clusters_file) if args.resume else set()

                existing_results = _read_jsonl(seed_results_file) if args.resume and seed_results_file.exists() else []
                existing_topic_ids = {str(r.get("topic_id", "")).strip() for r in existing_results if isinstance(r, dict)}

                async def _retrieve_cluster_papers(
                    *,
                    topic_id: str,
                    question_a: str,
                    out_log: Path,
                    cluster_papers: List[Dict[str, Any]],
                    checklist: str,
                ) -> Tuple[str, List[Dict[str, Any]]]:
                    async with retrieve_sem:
                        retrieve_res = await find_papers_for_direction(
                            question_a,
                            search_text=None,
                            max_rounds=args.max_rounds,
                            topk_return=-1,
                            output_log_dir=out_log,
                            expand_layers=2,
                            filter_score=0,
                            init_papers=cluster_papers,
                            checklist=checklist,
                        )
                    extra = retrieve_res.get("extra", {}) if isinstance(retrieve_res, dict) else {}
                    out: List[Dict[str, Any]] = []
                    if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
                        for p in extra["all_papers"]:
                            if not isinstance(p, dict):
                                continue
                            out.append(
                                {
                                    "title": p.get("title", ""),
                                    "abstract": p.get("abstract", ""),
                                    "arxiv_id": p.get("arxiv_id", ""),
                                    "doi": p.get("doi", ""),
                                    "select_score": p.get("select_score", None),
                                    "reason": p.get("reason", ""),
                                    "source": p.get("source", ""),
                                    "year": p.get("year", ""),
                                }
                            )
                    return topic_id, out

                pending: Dict[str, Dict[str, Any]] = {}
                retrieval_tasks: List[asyncio.Task] = []

                for topic_key, topic_obj in (filtered_topics or {}).items():
                    if not isinstance(topic_obj, dict):
                        continue
                    topic_id = str(topic_obj.get("topic_id", "")).strip()
                    if not topic_id:
                        continue

                    # Check if result already exists in results.jsonl (skip retrieval if exists)
                    if args.resume and topic_id in existing_topic_ids:
                        existing_result = next((r for r in existing_results if str(r.get("topic_id", "")).strip() == topic_id), None)
                        if isinstance(existing_result, dict) and isinstance(existing_result.get("paper_list"), list):
                            seed_results.append(existing_result)
                            if topic_id not in processed_clusters:
                                _append_line(processed_clusters_file, topic_id)
                                processed_clusters.add(topic_id)
                            _log_print(f"[SKIP] Cluster {topic_id} result already exists in {seed_results_file}, skipping retrieval")
                            continue

                    # cluster papers (all)
                    pids = topic_obj.get("paper_ids", [])
                    cluster_papers: List[Dict[str, Any]] = []
                    if isinstance(pids, list):
                        for pid in pids:
                            pid_s = str(pid).strip()
                            p = pool.get(pid_s, {}) if isinstance(pool, dict) else {}
                            kwobj = paper_keywords.get(pid_s, {}) if isinstance(paper_keywords, dict) else {}
                            if isinstance(p, dict) and (p.get("title") or p.get("abstract")):
                                cluster_papers.append(
                                    {
                                        "title": p.get("title", ""),
                                        "abstract": p.get("abstract", ""),
                                        "arxiv_id": p.get("arxiv_id", ""),
                                        "authors": p.get("authors", ""),
                                        "journal": p.get("journal", ""),
                                        "year": p.get("year", ""),
                                        "source": p.get("source", ""),
                                        "keywords": kwobj.get("keywords", []) if isinstance(kwobj, dict) else [],
                                    }
                                )
                    if not cluster_papers:
                        continue

                    core_topic = str(topic_obj.get("topic", topic_key)).strip()

                    # Generate questions if missing
                    variants = topic_obj.get("questions", [])
                    gen = None
                    if not isinstance(variants, list) or not variants:
                        max_retries = 3
                        for retry_idx in range(max_retries):
                            gen = _generate_abc_questions(
                                direction=args.direction,
                                topic=core_topic,
                                support=cluster_papers,
                                agent=question_agent,
                                all_papers_in_cluster=cluster_papers,
                            )
                            if gen is not None:
                                break
                            _log_print(f"[RETRY] cluster={topic_id} question gen failed ({retry_idx+1}/{max_retries})")
                        if gen is None:
                            _log_print(f"[WARNING] Failed to generate questions for cluster {topic_id}")
                            continue
                        variants = gen.get("variants", [])
                        topic_obj["questions"] = variants
                        topic_obj["reason"] = gen.get("reason", "")
                        topic_obj["questions_model"] = QUESTION_MODEL
                        topic_obj["questions_updated_at"] = datetime.now().isoformat(timespec="seconds")
                        try:
                            _write_json(seed_topics_file, filtered_topics)
                        except Exception:
                            pass

                    # Extract A/B/C
                    questions_dict: Dict[str, str] = {"A": "", "B": "", "C": ""}
                    for v in variants:
                        if not isinstance(v, dict):
                            continue
                        style = str(v.get("style", "")).strip().upper()
                        q = str(v.get("query", "")).strip()
                        ck = v.get("checklist", [])
                        if style in ("A", "B", "C") and q:
                            questions_dict[style] = q
                            if isinstance(ck, list):
                                questions_dict[f"{style}_checklist"] = "; ".join([str(x).strip() for x in ck if str(x).strip()])
                            else:
                                questions_dict[f"{style}_checklist"] = str(ck or "").strip()

                    question_a = questions_dict.get("A", "") or questions_dict.get("B", "") or questions_dict.get("C", "")
                    if not question_a:
                        continue

                    out_log = log_dir / f"seed_{safe_seed_id}" / f"cluster_{topic_id}"
                    _ensure_dir(out_log)
                    pending[topic_id] = {
                        "topic": core_topic,
                        "topic_id": topic_id,
                        "cluster_papers": cluster_papers,
                        "questions_dict": questions_dict,
                        "out_log": out_log,
                    }
                    retrieval_tasks.append(
                        asyncio.create_task(
                            _retrieve_cluster_papers(
                                topic_id=topic_id,
                                question_a=question_a,
                                out_log=out_log,
                                cluster_papers=cluster_papers,
                                checklist=str(questions_dict.get("A_checklist", "") or ""),
                            )
                        )
                    )

                if retrieval_tasks:
                    for t in asyncio.as_completed(retrieval_tasks):
                        try:
                            topic_id, retrieved_papers = await t
                        except Exception as e:
                            _log_print(f"[ERROR] Cluster retrieval task failed: {e}")
                            continue
                        meta = pending.get(topic_id)
                        if not isinstance(meta, dict):
                            continue
                        questions_dict = meta.get("questions_dict", {}) if isinstance(meta.get("questions_dict"), dict) else {}
                        core_topic = str(meta.get("topic", "")).strip()
                        cluster_papers = meta.get("cluster_papers", []) if isinstance(meta.get("cluster_papers"), list) else []

                        result = {
                            "question": [
                                str(questions_dict.get("A", "") or ""),
                                str(questions_dict.get("B", "") or ""),
                                str(questions_dict.get("C", "") or ""),
                            ],
                            "paper_list": retrieved_papers,
                            "topic": core_topic,
                            "topic_id": topic_id,
                            "seed_paper_id": seed_pid,
                            "seed_paper_title": seed_title,
                            "cluster_size": len(cluster_papers),
                        }
                        seed_results.append(result)
                        # Save immediately after each QA pair is generated
                        _append_jsonl(seed_results_file, result)
                        _append_line(processed_clusters_file, topic_id)
                        processed_clusters.add(topic_id)
                        _log_print(f"[SAVED] QA pair for cluster {topic_id} saved to {seed_results_file}")

                # Always ensure results.jsonl exists (placeholder if nothing)
                if not seed_results_file.exists():
                    placeholder = {
                        "question": ["", "", ""],
                        "paper_list": [],
                        "topic": "",
                        "topic_id": "",
                        "seed_paper_id": seed_pid,
                        "seed_paper_title": seed_title,
                        "cluster_size": 0,
                        "note": "no_results_generated_or_no_clusters",
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    _append_jsonl(seed_results_file, placeholder)

                # status.json
                status_data = {"seed_paper_id": seed_pid, "seed_paper_title": seed_title, "queries": {}}
                has_high_score_papers = False
                valid_results: List[Dict[str, Any]] = []
                for result in seed_results:
                    paper_list = result.get("paper_list", [])
                    if not isinstance(paper_list, list):
                        continue
                    questions = result.get("question", [])
                    query_text = str(questions[0]).strip() if isinstance(questions, list) and questions else ""
                    dist = {"77-100": 0, "61-77": 0, "46-60": 0, "30-46": 0, "15-30": 0, "0-15": 0, "no_score": 0}
                    for p in paper_list:
                        sc = (p or {}).get("select_score") if isinstance(p, dict) else None
                        if sc is None:
                            dist["no_score"] += 1
                        elif sc >= 77:
                            dist["77-100"] += 1
                            has_high_score_papers = True
                        elif sc >= 61:
                            dist["61-77"] += 1
                        elif sc >= 46:
                            dist["46-60"] += 1
                        elif sc >= 30:
                            dist["30-46"] += 1
                        elif sc >= 15:
                            dist["15-30"] += 1
                        else:
                            dist["0-15"] += 1
                    total = len(paper_list)
                    pct = {k: (round((v / total) * 100, 2) if total else 0.0) for k, v in dist.items()}
                    if query_text:
                        status_data["queries"][query_text] = {
                            "total_papers": total,
                            "score_distribution": dist,
                            "score_percentages": pct,
                            "topic_id": result.get("topic_id", ""),
                            "topic": result.get("topic", ""),
                        }
                    if dist["77-100"] > 0:
                        valid_results.append(result)

                _write_json(seed_dir / "status.json", status_data)
                if not has_high_score_papers:
                    async with shared_lock:
                        _append_jsonl(
                            rejected_log,
                            {
                                "seed_paper_id": seed_pid,
                                "seed_paper_title": seed_title,
                                "reason": "no_high_score_papers",
                                "num_clusters": len(filtered_clusters) if isinstance(filtered_clusters, list) else 0,
                                "num_results": len(seed_results),
                                "note": "directory_kept_with_status_json",
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            },
                        )

                async with shared_lock:
                    all_questions[seed_pid] = {
                        "seed_paper_id": seed_pid,
                        "seed_paper_title": seed_title,
                        "num_clusters": len(filtered_clusters) if isinstance(filtered_clusters, list) else 0,
                        "num_valid_results": len(valid_results),
                        "results": valid_results,
                    }
                    _append_line(paths.root / "processed_seeds.txt", seed_pid)
                    processed_seeds.add(seed_pid)

            except Exception as e:
                _log_print(f"[ERROR] Failed to process seed {seed_pid}: {e}")
                import traceback
                traceback.print_exc()
                return

        seed_papers_sorted = [p for p in (seed_papers or []) if isinstance(p, dict)]
        seed_papers_sorted = sorted(seed_papers_sorted, key=lambda x: str(x.get("arxiv_id", "") or ""))
        total_seeds = len(seed_papers_sorted)
        seed_concurrency = max(1, int(getattr(args, "seed_concurrency", 10) or 10))
        _log_print(f"[SEEDS] Processing seeds with concurrency={seed_concurrency} (total={total_seeds})")

        sem_seed = asyncio.Semaphore(seed_concurrency)

        async def _run_one_seed(idx: int, seed_paper: Dict[str, Any]) -> None:
            async with sem_seed:
                _log_seed_progress(idx + 1, total_seeds, seed_paper)
                t0 = time.time()
                await _process_one_seed(idx, seed_paper)
                _log_print(f"[SEED DONE] {idx+1}/{total_seeds} elapsed={time.time()-t0:.1f}s")

        tasks: List[asyncio.Task] = []
        for idx, seed_paper in enumerate(seed_papers_sorted):
            tasks.append(asyncio.create_task(_run_one_seed(idx, seed_paper)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    _log_print(f"[ERROR] seed task failed: {r}")
        else:
            _log_print("[SEEDS] No valid seed papers to process.")

        _write_json(paths.topic_questions_json, all_questions)
        _log_print(f"\n{'='*80}")
        _log_print(f"[DONE] Processed {len(processed_seeds)} seed papers")
        _log_print(f"[DONE] output_dir={paths.root}")
        _log_print(f"{'='*80}")

    def _count_completed_seeds(topic_root: Path) -> int:
        """Count seed directories that have results.jsonl file."""
        if not topic_root.exists() or not topic_root.is_dir():
            return 0
        count = 0
        try:
            for item in topic_root.iterdir():
                if item.is_dir() and item.name.startswith("seed_") and (item / "results.jsonl").exists():
                    count += 1
        except Exception:
            pass
        return count

    # For each topic: generate seeds -> run full pipeline (per-topic folder)
    matched_key, _subtopics, retrieval_queries = _resolve_direction_subtopics(args.direction)
    _log_print(f"[TOPIC LOOP] direction='{args.direction}' matched='{matched_key}' topics={len(retrieval_queries)}")

    for i, (topic_label, topic_desc) in enumerate(retrieval_queries):
        topic_slug = _slug(topic_label)
        paths = build_run_paths(master_paths.root, topic_slug)
        _ensure_dir(paths.root)

        # open per-topic log
        try:
            _topic_log_file_handle = (paths.root / "pipeline.log").open("a", encoding="utf-8")
            _topic_log_file_handle.write(f"[LOG] Topic log started at {datetime.now().isoformat()}\n")
            _topic_log_file_handle.flush()
        except Exception:
            _topic_log_file_handle = None

        _log_print(f"\n{'='*80}")
        _log_print(f"[TOPIC] {i+1}/{len(retrieval_queries)} topic='{topic_label}' folder={paths.root}")
        _log_print(f"{'='*80}")

        # Check if topic already has enough completed seeds
        completed_seeds = _count_completed_seeds(paths.root)
        if completed_seeds >= args.max_seeds:
            _log_print(f"[SKIP] Topic '{topic_label}' already has {completed_seeds} completed seeds (>= max_seeds={args.max_seeds}), skipping...")
            # close per-topic log
            if _topic_log_file_handle is not None:
                try:
                    _topic_log_file_handle.write(f"[LOG] Topic skipped (already completed) at {datetime.now().isoformat()}\n")
                    _topic_log_file_handle.close()
                except Exception:
                    pass
                _topic_log_file_handle = None
            continue

        use_sn = not bool(getattr(args, "no_sn", False))

        # Step 1 (per-topic): seed papers
        # Always try to load existing seed_papers.json first (even without --resume)
        if paths.seed_papers_json.exists():
            seed_papers = _read_json(paths.seed_papers_json, [])
            if not isinstance(seed_papers, list):
                seed_papers = []
            seed_papers = seed_papers[:args.max_seeds]
            _log_print(f"[TOPIC] Loaded cached seed_papers.json: {len(seed_papers)} seeds")
        else:
            kwc = max(1, int(getattr(args, "keyword_concurrency", 100) or 100))
            seed_papers = await _step_seed_papers_one_topic(
                direction=args.direction,
                topic_label=topic_label,
                topic_desc=topic_desc,
                out_dir=paths.root,
                max_rounds=args.max_rounds,
                use_sn=use_sn,
                retrieval_concurrency=10,
                keyword_concurrency=kwc,
            )
            seed_papers = seed_papers[:args.max_seeds]
            _write_json(paths.seed_papers_json, seed_papers)

        await _run_one_topic(paths, seed_papers)

        # close per-topic log
        if _topic_log_file_handle is not None:
            try:
                _topic_log_file_handle.write(f"[LOG] Topic log ended at {datetime.now().isoformat()}\n")
                _topic_log_file_handle.close()
            except Exception:
                pass
            _topic_log_file_handle = None

    # Close master log file
    if _log_file_handle is not None:
        try:
            _log_print(f"[LOG] Pipeline log ended at {datetime.now().isoformat()}")
            _log_file_handle.close()
            _log_file_handle = None
        except Exception:
            pass


def test_expand_refs():
    """
    Test function for step_expand_refs_single_seed.
    Tests reference expansion for a single seed paper.
    """
    _log_print("="*80)
    _log_print("TEST: Reference Expansion for Single Seed Paper")
    _log_print("="*80)
    
    # Create a test seed paper
    test_seed = {
        "title": "MAGDA: Multi-agent guideline-driven diagnostic assistance",
        "abstract": "In emergency departments, rural hospitals, or clinics in less developed regions, clinicians often lack fast image analysis by trained radiologists.",
        "arxiv_id": "2409.06351",
        "doi": "",
        "year": "2024",
        "authors": ["David Bani-Harouni", "Nassir Navab", "Matthias Keicher"],
        "journal": "arxiv",
        "source": "test",
    }
    
    # Create test directory (use /tmp; never delete real result dirs)
    test_dir = Path("/tmp/test_ref_expand_fn")
    test_dir.mkdir(parents=True, exist_ok=True)
    state_file = test_dir / "ref_expand_state.json"
    pool_file = test_dir / "paper_pool.json"
    
    seed_pid = _paper_id(test_seed)
    _log_print(f"\nTest seed paper:")
    _log_print(f"  ID: {seed_pid}")
    _log_print(f"  Title: {test_seed['title']}")
    _log_print(f"  Arxiv ID: {test_seed['arxiv_id']}")
    
    # Test expansion with 2 layers
    _log_print(f"\nExpanding references (2 layers)...")
    try:
        pool = step_expand_refs_single_seed(
            seed_paper=test_seed,
            n_layers=2,
            seed_paper_id=seed_pid,
            state_file=state_file,
            pool_file=pool_file,
            resume=False,  # Start fresh
            sn_batch=5,
            sn_fuzzy_threshold=80,
        )
        
        _log_print(f"\n{'='*80}")
        _log_print("EXPANSION RESULTS:")
        _log_print(f"{'='*80}")
        _log_print(f"Total papers in pool: {len(pool)}")
        
        # Count papers by depth
        depth_count = {}
        for pid, p in pool.items():
            depth = p.get("depth", -1)
            depth_count[depth] = depth_count.get(depth, 0) + 1
        
        _log_print(f"\nPapers by depth:")
        for depth in sorted(depth_count.keys()):
            _log_print(f"  Depth {depth}: {depth_count[depth]} papers")
        
        _log_print(f"\nSample papers:")
        for i, (pid, p) in enumerate(list(pool.items())[:5]):
            _log_print(f"  {i+1}. [{p.get('depth', '?')}] {p.get('title', 'N/A')[:60]}...")
            _log_print(f"      ID: {pid}")
        
        _log_print(f"\n{'='*80}")
        _log_print("TEST COMPLETED SUCCESSFULLY")
        _log_print(f"{'='*80}")
        
        # Cleanup (safe)
        import shutil
        td = str(test_dir.resolve())
        if "/tmp/" in td and "test_ref_expand" in td and test_dir.exists():
            shutil.rmtree(test_dir)
            _log_print(f"\nCleaned up test directory: {test_dir}")
        
        return pool
        
    except Exception as e:
        _log_print(f"\n{'='*80}")
        _log_print(f"TEST FAILED: {e}")
        _log_print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_full = sub.add_parser("full", help="direction -> seed -> ref expand -> keywords -> cluster -> ABC questions -> retrieve")
    p_full.add_argument("--direction", type=str, required=True)
    p_full.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    p_full.add_argument("--run-id", type=str, default=None, help="Optional run id; default is slug(direction).")
    p_full.add_argument("--ref-layers", type=int, default=2, help="Reference expansion layers (default 1).")
    p_full.add_argument("--cluster-sim", type=float, default=0.25, help="Jaccard similarity threshold for clustering (default 0.25). Lower values create larger clusters.")
    p_full.add_argument("--cluster-merge-sim", type=float, default=None, help="Optional cluster merge threshold (Jaccard on cluster cores). If not set, uses ~0.8*cluster-sim.")
    p_full.add_argument("--max-rounds", type=int, default=1)
    p_full.add_argument("--resume", action="store_true", help="Resume from existing caches if present.")
    p_full.add_argument("--min-pool-size", type=int, default=10, help="Reject seed if its paper pool (with abstracts) is smaller than this.")
    p_full.add_argument("--seed-concurrency", type=int, default=10, help="Number of seed papers to process concurrently (default 10). Each seed runs the FULL pipeline end-to-end.")
    p_full.add_argument("--keyword-concurrency", type=int, default=100, help="Keyword extraction concurrency (default 100).")
    p_full.add_argument("--retrieve-concurrency", type=int, default=10, help="Complex query paper retrieval concurrency per seed (default 10).")
    p_full.add_argument("--no-sn", action="store_true", help="Disable SN (Semantic Scholar/SN) fallback for ref expansion and seed abstract backfill; use arxiv-only.")
    p_full.add_argument("--max-seeds", type=int, default=30, help="Maximum number of seeds to process (default 100).")
    p_test = sub.add_parser("test-expand", help="Test reference expansion for a single seed paper")
    p_test.add_argument("--arxiv-id", type=str, default="2409.06351", help="Arxiv ID of test paper")
    p_test.add_argument("--layers", type=int, default=2, help="Number of expansion layers")
    p_test.add_argument("--keep-files", action="store_true", help="Keep test files after completion")

    args = parser.parse_args()
    if args.cmd == "full":
        asyncio.run(run_full(args))
        return
    elif args.cmd == "test-expand":
        # Enhanced test with custom arxiv_id
        test_seed = {
            "title": "MAGDA: Multi-agent guideline-driven diagnostic assistance",
            "abstract": "In emergency departments, rural hospitals, or clinics in less developed regions, clinicians often lack fast image analysis by trained radiologists.",
            "arxiv_id": args.arxiv_id,
            "doi": "",
            "year": "2024",
            "authors": ["Test Author"],
            "journal": "arxiv",
            "source": "test",
        }
        
        test_dir = Path("/tmp/test_ref_expand")
        test_dir.mkdir(exist_ok=True)
        state_file = test_dir / "ref_expand_state.json"
        pool_file = test_dir / "paper_pool.json"
        
        seed_pid = _paper_id(test_seed)
        _log_print(f"\n{'='*80}")
        _log_print("TEST: Reference Expansion")
        _log_print(f"{'='*80}")
        _log_print(f"Seed paper ID: {seed_pid}")
        _log_print(f"Arxiv ID: {args.arxiv_id}")
        _log_print(f"Layers: {args.layers}")
        _log_print(f"Test directory: {test_dir}")
        
        pool = step_expand_refs_single_seed(
            seed_paper=test_seed,
            n_layers=args.layers,
            seed_paper_id=seed_pid,
            state_file=state_file,
            pool_file=pool_file,
            resume=False,
            sn_batch=5,
            sn_fuzzy_threshold=80,
        )
        
        _log_print(f"\n{'='*80}")
        _log_print("RESULTS:")
        _log_print(f"{'='*80}")
        _log_print(f"Total papers: {len(pool)}")
        
        depth_count = {}
        for pid, p in pool.items():
            depth = p.get("depth", -1)
            depth_count[depth] = depth_count.get(depth, 0) + 1
        
        _log_print(f"\nPapers by depth:")
        for depth in sorted(depth_count.keys()):
            _log_print(f"  Depth {depth}: {depth_count[depth]} papers")
        
        if not args.keep_files:
            # Safety: only delete if it's clearly a temp test directory
            import shutil
            td = str(test_dir.resolve())
            if "/tmp/" in td and "test_ref_expand" in td and test_dir.exists():
                shutil.rmtree(test_dir)
                _log_print(f"\nCleaned up test directory")
        else:
            _log_print(f"\nTest files kept at: {test_dir}")
        
        return


if __name__ == "__main__":
    main()


