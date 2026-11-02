#!/usr/bin/env python3
# coding: utf-8
"""
Pipeline for:
1) discipline -> research directions (5-10)
2) direction -> find related papers via PasaMaster (Manager.run)
3) (direction + paper abstract) -> generate a more detailed/complex retrieval query
4) end-to-end: discipline -> directions -> papers -> complex queries

Outputs keep simple Python dict structures; a CLI is provided for batch runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.json"
PROMPTS_PATH = REPO_ROOT / "prompt" / "papermaster_prompt.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "result" / "dataset_final"
DEFAULT_TOPIC_FILE = REPO_ROOT / "create_dataset" / "data" / "topic_en.json"

# Retrieval models are FIXED (do not pass via CLI).
# - keyminder and ranker: qwen-flash
# - top_ranker: reason_rank
RETRIEVAL_KEYMINDER_MODEL = "qwen-72b"
RETRIEVAL_RANKER_MODEL = "qwen-72b"
RETRIEVAL_TOP_RANKER_MODEL = "reason_rank"

# Generation model default (topics + complex query)
DEFAULT_LLM_MODEL = "kimi2"

# Complex-query sampling policy (score buckets)
SCORE_BIN_SIZE = 10
SAMPLES_PER_BIN = 2

# Extracted from `areaIds` list (English names).
DISCIPLINES: List[str] = [
    "Geoscience",
    "Physics and Astrophysics",
    "Mathematics",
    "Agroforestry Science",
    "Material science",
    "Computer science",
    "Environmental Science and Ecology",
    "Chemistry",
    "Engineering Technology",
    "Biology",
    "Medicine",
    "Comprehensive Periodical",
    "Law",
    "Psychology",
    "Pedagogy",
    "Economics",
    "Management",
    "Humanities",
    "arXiv-Physics",
    "arXiv-Mathematics",
    "arXiv-Computer Science",
]


def get_discipline_list() -> List[str]:
    """Return the supported discipline names list."""
    return list(DISCIPLINES)


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _safe_json_extract(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    # try strict json first
    try:
        return json.loads(text)
    except Exception:
        pass
    # fallback: extract first json array/object block
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _norm_key(s: str) -> str:
    """Normalize a key for fuzzy matching (case-insensitive, alnum-only)."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def load_directions_from_topic_file(
    discipline_name: str,
    *,
    topic_file: Path = DEFAULT_TOPIC_FILE,
) -> Dict[str, str]:
    """
    Load discipline sub-directions from topic file.

    The expected schema of topic_en.json:
      {
        "<Discipline>": {
          "<SubDirectionName>": "<SubDirectionDescription>",
          ...
        },
        ...
      }

    Returns:
      Mapping {sub_direction_name: description}
    """
    if not discipline_name or not str(discipline_name).strip():
        raise ValueError("discipline_name is empty")

    obj = _read_json(topic_file, default={})
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid topic file schema (expected dict): {topic_file}")

    # try exact match first
    if discipline_name in obj and isinstance(obj.get(discipline_name), dict):
        mapping = obj.get(discipline_name) or {}
        return {str(k).strip(): str(v).strip() for k, v in mapping.items() if str(k).strip()}

    # fallback: fuzzy match (case-insensitive, ignore punctuation/spaces)
    target = _norm_key(discipline_name)
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if _norm_key(k) == target and isinstance(v, dict):
            return {str(sk).strip(): str(sv).strip() for sk, sv in v.items() if str(sk).strip()}

    # not found
    available = [str(k) for k in obj.keys() if isinstance(k, str)]
    raise ValueError(
        f"Discipline {discipline_name!r} not found in topic file: {topic_file}. "
        f"Available disciplines include: {available[:20]}{'...' if len(available) > 20 else ''}"
    )


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


def _safe_float(x: Any, default: float = -1.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_bool(v: Any, *, default: bool = False) -> bool:
    """
    Parse common boolean strings for CLI usage.
    Accepts: true/false, 1/0, yes/no, y/n, on/off (case-insensitive).
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _read_lines_set(path: Path) -> set[str]:
    """Read a newline-delimited text file into a set of stripped lines."""
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
    """Append a single line to a text file (with newline)."""
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(line).rstrip("\n") + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON object per line (jsonl)."""
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def _bucket_start(score: float, bin_size: int = SCORE_BIN_SIZE) -> int:
    """
    Score buckets: [0,10),[10,20),...,[90,100] (100 included into 90-100).
    Negative scores will be bucketed to 0.
    """
    if score < 0:
        return 0
    b = int(score // bin_size) * bin_size
    if b >= 100:
        b = 90
    if b < 0:
        b = 0
    return b


def _select_by_score_buckets(
    papers: List[Dict[str, Any]],
    *,
    bin_size: int = SCORE_BIN_SIZE,
    per_bin: int = SAMPLES_PER_BIN,
) -> List[Dict[str, Any]]:
    """
    Select up to `per_bin` papers from each `bin_size` score bucket.
    Within each bucket, prefer higher select_score.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for p in papers:
        if not isinstance(p, dict):
            continue
        sc = _safe_float(p.get("select_score", -1.0), -1.0)
        bs = _bucket_start(sc, bin_size=bin_size)
        buckets.setdefault(bs, []).append(p)

    selected: List[Dict[str, Any]] = []
    for bs in sorted(buckets.keys()):
        items = buckets[bs]
        items.sort(key=lambda x: _safe_float(x.get("select_score", -1.0), -1.0), reverse=True)
        selected.extend(items[:per_bin])
    return selected


def dataset_paths(dataset_dir: Path) -> Dict[str, Path]:
    """
    Cache / output files:
      1) discipline -> directions[]
      2) direction -> papers[]
      3) paper_title -> {"query": str, "checklist": [str, ...], "select_score": float|null}
      4) query_to_papers.jsonl: {"question": [A,B,C], "paper_list": [...]} (papers retrieved by Variant A)
         - Each line is an independent record for streaming writes & resume.
      5) query_to_papers.done.txt: newline-delimited keys for resume (normalized A-query key)
    """
    return {
        "discipline_to_directions": dataset_dir / "discipline_to_directions.json",
        "direction_to_papers": dataset_dir / "direction_to_papers.json",
        "title_to_complex_query": dataset_dir / "title_to_complex_query.json",
        "query_to_papers": dataset_dir / "query_to_papers.jsonl",
        "query_to_papers_done": dataset_dir / "query_to_papers.done.txt",
    }


def generate_research_directions(
    discipline_name: str,
    *,
    model_name: str = DEFAULT_LLM_MODEL,
    n_min: int = 5,
    n_max: int = 10,
) -> List[str]:
    """
    Function1:
      input discipline name -> output 5~10 different research directions (English list).
    """
    if not discipline_name or not str(discipline_name).strip():
        raise ValueError("discipline_name is empty")
    discipline_name = str(discipline_name).strip()

    # Import locally to avoid heavy imports for users who only want parsing helpers.
    sys.path.insert(0, str(REPO_ROOT))
    from model_vllm import Agent

    agent = Agent(model_name, config_path=str(CONFIG_PATH))
    prompt = f"""
You are an academic domain expert.
Given a discipline name, propose {n_min}-{n_max} DISTINCT research directions within that discipline.

Rules:
- Output MUST be a JSON array of strings.
- Each item is a concise English research direction (3-8 words).
- Items must be non-overlapping and diverse.
- No numbering, no extra text.

Discipline: {discipline_name}
""".strip()

    out = agent.infer(prompt, use_cot=False)
    obj = _safe_json_extract(out)
    if isinstance(obj, list):
        dirs = [str(x).strip() for x in obj if str(x).strip()]
        # de-dup while preserving order
        seen = set()
        res = []
        for d in dirs:
            k = d.lower()
            if k in seen:
                continue
            seen.add(k)
            res.append(d)
        return res[: max(n_min, min(n_max, len(res)))]

    # fallback: split lines/semicolons
    cand = re.split(r"[;\n]", out or "")
    cand = [c.strip(" \t-•\r") for c in cand if c.strip()]
    seen = set()
    res = []
    for d in cand:
        k = d.lower()
        if k in seen:
            continue
        seen.add(k)
        res.append(d)
    return res[: max(n_min, min(n_max, len(res)))]


async def find_papers_for_direction(
    direction: str,
    *,
    search_text: Optional[str] = None,
    end_date: Optional[str] = None,
    max_rounds: int = 1,
    topk_return: int = -1,
    output_log_dir: Optional[Path] = None,
    expand_layers: int = 2,
    filter_score: int = 0,
    init_papers = None,
    checklist = None,
) -> Dict[str, Any]:
    """
    Function2:
      input direction -> call PasaMaster (Manager.run) to find related papers.

    Returns:
      dict with keys:
        - query
        - extra: {..., all_papers: [...], final_checklist: [...], ...}
        - final_papers: [...]
    """
    if not direction or not str(direction).strip():
        raise ValueError("direction is empty")
    direction = str(direction).strip()

    sys.path.insert(0, str(REPO_ROOT))

    # PasaMaster reads config with a relative path ("config/config.json") at import-time,
    # so ensure cwd is repo root during import and execution.
    with _pushd(REPO_ROOT):
        from log import setup_logging
        from model_vllm import Agent
        from quick_thinker import PaperAgent
        from reason_agent import reason_agent as ReasonAgent
        from PasaMaster import Manager

        if output_log_dir is not None:
            output_log_dir.mkdir(parents=True, exist_ok=True)
            setup_logging(str((output_log_dir / "detail_run.log").resolve()))

        # Build a query that is naturally "paper search" oriented.
        date_str = datetime.now().strftime("%Y-%m-%d")
        q_text = (search_text or direction).strip()
        user_query = f"Find research papers about {q_text}. current date is {date_str}."
        # Print the exact search query before calling PasaMaster (green).
        print(f"\033[92m[PasaMaster search query]\033[0m {user_query}", flush=True)

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # Do NOT pass models via args; keep them fixed for stability.
        keyminder = Agent(RETRIEVAL_KEYMINDER_MODEL, config_path=str(CONFIG_PATH))
        top_ranker = Agent(RETRIEVAL_TOP_RANKER_MODEL, config_path=str(CONFIG_PATH))
        ranker = Agent(RETRIEVAL_RANKER_MODEL, config_path=str(CONFIG_PATH))

        quick_thinker = PaperAgent(
            task_id=f"dir_{re.sub(r'[^a-zA-Z0-9]+','_', direction)[:24]}",
            user_query=user_query,
            keyminder=keyminder,
            top_ranker=top_ranker,
            ranker=ranker,
            prompts_path=str(PROMPTS_PATH),
            expand_layers=expand_layers,
            search_queries=5,
            search_papers=10,
            expand_papers=20,
            threads_num=10,
            end_date=end_date,
            output_dir=str(output_log_dir) if output_log_dir is not None else str(REPO_ROOT / "result"),
            filter_score=filter_score,
            init_papers=init_papers,
            checklist=checklist,
        )

        reasoner = ReasonAgent(origin_query=user_query)
        coordinator = Manager(quick_thinker=quick_thinker, reason_agent_inst=reasoner)

        # Run minimal rounds (config.use_reasoner will still gate the slow agent)
        result = await coordinator.run(max_rounds=max_rounds)

    # Optionally truncate topk to reduce downstream load
    # Note: Only truncate if topk_return > 0; if topk_return == -1, return all papers
    try:
        extra = result.get("extra", {})
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            if topk_return > 0:
                extra["all_papers"] = extra["all_papers"][:topk_return]
            # If topk_return == -1, keep all papers (no truncation)
    except Exception:
        pass
    return result


def generate_complex_query(
    research_direction: str,
    paper_abstract: str,
    *,
    model_name: str = DEFAULT_LLM_MODEL,
) -> Dict[str, Any]:
    """
    Function3:
      input (paper abstract + research direction) -> generate a more detailed/complex retrieval query.
    """
    research_direction = (research_direction or "").strip()
    paper_abstract = (paper_abstract or "").strip()
    if not research_direction:
        raise ValueError("research_direction is empty")
    if not paper_abstract:
        paper_abstract = "(abstract missing)"

    sys.path.insert(0, str(REPO_ROOT))
    from model_vllm import Agent

    agent = Agent(model_name, config_path=str(CONFIG_PATH))

    def _empty_variants() -> List[Dict[str, Any]]:
        return [
            {"style": "A", "query": "", "checklist": []},
            {"style": "B", "query": "", "checklist": []},
            {"style": "C", "query": "", "checklist": []},
        ]

    def _validate_and_normalize(obj: Any) -> Optional[List[Dict[str, Any]]]:
        """Return normalized A/B/C variants if schema is valid; otherwise None."""
        if not isinstance(obj, dict):
            return None
        variants = obj.get("variants", None)
        if not isinstance(variants, list):
            return None

        norm: List[Dict[str, Any]] = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            style = str(v.get("style", "")).strip().upper()
            if style not in {"A", "B", "C"}:
                continue
            q = str(v.get("query", "")).strip().replace("\n", " ")
            # Guardrail: query must be plain text, not embedded JSON/code.
            if q.lstrip().startswith(("{", "[")) or ("[CLEANED_TOOL_RESPONSE]" in q):
                return None
            cl = v.get("checklist", [])
            if not isinstance(cl, list):
                cl = []
            cl2 = [str(x).strip() for x in cl if str(x).strip()]
            norm.append({"style": style, "query": q, "checklist": cl2})

        by_style = {str(x.get("style", "")).upper(): x for x in norm if isinstance(x, dict)}
        res_variants: List[Dict[str, Any]] = []
        for s in ["A", "B", "C"]:
            item = by_style.get(s) or {}
            res_variants.append(
                {
                    "style": s,
                    "query": str(item.get("query", "")).strip(),
                    "checklist": item.get("checklist", []) if isinstance(item.get("checklist", []), list) else [],
                }
            )
        # Require A query to exist (otherwise treat as invalid)
        if not str(res_variants[0].get("query", "")).strip():
            return None
        return res_variants
    prompt = f"""
You are an expert literature searcher.
Given a research direction and a seed paper abstract, generate THREE retrieval questions in English.
Each variant MUST contain:
1) "query": one continuous natural-language user question (1–3 sentences)
2) "checklist": a list of explicit constraints

The three variants MUST be:
- Variant A: multi-constraint, explicit, unambiguous
- Variant B: scenario / pain-point driven
- Variant C: exclusion / negative constraint (ONLY if the abstract clearly implies a real exclusion; otherwise write a query without exclusions but keep the style distinct)

The three variants should describe the same research direction, but with different styles and constraints.

Do NOT follow the reference examples' wording or sentence patterns. They are for intuition only.
Make the question forms as diverse as possible across A/B/C:
- vary the opening and voice (e.g., "Could you help...", "I'm trying to...", "What are papers that...", "Which methods address...", "Are there studies that...")
- vary clause structures (cause→effect, constraint stacking, contrast/except, requirement→evidence, evaluation-oriented, deployment-oriented, etc.)
- avoid repeating the same template across variants.

Ensure that:
- Variant A is general enough to typically retrieve at least **5 relevant papers**.
- All variants must use **2-5 domain-generic keywords** in the checklist, ensuring the retrieval is broad yet relevant.
- The "query" field must be plain English text and MUST NOT contain JSON/code (avoid '{{', '}}', '[' or ']' in the query).

Return STRICT JSON ONLY:
{{
  "variants": [
    {{
      "style": "A",
      "query": "...",
      "checklist": ["...", "..."]
    }},
    {{
      "style": "B",
      "query": "...",
      "checklist": ["...", "..."]
    }},
    {{
      "style": "C",
      "query": "...",
      "checklist": ["...", "..."]
    }}
  ]
}}

Reference examples (for style ONLY; do not copy entities unless supported by the abstract/direction):
Example A (multi-constraint, explicit, unambiguous):
"Please help me find papers on context management for LLM Agent when they are using tools. The methods must test on browsecomp benchmark."

Example B (scenario / pain-point driven):
"I'm building a single agent system with LLMs to use tools, but I find that the tool response is too long and it's hard to be used by the LLM Agent. Is there any paper that can help me to solve this problem?"

Example C (exclusion / negative constraint):
"Could you help check whether there are any papers that do context management for LLM Agent when they are using tools? It must be context management that solves the problem of long contexts caused by LLM agent calling tools, rather than simply reading long text and managing the context."

CRITICAL DESIGN RULES (HARD):
- Ask about a CLASS of papers, NOT this seed paper. Keep the query as general as possible.
- Do NOT mention seed identifiers (title/authors/venue/year/arXiv/DOI, "this paper").
- HIGH RECALL: Variant A must be general enough to typically retrieve >= 5 relevant papers.
- KEYWORDS: Each query must include 2-5 domain-generic KEYWORDS. 
- IMPORTANT: The "query" field must be plain English text and MUST NOT contain JSON/code (do not include '{{', '}}', '[' or ']' in the query).

Hard requirements:
- Output ONLY valid JSON.
- Checklist rules (for ALL variants):
  - Use EXACTLY 2-5 items.
  - Each item MUST express ONE verifiable requirement.
  - Each item MUST contain at least one keyword phrase.

Research direction: {research_direction}

Seed paper abstract:
{paper_abstract}
""".strip()

    out = agent.infer(prompt, use_cot=False)
    obj = _safe_json_extract(out or "")
    res_variants = _validate_and_normalize(obj)
    if res_variants is not None:
        return {"variants": res_variants, "_parsed_ok": True}

    # Parsing/validation failed: return empty variants and mark not ok (caller should NOT cache).
    return {"variants": _empty_variants(), "_parsed_ok": False}


async def run_pipeline(
    discipline_name: str,
    *,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_rounds: int = 1,
    output_dir: Optional[Path] = None,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    score_bin_size: int = SCORE_BIN_SIZE,
    per_bin: int = SAMPLES_PER_BIN,
    topic_file: Path = DEFAULT_TOPIC_FILE,
    force_topic_file: bool = False,
    concurrent_limit: int = 10,
    batch_retrieve: bool = True,
) -> Dict[str, Any]:
    """
    Function4:
      discipline -> directions -> papers -> complex queries
    """
    pipeline_start = time.time()
    stage_times: Dict[str, float] = {}
    
    _ensure_dir(dataset_dir)
    paths = dataset_paths(dataset_dir)

    discipline_to_directions: Dict[str, List[str]] = _read_json(paths["discipline_to_directions"], {})
    direction_to_papers: Dict[str, List[Dict[str, Any]]] = _read_json(paths["direction_to_papers"], {})
    # title -> {"query": str, "checklist": [str, ...], "select_score": float|null}
    title_to_complex_query: Dict[str, Any] = _read_json(paths["title_to_complex_query"], {})

    def normalize_query_key(query_text: str) -> str:
        """Normalize query text to ensure stable key (remove newlines, normalize whitespace)."""
        if not query_text:
            return ""
        return re.sub(r"\s+", " ", str(query_text).strip())

    # query_to_papers.jsonl: one record per complex-query group
    #   {"question": [A, B, C], "paper_list": [...]} where paper_list is retrieved by A only.
    # Resume uses a lightweight newline-delimited key file (normalized A query text).
    processed_keys = _read_lines_set(paths["query_to_papers_done"])

    # Best-effort rebuild of done keys if done file missing but jsonl exists.
    if not processed_keys and paths["query_to_papers"].exists():
        try:
            with paths["query_to_papers"].open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    q = obj.get("question", [])
                    if isinstance(q, list) and q:
                        qA = str(q[0] or "").strip()
                        qk = normalize_query_key(qA)
                        if qk:
                            processed_keys.add(qk)
            for k in sorted(processed_keys):
                _append_line(paths["query_to_papers_done"], k)
        except Exception:
            pass

    write_lock = asyncio.Lock()
    retrieve_semaphore = asyncio.Semaphore(concurrent_limit)
    retrieved_groups = [0]
    retrieved_papers_total = [0]

    def _questions_from_variants(variants: Any) -> List[str]:
        """Return [A, B, C] queries (may contain empty strings)."""
        if not isinstance(variants, list):
            return ["", "", ""]
        by_style = {
            str(v.get("style", "")).strip().upper(): v
            for v in variants
            if isinstance(v, dict)
        }
        return [
            str((by_style.get("A") or {}).get("query", "")).strip(),
            str((by_style.get("B") or {}).get("query", "")).strip(),
            str((by_style.get("C") or {}).get("query", "")).strip(),
        ]

    async def _retrieve_and_persist_group(question_list: List[str]) -> None:
        """
        Retrieve papers ONCE (Variant A query) and append one jsonl record:
          {"question": [A, B, C], "paper_list": [...]}
        Resume:
          - key = normalized Variant A query
          - if key exists in done.txt, skip
        Failure policy:
          - on exception, do NOT mark done (so rerun can retry)
        """
        if not isinstance(question_list, list) or not question_list:
            return
        qA = str((question_list[0] if len(question_list) > 0 else "") or "").strip()
        if not qA:
            # fallback: first non-empty query
            for q in question_list:
                if str(q or "").strip():
                    qA = str(q).strip()
                    break
        qk = normalize_query_key(qA)
        if not qk or qk in processed_keys:
            return

        async with retrieve_semaphore:
            res = await find_papers_for_direction(
                qA,
                search_text=None,
                max_rounds=max_rounds,
                topk_return=-1,
                output_log_dir=log_dir,
                expand_layers=2,
                filter_score=0,
            )
        extra = res.get("extra", {}) if isinstance(res, dict) else {}
        paper_list: List[Dict[str, Any]] = []
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            for p in extra["all_papers"]:
                if not isinstance(p, dict):
                    continue
                paper_list.append(
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "description": p.get("description", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "select_score": p.get("select_score", None),
                        "reason": p.get("reason", ""),
                        "source": p.get("source", ""),
                        "year": p.get("year", ""),
                        "authors": p.get("authors", ""),
                        "journal": p.get("journal", ""),
                        "doi": p.get("doi", ""),
                    }
                )

        record = {
            "question": [str(x or "").strip() for x in question_list],
            "paper_list": paper_list,
        }
        async with write_lock:
            _append_jsonl(paths["query_to_papers"], record)
            _append_line(paths["query_to_papers_done"], qk)
            processed_keys.add(qk)
            retrieved_groups[0] += 1
            retrieved_papers_total[0] += len(paper_list)

    # Stage 1: Load directions
    stage1_start = time.time()
    directions = discipline_to_directions.get(discipline_name) or []
    direction_desc_map: Dict[str, str] = {}
    if force_topic_file or not directions:
        direction_desc_map = load_directions_from_topic_file(discipline_name, topic_file=topic_file)
        directions = [k for k in direction_desc_map.keys() if k]
        if not directions:
            raise ValueError(
                f"No sub-directions found for discipline={discipline_name!r} in topic file: {topic_file}"
            )
        # write back to cache for future reuse (or overwrite when forced)
        discipline_to_directions[discipline_name] = directions
        _write_json(paths["discipline_to_directions"], discipline_to_directions)
    stage_times["Stage 1 (Load Directions)"] = time.time() - stage1_start

    out: Dict[str, Any] = {
        "discipline": discipline_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "directions": directions,
        "results": [],
    }

    log_dir = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir

    # Stage 2: Retrieve papers for each direction
    stage2_start = time.time()
    for d in directions:
        # Stage 2 cache: direction -> papers
        if d in direction_to_papers and isinstance(direction_to_papers[d], list) and direction_to_papers[d]:
            all_papers_cached = direction_to_papers[d]
        else:
            # If direction descriptions are available (from topic file), use them as search text
            # to make paper retrieval less ambiguous.
            search_text = None
            if not direction_desc_map:
                try:
                    direction_desc_map = load_directions_from_topic_file(discipline_name, topic_file=topic_file)
                except Exception:
                    direction_desc_map = {}
            if d in direction_desc_map and direction_desc_map[d]:
                search_text = direction_desc_map[d]
            res = await find_papers_for_direction(
                d,
                search_text=search_text,
                max_rounds=max_rounds,
                topk_return=-1,
                output_log_dir=log_dir,
                expand_layers=1,
            )
            extra = res.get("extra", {}) if isinstance(res, dict) else {}
            all_papers_cached = []
            if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
                # store only fields we need (keep cache smaller and stable)
                for p in extra["all_papers"]:
                    if not isinstance(p, dict):
                        continue
                    all_papers_cached.append(
                        {
                            "title": p.get("title", ""),
                            "abstract": p.get("abstract", ""),
                            "description": p.get("description", ""),
                            "arxiv_id": p.get("arxiv_id", ""),
                            "select_score": p.get("select_score", None),
                            "reason": p.get("reason", ""),
                            "source": p.get("source", ""),
                            "year": p.get("year", ""),
                            "authors": p.get("authors", ""),
                            "journal": p.get("journal", ""),
                            "doi": p.get("doi", ""),
                        }
                    )
            direction_to_papers[d] = all_papers_cached
            _write_json(paths["direction_to_papers"], direction_to_papers)
    stage_times["Stage 2 (Retrieve Papers for Directions)"] = time.time() - stage2_start

    # Stage 3: Generate complex queries
    stage3_start = time.time()
    for d in directions:
        # Get cached papers for this direction
        all_papers_cached = direction_to_papers.get(d, [])
        if not isinstance(all_papers_cached, list):
            all_papers_cached = []

        # Select papers by score buckets: 10 points per bucket, 2 papers per bucket by default.
        selected_papers = _select_by_score_buckets(
            [p for p in all_papers_cached if isinstance(p, dict)],
            bin_size=score_bin_size,
            per_bin=per_bin,
        )

        complex_queries = []
        for p in selected_papers:
            abstract = (p.get("abstract") or p.get("description") or "").strip()
            title = (p.get("title") or "").strip()
            sc = _safe_float(p.get("select_score", None), -1.0)
            bs = _bucket_start(sc, bin_size=score_bin_size)
            bucket_label = f"{bs}-{bs+score_bin_size}"
            cached = title_to_complex_query.get(title) if title else None
            if isinstance(cached, dict) and isinstance(cached.get("variants"), list):
                complex_obj = cached
            elif isinstance(cached, dict) and isinstance(cached.get("query"), str):
                # legacy cache: {"query": str, "checklist": [...]}
                q_legacy = str(cached.get("query", "")).strip()
                cl_legacy = cached.get("checklist", [])
                if not isinstance(cl_legacy, list):
                    cl_legacy = []
                complex_obj = {
                    "variants": [
                        {"style": "A", "query": q_legacy, "checklist": cl_legacy},
                        {"style": "B", "query": "", "checklist": []},
                        {"style": "C", "query": "", "checklist": []},
                    ],
                    "select_score": cached.get("select_score", None),
                }
            elif isinstance(cached, str) and cached.strip():
                # legacy cache: value is a query string
                complex_obj = {
                    "variants": [
                        {"style": "A", "query": cached.strip(), "checklist": []},
                        {"style": "B", "query": "", "checklist": []},
                        {"style": "C", "query": "", "checklist": []},
                    ],
                    "select_score": None,
                }
            else:
                gen = generate_complex_query(d, abstract, model_name=llm_model)
                complex_obj = {
                    "variants": gen.get("variants", []) if isinstance(gen, dict) else [],
                    "select_score": sc if sc >= 0 else None,
                }
                if title:
                    # Only cache if parsing/validation succeeded.
                    if isinstance(gen, dict) and bool(gen.get("_parsed_ok", False)):
                        title_to_complex_query[title] = complex_obj
                        _write_json(paths["title_to_complex_query"], title_to_complex_query)
            complex_queries.append(
                {
                    "paper_title": p.get("title", ""),
                    "paper_arxiv_id": p.get("arxiv_id", ""),
                    "paper_select_score": p.get("select_score", None),
                    "score_bucket": bucket_label,
                    "variants": complex_obj.get("variants", []),
                }
            )

            # Streaming mode: retrieve papers immediately per complex-query GROUP.
            # - "question" is [A,B,C]
            # - "paper_list" is retrieved ONLY by Variant A
            # - resume via query_to_papers.done.txt (normalized A query key)
            if not batch_retrieve:
                q_list = _questions_from_variants(complex_obj.get("variants", []))
                try:
                    await _retrieve_and_persist_group(q_list)
                except Exception as e:
                    # Do not crash the whole pipeline; missing groups can be retried in next run.
                    print(f"  [STREAM_RETRIEVE_ERROR] {e}", flush=True)

        out["results"].append(
            {
                "direction": d,
                "selected_papers": [
                    {
                        "title": p.get("title", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "select_score": p.get("select_score", None),
                        "abstract": p.get("abstract", ""),
                        "source": p.get("source", ""),
                    }
                    for p in selected_papers
                ],
                "complex_queries": complex_queries,
            }
        )

    stage_times["Stage 3 (Generate Complex Queries)"] = time.time() - stage3_start

    # Stage 3 summary statistics
    total_directions = len(out["results"])
    total_selected_papers = sum(len(r.get("selected_papers", [])) for r in out["results"])
    total_complex_queries = sum(
        sum(len(cq.get("variants", [])) for cq in r.get("complex_queries", []))
        for r in out["results"]
    )
    print(f"\n[Stage 3 Summary]")
    print(f"  - Total directions: {total_directions}")
    print(f"  - Total selected papers: {total_selected_papers}")
    print(f"  - Total complex queries (variants): {total_complex_queries}", flush=True)

    # Stage 4: For each complex query variant, retrieve papers via PasaMaster
    # (supports resume: skip queries that already exist in cache, concurrent with limit)
    stage4_start = time.time()
    if not batch_retrieve:
        # In streaming mode, retrieval is already done during Stage 3.
        print("\n[Stage 4] Skipped (stream mode: retrieve during Stage 3).", flush=True)
        stage_times["Stage 4 (Retrieve Papers for Complex Queries)"] = time.time() - stage4_start
        # Continue to final summary
        stage4_time = time.time() - stage4_start
        retrieved_count = 0
    else:
        print(
            f"\n[Stage 4] Retrieving papers for complex-query groups (Variant A only, concurrent limit: {concurrent_limit})...",
            flush=True,
        )
    
    if batch_retrieve:
        groups_to_retrieve: List[List[str]] = []
        skipped_groups = 0
        scheduled_keys = set(processed_keys)
        for result_item in out["results"]:
            for cq_item in result_item.get("complex_queries", []):
                variants = cq_item.get("variants", [])
                q_list = _questions_from_variants(variants)
                qA_key = normalize_query_key(q_list[0] if q_list else "")
                if not qA_key:
                    continue
                if qA_key in scheduled_keys:
                    skipped_groups += 1
                    continue
                scheduled_keys.add(qA_key)
                groups_to_retrieve.append(q_list)

        if skipped_groups > 0:
            print(f"  [Resume] Found {skipped_groups} processed groups, will skip them", flush=True)
        print(f"  [Batch] Will retrieve {len(groups_to_retrieve)} groups (Variant A only)", flush=True)

        if groups_to_retrieve:
            await asyncio.gather(*[_retrieve_and_persist_group(g) for g in groups_to_retrieve])

        stage4_time = time.time() - stage4_start
        retrieved_count = len(groups_to_retrieve)
        stage_times["Stage 4 (Retrieve Papers for Complex Queries)"] = stage4_time

    # Final summary statistics
    pipeline_total_time = time.time() - pipeline_start
    total_papers_in_query_results = int(retrieved_papers_total[0])
    avg_papers_per_group = (
        total_papers_in_query_results / int(retrieved_groups[0])
        if int(retrieved_groups[0]) > 0
        else 0.0
    )

    print(f"\n{'='*60}")
    print(f"[Pipeline Summary]")
    print(f"  - Total directions: {total_directions}")
    print(f"  - Total selected papers: {total_selected_papers}")
    print(f"  - Total complex queries (variants): {total_complex_queries}")
    print(f"  - Retrieved groups (this run): {int(retrieved_groups[0])}")
    print(f"  - Average papers per group (this run): {avg_papers_per_group:.2f}")
    print(f"\n[Time Breakdown]")
    for stage_name, stage_time in stage_times.items():
        print(f"  - {stage_name}: {stage_time:.2f}s ({stage_time/pipeline_total_time*100:.1f}%)")
    print(f"  - Total pipeline time: {pipeline_total_time:.2f}s")
    print(f"{'='*60}", flush=True)

    if output_dir is not None:
        (output_dir / "pipeline_output.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return out


def main() -> None:
    """
    CLI entrypoint with subcommands to test each module independently.

    Examples:
      - list disciplines:
          python query_pipeline.py list-disciplines
      - test function1:
          python query_pipeline.py directions --discipline "Computer science"
      - test function2:
          python query_pipeline.py papers --direction "large language model agents" --max-rounds 1
      - test function3:
          python query_pipeline.py complex-query --direction "LLM agents" --abstract-file /path/to/abs.txt
      - run full pipeline (function4):
          python query_pipeline.py full --discipline "Computer science" --output-dir /tmp/out
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-disciplines", help="Print discipline name list (English).")

    p_dir = sub.add_parser("directions", help="Function1: discipline -> research directions.")
    p_dir.add_argument("--discipline", type=str, required=True)
    p_dir.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="LLM for topic generation.")
    p_dir.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    p_dir.add_argument(
        "--topic-file",
        type=str,
        default=str(DEFAULT_TOPIC_FILE),
        help="If provided with --use-topic-file, load sub-directions from this JSON file.",
    )
    p_dir.add_argument(
        "--use-topic-file",
        action="store_true",
        help="If set, do NOT call LLM; directly load sub-directions from topic file and cache them.",
    )

    p_papers = sub.add_parser("papers", help="Function2: direction -> find related papers via PasaMaster.")
    p_papers.add_argument("--direction", type=str, required=True)
    p_papers.add_argument("--max-rounds", type=int, default=1)
    p_papers.add_argument("--topk-return", type=int, default=-1)
    p_papers.add_argument("--output-dir", type=str, default=None, help="If set, write papers_result.json and logs there.")
    p_papers.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    p_papers.add_argument("--expand-layers", type=int, default=2)

    p_cq = sub.add_parser("complex-query", help="Function3: (direction + abstract) -> complex query.")
    p_cq.add_argument("--direction", type=str, required=True)
    p_cq.add_argument("--paper-title", type=str, default=None, help="If set, cache by this title as key.")
    p_cq.add_argument("--abstract", type=str, default=None, help="Abstract text.")
    p_cq.add_argument("--abstract-file", type=str, default=None, help="Path to a file containing abstract.")
    p_cq.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="LLM for complex query generation.")
    p_cq.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))

    p_full = sub.add_parser("full", help="Function4: discipline -> directions -> papers -> complex queries.")
    p_full.add_argument("--discipline", type=str, required=True)
    p_full.add_argument("--output-dir", type=str, default=None)
    p_full.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="LLM for topic+query generation.")
    p_full.add_argument("--max-rounds", type=int, default=1)
    p_full.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    p_full.add_argument("--score-bin-size", type=int, default=SCORE_BIN_SIZE, help="Score bucket size (default 10).")
    p_full.add_argument("--per-bin", type=int, default=SAMPLES_PER_BIN, help="Papers per score bucket (default 2).")
    p_full.add_argument(
        "--topic-file",
        type=str,
        default=str(DEFAULT_TOPIC_FILE),
        help="Topic file to load discipline sub-directions when cache is missing.",
    )
    p_full.add_argument(
        "--force-topic-file",
        action="store_true",
        help="If set, ALWAYS load sub-directions from topic file (ignore cached directions).",
    )
    p_full.add_argument(
        "--concurrent-limit",
        type=int,
        default=10,
        help="Maximum concurrent queries for Stage 4 paper retrieval (default 10).",
    )
    p_full.add_argument(
        "--batch-retrieve",
        type=str,
        default="true",
        help=(
            "Whether to retrieve papers ONLY after all complex queries are generated. "
            "Set to 'true' (default) for batch mode (Stage 3 then Stage 4). "
            "Set to 'false' for stream mode (retrieve immediately per complex query during Stage 3)."
        ),
    )

    args = parser.parse_args()

    if args.cmd == "list-disciplines":
        print(json.dumps(get_discipline_list(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "directions":
        if args.use_topic_file:
            mapping = load_directions_from_topic_file(args.discipline, topic_file=Path(args.topic_file))
            dirs = [k for k in mapping.keys() if k]
        else:
            dirs = generate_research_directions(args.discipline, model_name=args.llm_model)
        ds_dir = Path(args.dataset_dir)
        _ensure_dir(ds_dir)
        paths = dataset_paths(ds_dir)
        discipline_to_directions = _read_json(paths["discipline_to_directions"], {})
        discipline_to_directions[args.discipline] = dirs
        _write_json(paths["discipline_to_directions"], discipline_to_directions)
        print(json.dumps(dirs, ensure_ascii=False, indent=2))
        return

    if args.cmd == "papers":
        import asyncio

        out_dir = Path(args.output_dir) if args.output_dir else None
        res = asyncio.run(
            find_papers_for_direction(
                args.direction,
                max_rounds=args.max_rounds,
                topk_return=args.topk_return,
                output_log_dir=out_dir,
                expand_layers=args.expand_layers,
            )
        )
        # cache: direction -> papers
        ds_dir = Path(args.dataset_dir)
        _ensure_dir(ds_dir)
        paths = dataset_paths(ds_dir)
        direction_to_papers = _read_json(paths["direction_to_papers"], {})
        extra = res.get("extra", {}) if isinstance(res, dict) else {}
        all_papers_cached = []
        if isinstance(extra, dict) and isinstance(extra.get("all_papers"), list):
            for p in extra["all_papers"]:
                if not isinstance(p, dict):
                    continue
                all_papers_cached.append(
                    {
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "description": p.get("description", ""),
                        "arxiv_id": p.get("arxiv_id", ""),
                        "select_score": p.get("select_score", None),
                        "source": p.get("source", ""),
                        "year": p.get("year", ""),
                        "authors": p.get("authors", ""),
                        "journal": p.get("journal", ""),
                    }
                )
        direction_to_papers[args.direction] = all_papers_cached
        _write_json(paths["direction_to_papers"], direction_to_papers)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "papers_result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(res.get("extra", {}).get("final_checklist", []), ensure_ascii=False, indent=2))
        return

    if args.cmd == "complex-query":
        abstract = args.abstract
        if abstract is None and args.abstract_file:
            abstract = Path(args.abstract_file).read_text(encoding="utf-8")
        if abstract is None:
            raise SystemExit("Provide --abstract or --abstract-file")
        obj = generate_complex_query(args.direction, abstract, model_name=args.llm_model)
        if args.paper_title:
            ds_dir = Path(args.dataset_dir)
            _ensure_dir(ds_dir)
            paths = dataset_paths(ds_dir)
            title_to_complex_query = _read_json(paths["title_to_complex_query"], {})
            # Only cache if parsing/validation succeeded.
            if isinstance(obj, dict) and bool(obj.get("_parsed_ok", False)):
                title_to_complex_query[args.paper_title] = {"variants": obj.get("variants", [])}
                _write_json(paths["title_to_complex_query"], title_to_complex_query)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return

    if args.cmd == "full":
        import asyncio

        out_dir = Path(args.output_dir) if args.output_dir else None
        res = asyncio.run(
            run_pipeline(
                args.discipline,
                llm_model=args.llm_model,
                max_rounds=args.max_rounds,
                output_dir=out_dir,
                dataset_dir=Path(args.dataset_dir),
                score_bin_size=args.score_bin_size,
                per_bin=args.per_bin,
                topic_file=Path(args.topic_file),
                force_topic_file=bool(args.force_topic_file),
                concurrent_limit=args.concurrent_limit,
                batch_retrieve=_parse_bool(getattr(args, "batch_retrieve", True), default=True),
            )
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()


