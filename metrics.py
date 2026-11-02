# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import json
import glob
import argparse
import re
import sys
import concurrent.futures
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from rapidfuzz import fuzz, process  # fuzzy matching

# Toggle here whether to print intermediate results: True/False
VERBOSE = False

# High-score paper threshold: score >= 70
HIGH_SCORE_THRESHOLD = 80.0

# Fuzzy match similarity threshold: >= 0.9 is considered a match
FUZZY_MATCH_THRESHOLD = 0.85

# Matching threshold between query and bench question: similarity > 0.9 is treated as the same query
QUERY_MATCH_THRESHOLD = 0.9

# When the number of evaluation JSON files in the folder reaches this threshold, aggregate Top20 metrics by subject from the bench JSONL
SUBJECT_BREAKDOWN_MIN_JSON_FILES = 200

DEFAULT_BENCH_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pasamaster_bench.jsonl")

# Per-subject high-score subset summary: subject output order for the terminal and by_subject_high_score
SUBJECT_OUTPUT_ORDER = (
    "Artificial Intelligence and Computing",
    "Engineering and Technology",
    "Medicine and Life Sciences",
    "Fundamental Sciences",
    "Other",
)


def iter_subjects_preferred(subject_keys):
    """First follow SUBJECT_OUTPUT_ORDER, then append remaining subjects in alphabetical order at the end."""
    keys = set(subject_keys)
    seen = set()
    for s in SUBJECT_OUTPUT_ORDER:
        if s in keys:
            seen.add(s)
            yield s
    for s in sorted(k for k in keys if k not in seen):
        yield s


# ==================== LLM judge (optional) ====================

_TAG_RE_CACHE = {}


def _extract_tag(text: str, tag: str):
    """
    Extract content from output of the form <tag>...</tag>. Returns None if not found.
    """
    if not text:
        return None
    pat = _TAG_RE_CACHE.get(tag)
    if pat is None:
        pat = re.compile(
            rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE
        )
        _TAG_RE_CACHE[tag] = pat
    m = pat.search(text)
    return m.group(1).strip() if m else None


def _parse_constraint_scores(output_text: str):
    """
    Parse each constraint's Score:<1-5> from the <constraint> section of point_wise_rank_english.
    Returns (scores:list[int], avg:float|None).
    """
    constraint_text = _extract_tag(output_text or "", "constraint") or ""
    # Handle both Score:5 and Score: 5
    scores = [int(x) for x in re.findall(r"Score\s*:\s*([1-5])", constraint_text)]
    if not scores:
        return [], None
    return scores, sum(scores) / len(scores)

def _try_float(x):
    """
    Try to convert x to float.
    - Success: return float
    - Failure/missing: return None
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _truncate_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if not isinstance(max_chars, int) or max_chars <= 0:
        return ""
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]..."


def _build_paper_meta_info_value(paper: dict) -> dict:
    """
    Align with quick_thinker: the LLM judge's meta_info only provides "metadata values" and does not mix in chunk evidence text.
    """
    if not isinstance(paper, dict):
        return {}
    return {
        "authors": paper.get("authors", None),
        "publication_date": paper.get("year", None),
        "citations": paper.get("citations", None),
        "journal": paper.get("journal", None),
        "CCF": paper.get("CCF", None),
        "url": paper.get("url", None),
    }





def _load_pointwise_prompt_template(prompt_path: str):
    """
    Read ranker.point_wise_rank_english from papermaster_prompt.json.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["ranker"]["point_wise_rank_english"]


def _format_pointwise_prompt(
    template: str,
    *,
    title: str,
    abstract: str,
    user_query: str,
    constraints,
    meta_info: str = "",
):
    if constraints is None:
        constraints = []
    if isinstance(constraints, (list, tuple)):
        # Align with quick_thinker: numbering + line breaks
        lines = []
        for idx, item in enumerate(constraints):
            s = str(item).strip()
            if not s:
                continue
            lines.append(f"{idx+1}. {s}")
        constraints_str = "\n".join(lines)
    else:
        constraints_str = str(constraints).strip()
    return template.format(
        title=title or "",
        abstract=abstract or "",
        user_query=user_query or "",
        meta_info=meta_info or "",
        constraints=constraints_str,
    )


def _ensure_pasa_on_path():
    """
    Allow `tools.llm_call` to be imported even when metrics.py is run as a script.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)


def _llm_judge_candidates(
    *,
    candidates,  # list[paper_data dict] with title/abstract
    user_query: str,
    constraints,
    prompt_template: str,
    model_name: str,
    timeout: int,
    max_papers: int,
    concurrency: int = 50,
    max_meta_chars: int = 6000,
):
    """
    Perform per-constraint LLM scoring (1-5) on candidate papers.

    Returns:
      - judge5_titles: list[str] (non-GT papers with "average score = 5")
      - judge4_titles: list[str] (non-GT papers with "average score = 4")
      - judge_detail: dict[norm_title] -> {avg, scores, raw}

    Note: per the requirement, "average = 5" is interpreted strictly: all constraint scores are 5 (otherwise the average cannot be 5).
    """
    if not candidates:
        return [], [], {}

    _ensure_pasa_on_path()
    from tools.llm_call import llm_call  # lazy import to avoid pulling in the dependency when disabled

    # Limit the maximum number of calls to avoid an explosion of calls in a single evaluation
    if isinstance(max_papers, int) and max_papers > 0:
        candidates = candidates[:max_papers]

    judge5_titles: list[str] = []
    judge4_titles: list[str] = []
    judge_detail: dict = {}

    def _judge_one(paper: dict):
        title = (paper.get("title") or "").strip()
        if not title:
            return None
        abstract = (paper.get("abstract") or "").strip()
        description = (paper.get("description") or "").strip()
        # Align with quick_thinker: the abstract input includes description (chunk)
        if description:
            description = _truncate_text(description, max_chars=int(max_meta_chars))
            abstract = (abstract + "\n\n" + description).strip()

        # Align with quick_thinker: meta_info only provides metadata "values"
        meta_info_value = _build_paper_meta_info_value(paper)
        meta_info = json.dumps(meta_info_value, ensure_ascii=False)

        prompt = _format_pointwise_prompt(
            prompt_template,
            title=title,
            abstract=abstract,
            user_query=user_query,
            constraints=constraints,
            meta_info=meta_info,
        )
        try:
            out = llm_call(prompt, model_name=model_name, timeout=timeout)
        except Exception as e:
            out = f"<error>{type(e).__name__}: {e}</error>"

        scores, avg_score = _parse_constraint_scores(out)
        llm_reason = out
        llm_relevant = _extract_tag(out or "", "relevant") or ""
        norm_t = normalize_title(title)

        judge5 = bool(scores) and all(s == 5 for s in scores)
        avg_round = None
        if scores and avg_score is not None:
            avg_round = int(math.floor(avg_score + 0.5))

        return norm_t, title, scores, avg_score, avg_round, llm_reason, llm_relevant, out, judge5

    # Concurrency cap is fixed at 50 (per the requirement)
    try:
        max_workers = int(concurrency)
    except Exception:
        max_workers = 50
    if max_workers <= 0:
        max_workers = 50
    max_workers = min(50, max_workers, max(1, len(candidates)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_judge_one, p) for p in candidates]
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception:
                continue
            if not res:
                continue
            (
                norm_t,
                title,
                scores,
                avg_score,
                avg_round,
                llm_reason,
                llm_relevant,
                out,
                judge5,
            ) = res
            judge_detail[norm_t] = {
                "avg": avg_score,
                "avg_round": avg_round,
                "scores": scores,
                "llm_reason": llm_reason,
                "llm_relevant": llm_relevant,
                "raw": out,
            }

            if judge5:
                judge5_titles.append(title)
            elif scores and avg_round == 4:
                judge4_titles.append(title)

    return judge5_titles, judge4_titles, judge_detail


# ==================== Utility functions ====================


def dcg_at_k(rels, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(ranked_list, relevant_set, k):
    """
    Compute NDCG@k based on ranked_list (a pre-sorted list of normalized titles)
    and relevant_set (the normalized set of answers).

    To avoid "double-counting" when the same answer appears multiple times,
    each relevant item is counted as a hit only once here.
    """
    hits = []
    seen_rels = set()
    for x in ranked_list[:k]:
        if x in relevant_set and x not in seen_rels:
            hits.append(1)
            seen_rels.add(x)
        else:
            hits.append(0)

    dcg = dcg_at_k(hits, k)

    m = min(k, len(relevant_set))
    ideal = [1] * m + [0] * max(0, k - m)
    idcg = dcg_at_k(ideal, k)

    return 0.0 if idcg == 0 else dcg / idcg


def normalize_title(s: str) -> str:
    """
    All title matching goes through here:
    - Remove all spaces / newlines / tabs and other isspace() characters
    - Convert everything to lowercase
    """
    if s is None:
        return ""
    s = s.lower()
    return "".join(ch for ch in s if not ch.isspace())


def cal_micro(pred_set, label_set):
    """Compute TP / FP / FN and their corresponding sets"""
    if len(label_set) == 0:
        return 0, 0, 0, set(), set(), set()

    if len(pred_set) == 0:
        return 0, 0, len(label_set), set(), set(), label_set

    tp_set = pred_set & label_set
    fp_set = pred_set - label_set
    fn_set = label_set - pred_set

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    assert tp + fn == len(label_set)
    return tp, fp, fn, tp_set, fp_set, fn_set


def safe_div(a, b):
    return a / (b if b > 0 else 1e-9)

def f1_score(precision: float, recall: float) -> float:
    """
    F1 = 2PR/(P+R); returns 0 if P+R=0.
    """
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else 0.0


def pct_score(x):
    """
    Convert a 0~1 evaluation metric to a 0~100 score (both the terminal and eval_result use the percentage scale consistently).
    """
    if x is None:
        return None
    try:
        return round(float(x) * 100.0, 2)
    except (TypeError, ValueError):
        return x


def parse_year_to_date(year_str):
    """
    Parse the year field into a date object.
    Supports multiple formats: '2023-09-19', '2023-09-8', '2023', '2023-09', etc.
    Returns a datetime object, or None if parsing fails.
    """
    if not year_str or not isinstance(year_str, str):
        return None

    year_str = year_str.strip()

    # Try to parse full date formats: YYYY-MM-DD or YYYY-M-D
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(year_str, fmt)
        except ValueError:
            continue

    return None


def is_year_after_end_date(year_str, end_date):
    """
    Check whether year_str is later than end_date.
    year_str: the paper's year field (string)
    end_date: the cutoff date (datetime object or None)
    Returns True if year_str is later than end_date, otherwise False.
    If end_date is None, returns False (no filtering).
    """
    if end_date is None:
        return False

    paper_date = parse_year_to_date(year_str)
    if paper_date is None:
        # If the date cannot be parsed, do not filter by default (keep this paper)
        return False

    return paper_date > end_date


def build_fuzzy_alias(pred_titles, answer_titles, threshold=FUZZY_MATCH_THRESHOLD):
    """
    Find the most similar answer title for each predicted title:
    - When similarity >= threshold (default 0.9), treat "this prediction as hitting that answer"
    - Both input and output use "normalized titles"
    - Returns dict: pred_norm_title -> (answer_norm_title, similarity)
    """
    alias = {}
    for p in pred_titles:
        if not answer_titles:
            break
        best_match = process.extractOne(p, answer_titles, scorer=fuzz.ratio)
        if not best_match:
            continue
        best_ans, score, _ = best_match
        sim = score / 100.0
        if sim >= threshold:
            alias[p] = (best_ans, sim)
    return alias


def count_eval_json_files(input_folder: str) -> int:
    """
    Count the number of *.json files in the folder that are treated as "evaluation results" (files whose names contain eval are excluded, consistent with the main loop in evaluate_folder).
    """
    pred_files = glob.glob(os.path.join(input_folder, "*.json"))
    n = 0
    for p in pred_files:
        if "eval" in os.path.basename(p).lower():
            continue
        n += 1
    return n


def normalize_query_text(s: str) -> str:
    """
    Loosely normalize the query to make matching easier:
    - Remove appended info of the form "current date is YYYY-MM-DD."
    - Collapse extra whitespace
    - Convert to lowercase
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(
        r"\bcurrent\s+date\s+is\s+\d{4}-\d{2}-\d{2}\.\s*",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    return " ".join(s.lower().split())


def extract_numeric_qid(qid_value):
    """
    Extract an integer id from qid, supporting:
    - pure numeric string / integer
    - pasamasterbench_123
    """
    if qid_value is None:
        return None
    if isinstance(qid_value, int):
        return qid_value
    s = str(qid_value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.search(r"(\d+)$", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_bench_index(bench_jsonl_path: str) -> dict:
    """
    Read the PaSaMaster bench JSONL and build:
    - line_to_subject: line number (0-based) -> subject
    - query_to_entry: normalized query -> same as above (exact match)
    - query_entries: [(normalized query, entry)] (used for similarity matching)
    """
    bench_index = {
        "line_to_subject": {},
        "query_to_entry": {},
        "query_entries": [],
    }
    if not bench_jsonl_path or not os.path.isfile(bench_jsonl_path):
        return bench_index

    with open(bench_jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = (obj.get("question") or "").strip()
            answer = obj.get("answer", []) or []
            if not isinstance(answer, list):
                answer = [answer]

            sm = obj.get("subject_meta")
            subj = None
            if isinstance(sm, dict):
                subj = sm.get("subject")
            if not subj:
                subj = "unknown"
            else:
                subj = str(subj).strip() or "unknown"
            bench_index["line_to_subject"][i] = subj

            entry = {
                "question": question,
                "answer": answer,
                "subject": subj,
                "bench_qid": obj.get("qid"),
                "line_idx": i,
            }

            norm_query = normalize_query_text(question)
            if norm_query and norm_query not in bench_index["query_to_entry"]:
                bench_index["query_to_entry"][norm_query] = entry
            if norm_query:
                bench_index["query_entries"].append((norm_query, entry))

    return bench_index


def load_bench_line_index_to_subject(bench_jsonl_path: str) -> dict:
    """
    Read the PaSaMaster bench JSONL: the line number (0-based) corresponds to N in the result file name N.json.
    Each line uses subject_meta.subject as the subject name.
    """
    return load_bench_index(bench_jsonl_path).get("line_to_subject", {})


def resolve_bench_entry_for_query(query_text: str, bench_index: dict):
    """
    Match the bench by normalized query text:
    1. First try an exact match
    2. Then try a similarity match, requiring similarity > QUERY_MATCH_THRESHOLD
       If there are multiple hits, take the one with the highest similarity
    """
    if not isinstance(bench_index, dict) or not bench_index:
        return None, None

    norm_query = normalize_query_text(query_text)
    query_to_entry = bench_index.get("query_to_entry") or {}
    if norm_query and norm_query in query_to_entry:
        return query_to_entry[norm_query], "query"

    best_entry = None
    best_score = -1.0
    for bench_norm_query, entry in bench_index.get("query_entries") or []:
        if not bench_norm_query:
            continue
        sim = fuzz.ratio(norm_query, bench_norm_query) / 100.0
        if sim > QUERY_MATCH_THRESHOLD and sim > best_score:
            best_entry = entry
            best_score = sim
    if best_entry is not None:
        return best_entry, "query_fuzzy"

    return None, None


def subject_for_result_qid(qid: int, line_to_subject: dict) -> str:
    """Look up the subject from the bench line mapping using the result file qid (integer); unknown if there is no mapping."""
    if not line_to_subject:
        return "unknown"
    return line_to_subject.get(int(qid), "unknown")


# ==================== Single-query evaluation logic ====================


def evaluate_query(paper_root, file_name, qid, args, bench_index=None):
    """
    Evaluate a single JSON result file and return:
      - query_detail: the detailed results for this query
      - stats: this entry's contribution to the global statistics (various lists)
      - If this query is not matched in the bench, return (None, None) to indicate skipping this sample

    Args:
        paper_root: the root object of the JSON result file
        file_name: the file name
        qid: query ID
        args.end_date: the cutoff date (datetime object); if a paper's year is later than this date it is excluded from the computation
        args.sn_flag: if False, filter out papers whose source field contains "sn"
        args.llm_judge: whether to enable the LLM judge (default False)
    """
    extra = paper_root.get("extra", {})
    query_text = (paper_root.get("query") or "").strip()

    # ---------- Ground truth ----------
    bench_entry, bench_match_type = resolve_bench_entry_for_query(
        query_text, bench_index
    )
    if bench_entry is not None:
        gt_answer_list_original = bench_entry.get("answer", []) or []
        gt_source = f"bench:{bench_match_type}"
        bench_subject = (bench_entry.get("subject") or "unknown").strip() or "unknown"
    else:
        print(f"⚠️ GT not found in bench: file={file_name}, query={query_text}")
        return None, None
    gt_answer_set = set(normalize_title(p) for p in gt_answer_list_original)
    normalized_answer_to_original = {
        normalize_title(ans): ans for ans in gt_answer_list_original
    }
    gt_answer_count = len(gt_answer_set)

    # The "effective answer set" used for LLM expansion (defaults to GT)
    effective_answer_set = set(gt_answer_set)
    llm_judge_5_titles = []
    llm_judge_4_titles = []
    llm_judge_detail = {}
    llm_judge_candidates_count = 0
    any_score_present = False  # whether this query has any usable score
    all_scores_missing = True  # whether all papers in this query have no score (or cannot be parsed)
    high_score_use_all_papers = False

    # ---------- Retrieved papers (used for @5/@10/@20) ----------
    crawled_papers = []  # [ [norm_title, score], ... ]
    crawled_paper_set = set()
    norm_to_original = {}  # norm_title -> original title
    norm_to_paper = {}  # norm_title -> paper_data (used for LLM judge)

    # High-score papers (score >= 80)
    high_score_titles = []  # [ (norm_title, score), ... ]
    high_score_titles_original = []  # original title

    # Raw all_papers records (used later for fuzzy-match hit statistics)
    all_papers_raw = []  # [ (norm_title, original_title, score), ... ]
    all_papers_hit_titles_original = set()
    hit_in_gt_titles_original = set()
    hit_llm5_titles_original = set()

    # Prefer the new framework's all_papers
    if "all_papers" in extra and extra["all_papers"]:
        for paper_data in extra["all_papers"]:
            # Check the year field; skip if it is later than end_date
            year_str = paper_data.get("year", "") or ""
            if is_year_after_end_date(year_str, args.end_date):
                continue

            # Check the source field; skip if sn_flag is False and source contains "sn"
            if not args.sn_flag:
                source_str = paper_data.get("source", "") or ""
                if source_str and "sn" in source_str.lower():
                    continue

            original_title = paper_data.get("title", "") or ""
            norm_title = normalize_title(original_title)
            score_raw = paper_data.get("select_score", None)
            score_f = _try_float(score_raw)
            if score_f is not None:
                any_score_present = True
                all_scores_missing = False
            # Numeric score used for sorting/threshold checks (use -1 if missing)
            score = score_f if score_f is not None else -1.0

            if norm_title:
                # Record a paper_data for the LLM judge (keep the one with the higher select_score when possible)
                prev = norm_to_paper.get(norm_title)
                if prev is None or (prev.get("select_score", -1.0) or -1.0) < score:
                    norm_to_paper[norm_title] = paper_data

            # Record the raw all_papers info
            all_papers_raw.append((norm_title, original_title, score))

            if norm_title not in crawled_paper_set:
                crawled_paper_set.add(norm_title)
                crawled_papers.append([norm_title, score])
                norm_to_original[norm_title] = original_title

            # High-score papers
            if score_f is None:
                # When score is missing, do not decide here whether to include in the high-score set; handle it uniformly later
                pass
            elif score_f >= HIGH_SCORE_THRESHOLD:
                high_score_titles.append((norm_title, score))
                high_score_titles_original.append(original_title)

    elif "sorted_papers" in extra and extra["sorted_papers"]:
        # Degenerate case: only sorted_papers is available, use this set to compute the basic metrics
        for paper_data in extra["sorted_papers"]:
            # Check the year field; skip if it is later than end_date
            year_str = paper_data.get("year", "") or ""
            if is_year_after_end_date(year_str, args.end_date):
                continue

            # Check the source field; skip if sn_flag is False and source contains "sn"
            if not args.sn_flag:
                source_str = paper_data.get("source", "") or ""
                if source_str and "sn" in source_str.lower():
                    continue

            original_title = paper_data.get("title", "") or ""
            norm_title = normalize_title(original_title)
            score_raw = paper_data.get("select_score", None)
            score_f = _try_float(score_raw)
            if score_f is not None:
                any_score_present = True
                all_scores_missing = False
            score = score_f if score_f is not None else -1.0

            if norm_title not in crawled_paper_set:
                crawled_paper_set.add(norm_title)
                crawled_papers.append([norm_title, score])
                norm_to_original[norm_title] = original_title

    elif "child" in paper_root and paper_root["child"]:
        # Old framework's child tree; flatten all papers
        queue = [paper_root]
        while queue:
            node = queue.pop(0)
            for _, v in node.get("child", {}).items():
                for child in v:
                    queue.append(child)
                    # Check the year field; skip if it is later than end_date
                    year_str = child.get("year", "") or ""
                    if is_year_after_end_date(year_str, args.end_date):
                        continue

                    # Check the source field; skip if sn_flag is False and source contains "sn"
                    if not args.sn_flag:
                        source_str = child.get("source", "") or ""
                        if source_str and "sn" in source_str.lower():
                            continue

                    original_title = child.get("title", "") or ""
                    norm_title = normalize_title(original_title)
                    score_raw = child.get("select_score", None)
                    score_f = _try_float(score_raw)
                    if score_f is not None:
                        any_score_present = True
                        all_scores_missing = False
                    score = score_f if score_f is not None else -1.0
                    if norm_title not in crawled_paper_set:
                        crawled_paper_set.add(norm_title)
                        crawled_papers.append([norm_title, score])
                        norm_to_original[norm_title] = original_title
    else:
        if VERBOSE:
            print(f"⚠️ Warning: no paper data found in file {file_name}")

    # ======== LLM judge: score relevance of non-GT candidate papers (optional) ========
    if getattr(args, "llm_judge", False):
        user_query = query_text
        constraints = extra.get("final_checklist") or extra.get("base_checklist") or []
        if isinstance(constraints, str):
            constraints = [constraints]
        if not isinstance(constraints, (list, tuple)):
            constraints = []

        # Candidates: non-GT papers sorted by select_score
        # To avoid an explosion of calls in a single evaluation, only take the top N by default (args.llm_judge_max_papers_per_query)
        # Only for:
        # - has score: select_score > 80 and not in GT
        # - no score: include in the judge directly (and not in GT)
        non_gt_norms = []
        for t, _s in crawled_papers:
            if not t or t in gt_answer_set:
                continue
            p = norm_to_paper.get(t) or {}
            score_f = _try_float(p.get("select_score", None))
            if score_f is None:
                non_gt_norms.append(t)
            elif score_f > HIGH_SCORE_THRESHOLD:
                non_gt_norms.append(t)
        # Sort in descending order
        score_map_for_sort = {t: s for t, s in crawled_papers}
        non_gt_norms.sort(key=lambda t: score_map_for_sort.get(t, -1.0), reverse=True)

        max_papers = getattr(args, "llm_judge_max_papers_per_query", 50)
        non_gt_norms = non_gt_norms[: max(0, int(max_papers))]

        # Build the paper_data list
        llm_candidates = []
        for nt in non_gt_norms:
            p = norm_to_paper.get(nt)
            if isinstance(p, dict) and (p.get("title") or "").strip():
                llm_candidates.append(p)
        llm_judge_candidates_count = len(llm_candidates)

        prompt_path = getattr(
            args,
            "llm_judge_prompt_path",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt", "papermaster_prompt.json"),
        )
        prompt_template = _load_pointwise_prompt_template(prompt_path)
        model_name = getattr(args, "llm_judge_model", "deepseek-v3.2")
        timeout = int(getattr(args, "llm_judge_timeout", 60))
        concurrency = int(getattr(args, "llm_judge_concurrency", 50))

        llm_judge_5_titles, llm_judge_4_titles, llm_judge_detail = _llm_judge_candidates(
            candidates=llm_candidates,
            user_query=user_query,
            constraints=constraints,
            prompt_template=prompt_template,
            model_name=model_name,
            timeout=timeout,
            max_papers=max_papers,
            concurrency=concurrency,
            max_meta_chars=int(getattr(args, "llm_judge_max_meta_chars", 6000) or 0),
        )

        llm5_norm_set = set(normalize_title(x) for x in llm_judge_5_titles if x)
        # Expand the effective answer set: GT + LLM (average = 5)
        effective_answer_set |= llm5_norm_set
        for t in llm_judge_5_titles:
            nt = normalize_title(t)
            if nt and nt not in normalized_answer_to_original:
                normalized_answer_to_original[nt] = t

    # ======== Save LLM judge result details (includes query/paper info as required) ========
    # Only save non-GT judge=5/judge=4 papers (judge=5 papers also count as "expanded answers")
    judge_5_papers_detail = []
    judge_4_papers_detail = []
    judge_papers_by_score_detail = {str(i): [] for i in range(1, 6)}
    if getattr(args, "llm_judge", False) and llm_judge_detail:
        def _paper_snapshot(norm_t: str):
            p = norm_to_paper.get(norm_t) or {}
            title = (p.get("title") or norm_t or "").strip()
            abstract = (p.get("abstract") or "").strip()
            select_score = p.get("select_score", None)
            paper_reason = p.get("reason", "") or ""

            jd = llm_judge_detail.get(norm_t) or {}
            return {
                "query": query_text,
                "title": title,
                "abstract": abstract,
                "select_score": select_score,
                "reason": paper_reason,
                "llm": {
                    "model": getattr(args, "llm_judge_model", None),
                    "relevant": (jd.get("llm_relevant") or "").strip(),
                    "avg": jd.get("avg", None),
                    "avg_round": jd.get("avg_round", None),
                    "scores": jd.get("scores", []),
                    "llm_reason": jd.get("llm_reason", ""),
                    # raw output is not saved by default (too large); can be enabled via a parameter if needed
                    "raw": jd.get("raw") if getattr(args, "llm_judge_save_detail", False) else None,
                },
            }

        for nt, jd in llm_judge_detail.items():
            if not isinstance(jd, dict):
                continue
            avg_round = jd.get("avg_round", None)
            scores = jd.get("scores", []) or []
            # Bucket by score 1/2/3/4/5 (using the rounded avg_round result)
            try:
                ar = int(avg_round) if avg_round is not None else None
            except Exception:
                ar = None
            if scores and ar in (1, 2, 3, 4, 5):
                judge_papers_by_score_detail[str(ar)].append(_paper_snapshot(nt))
            if scores and all(s == 5 for s in scores):
                judge_5_papers_detail.append(_paper_snapshot(nt))
            elif scores and avg_round == 4:
                judge_4_papers_detail.append(_paper_snapshot(nt))

        # For stable output, sort by select_score in descending order
        def _score_key(x):
            s = x.get("select_score", None)
            try:
                return float(s)
            except Exception:
                return -1.0

        judge_5_papers_detail.sort(key=_score_key, reverse=True)
        judge_4_papers_detail.sort(key=_score_key, reverse=True)
        for k, lst in judge_papers_by_score_detail.items():
            if isinstance(lst, list):
                lst.sort(key=_score_key, reverse=True)

    # --------- Build the fuzzy alias mapping based on all predicted titles ---------
    all_pred_norm_titles = [t for t, _ in crawled_papers]
    fuzzy_alias = build_fuzzy_alias(all_pred_norm_titles, list(effective_answer_set))

    # Map a "predicted title" to the "canonical title" used for evaluation:
    # - If it is itself one of the ground-truth answers: use itself
    # - If its similarity to some answer >= threshold: use the corresponding answer title (normalized)
    # - Otherwise: keep it as is (treated as a miss)
    def map_pred_title(norm_title: str) -> str:
        if norm_title in effective_answer_set:
            return norm_title
        if norm_title in fuzzy_alias:
            return fuzzy_alias[norm_title][0]
        return norm_title

    # -------- Sort, used for overall @K + NDCG --------
    crawled_papers.sort(key=lambda x: x[1], reverse=True)
    ranked_titles_all = [map_pred_title(t) for t, _ in crawled_papers]

    # Build rank / score mappings (used for printing & saving hit info)
    rank_map = {}
    score_map = {}
    for idx, (t, s) in enumerate(crawled_papers):
        rank_map[t] = idx + 1  # rank starts at 1
        score_map[t] = s

    # ========= 0. Additionally save to eval_result: papers within topK with score>=threshold (title as key) =========
    # Notes:
    # - topK is based on crawled_papers (sorted by score in descending order)
    # - Only collect papers with score>=HIGH_SCORE_THRESHOLD
    # - hit_gt: whether this paper hits GT (fuzzy matching supported, but only against the GT set)
    high_score_topk_titles = {
        "threshold": HIGH_SCORE_THRESHOLD,
        "top5": {},
        "top10": {},
        "top20": {},
    }

    def _is_hit_gt(pred_norm: str) -> bool:
        """
        Determine whether GT is hit using the "mapping rules used for evaluation", to avoid:
        - pred itself not being in gt_answer_set
        - but actually hitting some GT after mapping via fuzzy_alias
        """
        if not pred_norm:
            return False
        mapped = map_pred_title(pred_norm)
        return mapped in gt_answer_set

    def _upsert_title(bucket: dict, title: str, *, hit_gt: bool, score: float):
        title = (title or "").strip()
        if not title:
            return
        prev = bucket.get(title)
        if not isinstance(prev, dict):
            bucket[title] = {"hit_gt": bool(hit_gt), "score": score}
            return
        prev_score = _try_float(prev.get("score", None))
        if prev_score is None or score > prev_score:
            bucket[title] = {"hit_gt": bool(hit_gt) or bool(prev.get("hit_gt", False)), "score": score}
        else:
            prev["hit_gt"] = bool(prev.get("hit_gt", False)) or bool(hit_gt)

    for idx, (t, s) in enumerate(crawled_papers[:20]):
        if s < HIGH_SCORE_THRESHOLD:
            continue
        title_key = (
            (norm_to_original.get(t) or "").strip()
            or ((norm_to_paper.get(t) or {}).get("title") or "").strip()
            or (t or "").strip()
        )
        hit_gt = _is_hit_gt(t)
        if idx < 5:
            _upsert_title(high_score_topk_titles["top5"], title_key, hit_gt=hit_gt, score=s)
        if idx < 10:
            _upsert_title(high_score_topk_titles["top10"], title_key, hit_gt=hit_gt, score=s)
        _upsert_title(high_score_topk_titles["top20"], title_key, hit_gt=hit_gt, score=s)

    # ========= 1. Overall @5/@10/@20 Recall / Precision / NDCG =========
    k_values = [5, 10, 20]
    top_sets = {k: set() for k in k_values}

    for idx, (t, _) in enumerate(crawled_papers):
        mapped_t = map_pred_title(t)
        if idx < 5:
            top_sets[5].add(mapped_t)
        if idx < 10:
            top_sets[10].add(mapped_t)
        if idx < 20:
            top_sets[20].add(mapped_t)
        if idx >= max(k_values):
            break

    metrics_at_k = {}
    for k in k_values:
        tp, fp, fn, _, _, _ = cal_micro(top_sets[k], effective_answer_set)
        recall_k = safe_div(tp, tp + fn)
        precision_k = safe_div(tp, k)  # denominator fixed to K
        f1_k = f1_score(precision_k, recall_k)
        metrics_at_k[k] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": pct_score(recall_k),
            "precision": pct_score(precision_k),
            "f1": pct_score(f1_k),
        }

    ndcg_5 = pct_score(ndcg_at_k(ranked_titles_all, effective_answer_set, 5))
    ndcg_10 = pct_score(ndcg_at_k(ranked_titles_all, effective_answer_set, 10))
    ndcg_20 = pct_score(ndcg_at_k(ranked_titles_all, effective_answer_set, 20))

    # ========= 2. High-score paper (score >= 80) subset =========
    # If this query has no usable score, the "high-score subset (>=80)" degenerates to "computed using all papers"
    if all_scores_missing:
        high_score_use_all_papers = True
        high_score_titles = list(crawled_papers)
        high_score_titles_original = [
            norm_to_original.get(t, t) for t, _ in high_score_titles if t
        ]

    high_score_norm_titles = [
        map_pred_title(t)
        for t, _ in sorted(high_score_titles, key=lambda x: x[1], reverse=True)
    ]
    high_score_set = set(high_score_norm_titles)

    high_tp, high_fp, high_fn, _, _, _ = cal_micro(high_score_set, effective_answer_set)
    high_recall_raw = safe_div(high_tp, high_tp + high_fn)
    high_precision_raw = safe_div(high_tp, len(high_score_set)) if high_score_set else 0.0
    high_f1_raw = f1_score(high_precision_raw, high_recall_raw)
    high_recall = pct_score(high_recall_raw)
    high_precision = pct_score(high_precision_raw)
    high_f1 = pct_score(high_f1_raw)

    # High-score subset NDCG: k = the number of papers with score>=threshold under this query (use however many there are), then average over the whole dataset
    high_ndcg_k = len(high_score_norm_titles)
    high_ndcg = pct_score(
        ndcg_at_k(high_score_norm_titles, effective_answer_set, high_ndcg_k)
        if high_ndcg_k > 0
        else 0.0
    )

    high_hit_count = len(high_score_set & effective_answer_set)
    high_total = len(high_score_set)

    # ========= 3. Number of answer hits within all_papers (fuzzy matching supported) =========
    for norm_title, original_title, _ in all_papers_raw:
        mapped = map_pred_title(norm_title)
        if mapped in effective_answer_set:
            all_papers_hit_titles_original.add(
                normalized_answer_to_original.get(mapped, original_title)
            )
            if mapped in gt_answer_set:
                hit_in_gt_titles_original.add(
                    normalized_answer_to_original.get(mapped, original_title)
                )
            else:
                # In theory this only falls into the LLM judge=5 expansion set
                hit_llm5_titles_original.add(
                    normalized_answer_to_original.get(mapped, original_title)
                )

    all_papers_hit_count = len(all_papers_hit_titles_original)

    # ========= 4. per-query hit paper details (used for saving and printing) =========
    fuzzy_matches_detail = []
    # As long as pred_norm appears in fuzzy_alias, this prediction hit some answer
    for pred_norm, (ans_norm, sim) in sorted(
        fuzzy_alias.items(), key=lambda x: x[1][1], reverse=True
    ):
        fuzzy_matches_detail.append(
            {
                "pred_title": norm_to_original.get(pred_norm, pred_norm),
                "answer_title": normalized_answer_to_original.get(ans_norm, ans_norm),
                "similarity": round(sim, 2),
                "rank": rank_map.get(pred_norm, -1),
                "score": score_map.get(pred_norm, None),
            }
        )

    # ========= 5. per-query detailed results =========
    query_detail = {
        "query_id": qid,
        "query_file": file_name,
        "query": query_text,
        "gt_source": gt_source,
        "bench_subject": bench_subject,
        "gt_answer_count": gt_answer_count,
        "gt_answer_titles": gt_answer_list_original,
        "effective_answer_count": len(effective_answer_set),
        "effective_answer_titles": [
            normalized_answer_to_original.get(t, t) for t in sorted(list(effective_answer_set))
        ],
        # New: "GT" papers newly expanded relative to the original GT (usually from llm_judge=5)
        "new_found_gt_titles": [
            normalized_answer_to_original.get(t, t)
            for t in sorted(list(effective_answer_set - gt_answer_set))
        ],
        "overall_at_k": {
            "5": {
                "recall": metrics_at_k[5]["recall"],
                "precision": metrics_at_k[5]["precision"],
                "f1": metrics_at_k[5]["f1"],
                "ndcg": ndcg_5,
            },
            "10": {
                "recall": metrics_at_k[10]["recall"],
                "precision": metrics_at_k[10]["precision"],
                "f1": metrics_at_k[10]["f1"],
                "ndcg": ndcg_10,
            },
            "20": {
                "recall": metrics_at_k[20]["recall"],
                "precision": metrics_at_k[20]["precision"],
                "f1": metrics_at_k[20]["f1"],
                "ndcg": ndcg_20,
            },
        },
        "high_score_subset": {
            "threshold": HIGH_SCORE_THRESHOLD,
            "use_all_papers_when_score_missing": high_score_use_all_papers,
            "paper_count": len(high_score_titles_original),
            "paper_titles": high_score_titles_original,
            "hit_in_answer_count": high_hit_count,
            "metrics": {
                "recall": high_recall,
                "precision": high_precision,
                "f1": high_f1,
                "ndcg_k": high_ndcg_k,
                "ndcg": high_ndcg,
            },
        },
        "all_papers_answer_hit": {
            "hit_in_answer_count": all_papers_hit_count,
            "hit_titles": sorted(list(all_papers_hit_titles_original)),
        },
        # Grouped for output as required: GT hits / non-GT but LLM=5 / non-GT but LLM=4
        "hit_grouped": {
            "hit_in_gt_titles": sorted(list(hit_in_gt_titles_original)),
            "hit_not_in_gt_llm_5_titles": sorted(list(hit_llm5_titles_original)),
            "hit_not_in_gt_llm_4_titles": sorted(list(set(llm_judge_4_titles))),
        },
        "llm_judge": {
            "enabled": bool(getattr(args, "llm_judge", False)),
            "model": getattr(args, "llm_judge_model", None),
            "max_papers_per_query": getattr(args, "llm_judge_max_papers_per_query", None),
            "concurrency": getattr(args, "llm_judge_concurrency", None),
            "candidates_count": llm_judge_candidates_count,
            "judge_5_titles": sorted(list(set(llm_judge_5_titles))),
            "judge_4_titles": sorted(list(set(llm_judge_4_titles))),
            "judge_5_papers": judge_5_papers_detail,
            "judge_4_papers": judge_4_papers_detail,
            # New: saved bucketed by score 1/2/3/4/5
            "papers_by_score": judge_papers_by_score_detail,
            # To avoid eval_result.json becoming too large, raw is not saved by default; a parameter switch can be added if needed
            "detail": llm_judge_detail if getattr(args, "llm_judge_save_detail", False) else None,
        },
        # For each query: "hit papers + corresponding answer + similarity + rank/score"
        "fuzzy_matches": fuzzy_matches_detail,
        # For each query: papers within top5/top10/top20 with score>=threshold (title as key, with hit_gt/score)
        "high_score_topk_titles": high_score_topk_titles,
    }

    if VERBOSE:
        print(f"\n===== Query {qid} ({file_name}) =====")
        print(f"Answer count: effective={len(effective_answer_set)} / gt={gt_answer_count}")
        print(
            "Recall@5/10/20 = "
            f"{metrics_at_k[5]['recall']:.2f} / "
            f"{metrics_at_k[10]['recall']:.2f} / "
            f"{metrics_at_k[20]['recall']:.2f}"
        )
        print(
            "Precision@5/10/20 = "
            f"{metrics_at_k[5]['precision']:.2f} / "
            f"{metrics_at_k[10]['precision']:.2f} / "
            f"{metrics_at_k[20]['precision']:.2f}"
        )
        print("NDCG@5/10/20 = " f"{ndcg_5:.2f} / {ndcg_10:.2f} / {ndcg_20:.2f}")
        print(
            f"High-score papers (>= {HIGH_SCORE_THRESHOLD}) count: {len(high_score_titles_original)}, "
            f"answer hits: {high_hit_count}"
        )
        print(f"answer hits within all_papers: {all_papers_hit_count}")
        print("Fuzzy match count (hit papers):", len(fuzzy_matches_detail))

        if fuzzy_matches_detail:
            print("Detailed list of hit papers:")
            for m in fuzzy_matches_detail:
                rank_str = f"rank={m['rank']}" if m["rank"] != -1 else "rank=N/A"
                print(
                    f"  [{rank_str}, score={m['score']}] "
                    f"pred='{m['pred_title']}'  <--->  "
                    f"answer='{m['answer_title']}', sim={m['similarity']}"
                )

    # This query's contribution to the global statistics
    stats = {
        "recall_at_k": {
            5: metrics_at_k[5]["recall"],
            10: metrics_at_k[10]["recall"],
            20: metrics_at_k[20]["recall"],
        },
        "precision_at_k": {
            5: metrics_at_k[5]["precision"],
            10: metrics_at_k[10]["precision"],
            20: metrics_at_k[20]["precision"],
        },
        "f1_at_k": {
            5: metrics_at_k[5]["f1"],
            10: metrics_at_k[10]["f1"],
            20: metrics_at_k[20]["f1"],
        },
        "ndcg_at_k": {
            5: ndcg_5,
            10: ndcg_10,
            20: ndcg_20,
        },
        "high_score": {
            "recall": high_recall,
            "precision": high_precision,
            "f1": high_f1,
            "ndcg": high_ndcg,
            "ndcg_k": high_ndcg_k,
            "hit_count": high_hit_count,
            "total_count": high_total,
        },
        "all_papers_hit_count": all_papers_hit_count,
        "answer_count": len(effective_answer_set),
        "gt_answer_count": gt_answer_count,
    }

    return query_detail, stats


# ==================== Aggregation and main flow ====================


def evaluate_folder(args):
    global VERBOSE
    VERBOSE = VERBOSE or args.verbose

    # Default output: input_folder/eval_result.json
    if args.output_file is None:
        args.output_file = os.path.join(args.input_folder, "eval_result.json")

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pred_files = glob.glob(os.path.join(args.input_folder, "*.json"))

    json_file_count = count_eval_json_files(args.input_folder)
    do_subject_breakdown = json_file_count >= SUBJECT_BREAKDOWN_MIN_JSON_FILES
    bench_path = args.bench_jsonl or DEFAULT_BENCH_JSONL
    bench_index = {}
    line_to_subject = {}
    if os.path.isfile(bench_path):
        bench_index = load_bench_index(bench_path)
        line_to_subject = bench_index.get("line_to_subject", {})
    else:
        print(f"⚠️ Bench file not found: {bench_path}; GT will fall back to extra.answer in the result files.")
        if do_subject_breakdown:
            print(
                f"⚠️ Per-subject evaluation is enabled (JSON≥{SUBJECT_BREAKDOWN_MIN_JSON_FILES}), "
                f"but bench file not found: {bench_path}; the per-subject summary will be skipped."
            )
            do_subject_breakdown = False

    # Aggregate by subject: use the overall Top20 recall/precision/f1/ndcg
    by_subject = defaultdict(
        lambda: {
            "recall_at_20": [],
            "precision_at_20": [],
            "f1_at_20": [],
            "ndcg_at_20": [],
        }
    )

    # Global statistics containers
    recall_at_5_list = []
    recall_at_10_list = []
    recall_at_20_list = []

    precision_at_5_list = []
    precision_at_10_list = []
    precision_at_20_list = []

    f1_at_5_list = []
    f1_at_10_list = []
    f1_at_20_list = []

    ndcg_at_5_list = []
    ndcg_at_10_list = []
    ndcg_at_20_list = []

    high_score_recall_list = []
    high_score_precision_list = []
    high_score_f1_list = []
    high_score_ndcg_list = []
    high_score_ndcg_k_list = []
    high_score_hit_counts = []
    high_score_total_counts = []

    all_papers_hit_counts = []
    answer_counts = []

    # Execution time statistics
    paper_agent_durations = []
    total_batch_times = []

    detailed_query_results = []

    for pred_file in tqdm(pred_files):
        if "eval" in pred_file:
            continue

        file_name = os.path.basename(pred_file)
        try:
            qid = int(file_name.split(".")[0])
        except ValueError:
            # Skip files whose name is not purely numeric
            continue

        if args.test_num >= 0 and qid >= args.test_num:
            continue

        with open(pred_file, "r", encoding="utf-8") as f:
            paper_root = json.load(f)

        query_detail, stats = evaluate_query(
            paper_root, file_name, qid, args, bench_index=bench_index
        )
        if query_detail is None or stats is None:
            continue
        if do_subject_breakdown and line_to_subject:
            subj = (query_detail.get("bench_subject") or "unknown").strip() or "unknown"
            bucket = by_subject[subj]
            bucket["recall_at_20"].append(stats["recall_at_k"][20])
            bucket["precision_at_20"].append(stats["precision_at_k"][20])
            bucket["f1_at_20"].append(stats["f1_at_k"][20])
            bucket["ndcg_at_20"].append(stats["ndcg_at_k"][20])

        detailed_query_results.append(query_detail)

        # Collect execution time
        execution_time = paper_root.get("execution_time", {})
        if execution_time:
            paper_agent_duration = execution_time.get("paper_agent_run_duration")
            total_batch_time = execution_time.get("total_batch_time")
            if paper_agent_duration is not None:
                paper_agent_durations.append(paper_agent_duration)
            if total_batch_time is not None:
                total_batch_times.append(total_batch_time)

        # Collect overall @K
        recall_at_5_list.append(stats["recall_at_k"][5])
        recall_at_10_list.append(stats["recall_at_k"][10])
        recall_at_20_list.append(stats["recall_at_k"][20])

        precision_at_5_list.append(stats["precision_at_k"][5])
        precision_at_10_list.append(stats["precision_at_k"][10])
        precision_at_20_list.append(stats["precision_at_k"][20])

        f1_at_5_list.append(stats["f1_at_k"][5])
        f1_at_10_list.append(stats["f1_at_k"][10])
        f1_at_20_list.append(stats["f1_at_k"][20])

        ndcg_at_5_list.append(stats["ndcg_at_k"][5])
        ndcg_at_10_list.append(stats["ndcg_at_k"][10])
        ndcg_at_20_list.append(stats["ndcg_at_k"][20])

        # High-score subset
        high_score_recall_list.append(stats["high_score"]["recall"])
        high_score_precision_list.append(stats["high_score"]["precision"])
        high_score_f1_list.append(stats["high_score"]["f1"])
        high_score_ndcg_list.append(stats["high_score"]["ndcg"])
        high_score_ndcg_k_list.append(stats["high_score"]["ndcg_k"])
        high_score_hit_counts.append(stats["high_score"]["hit_count"])
        high_score_total_counts.append(stats["high_score"]["total_count"])

        # all_papers & answer statistics
        all_papers_hit_counts.append(stats["all_papers_hit_count"])
        answer_counts.append(stats["answer_count"])

    # ======== Global averages and summary ========
    avg_recall_5 = avg(recall_at_5_list)
    avg_recall_10 = avg(recall_at_10_list)
    avg_recall_20 = avg(recall_at_20_list)

    avg_precision_5 = avg(precision_at_5_list)
    avg_precision_10 = avg(precision_at_10_list)
    avg_precision_20 = avg(precision_at_20_list)

    avg_f1_5 = avg(f1_at_5_list)
    avg_f1_10 = avg(f1_at_10_list)
    avg_f1_20 = avg(f1_at_20_list)

    avg_ndcg_5 = avg(ndcg_at_5_list)
    avg_ndcg_10 = avg(ndcg_at_10_list)
    avg_ndcg_20 = avg(ndcg_at_20_list)

    avg_high_recall = avg(high_score_recall_list)
    avg_high_precision = avg(high_score_precision_list)
    avg_high_f1 = avg(high_score_f1_list)
    avg_high_ndcg = avg(high_score_ndcg_list)
    avg_high_ndcg_k = avg(high_score_ndcg_k_list)
    avg_high_hit = avg(high_score_hit_counts)
    avg_high_total = avg(high_score_total_counts)

    avg_all_papers_hit = avg(all_papers_hit_counts)
    total_answer_count = sum(answer_counts)
    avg_answer_per_query = avg(answer_counts)

    # Compute average execution time
    avg_paper_agent_duration = avg(paper_agent_durations)
    avg_total_batch_time = avg(total_batch_times)

    # Console summary
    print("\n========== Global Evaluation Summary ==========")
    print(
        "Recall@5/10/20: {:.2f} / {:.2f} / {:.2f}".format(
            avg_recall_5, avg_recall_10, avg_recall_20
        )
    )
    print(
        "Precision@5/10/20: {:.2f} / {:.2f} / {:.2f}".format(
            avg_precision_5, avg_precision_10, avg_precision_20
        )
    )
    print(
        "F1@5/10/20: {:.2f} / {:.2f} / {:.2f}".format(
            avg_f1_5, avg_f1_10, avg_f1_20
        )
    )
    print(
        "NDCG@5/10/20: {:.2f} / {:.2f} / {:.2f}".format(
            avg_ndcg_5, avg_ndcg_10, avg_ndcg_20
        )
    )

    print(
        "\n[High-score subset >= {:.1f}] Avg Recall / Precision: {:.2f} / {:.2f}".format(
            HIGH_SCORE_THRESHOLD, avg_high_recall, avg_high_precision
        )
    )
    print(
        "[High-score subset >= {:.1f}] Avg F1: {:.2f}".format(HIGH_SCORE_THRESHOLD, avg_high_f1)
    )
    print(
        "[High-score subset >= {:.1f}] Avg NDCG@k (k=number of high-score papers for this query): {:.2f}, avg k: {:.2f}".format(
            HIGH_SCORE_THRESHOLD, avg_high_ndcg, avg_high_ndcg_k
        )
    )
    print(
        "[High-score subset >= {:.1f}] Avg high-score papers per query: {:.2f}, avg answer hits: {:.2f}".format(
            HIGH_SCORE_THRESHOLD, avg_high_total, avg_high_hit
        )
    )

    by_subject_top20 = {}
    if do_subject_breakdown and line_to_subject and len(by_subject) > 0:
        print(
            "\n========== By subject (Top20, matched by bench query) =========="
        )
        bench_path_used = args.bench_jsonl or DEFAULT_BENCH_JSONL
        for subj in iter_subjects_preferred(by_subject.keys()):
            b = by_subject[subj]
            nq = len(b["recall_at_20"])
            if nq == 0:
                continue
            ar = avg(b["recall_at_20"])
            ap = avg(b["precision_at_20"])
            af = avg(b["f1_at_20"])
            an = avg(b["ndcg_at_20"])
            by_subject_top20[subj] = {
                "query_count": nq,
                "ndcg_at_20": an,
                "recall_at_20": ar,
                "precision_at_20": ap,
                "f1_at_20": af,
            }
            print(
                "[{}] queries={}  NDCG@20={:.2f}  Recall@20={:.2f}  Precision@20={:.2f}  F1@20={:.2f}".format(
                    subj, nq, an, ar, ap, af
                )
            )
        print(f"(bench: {bench_path_used}, JSON files in folder: {json_file_count})")

    print(
        "\n[all_papers] Avg answer hits per query: {:.2f}".format(
            avg_all_papers_hit
        )
    )
    print(
        "[all_papers] Avg answer count per query: {:.2f}".format(avg_answer_per_query)
    )

    print("\n[Execution time statistics]")
    if paper_agent_durations:
        print(
            "Avg paper_agent_run_duration: {:.2f} s".format(avg_paper_agent_duration)
        )
    if total_batch_times:
        print("Avg total_batch_time: {:.2f} s".format(avg_total_batch_time))

    # Write JSON
    # Extra output: keyed by query, save the details of "LLM judge=5 (treated as GT)" papers
    llm_judge_gt_papers_by_query = {}
    # Extra output: keyed by query, save the details of "LLM judge bucketed by score 1/2/3/4/5" papers
    llm_judge_papers_by_query_by_score = {}

    def _dedupe_papers_by_title(papers: list) -> list:
        if not papers:
            return []
        seen = set()
        out = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").strip()
            key = normalize_title(title) if title else json.dumps(p, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out
    for qd in detailed_query_results:
        qtext = (qd.get("query") or "").strip()
        papers = ((qd.get("llm_judge") or {}).get("judge_5_papers") or [])
        if not qtext or not papers:
            continue

        if qtext in llm_judge_gt_papers_by_query:
            # merge + dedupe
            merged = llm_judge_gt_papers_by_query[qtext] + papers
            llm_judge_gt_papers_by_query[qtext] = _dedupe_papers_by_title(merged)
        else:
            llm_judge_gt_papers_by_query[qtext] = _dedupe_papers_by_title(papers)

        # Also aggregate: papers bucketed by score 1/2/3/4/5
        by_score = ((qd.get("llm_judge") or {}).get("papers_by_score") or {})
        if not isinstance(by_score, dict) or not by_score:
            continue
        if qtext not in llm_judge_papers_by_query_by_score:
            llm_judge_papers_by_query_by_score[qtext] = {str(i): [] for i in range(1, 6)}
        for k in ["1", "2", "3", "4", "5"]:
            cur = llm_judge_papers_by_query_by_score[qtext].get(k) or []
            add = by_score.get(k) or []
            merged = cur + (add if isinstance(add, list) else [])
            llm_judge_papers_by_query_by_score[qtext][k] = _dedupe_papers_by_title(merged)

    eval_result = {
        "evaluation_summary": {
            "total_queries": len(detailed_query_results),
            "input_folder": args.input_folder,
            "output_file": args.output_file,
            "ensemble_folder": args.output_folder_ensemble,
            "high_score_threshold": HIGH_SCORE_THRESHOLD,
            "fuzzy_match_threshold": FUZZY_MATCH_THRESHOLD,
            "end_date": args.end_date.strftime("%Y-%m-%d") if args.end_date else None,
            "sn_flag": args.sn_flag,
            "eval_json_file_count": json_file_count,
            "subject_breakdown_enabled": bool(len(by_subject_top20) > 0),
            "subject_breakdown_metric": "top20",
            "subject_breakdown_threshold_json_files": SUBJECT_BREAKDOWN_MIN_JSON_FILES,
            "bench_jsonl": (args.bench_jsonl or DEFAULT_BENCH_JSONL)
            if do_subject_breakdown
            else None,
        },
        "by_subject_top20": by_subject_top20
        if by_subject_top20
        else {},
        # Kept for backward-compatible field name: content has been switched to Top20 per-subject results.
        "by_subject_high_score": by_subject_top20
        if by_subject_top20
        else {},
        "overall_at_k": {
            "recall": {"5": avg_recall_5, "10": avg_recall_10, "20": avg_recall_20},
            "precision": {
                "5": avg_precision_5,
                "10": avg_precision_10,
                "20": avg_precision_20,
            },
            "f1": {"5": avg_f1_5, "10": avg_f1_10, "20": avg_f1_20},
            "ndcg": {"5": avg_ndcg_5, "10": avg_ndcg_10, "20": avg_ndcg_20},
        },
        "high_score_subset": {
            "threshold": HIGH_SCORE_THRESHOLD,
            "average_metrics": {
                "recall": avg_high_recall,
                "precision": avg_high_precision,
                "f1": avg_high_f1,
                "ndcg": avg_high_ndcg,
                "avg_ndcg_k": avg_high_ndcg_k,
                "avg_high_score_paper_count": avg_high_total,
                "avg_hit_in_answer_count": avg_high_hit,
            },
        },
        "all_papers_coverage": {
            "total_answer_count": total_answer_count,
            "avg_answer_per_query": avg_answer_per_query,
            "avg_hit_in_answer_per_query": avg_all_papers_hit,
            "per_query_hit_counts": all_papers_hit_counts,
        },
        "execution_time_stats": {
            "avg_paper_agent_run_duration": avg_paper_agent_duration,
            "avg_total_batch_time": avg_total_batch_time,
            "total_queries_with_time": len(paper_agent_durations),
        },
        # query as key: the list of LLM judge=5 (treated as GT) papers for this query (with query/title/abstract/select_score/reason/llm_reason)
        "llm_judge_gt_papers_by_query": llm_judge_gt_papers_by_query,
        # Separate key: aggregate all LLM judge results (bucketed by score 1/2/3/4/5)
        "llm_judge": {
            "papers_by_query_by_score": llm_judge_papers_by_query_by_score,
            "gt_papers_by_query": llm_judge_gt_papers_by_query,
        },
        # Each query includes fuzzy_matches (hit paper details)
        "queries": detailed_query_results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Evaluation results saved to: {args.output_file}")


# ==================== Argument parsing & main ====================


def str_to_bool(v):
    """Convert a string to a boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected (yes/no, true/false, t/f, 1/0), but got: {v}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the performance metrics of the literature search system (only @5/@10/@20 are kept, fuzzy matching supported)"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "evaluation"),
        help="Path to the input folder of JSON result files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the evaluation results, defaults to input_folder/eval_result.json",
    )
    parser.add_argument(
        "--output_folder_ensemble",
        type=str,
        default=None,
        help="Path to the second result folder for ensemble evaluation (optional)",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=-1,
        help="Maximum number of queries to evaluate (by the numeric qid in the file name, only processes qid < test_num). Default -1 means no limit and all files in the folder are evaluated.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate results for each query (including the list of hit papers)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-11-01",
        help="Cutoff date (format: YYYY-MM-DD); papers whose year is later than this date are excluded from the computation. If not specified, no date filtering is applied.",
    )
    parser.add_argument(
        "--sn_flag",
        type=str_to_bool,
        default=True,
        help="If False, filter out papers whose source field contains 'sn'. Defaults to True (no filtering). Accepted values: yes/no, true/false, t/f, 1/0",
    )
    parser.add_argument(
        "--bench_jsonl",
        type=str,
        default=DEFAULT_BENCH_JSONL,
        help=(
            "Path to the PaSaMaster bench JSONL; each line's subject_meta.subject is used for the per-subject summary. "
            f"When the number of evaluation JSON files under input_folder is >= {SUBJECT_BREAKDOWN_MIN_JSON_FILES}, "
            "line numbers are aligned with N in the result file N.json, and the Top20 "
            "recall/precision/f1/ndcg are reported."
        ),
    )

    # ===== LLM judge (optional) =====
    parser.add_argument(
        "--llm_judge",
        type=str_to_bool,
        default=False,
        help="Whether to enable the LLM judge: use deepseek-v3.2 to evaluate the relevance of papers not in GT (answer); a paper is treated as satisfying the answer and counted as a hit only when all constraints score 5.",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="deepseek-v3.2",
        help="Model name used by the LLM judge (default deepseek-v3.2).",
    )
    parser.add_argument(
        "--llm_judge_timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for a single LLM judge request.",
    )
    parser.add_argument(
        "--llm_judge_max_papers_per_query",
        type=int,
        default=100,
        help="Maximum number of non-GT papers per query to run the LLM judge on (truncated by select_score from high to low).",
    )
    parser.add_argument(
        "--llm_judge_prompt_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "prompt",
            "papermaster_prompt.json",
        ),
        help="Path to papermaster_prompt.json (used to read ranker.point_wise_rank_english).",
    )
    parser.add_argument(
        "--llm_judge_save_detail",
        type=str_to_bool,
        default=False,
        help="Whether to save each judged paper's raw output in eval_result.json (can be very large, default False).",
    )
    parser.add_argument(
        "--llm_judge_concurrency",
        type=int,
        default=50,
        help="LLM judge concurrency (default 50, concurrency cap fixed at 50).",
    )
    parser.add_argument(
        "--llm_judge_max_meta_chars",
        type=int,
        default=50000,
        help="Maximum number of characters of extra evidence text (e.g. description/chunks/sections) for the LLM judge. 0 means do not append.",
    )

    args = parser.parse_args()

    # Parse the end_date string into a datetime object
    if args.end_date:
        try:
            args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            parser.error(
                f"Invalid date format: {args.end_date}. Please use the YYYY-MM-DD format, for example: 2023-12-31"
            )
    else:
        args.end_date = None

    return args


def main():
    args = parse_args()
    evaluate_folder(args)


if __name__ == "__main__":
    main()