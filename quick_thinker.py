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
from config_loader import get_config_path
import json
import ast
import re
import select
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional
from rapidfuzz import fuzz, process

from log import quick_logger as logger
from model_vllm import Agent
from paper_node import PaperNode

from tools.web_search import arxiv_search_id, google_search_paper
from utils import search_paper_by_title
import os
import traceback
from tools.sn_server import sn_paper_search, sn_get_paper_info

current_path = os.path.dirname(os.path.abspath(__file__))

from utils import search_abs_by_id
from search_utils.call_mtr import CallMetricsTracker
from tools.search_info import search_meta_info
from log import setup_logging


with open(get_config_path(), "r") as f:
    config = json.load(f)
USE_LOCAL_SEARCH = config["use_local_search"]
USE_CHUNK = config.get("use_chunk", False)
FREEZE_MODE = False
FREEZE_MODE2 = False
FREEZE_PATH = os.path.join(current_path, "result", "frozen")

# Load the journal_metainfo.jsonl file
JOURNAL_METAINFO_CACHE = {}
JOURNAL_METAINFO_LIST = []
JOURNAL_METAINFO_LOADED = False

def _load_journal_metainfo():
    """Load the journal_metainfo.jsonl file into memory"""
    global JOURNAL_METAINFO_CACHE, JOURNAL_METAINFO_LIST, JOURNAL_METAINFO_LOADED
    if JOURNAL_METAINFO_LOADED:
        return
    
    journal_metainfo_path = f"{current_path}/config/journal_metainfo.jsonl"
    if not os.path.isfile(journal_metainfo_path):
        JOURNAL_METAINFO_LOADED = True
        logger.info("Optional journal metadata is absent; skipping local journal-metric enrichment")
        return
    try:
        with open(journal_metainfo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    journal_full = entry.get("journal_full", "")
                    if journal_full:
                        JOURNAL_METAINFO_CACHE[journal_full.lower()] = entry
                        JOURNAL_METAINFO_LIST.append(journal_full)
                        # Also handle journal_alias
                        journal_alias = entry.get("journal_alias", [])
                        if isinstance(journal_alias, list):
                            for alias in journal_alias:
                                if alias:
                                    JOURNAL_METAINFO_CACHE[alias.lower()] = entry
                except Exception as e:
                    logger.warning(f"Failed to parse journal_metainfo line: {e}")
                    continue
        JOURNAL_METAINFO_LOADED = True
        logger.info(f"Loaded {len(JOURNAL_METAINFO_LIST)} journal entries from journal_metainfo.jsonl")
    except Exception as e:
        logger.error(f"Failed to load journal_metainfo.jsonl: {e}")
        
_load_journal_metainfo()

def _match_journal_metainfo(journal: str, threshold: int = 80) -> Optional[Dict]:
    """
    Use fuzzy matching to match a journal against journal_metainfo.jsonl and return its meta info

    Args:
        journal: the journal name to match
        threshold: matching threshold (0-100), default 80

    Returns:
        The matched entry, containing h5_index, IF, CCF and other info; returns None if no match is found
    """
    if not journal or not journal.strip():
        return None

    if not JOURNAL_METAINFO_LIST:
        return None
    
    # First try an exact match (case-insensitive)
    journal_lower = journal.strip().lower()
    if journal_lower in JOURNAL_METAINFO_CACHE:
        return JOURNAL_METAINFO_CACHE[journal_lower]
    
    # Use fuzzy matching
    try:
        cleaned_journal = journal.replace("\n", " ").strip()
        best_match, score, idx = process.extractOne(
            cleaned_journal, JOURNAL_METAINFO_LIST, scorer=fuzz.ratio
        )
        if score >= threshold:
            matched_journal = JOURNAL_METAINFO_LIST[idx]
            return JOURNAL_METAINFO_CACHE.get(matched_journal.lower())
    except Exception as e:
        logger.warning(f"Fuzzy match failed for journal '{journal}': {e}")
    
    return None

MAX_LEN = 30000


class PaperAgent:
    def __init__(
        self,
        task_id: str,
        user_query: str,
        keyminder: Agent,  # New architecture: keyword extractor and query generator
        top_ranker: Agent,  # New architecture: locator
        ranker: Agent,  # Scorer
        end_date: str = datetime.now().strftime("%Y%m%d"),
        prompts_path: str = "prompt/papermaster_prompt.json",
        expand_layers: int = 2,
        search_queries: int = 5,  # Backward-compatible parameter
        search_papers: int = 10,
        expand_papers: int = 20,
        threads_num: int = 20,
        max_num: int = -1,
        time_limit: int = -1,
        build_ranker_sample: bool = False,
        further_sample_limit: int = 50,  # Upper limit on the number of papers sampled in further_search
        idx: int = 0,
        output_dir: str = f"{current_path}/result",
        init_papers = None,
        checklist = None,
    ) -> None:
        self.user_query = user_query
        self.keyminder = keyminder
        self.top_ranker = top_ranker
        self.ranker = ranker
        self.end_date = end_date
        self.idx = idx
        self.output_dir = output_dir
        self.further_sample_limit = further_sample_limit
        self.logger = logger.bind(task_id=task_id)
        self.task_id = task_id
        self.queries = {}
        self.messages_queue = []
        # Add necessary parameter checks
        if self.keyminder is None:
            raise ValueError("keyminder parameter cannot be None; please pass a valid Agent instance")
        if self.top_ranker is None:
            raise ValueError("top_ranker parameter cannot be None; please pass a valid Agent instance")
        if self.ranker is None:
            raise ValueError("ranker parameter cannot be None; please pass a valid Agent instance")

        with open(prompts_path, "r") as f:
            self.prompts = json.load(f)

        # Verify that the required prompts exist
        required_prompts = [
            ("keyminder", "extract_keywords_and_queries"),
        ]

        for module, prompt_key in required_prompts:
            if module not in self.prompts:
                raise ValueError(f"prompts file is missing the '{module}' module")
            if prompt_key not in self.prompts[module]:
                raise ValueError(f"prompts file is missing the '{module}.{prompt_key}' prompt")

        self.root = PaperNode(
            {
                "title": user_query,
                "extra": {
                    "touch_ids": [],
                    "recall_papers": [],
                    "recall_papers_abstract": [],
                    "arxiv_ids": [],
                    "score": [],
                    "sorted_papers": [],  # Store the full sorted paper info
                    "final_selected_papers_title": [],  # Store the sorted paper titles
                    "all_papers": [],  # Store all searched papers (regardless of score)
                    "all_papers_title": [],  # Store the titles of all searched papers
                    "subqueries": {},  # Store the subqueries generated by keyminder
                    "false_sample": {
                        "prompt": [],
                        "answer": [],
                        "title": [],
                    },
                },
                "relevant_section": {},  # Initialize the relevant_section field
                "relevant_section_ref": {},  # Initialize the relevant_section_ref field
            }
        )

        # hyperparameters
        self.max_num = max_num
        self.time_limit = time_limit
        self.start_time = datetime.now()
        self.expand_layers = expand_layers
        self.search_queries = search_queries  # Kept for backward compatibility
        self.search_papers = search_papers
        self.expand_papers = expand_papers
        self.threads_num = threads_num
        self.build_ranker_sample = build_ranker_sample
        if init_papers is not None:
            self.papers_queue = [self.get_paper_info(p, None, 0) for p in init_papers]
        else:
            self.papers_queue = []
        self.expand_start = 0
        self.lock = threading.Lock()
        self.checklist = checklist
        self.chunk_query: List[str] = []  # Store the split chunk retrieval queries
        self.language = "English"

        self.depth_stats = {}  # Record the number of papers at each depth
        # Record already-searched queries to avoid duplicates
        self.searched_queries = set()
        # Record the initial queries originally generated for the user
        self.original_queries = []
        self.embedding_paper_list = []
        # ========== NEW: cooperative mode state ==========
        self.coop_initialized = (
            False  # Whether initialization has been done (translate + keyminder + initial retrieval)
        )
        self.current_depth_processed = 0  # The max depth already processed (next round starts here)
        self.need_reasoner: bool = False
        self.uncertain_content: str = ""

    # ========== NEW: top-k interface for the Reasoner ==========
    def get_top_k_papers(self, k: int = 10):
        """Return the top-k papers in the current queue sorted by score (only scored papers are considered)"""
        candidates = [
            p for p in self.papers_queue if getattr(p, "select_score", -1) != -1
        ]
        candidates = sorted(candidates, key=lambda p: p.select_score, reverse=True)
        return candidates[:k]

    def get_paper_info(self, paper, meta_info, depth):
        try:
            if meta_info is not None:
                if (
                    meta_info["paired_journal"] is not None
                    and meta_info["paired_journal"] != ""
                ):
                    journal = meta_info["paired_journal"]
                elif meta_info["journal"] is not None and meta_info["journal"] != "":
                    journal = meta_info["journal"]
                elif "arxiv" in paper["source"]:
                    journal = "arxiv"
                else:
                    journal = paper["journal"]
            else:
                journal = paper["journal"]
            
            h5_index = -1
            if_value = -1
            ccf_value = ""
            
            if meta_info is not None:
                if meta_info.get("h5_index") is not None:
                    h5_index = meta_info["h5_index"]
                if meta_info.get("IF") is not None:
                    if_val = meta_info["IF"]
                    try:
                        if_value = float(if_val) if if_val else -1
                    except (ValueError, TypeError):
                        if_value = -1
                if meta_info.get("CCF") is not None:
                    ccf_value = str(meta_info["CCF"]) if meta_info["CCF"] else ""
            
            if (h5_index == -1 or if_value == -1 or ccf_value == "") and journal and journal != "arxiv":
                journal_meta = _match_journal_metainfo(journal)
                if journal_meta:
                    if h5_index == -1 and journal_meta.get("h5_index") is not None:
                        h5_index = journal_meta["h5_index"]
                    if if_value == -1 and journal_meta.get("IF") is not None:
                        if_str = journal_meta["IF"]
                        try:
                            if_value = float(if_str) if if_str else -1
                        except (ValueError, TypeError):
                            if_value = -1
                    if ccf_value == "" and journal_meta.get("CCF") is not None:
                        ccf_value = str(journal_meta["CCF"]) if journal_meta["CCF"] else ""
                
            paper_node = PaperNode(
                {
                    "title": paper["title"],
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "doi": paper.get("doi", ""),
                    "depth": depth,
                    "abstract": paper["abstract"],
                    "sections": paper.get("sections", ""),
                    "journal": journal,
                    "authors": paper.get("authors", ""),
                    "year": paper.get("year", ""),
                    "h5": h5_index,
                    "IF": if_value,
                    "CCF": ccf_value,
                    "citations": (
                        meta_info["citedBy"]
                        if meta_info is not None and meta_info["citedBy"] is not None
                        else -1
                    ),
                    "source": paper.get("source", ""),
                    "select_score": -1,
                    "extra": {},
                    "relevant_section": {},  # Initialize the relevant_section field
                    "relevant_section_ref": {},  # Initialize the relevant_section_ref field
                }
            )
        except:
            paper_node = PaperNode(
                {
                    "title": paper["title"],
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "doi": paper.get("doi", ""),
                    "depth": depth,
                    "abstract": paper["abstract"],
                    "sections": paper.get("sections", ""),
                    "authors": paper.get("authors", ""),
                    "year": paper.get("year", ""),
                    "source": paper.get("source", ""),
                    "select_score": -1,
                    "extra": {},
                    "relevant_section": {},  # Initialize the relevant_section field
                    "relevant_section_ref": {},  # Initialize the relevant_section_ref field
                }
            )

        return paper_node

    def get_paper_ref(self, papers):
        """Fetch missing paper references through the SN service."""
        from tools.sn_server import sn_get_ref

        self.logger.info(f"📖 [PROCESS] 3.2 Fetching paper details...")
        # Retrieve missing references through the SN service.
        failed_papers = [p for p in papers if not p.sections]
        if failed_papers:

            titles = [p.title for p in failed_papers]
            try:
                sn_refs = sn_get_ref(titles)  # Batch call
                success_count = 0
                # sn_refs should correspond one-to-one with failed_papers in order
                no_found_papers = []
                if len(sn_refs) == len(failed_papers):
                    for paper, refs in zip(failed_papers, sn_refs):
                        if refs:
                            paper.sections = refs
                            success_count += 1
                        else:
                            no_found_papers.append(
                                [paper.title, paper.arxiv_id, paper.doi, paper.source]
                            )
                else:
                    self.logger.error(
                        f"❌ [FetchContent] SN returned a mismatched count: exp={len(failed_papers)}:{[p.title for p in failed_papers]}, act={len(sn_refs)}:{[p['title'] for p in sn_refs]}"
                    )

                self.logger.info(
                    f"✅ [FetchContent] SN reference retrieval succeeded: {success_count}/{len(failed_papers)}"
                )
                self.logger.info(
                    f"🔄 [FetchContent] Total papers:{len(papers)}, already have references:{len(papers)-len(failed_papers)}, tried to supplement references for {len(failed_papers)} papers via the SN interface, successfully supplemented {success_count}, success rate {success_count/len(failed_papers):.2f}. Papers with no references found:{no_found_papers}"
                )
            except Exception as e:
                self.logger.exception(f"❌ [FetchContent] SN reference retrieval failed: {e}")

        # Collect all papers that have content, keeping the original order
        papers_with_content = [p for p in papers if p.sections]
        self.logger.info(
            f"📖 [FetchContent] Final number of papers with content: {len(papers_with_content)}/{len(papers)}"
        )

        return papers_with_content
    
    def _split_query_keywords_with_llm(
        self, user_query: str, checklist: List[str] = None, max_terms: int = 5
    ) -> List[str]:
        """
        Use the LLM to split a query into multiple chunk retrieval queries
        """
        q = (user_query or "").strip()
        if not q:
            return []

        cl_items: List[str] = []
        if checklist is None:
            cl_items = []
        elif isinstance(checklist, list):
            cl_items = [str(x).strip() for x in checklist if str(x).strip()]
        else:
            s = str(checklist).strip()
            if s:
                cl_items = [x.strip() for x in s.splitlines() if x.strip()]

        eff_max_terms = max_terms

        checklist_block = ""
        if cl_items:
            checklist_block = (
                "Checklist:\n" + "\n".join([f"- {x}" for x in cl_items]) + "\n\n"
            )

        prompt = (
            "You are a keyword extraction expert for embedding-based retrieval.\n"
            f"Given a user query and a checklist, output the SMALLEST possible set of search terms (from the checklist only) that still pinpoints the right paper chunks.\n"
            f"Hard cap: at most {eff_max_terms} terms — but prefer far fewer: often 1–2 terms is enough; add a third only if it removes real ambiguity.\n\n"
            "Goal: Minimize noise. Every term must be indispensable; drop nice-to-haves, synonyms, and peripheral checklist items.\n\n"
            "Rules:\n"
            "- Output STRICT JSON only (no markdown, no code blocks).\n"
            '- Schema: {"terms": ["...", "..."]}\n'
            "- Each term MUST be a very concise noun phrase (1-3 words max, e.g., 'contrastive learning', 'ablation study').\n"
            "- Only keep methods, entities, metrics, or hard constraints that are central to what must be retrieved; skip generic or vague words.\n"
            "- Do NOT pad the list to use the cap — fewer, sharper terms beats more terms.\n"
            f"User query: {q}\n"
            f"{checklist_block}"
            "JSON:\n"
        )

        out = self.keyminder.infer(prompt, use_cot=False)
        obj = self._safe_json_extract(out or "")
        terms: List[str] = []
        if isinstance(obj, dict) and isinstance(obj.get("terms"), list):
            for t in obj["terms"]:
                s = str(t or "").strip()
                if not s:
                    continue
                s = re.sub(r"\s+", " ", s)
                terms.append(s)
        
        # fallback
        if not terms:
            if cl_items:
                for it in cl_items:
                    s = re.sub(r"\s+", " ", f"{q} {it}".strip())
                    if s:
                        terms.append(s[:200])
            else:
                parts = re.split(r"[;,]| and | or | with | involving ", q, flags=re.IGNORECASE)
                for p in parts:
                    p = re.sub(r"\s+", " ", (p or "").strip())
                    if 2 <= len(p.split()) <= 12:
                        terms.append(p)
                    if len(terms) >= eff_max_terms:
                        break

        terms = [t for t in terms if str(t).strip()]
        return terms[:eff_max_terms] if terms else [q]

    def _safe_json_extract(self, text: str):
        """Best-effort JSON dict extraction from LLM output."""
        if not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _chunk_to_text(self, chunks) -> str:
        """Convert a list of chunks into text"""
        if not chunks:
            return ""
        parts: List[str] = []
        if isinstance(chunks, list):
            for c in chunks:
                if isinstance(c, list) and len(c) > 0:
                    # chunks format: [[content, score], ...]
                    for item in c:
                        if isinstance(item, list) and len(item) >= 1:
                            txt = str(item[0]).strip()
                            if len(item) > 1:
                                sc = item[1]
                                parts.append(f"[score={sc}] {txt}".strip())
                            else:
                                parts.append(txt)
                elif isinstance(c, dict):
                    txt = (
                        c.get("chunk")
                        or c.get("text")
                        or c.get("content")
                        or c.get("passage")
                        or ""
                    )
                    sc = c.get("score", None)
                    if sc is not None:
                        parts.append(f"[score={sc}] {txt}".strip())
                    else:
                        parts.append(str(txt).strip())
                else:
                    parts.append(str(c).strip())
        else:
            parts.append(str(chunks).strip())
        return "\n\n---\n\n".join([p for p in parts if p])

    def _truncate_parts_evenly(self, parts: List[str], max_len: int) -> List[str]:
        """Truncate each part evenly so the total length <= max_len"""
        parts = [p for p in parts if p]
        if not parts:
            return []
        if sum(len(p) for p in parts) <= max_len:
            return parts
        per = max(1, max_len // len(parts))
        return [p[:per] for p in parts]

    def _enrich_papers_with_chunks(self, papers,expand_depth):
        """
        Retrieve chunks for papers and fill them into the description field
        See create_dataset/score_ai10_answers_with_chunks.py:393-446
        """
        if not USE_CHUNK or not self.chunk_query:
            return
        
        try:
            from tools.chunk import get_paper_chunk
        except ImportError:
            self.logger.warning("[CHUNK] get_paper_chunk not available, skipping chunk enrichment")
            return

        self.logger.info(f"[CHUNK] Enriching {len(papers)} papers with chunks using queries: {self.chunk_query}")

        def _fetch_desc(arxiv_id: str) -> str:
            try:
                texts: List[str] = []
                for term in self.chunk_query:
                    chunks = get_paper_chunk(arxiv_id, term, topk=3)
                    t = self._chunk_to_text(chunks).strip()
                    if t:
                        texts.append(t)
                if not texts:
                    return ""
                # Merge texts, limiting the total length
                texts = self._truncate_parts_evenly(texts, MAX_LEN)
                merged = "\n\n====\n\n".join([t for t in texts if t])
                return merged[:MAX_LEN] if len(merged) > MAX_LEN else merged
            except Exception as e:
                self.logger.warning(f"[CHUNK] Failed to fetch chunks for {arxiv_id}: {e}")
                return ""

        # Deduplicate arxiv_id
        unique_ids: List[str] = []
        seen_ids: set = set()
        id_to_papers: Dict[str, List] = {}
        for pn in papers:
            if pn.depth != expand_depth:
                continue
            pid = str(getattr(pn, "arxiv_id", "") or "").strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            unique_ids.append(pid)
            if pid not in id_to_papers:
                id_to_papers[pid] = []
            id_to_papers[pid].append(pn)

        # Retrieve chunks concurrently
        id_to_desc: Dict[str, str] = {}
        if unique_ids:
            with ThreadPoolExecutor(max_workers=50) as ex:
                futs = {ex.submit(_fetch_desc, pid): pid for pid in unique_ids}
                for fut in futs:
                    pid = futs[fut]
                    try:
                        id_to_desc[pid] = fut.result()
                    except Exception as e:
                        self.logger.warning(f"[CHUNK] Error fetching chunk for {pid}: {e}")
                        id_to_desc[pid] = ""

        # Write the retrieved chunks into paper.description
        updated = 0
        for pid, desc in id_to_desc.items():
            if desc:
                for pn in id_to_papers.get(pid, []):
                    # If a description already exists, append; otherwise set it directly
                    if getattr(pn, "description", ""):
                        pn.description = f"{pn.description}\n\n====\n\n{desc}"
                    else:
                        pn.description = desc
                    updated += 1

        self.logger.info(f"[CHUNK] Updated {updated} papers with chunk descriptions")

    def expand_iteration(self, depth):
        """Run one expansion iteration - deprecated, kept for backward compatibility"""
        self.logger.info(
            f"\033[93mWarning: expand_iteration is deprecated, use process() instead\033[0m"
        )
        # Keep an empty implementation to avoid breaking old code
        pass

    def save_paper_nodes(self, paper_nodes):
        paper_list = []
        for paper_node in paper_nodes:
            paper_list.append(paper_node.todic())
        return paper_list

    def read_paper_nodes(self, paper_list):
        paper_nodes = []
        for paper_dic in paper_list:
            paper_node = PaperNode(
                {
                    "title": paper_dic["title"],
                    "depth": paper_dic["depth"],
                    "arxiv_id": paper_dic.get("arxiv_id", ""),
                    "doi": paper_dic.get("doi", ""),
                    "abstract": paper_dic["abstract"],
                    "sections": paper_dic.get("sections", ""),
                    "content_sections": paper_dic.get("content_sections", ""),
                    "source": paper_dic.get("source", ""),
                    "select_score": paper_dic.get("select_score", -1),
                    "extra": paper_dic.get("extra", {}),
                }
            )
            paper_nodes.append(paper_node)
        return paper_nodes

    def add_paper_to_queue(self, paper_node, threshold: int = 90):
        """
        Use fuzzy matching to check whether a paper with a similar title already exists:
        - When score >= threshold, treat it as a duplicate and do not add it to the queue
        - Also, if the existing node lacks arxiv_id / doi, fill it in using the new node's info
        """
        with self.lock:
            target_title_raw = paper_node.title or ""
            target_title_clean = target_title_raw.replace("\n", " ").strip().lower()

            existing_titles_clean = []
            existing_papers = []

            for paper in self.papers_queue:
                title_raw = paper.title or ""
                cleaned = title_raw.replace("\n", " ").strip().lower()
                existing_titles_clean.append(cleaned)
                existing_papers.append(paper)

            if not existing_titles_clean:
                self.papers_queue.append(paper_node)
                return True

            try:
                best_match, score, idx = process.extractOne(
                    target_title_clean, existing_titles_clean, scorer=fuzz.ratio
                )
            except Exception as e:
                # On a matching error, fall back to "add directly"
                logger.error(
                    f"\033[91m Error during fuzzy match in add_paper_to_queue: {e}, "
                    f"target_title: {target_title_raw!r} \033[0m"
                )
                self.papers_queue.append(paper_node)
                return True

            if score >= threshold:
                duplicate_paper = existing_papers[idx]

                logger.info(
                    f"\033[93m Detected duplicate paper by fuzzy match (score={score}):\n"
                    f"  New:      {target_title_raw!r}\n"
                    f"  Existing: {duplicate_paper.title!r}\033[0m"
                )

                if (
                    getattr(duplicate_paper, "arxiv_id", "") == ""
                    and getattr(paper_node, "arxiv_id", "") != ""
                ):
                    duplicate_paper.arxiv_id = paper_node.arxiv_id
                if (
                    getattr(duplicate_paper, "doi", "") == ""
                    and getattr(paper_node, "doi", "") != ""
                ):
                    duplicate_paper.doi = paper_node.doi

                return False
            self.papers_queue.append(paper_node)
            return True

    @staticmethod
    def do_parallel(func, args, num):
        threads = []
        for _ in range(num):
            thread = threading.Thread(target=func, args=args)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def check_limit(self):
        if self.max_num != -1 and len(self.root.extra["recall_papers"]) >= self.max_num:
            return True
        if (
            self.time_limit != -1
            and (datetime.now() - self.start_time).seconds >= self.time_limit
        ):
            return True
        return False

    def get_depth_statistics(self):
        """Get the current depth statistics"""
        depth_stats = {}
        for paper in self.papers_queue:
            depth_stats[paper.depth] = depth_stats.get(paper.depth, 0) + 1
        return depth_stats

    def print_progress_summary(self, current_depth=None):
        """Print a progress summary"""
        stats = self.get_depth_statistics()
        total_papers = len(self.papers_queue)

        self.logger.info(f"📈 [PROGRESS] =============== Progress summary ===============")
        self.logger.info(f"📊 [PROGRESS] Total papers: {total_papers}")
        self.logger.info(f"📈 [PROGRESS] Depth distribution: {stats}")
        if current_depth is not None:
            self.logger.info(f"🎯 [PROGRESS] Current processing depth: {current_depth}")
        self.logger.info(f"📈 [PROGRESS] =========================================\n")

    def extract_keywords_and_queries(self):
        """Keyminder: extract keywords + weights + search queries from the user query"""
        self.logger.info(f"\n🔍 [Keyminder] Starting to extract keywords and queries...")
        self.logger.info(f"📝 [Keyminder] Input query: {self.user_query}")

        start_time = time.time()
        prompt_template = self.prompts["keyminder"]["extract_keywords_and_queries"]
        prompt = prompt_template.format(user_query=self.user_query)
        prompt += f"Current date is {datetime.now().strftime('%Y-%m-%d')}."
        max_retry = 3
        for i in range(max_retry):
            response = self.keyminder.infer(prompt, use_cot=False)
            elapsed_time = time.time() - start_time

            try:
                query_begin = response.find("<query>") + len("<query>")
                query_end = response.find("</query>")
                queries = response[query_begin:query_end].strip().split(";")
                checkpoints_begin = response.find("<checkpoints>") + len(
                    "<checkpoints>"
                )
                checkpoints_end = response.find("</checkpoints>")
                if self.checklist is None:
                    self.checklist = (
                        response[checkpoints_begin:checkpoints_end].strip().split(";")
                    )
                    self.logger.info(f"checklist:{self.checklist}")
                else:
                    self.logger.info(f"already have checklist:{self.checklist}")

                if not queries:
                    logger.info(f"⚠️ [Keyminder] No queries were generated, using the original query")
                    queries = [self.user_query]
                self.queries["0"] = queries  # Save the original query
                self.root.extra["subqueries"][
                    "0"
                ] = queries  # Save the subqueries generated by keyminder
                # Record the initial queries
                self.original_queries = list(queries)

                self.logger.info(f"📊 \033[32m[Keyminder] Final result:")
                self.logger.info(f"  Queries: {queries}\033[0m")

                return queries
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.exception(f"❌ [Keyminder] JSON parsing failed: {e}")
                if i == max_retry - 1:
                    self.logger.info(f"🔄 [Keyminder] Using fallback: the original query")
                    fallback_result = [self.user_query]
                    self.logger.info(f"📊 [Keyminder] Fallback result: {fallback_result}")
                    return fallback_result

    def search_papers_with_queries(self, queries, current_depth=0):
        """Search papers directly with queries and add them to the queue"""
        self.logger.info(f"\n🔍 [DirectSearch] Starting to search papers...")

        start_time = time.time()

        # Limit the number of queries
        original_count = len(queries)
        queries = (
            queries[: self.search_queries] if self.search_queries > 0 else queries[:5]
        )
        # Record the queries
        for q in queries:
            self.searched_queries.add(q)
    
        # Search in parallel
        PaperAgent.do_parallel(
            self.search_paper_worker,
            (queries, current_depth, -1),
            min(len(queries), self.threads_num),
        )

        elapsed_time = time.time() - start_time

        self.logger.info(f"⏱️ [DirectSearch] Search took: {elapsed_time:.2f}s")
        self.logger.info(
            f"📊 [DirectSearch] Current number of papers in the queue: {len(self.papers_queue)}"
        )

    def search_arxiv_paper(self, arxiv_ids, thread_id, searched_papers):
        success_count = 0
        for i, arxiv_id in enumerate(arxiv_ids):
            arxiv_id = arxiv_id.split("v")[0]

            with self.lock:
                if arxiv_id not in self.root.extra["touch_ids"]:
                    self.root.extra["touch_ids"].append(arxiv_id)
                else:
                    continue

            paper = search_abs_by_id(arxiv_id)
            if paper is not None:
                searched_papers.append(paper)
                self.logger.info(
                    f"✅ [SearchWorker-{thread_id}] Successfully fetched: {paper['title']}"
                )
                success_count += 1
            else:
                self.logger.info(f"❌ [SearchWorker-{thread_id}] Fetch failed: {arxiv_id}")
        self.logger.info(
            f"✅ [SearchWorker-{thread_id}] Successfully fetched arxiv papers: {success_count}"
        )

    def search_sn_paper(self, paper_titles, thread_id, searched_papers):
        success_count = 0
        begin_time = time.time()
        sn_papers = sn_get_paper_info(paper_titles)
        end_time = time.time()
        self.logger.info(
            f"📊 [SearchWorker-{thread_id}] SN paper info search took: {end_time - begin_time:.2f}s, retrieved in total:{len(sn_papers)}"
        )
        if sn_papers:
            for paper in sn_papers:
                if paper != {}:
                    with self.lock:
                        if paper.get("doi", "") not in self.root.extra["touch_ids"]:
                            self.root.extra["touch_ids"].append(paper.get("doi", ""))
                        else:
                            continue
                    searched_papers.append(paper)
                    self.logger.info(
                        f"✅ [SearchWorker-{thread_id}] Successfully fetched: {paper['title']}"
                    )
                    success_count += 1
        else:
            self.logger.info(
                f"❌ [SearchWorker-{thread_id}] SN search failed: {paper_titles}"
            )
        logger.info(f"✅ [SearchWorker-{thread_id}] Successfully fetched SN papers: {success_count}")

    def search_paper_worker(self, queries, current_depth=0, num=-1):
        """Worker thread for searching papers"""
        thread_id = threading.current_thread().name
        while queries:
            with self.lock:
                if queries:
                    query = queries.pop()
                else:
                    break

            # First try Google search
            paper_titles = []
            if USE_LOCAL_SEARCH:
                arxiv_ids = arxiv_search_id(query, self.search_papers, self.end_date)
                self.logger.info(
                    f"📊 [SearchWorker-{thread_id}] Local search found {len(arxiv_ids)} paper IDs"
                )
            else:
                arxiv_ids, paper_titles = google_search_paper(
                    query, self.search_papers, self.end_date
                )
                self.logger.info(
                    f"📊 [SearchWorker-{thread_id}] Google search found {len(arxiv_ids)} arxiv paper IDs, {len(paper_titles)} other paper titles"
                )

            if not arxiv_ids and not paper_titles:
                self.logger.error(
                    f"📊 \033[091m [SearchWorker-{thread_id}] Google search failed!!!,query:{query}\033[0m"
                )
                return
            if num != -1:
                arxiv_ids = arxiv_ids[:num]
                if len(arxiv_ids) < num:
                    paper_titles = paper_titles[: num - len(arxiv_ids)]
            searched_papers = []
            if arxiv_ids:
                self.search_arxiv_paper(arxiv_ids, thread_id, searched_papers)
            # Supplement scholar-found papers from the SN library
            if paper_titles:
                self.search_sn_paper(paper_titles, thread_id, searched_papers)
            # Supplement the search via the SN interface (synchronous call)
            begin_time = time.time()
            if num == 3:
                try:
                    sn_papers = sn_paper_search(
                        query, self.search_papers, self.end_date
                    )
                    if sn_papers:
                        self.logger.info(
                            f"📊 [SearchWorker-{thread_id}] SN search obtained {len(sn_papers)} candidate papers,query:{query}"
                        )
                        for paper in sn_papers:
                            if paper.get("doi", "") not in self.root.extra["touch_ids"]:
                                self.logger.info(f"Paper found by SN search: {paper['title']}")
                                self.root.extra["touch_ids"].append(
                                    paper.get("doi", "")
                                )
                                searched_papers.append(paper)
                            else:
                                self.logger.info(
                                    f"♻️ [SearchWorker-{thread_id}] Duplicate paper, skipping: {paper.get('doi', '')}"
                                )
                                continue
                except Exception as e:
                    self.logger.exception(
                        f"❌ [SearchWorker-{thread_id}] SN search failed: {e}"
                    )
                end_time = time.time()
                self.logger.info(
                    f"📊 [SearchWorker-{thread_id}] SN search took: {end_time - begin_time:.2f}s"
                )
            # Add the searched papers to the queue
            searched_paper_titles = [p["title"] for p in searched_papers]
            # TODO: how to add meta info for sn
            meta_infos = search_meta_info(searched_paper_titles)
            for paper, meta_info in zip(searched_papers, meta_infos):

                paper_node = self.get_paper_info(paper, meta_info, current_depth)
                self.add_paper_to_queue(paper_node)
                # Update depth statistics
                with self.lock:
                    self.depth_stats[current_depth] = (
                        self.depth_stats.get(current_depth, 0) + 1
                    )
            self.logger.info(
                f"📊 [SearchWorker-{thread_id}] This query is done, obtained {len(searched_papers)} valid papers, task ID: {self.task_id}"
            )
        
        # logger.info(f"🏁 [SearchWWorker-{thread_id}] Thread finished")

    def further_search(self, papers_with_content, target_depth):
        """Generate additional search queries based on the previous round's paper abstracts and retrieve

        - Directly take the top 10 papers and assemble one prompt to generate extra queries (for speed)
        - Pass the already-searched queries to the model to avoid duplicates
        - Retrieve for the newly generated queries and add the resulting papers to this round (target_depth)
        """
        try:
            begin = time.time()
            if not papers_with_content:
                self.logger.info("⚠️ [FurtherSearch] No paper content available to generate extra queries, skipping")
                return
            # For speed: each time only take the top 10 papers to generate search terms
            papers_sample = papers_with_content[:10]

            # Assemble the abstract text (all at once: top10)
            prompts = []
            self.logger.info(
                f"      📦 [FurtherSearch] Papers this round: {len(papers_with_content)} | Using top10: {len(papers_sample)}"
            )

            abstracts_lines = []
            for i, paper in enumerate(papers_sample, 1):
                title = getattr(paper, "title", "") or ""
                abstract = getattr(paper, "abstract", "") or ""
                abstracts_lines.append(f"{i}. Title: {title}\nAbstract: {abstract}")
            abstracts_block = "\n\n".join(abstracts_lines)

            # Distinguish the original queries from the follow-up queries
            original_qs = list(self.original_queries) if self.original_queries else []
            expanded_qs = [
                q for q in sorted(self.searched_queries) if q not in set(original_qs)
            ]
            original_queries_text = "\n".join(original_qs) if original_qs else "(none)"
            expanded_queries_text = "\n".join(expanded_qs) if expanded_qs else "(none)"

            prompt_template = self.prompts.get("further", {}).get(
                "generate_extra_queries", ""
            )
            if not prompt_template:
                self.logger.info(
                    "⚠️ [FurtherSearch] Missing the further.generate_extra_queries prompt, skipping further search"
                )
                return
            prompt = prompt_template.format(
                user_query=self.user_query,
                original_queries=original_queries_text,
                expanded_queries=expanded_queries_text,
                abstracts_block=abstracts_block,
            )
            prompts.append(prompt)
            if not prompts:
                self.logger.info("⚠️ [FurtherSearch] No usable prompt, skipping")
                return

            batch_size = min(len(prompts), self.threads_num)
            responses = self.keyminder.batch_infer_safe(
                prompts, batch_size=batch_size, use_cot=False
            )

            new_queries = []
            for resp in responses:
                q_start = resp.find("<query>")
                q_end = resp.find("</query>")
                if q_start == -1 or q_end == -1:
                    self.logger.info(f"⚠️ [FurtherSearch] Cannot parse the query tag, skipping this segment")
                    continue

                # Parse, drop empties, and filter out already-searched queries
                raw_queries = resp[q_start + len("<query>") : q_end].strip().split(";")
                candidates = [q.strip() for q in raw_queries if q.strip()]

                # Filter out globally already-searched ones, and limit to at most 5 per batch
                valid_batch = [q for q in candidates if q not in self.searched_queries][
                    :5
                ]
                new_queries.extend(valid_batch)

            # Deduplicate within this round's results (preserving order)
            filtered = list(dict.fromkeys(new_queries))

            if not filtered:
                self.logger.info("⚠️ [FurtherSearch] No new valid queries, skipping retrieval")
                return

            self.logger.info(f"📝 [FurtherSearch] Number of newly generated queries: {len(filtered)}")
            self.logger.info(
                f"📝 \033[92m [FurtherSearch] Newly generated query questions: {filtered}\033[0m"
            )

            # Update records
            self.queries[f"{target_depth}"] = filtered
            self.root.extra["subqueries"][f"{target_depth}"] = filtered
            self.searched_queries.update(filtered)

            # Retrieve and add to the specified depth
            prev_queue_size = len(self.papers_queue)
            self.search_papers_with_queries(filtered, target_depth)
            added = len(self.papers_queue) - prev_queue_size
            self.logger.info(f"📊 [FurtherSearch] Papers added via extra queries: {added}")
            end = time.time()
            self.logger.info(f"🔍 [FurtherSearch] This round of extra queries is done, took {end-begin:.2f}s")
        except Exception as e:
            logger.exception(f"❌ [FurtherSearch] An error occurred: {e}")

    def locate_relevant_papers(self, papers_with_content, target_depth=None):

        from tools.ref_extend import embedding_retrieval
        papers_with_content.sort(key=lambda x: x.select_score, reverse=True)
        refs = []
        for paper in papers_with_content:
            section_ref = paper.sections
            if section_ref:
                for key, value in section_ref.items():
                    refs.append(value)

        # Fuzzy-match refs against the paper titles in the current queue to deduplicate: keep only candidates "not in the queue"
        def _flatten_to_titles(items):
            out = []
            for x in items or []:
                if not x:
                    continue
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    out.extend([v for v in x.values() if isinstance(v, str) and v.strip()])
                elif isinstance(x, (list, tuple, set)):
                    out.extend([str(v) for v in x if v and str(v).strip()])
                else:
                    out.append(str(x))
            # Deduplicate (preserving order)
            return list(dict.fromkeys([s.replace("\n", " ").strip() for s in out if str(s).strip()]))

        def _normalize_title_local(text: str) -> str:
            s = (text or "").lower()
            s = re.sub(r"[^a-z0-9]+", "", s)
            return s

        def _get_queue_titles() -> List[str]:
            titles: List[str] = []
            for p in getattr(self, "papers_queue", []) or []:
                if isinstance(p, dict):
                    t = p.get("title", "") or ""
                else:
                    t = getattr(p, "title", "") or ""
                t = t.replace("\n", " ").strip()
                if t:
                    titles.append(t)
            return titles

        def _filter_out_existing_titles(
            candidates: List[str], existing_titles: List[str], threshold: int = 90
        ) -> List[str]:
            if not candidates:
                return []
            if not existing_titles:
                return list(dict.fromkeys(candidates))

            existing_norm = {_normalize_title_local(t) for t in existing_titles if t}
            kept: List[str] = []
            for cand in candidates:
                cleaned = (cand or "").replace("\n", " ").strip()
                if not cleaned:
                    continue
                # First deduplicate using strict normalization (faster and safer)
                if _normalize_title_local(cleaned) in existing_norm:
                    continue
                best = process.extractOne(cleaned, existing_titles, scorer=fuzz.ratio)
                if not best:
                    kept.append(cleaned)
                    continue
                _best_match, score, _idx = best
                if score < threshold:
                    kept.append(cleaned)
            return list(dict.fromkeys(kept))

        queue_titles = _get_queue_titles()
        origin_refs_len = len(refs)
        refs = _filter_out_existing_titles(_flatten_to_titles(refs), queue_titles, threshold=90)
        logger.info(f"🔍 [Locate] Number of refs before local retrieval: {origin_refs_len}, number of refs after filtering for local retrieval: {len(refs)}")

        if not refs:
            return

        search_query = [self.user_query]
        search_query.extend(self.queries[str(target_depth-1)])
        begin = time.time()
        extend_refs = embedding_retrieval(search_query, refs,expand_papers=self.expand_papers,embedding_paper_list = self.embedding_paper_list)
        end = time.time()
        self.logger.info(f"🔍 [Locate] Local retrieval done, took {end-begin:.2f}s")

        # Do another title-level deduplication against the current queue to avoid duplicate retrieval / duplicate enqueuing
        extend_refs = _filter_out_existing_titles(_flatten_to_titles(extend_refs), queue_titles, threshold=90)
        if not extend_refs:
            return
        # --- Change: query locally first, then query SN ---
        found_papers = []
        missing_titles = []
        begin = time.time()
        # 1. Try local search
        for title in extend_refs:
            paper = search_paper_by_title(title)
            if paper:
                found_papers.append(paper)
            else:
                missing_titles.append(title)
        arxiv_found_papers_num = len(found_papers)
        end = time.time()
        self.logger.info(f"🔍 [Locate] arxiv paper retrieval done, took {end-begin:.2f}s")
        # 2. For those not found locally, go through SN in batch
        if missing_titles:
            self.logger.info(
                f"🔍 [Locate] {len(missing_titles)} not found locally, trying to supplement via SN..."
            )
            try:
                begin_time = time.time()
                sn_papers = sn_get_paper_info(missing_titles)
                end_time = time.time()
                self.logger.info(
                    f"📊 [Locate] SN paper info search took: {end_time - begin_time:.2f}s, retrieved in total:{len(sn_papers)}"
                )
                for i, p in enumerate(sn_papers):
                    if p and p.get("title"):
                        found_papers.append(p)
                    else:
                        self.logger.info(
                            f"❌ [Locate] SN also did not find the paper: {missing_titles[i]}"
                        )
            except Exception as e:
                self.logger.error(f"❌ [Locate] SN supplement failed: {e}")
            
        logger.info(
            f"🔍 [Locate] Found {len(found_papers)} papers, of which papers found via arxiv: {arxiv_found_papers_num}, papers found via sn: {len(found_papers)-arxiv_found_papers_num}, total papers: {len(extend_refs)}"
        )
        self.add_found_papers(found_papers, target_depth)

    def add_found_papers(self, paper_list, depth):
        """Add the list of already-searched papers to the queue"""
        for paper in paper_list:
            if not paper:
                continue

            arxiv_id = paper.get("arxiv_id", "")
            doi = paper.get("doi", "")
            title = paper.get("title", "")

            with self.lock:
                is_touched = False
                # Use OR logic: hitting any one ID counts as a duplicate
                if (arxiv_id and arxiv_id in self.root.extra["touch_ids"]) or (
                    doi and doi in self.root.extra["touch_ids"]
                ):
                    is_touched = True

                if not is_touched:
                    if arxiv_id:
                        self.root.extra["touch_ids"].append(arxiv_id)
                    if doi:
                        self.root.extra["touch_ids"].append(doi)
                else:
                    continue

            try:
                searched_paper_titles = [title]
                meta_infos = search_meta_info(searched_paper_titles)
                meta_info = meta_infos[0] if meta_infos else None
                paper_node = self.get_paper_info(paper, meta_info, depth)
                self.add_paper_to_queue(paper_node)
            except Exception as e:
                self.logger.error(f"❌ [AddPaper] Failed to add paper {title}: {e}")

    def translate(self, query):
        prompt_template = self.prompts["keyminder"]["translate"]
        prompt = prompt_template.format(query=query)

        response = self.keyminder.infer(prompt, use_cot=False)

        m_q = re.search(
            r"<query>\s*(.*?)\s*<(?:\\|/)query>", response, re.IGNORECASE | re.DOTALL
        )
        new_query = m_q.group(1).strip() if m_q else query

        m_t = re.search(
            r"<language>\s*(.*?)\s*<(?:\\|/)language>",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        self.language = m_t.group(1).strip() if m_t else self.language
        logger.info(f"🧠 [Translate] language:{self.language}")

        m_r = re.search(
            r"<reasoner>\s*(.*?)\s*<(?:\\|/)reasoner>",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        need_reasoner = False
        if m_r:
            flag = m_r.group(1).strip().lower()
            if flag in ("true"):
                need_reasoner = True
        self.need_reasoner = need_reasoner

        m_r = re.search(
            r"<uncertain_content>\s*(.*?)\s*<(?:\\|/)uncertain_content>",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        self.uncertain_content = m_r.group(1).strip() if m_r else ""

        self.logger.info(f"🧠 [Translate] response:{response}")

        return new_query

    def batch_rank(
        self,
        all_prompts,
        paper_mapping,
        filter_num,
        expand_depth,
        filter=False,
        batch_size=-1,
    ):
        if batch_size == -1:
            batch_size = self.threads_num
        for idx in range(0, len(all_prompts), batch_size):
            begin = time.time()
            prompts = all_prompts[idx : idx + batch_size]
            responses = self.ranker.batch_infer_safe(
                prompts, batch_size=batch_size, use_cot=False, logprobs=True
            )
            end = time.time()
            self.logger.info(f"🔍 [Ranker] LLM call finished inference for {len(paper_mapping)} papers, took {end-begin:.2f}s")
            paper_list = []
            for paper, response in zip(
                paper_mapping[idx : idx + batch_size], responses
            ):
                # Get the corresponding prompt
                res, prob = response
                try:
                    # Extract <relevant> True/False </relevant>
                    relevant_begin = res.find("<relevant>") + len("<relevant>")
                    relevant_end = res.find("</relevant>")
                    relevant_text = res[relevant_begin:relevant_end].strip() if relevant_end > relevant_begin else ""
                    
                    # Extract the <reason> content
                    reason_begin = res.find("<reason>") + len("<reason>")
                    reason_end = res.find("</reason>")
                    reason = res[reason_begin:reason_end].strip() if reason_end > reason_begin else ""
                    
                    # Extract the scores of all constraints in <constraint>
                    constraint_begin = res.find("<constraint>") + len("<constraint>")
                    constraint_end = res.find("</constraint>")
                    constraint_text = res[constraint_begin:constraint_end].strip() if constraint_end > constraint_begin else ""
                    constraint_text = constraint_text.split(";")
                    constraint_text = [c.strip() for c in constraint_text if c.strip()]
                    constraint_text = "\n".join(constraint_text)

                    irrelevant_begin = res.find("<irrelevant>") + len("<irrelevant>")
                    irrelevant_end = res.find("</irrelevant>")
                    irrelevant_text = res[irrelevant_begin:irrelevant_end].strip() if irrelevant_end > irrelevant_begin else ""
                    reason = f"Conclusion: {reason}\n\nConstraint checks: \n{constraint_text}\n\nIrrelevance/risk points/gaps: \n{irrelevant_text}"
                    # Parse each constraint's score Score:<1-5>
                    constraint_scores = []
                    if constraint_text:
                        # Match all Score:<number> patterns
                        score_matches = re.findall(r'Score:\s*(\d+)', constraint_text)
                        constraint_scores = [int(s) for s in score_matches if 1 <= int(s) <= 5]
                    
                    if res == "":
                        raise "response error"
                    
                    # Compute score: sum(constraint_scores) + prob, then divide by 5 * (number of constraints + 1), then multiply by 100
                    num_constraints = len(constraint_scores)
                    if num_constraints == 0:
                        logger.info(f"⚠️ [Ranker] Paper: {paper.title} | Inference anomaly")
                        score = -1
                    else:
                        # Weighting method: L2-norm (normalized sum of squares, then square root)
                        # Assuming constraint_scores = [a, b], then score = sqrt((a^2 + b^2) / (5 * len(constraint_scores))) * 100
                        mean_sq = sum(s * s for s in constraint_scores) / (25 * num_constraints)
                        score = math.sqrt(mean_sq) * 100
                        if prob != 0:
                            score = (score+prob*10)/110*100
                        # Even distribution
                        # score = (sum(constraint_scores)+prob) / (num_constraints*5+1)*100
                        # Leave 5 points of differentiation room for toprank
                        score = score*95/100
                except Exception as e:
                    logger.exception(
                        f"❌ [Ranker] Paper: {paper.title} | Cannot parse score and reasoning,{e}\n response:\n{response}"
                    )
                    score = -1
                    reason = res
                
                if score == -1:
                    paper.select_score = -1
                else:
                    paper.select_score = score
                paper.reason = reason

                if filter:
                    if paper.select_score < 30:
                        self.papers_queue.remove(paper)
                        filter_num += 1
                # self.logger.info(
                #     f"🔍 [Ranker] Paper: {paper.title} | Score: {paper.select_score} | Reasoning: {paper.reason}"
                # )
            end = time.time()
            self.logger.info(
                f"🔍 [Ranker] Finished inference for {len(paper_mapping)} papers, took {end-begin:.2f}s"
            )

    
    def rank_papers(
        self, papers, expand_depth, filter=False, batch_size=-1, all_rank=False
    ):
        """Ranker: compute relevance scores against each keyword based on the paper abstract and query"""
        name = "Ranker"
        if filter:
            name = "Selector"
        start_time = time.time()
        
        if not papers:
            return papers
        
        # If chunk retrieval is enabled, retrieve chunks before rank_papers
        if USE_CHUNK:
            if len(self.chunk_query) == 0:
                self.chunk_query = self.checklist
            self._enrich_papers_with_chunks(papers,expand_depth)

        # Generate a relevance-evaluation prompt against all keywords for each paper
        all_prompts = []
        paper_mapping = []
        for paper in papers:
            if not FREEZE_MODE and not all_rank:
                if paper.select_score != -1:
                    continue
            if "chinese" in self.language.lower():
                prompt_template = self.prompts["ranker"]["point_wise_rank_chinese"]
            else:
                prompt_template = self.prompts["ranker"]["point_wise_rank_english"]
            # constraints = [["llm agent based paper search(must be paper search not web search","not include rag based paper search"],["benchmark for evaluating the ablility of agentic paper seach, must evaluating paper search not web search","paper search tasks based on real-world network environments"],["agentic multi-tun-tool-using pipeline,must solving the problem of long context caused by tool-use ","method to solve long context during the inference"],["benchmark for evaluating the capabilities of deep-search agents, must evaluating deep-search agents","in open-web environment","LLM as judge to evaluate the results and do not rely on pre-annotated ground truths"]]

            meta_info = {
                "authors": paper.authors,
                "publication_date": paper.year,
                "citations": paper.citations,
                "journal": paper.journal,
                "CCF": paper.ccf,
                "url": paper.url,
            }
            checklist = ""
            for idx, item in enumerate(self.checklist):
                checklist += f"{idx+1}. {item}\n"
            prompt = prompt_template.format(
                title=paper.title,
                abstract=f"{paper.abstract}{paper.description}",
                user_query=self.user_query,
                meta_info=meta_info,
                constraints=checklist,
            )
            all_prompts.append(prompt)
            paper_mapping.append(paper)
        filter_num = 0
        if all_prompts:
            logger.info(f"🔍 [{name}] Starting batch inference for {len(all_prompts)} papers, total number of papers {len(papers)}")
            self.batch_rank(
                all_prompts, paper_mapping, filter_num, expand_depth, filter, batch_size
            )
        if filter:
            candidates = sorted(papers, key=lambda p: p.select_score, reverse=True)
            high_score_papers = [
                p
                for p in candidates
                if p.select_score >= 50 or p.select_score == -1
            ]
            papers = high_score_papers
            logger.info(
                f"🔍 [{name}] Filtered out a total of {filter_num} papers, selected {len(papers)} papers,"
            )
        # Sort by score
        # Check whether all papers have a score
        for paper in papers:
            if paper.select_score == -1:
                logger.info(
                    f"❌ [{name}] Paper: {paper.title} | No score found, depth:{paper.depth}, skipping"
                )
        return sorted(papers, key=lambda p: p.select_score, reverse=True)

    def top_rank(self, papers_queue):
        """Look at the best papers with consistent scores all at once, sort them based on the paper abstract and query, and assign relative scores"""
        papers = []
        for i, paper in enumerate(papers_queue):
            if i >= 20:
                break
            if (
                paper.select_score > 80
                and paper.year.replace("-", "") < self.end_date
            ):
                papers.append(paper)
            else:
                break
        if len(papers) <= 1:
            logger.info(f"[TopRanker] No high-scoring papers need fine ranking")
            return
        logger.info(f"🔍 [TopRanker] {len(papers)} papers before re-ranking")
        sorted_papers = sorted(papers, key=lambda p: p.select_score, reverse=True)

        template = self.prompts["ranker"]["top_ranker"]
        paper_info = {}
        for idx, paper in enumerate(papers):
            paper_info[f"{idx}"] = {
                "title": paper.title,
                "abstract": f"{paper.abstract}{paper.description[:int(MAX_LEN/(len(papers)))]}",
                # "description": paper.description[:MAX_LEN/len(papers)],
                "authors": paper.authors[:int(MAX_LEN/(len(papers)*10))],
                "publication_date": paper.year,
                "citations": paper.citations,
                "journal": paper.journal,
                "CCF": paper.ccf,
            }
        checklist = ""
        for idx, item in enumerate(self.checklist):
            checklist += f"{idx+1}. {item}\n"
        prompt = template.format(
            paper_info=json.dumps(paper_info),
            user_query=self.user_query,
            checklist=checklist,
            num_papers=len(papers),
        )
        response = self.top_ranker.infer(prompt, use_cot=False)
        try:
            order_begin = response.find("<order>") + len("<order>")
            order_end = response.find("</order>")
            orders = response[order_begin:order_end].strip().split(">")
            order_list = [int(id) for id in orders]
            origin_rank_labels = [
                f"{idx}.{paper.title},score:{paper.select_score}"
                for idx, paper in enumerate(papers)
            ]
            logger.info(f"order list {order_list}, origin rank:{origin_rank_labels}")
            if len(order_list) != len(papers):
                logger.error(
                    f"error,output order list {len(order_list)} != papers length {len(papers)}, response:{response}"
                )
                order_list = order_list[: len(papers)]
            
            # Compute the highest and lowest score of this batch of papers
            max_score = max(p.select_score for p in papers)+5
            min_score = min(p.select_score for p in papers)
            score_range = max_score - min_score
            
            # Re-ranking score: within this batch of papers, evenly distribute (highest score - lowest score) to get the final score
            for idx, paper_id in enumerate(order_list):
                if len(papers) == 1:
                    papers[paper_id].origin_score = papers[paper_id].select_score
                    papers[paper_id].select_score = max_score
                else:
                    # Even distribution: uniformly distributed from the highest score to the lowest score
                    ratio = idx / (len(papers) - 1) if len(papers) > 1 else 0
                    papers[paper_id].origin_score = papers[paper_id].select_score
                    papers[paper_id].select_score = max_score - (ratio * score_range)
            
            sorted_papers = sorted(papers, key=lambda p: p.select_score, reverse=True)
            logger.info(
                f"new rank:{[(paper.title,paper.select_score) for paper in sorted_papers]}\nresponse:{response}"
            )
        except Exception as e:
            logger.exception(f"❌ [TopRanker] Paper ranking failed: {e},response:{response}")

    def _expand_one_depth(self, current_depth: int) -> int:
        """
        Run one round of the "expand from the current depth to the next depth" logic,
        equivalent to one iteration of the for-loop body in the original process_coop.
        Returns the number of newly added papers this round.
        """
        # 0) Limit check
        if self.check_limit():
            self.logger.info(
                f"⏰ [PROCESS] Reached the limit condition, stopping expansion at depth {current_depth}"
            )
            return 0

        self.logger.info(
            f"\n🔄 --------------- Processing depth {current_depth} ---------------"
        )
        self.logger.info(f"\n📖 [PROCESS] Starting to filter expandable papers...")

        # 1) Only take papers at the current depth
        begin = time.time()
        papers = [p for p in self.papers_queue if p.depth == current_depth]
        if not papers:
            self.logger.info(f"📭 [PROCESS] Depth {current_depth} has no papers to process")
            return 0

        current_depth_papers = self.rank_papers(papers, current_depth, filter=True)
        end = time.time()
        self.logger.info(
            f"🔍 \033[92m [SELECTOR] Filtering expandable papers took: {end-begin:.2f} \033[0m s, number of papers processed {len(papers)}"
        )

        high_score_papers = [
            p for p in self.papers_queue if p.select_score > 30
        ]
        hight_score_papers = sorted(
            high_score_papers, key=lambda p: p.select_score, reverse=True
        )[:self.expand_papers]
        title_list = [p.title for p in hight_score_papers]
        self.logger.info(
            f"📊 \033[093m Depth {current_depth} high-scoring papers: count {len(title_list)}\n\033[0m,{title_list}"
        )

        if not current_depth_papers:
            self.logger.info(f"📭 [PROCESS] Depth {current_depth} has no papers that passed the filter")
            return 0

        self.logger.info(
            f"📊 [PROCESS] Number of papers at depth {current_depth}: {len(current_depth_papers)}"
        )
        # for i, paper in enumerate(current_depth_papers):
        #     self.logger.info(f"  📄 {i+1}. {paper.title[:60]}... (depth={paper.depth})")

        # 2) Fetch the paper details
        self.logger.info(
            f"\n📖 [PROCESS] 3.2 Fetching the details of depth {current_depth} papers..."
        )
        begin = time.time()
        papers_with_content = self.get_paper_ref(current_depth_papers)
        end = time.time()
        self.logger.info(
            f"📖 \033[92m [FetchContent] Fetching paper content took: {end-begin:.2f} \033[0m s"
        )

        # 3) Reflective retrieval + ref location expansion
        self.logger.info(
            f"\n🎯 [PROCESS] 3.3 Locating relevant sections and expanding to depth {current_depth + 1}..."
        )
        old_queue_size = len(self.papers_queue)
        begin = time.time()
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(
                self.further_search, papers_with_content, target_depth=current_depth + 1
            )
            future2 = executor.submit(
                self.locate_relevant_papers,
                papers_with_content,
                target_depth=current_depth + 1,
            )
            _ = future1.result()
            _ = future2.result()
        end = time.time()
        self.logger.info(
            f"🎯 \033[92m [FurtherSearch+Locater] Reflective retrieval + ref location expansion took: {end-begin:.2f} \033[0m s"
        )

        new_papers_count = len(self.papers_queue) - old_queue_size
        self.logger.info(
            f"📊 [PROCESS] Expanded from depth {current_depth} to depth {current_depth + 1}: {new_papers_count} new papers"
        )
        self.logger.info(f"📊 [PROCESS] Total paper queue size: {len(self.papers_queue)}")
        self.print_progress_summary(current_depth + 1)

        if new_papers_count == 0:
            self.logger.info(f"📭 [PROCESS] Depth {current_depth} did not expand any new papers")

        return new_papers_count

    def process_step(self, extra_queries, current_step) -> int:
        """
        One "fast agent step" in cooperative mode:
        - On the first call, it handles translate + keyminder + initial retrieval;
        - Afterwards, each call only expands at most max_new_layers depths from the current frontier;
        - If extra_queries is provided, run a retrieval with those queries before this round starts,
          with the depth set to the current frontier depth.
        Returns: the number of papers newly added in this step.
        """
        start_time = time.time()
        initial_cnt = len(self.papers_queue)

        # 0) First time: initialization phase (equivalent to Step 0/1/2 of the original process_coop)
        if not self.coop_initialized:

            self.search_paper_worker(
                [self.user_query.split("current date is")[0]], num=3
            )
            # Keyminder
            self.logger.info(
                f"\n🎯 =============== [COOP] Keyminder phase ==============="
            )
            begin = time.time()
            queries = self.extract_keywords_and_queries()
            if not queries:
                queries = list(extra_queries or [self.user_query])
            end = time.time()
            self.logger.info(
                f"🔍 \033[92m [Keyminder] Understanding the user question and generating queries took: {end-begin:.2f} \033[0m s,search_queries: {queries}"
            )

            # DirectSearch
            self.logger.info(
                f"\n🔍 =============== [COOP] DirectSearch phase ==============="
            )
            begin = time.time()
            self.search_papers_with_queries(queries)
            end = time.time()
            self.logger.info(
                f"🔍 \033[92m [DirectSearch] Direct search took: {end-begin:.2f} \033[0m s"
            )
            self.logger.info(f"📊 [COOP] Initial paper queue size: {len(self.papers_queue)}")

            self.coop_initialized = True

        # 1) Extra queries given by the Reasoner: run a retrieval at the current frontier depth before expanding
        seen = {str(query).strip() for query in self.searched_queries}
        fresh_queries = []
        for query in extra_queries or []:
            query = str(query).strip()
            if query and query not in seen:
                fresh_queries.append(query)
                seen.add(query)
        extra_queries = fresh_queries
        if extra_queries:
            self.logger.info(
                f"\n🔍 [COOP] Retrieving using the extra queries provided by the Reasoner, depth={self.current_depth_processed}"
            )
            self.search_papers_with_queries(
                extra_queries, current_depth=self.current_depth_processed
            )
            if str(self.current_depth_processed) in self.queries.keys():
                self.queries[str(self.current_depth_processed)].extend(extra_queries)
            else:
                self.queries[str(self.current_depth_processed)] = extra_queries

        # 2) Layered expansion: start from current_depth_processed, expanding at most max_new_layers layers
        logger.info(
            f"current_depth_processed: {self.current_depth_processed}, expand_layers: {self.expand_layers},current_step: {current_step}"
        )
        while self.current_depth_processed < current_step:
            new_cnt = self._expand_one_depth(self.current_depth_processed)
            self.current_depth_processed += 1
            if new_cnt == 0:  # This layer expanded no new papers, stop early
                break

        added = len(self.papers_queue) - initial_cnt
        self.logger.info(
            f"✅ [COOP] This step is done: added {added} papers, "
            f"took {time.time()-start_time:.2f} s"
        )
        return added

    def process_coop(self):
        """Correct processing flow: Keyminder → DirectSearch → [Locater → Expand] × N rounds → Ranker"""
        start_time = time.time()
        if FREEZE_MODE2 == False:
            if FREEZE_MODE == False:
                self.logger.info(
                    f"\n🚀 =============== [PROCESS] Starting the paper search flow ==============="
                )
                self.logger.info(f"🔍 [PROCESS] User query: {self.user_query}")
                translated = self.translate(self.user_query)
                if isinstance(translated, str):
                    self.user_query = translated
                # NEW: use the cooperative interface to run all expand_layers layers in one step
                self.process_step()

                all_papers = list(self.papers_queue)
                begin = time.time()
                file_path = f"{self.output_dir}/record"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_path = os.path.join(file_path, f"{self.idx}.json")
                save_papers = self.save_paper_nodes(all_papers)
                data = {
                    "all_papers": save_papers,
                }
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                self.logger.info(
                    f"\n🚀 \033[93m=============== Freeze mode, reading the record file directly, starting reranker \033[0m ==============="
                )
                self.logger.info(f"🔍 [PROCESS] User query: {self.user_query}")
                begin = time.time()
                file_path = f"{FREEZE_PATH}/record/{self.idx}.json"
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_papers = self.read_paper_nodes(data["all_papers"])
                for paper in all_papers:
                    self.papers_queue.append(paper)

        else:
            self.logger.info(
                f"\n🚀 \033[93m=============== Freeze mode, reading the record file directly, starting top ranker \033[0m ==============="
            )
            self.logger.info(f"🔍 [PROCESS] User query: {self.user_query}")
            path = f"{FREEZE_PATH}/{self.idx}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)["extra"]["all_papers"]
            self.papers_queue = self.read_paper_nodes(data)

        return self.papers_queue


if __name__ == "__main__":
    import argparse
    import time
    import os
    from log import setup_logging  # Use your previous logging initialization
    from model_vllm import Agent  # Already imported at the top of the file, same package

    parser = argparse.ArgumentParser(description="Quick thinker (PaperAgent) demo")
    parser.add_argument(
        "--query",
        type=str,
        default="protective coatings for porous transport layers in proton exchange membrane water electrolyzers",
        help="User query (English or Chinese)",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="How many top papers to print as the result display"
    )
    args = parser.parse_args()

    # Logging & output directory
    output_dir = os.path.join(current_path, "result")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "quick_thinker_demo.log")
    setup_logging(log_path)

    logger.info("=============== Quick Thinker Demo ===============")
    logger.info(f"User query: {args.query}")

    # Construct the three Agents (consistent with dual_agent_coordinator)
    keyminder = Agent("qwen-flash")  # Extract keywords + generate queries
    top_ranker = Agent("reason_rank")  # Small/large model used for top-rank
    ranker = Agent("qwen-flash")  # Scoring model

    paper_agent = PaperAgent(
        task_id="quick_thinker_demo",
        user_query=args.query,
        keyminder=keyminder,
        top_ranker=top_ranker,
        ranker=ranker,
        prompts_path="prompt/papermaster_prompt.json",
        expand_layers=2,  # Max allowed number of layers (overall cap), consistent with cooperative mode
        search_queries=5,
        search_papers=10,
        expand_papers=20,
        threads_num=10,
        output_dir=output_dir,
    )

    start = time.time()
    added = paper_agent.process_step(extra_queries=None)
    end = time.time()

    logger.info(f"Quick thinker step finished. New papers added: {added}")
    logger.info(f"Total papers in queue: {len(paper_agent.papers_queue)}")
    logger.info(f"Time elapsed: {end - start:.2f}s")

    # Scoring is already done inside process_step (Selector / Ranker); here just sort and take topk
    papers_sorted = sorted(
        paper_agent.papers_queue,
        key=lambda p: getattr(p, "select_score", -1),
        reverse=True,
    )

    topk = min(args.topk, len(papers_sorted))
    print("\n================ Top Papers (Quick Thinker) ================")
    for i, p in enumerate(papers_sorted[:topk], 1):
        score = getattr(p, "select_score", -1)
        depth = getattr(p, "depth", None)
        print(f"{i}. [{score:.2f}] (depth={depth}) {p.title}")
        print(f"    Source: {getattr(p, 'source', '')}")
        print(f"    Year: {getattr(p, 'year', '')}")
        print(f"    Journal: {getattr(p, 'journal', '')}")
        print(f"    Abstract: {p.abstract[:300]}...")
        print()

    print("============================================================")
    print(
        f"Total papers: {len(paper_agent.papers_queue)}, time: {end - start:.2f}s, log: {log_path}"
    )
