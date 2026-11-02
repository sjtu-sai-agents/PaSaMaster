# dual_agent_coordinator.py
from config_loader import get_config_path
import asyncio
import threading
import json
import os
import re
from typing import Any, Dict, List, Optional

from quick_thinker import PaperAgent  # the quick_thinker class
from reason_agent import reason_agent as ReasonAgent
from paper_node import PaperNode  # your existing class
from log import manager_logger as logger
from log import setup_logging
from datetime import datetime
from tools.sn_server import sn_get_paper_info
import time
current_path = os.path.dirname(os.path.abspath(__file__))
with open(get_config_path(), "r") as f:
    config = json.load(f)
    use_reasoner = config["use_reasoner"]
    use_chunk = config.get("use_chunk", False)


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError("Expected true or false")


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError("Expected a positive integer")
    return number


class SharedMind:
    """
    Thread-safe shared mind:
    - Stores the reasoner's results (parsed_query / checklist / search_query / state / paper_list)
    - Stores quick_thinker's top-k summary for each round (for the reasoner to see)
    """

    def __init__(self, user_query: str) -> None:
        self.user_query = user_query
        self._lock = threading.Lock()
        self._reason_result: Dict[str, Any] = {}
        # The top-k that quick_thinker shares with the Reasoner each round
        self._latest_top_papers: List[Dict[str, Any]] = []
        self._round_logs: List[Dict[str, Any]] = []

    def update_from_reasoner(self, result: Dict[str, Any]) -> None:
        with self._lock:
            # Handle missing fields (different versions of reason_agent may not return reasoner_state, etc.)
            parsed_query = (result.get("parsed_query") or "").strip()
            if parsed_query:
                self._reason_result["parsed_query"] = parsed_query

            checklist = result.get("checklist") or []
            if checklist:
                self._reason_result["checklist"] = list(checklist)

            if "search_query" in result:
                self._reason_result["search_query"] = list(result["search_query"] or [])

            quick_state = (result.get("quick_state") or "").strip()
            if quick_state:
                self._reason_result["quick_state"] = quick_state


            paper_list = result.get("paper_list") or []
            if paper_list:
                self._reason_result["paper_list"] = list(paper_list)

            trace = result.get("trace") or []
            if trace:
                self._reason_result["trace"] = list(trace)

    def get_reason_result(self) -> Dict[str, Any]:
        with self._lock:
            reason_result = self._reason_result.copy()
            return dict(reason_result)

    def get_reason_checklist(self) -> List[str]:
        with self._lock:
            cl = self._reason_result.get("checklist", []) or []
            return list(cl)

    def get_reason_quick_state(self) -> str:
        with self._lock:
            return self._reason_result.get("quick_state", "stop")

    def get_reason_reasoner_state(self) -> str:
        with self._lock:
            return self._reason_result.get("reasoner_state", "stop")

    def get_reason_search_query(self) -> List[str]:
        with self._lock:
            return self._reason_result.get("search_query", [])

    def get_reason_papers(self) -> List[Dict[str, Any]]:
        with self._lock:
            papers = self._reason_result.get("paper_list", []) or []
            return [dict(p) for p in papers]

    # quick_thinker writes the top-k papers here each round
    def update_quick_round(
        self, round_idx: int, top_papers: List[Dict[str, Any]]
    ) -> None:
        with self._lock:
            snapshot = {
                "round": round_idx,
                "top_papers": [dict(p) for p in top_papers],
            }
            self._latest_top_papers = snapshot["top_papers"]
            self._round_logs.append(snapshot)

    def get_latest_top_papers(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._latest_top_papers]

    def get_round_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self._round_logs]


class Manager:
    """
    The overall control class that coordinates the two agents (PasaMaster):
    - quick_thinker: responsible for actual retrieval, expansion, and scoring
    - reasoner: slow thinking enabled on demand, producing checklist / search_query / quick_state / parsed_query
    - SharedMind: used to share checklist / search_query / per-round top-k / reasoner history results

    When use_reasoner=True (serial mode):
    1. Translate user_query
    2. First call the reasoner to analyze the query and obtain parsed_query / checklist / search_query
    3. Use the reasoner's output to guide quick_thinker:
       - query becomes "[Origin Query]: {original query}\n[Parsed Query]: {parsed_query}"
       - override quick_thinker's checklist with the reasoner's checklist
       - use the reasoner's search_query as extra_queries
    4. use_chunk splitting is performed after the reasoner finishes analysis
    5. Iterate: quick_thinker retrieves -> reasoner decides whether to continue, up to max_rounds rounds

    When use_reasoner=False:
    Only quick_thinker runs directly, without involving the reasoner
    """

    def __init__(
        self,
        quick_thinker: PaperAgent,
        reason_agent_inst: ReasonAgent,
        use_reasoner: bool = False,
    ) -> None:
        self.quick_thinker = quick_thinker
        self.reason_agent = reason_agent_inst
        self.shared_mind = SharedMind(user_query=self.quick_thinker.user_query)

        self.use_reasoner: bool = use_reasoner
        self._reasoner_task: Optional[asyncio.Task] = None
        self.origin_checklist = []
        self.original_user_query = self.quick_thinker.user_query
        self.origin_query = self.quick_thinker.user_query
        self.query_fixkey: List[Dict[str, Any]] = []
        self._needs_rescore = False
        self.stop_reason = ""

    # ========== A single quick_thinker step ==========

    async def _run_quick_step(
        self,
        extra_queries: Optional[List[str]],
        current_step: int,
    ) -> int:
        """
        Call quick_thinker.process_step via a thread pool to perform one quick-agent step.
        Returns the number of papers added in this round.
        """
        logger.info(
            f"quick_thinker performing one step, extra_queries: {extra_queries}, expand layers: {self.quick_thinker.expand_layers}"
        )
        loop = asyncio.get_running_loop()
        if self._needs_rescore and self.quick_thinker.papers_queue:
            await loop.run_in_executor(
                None,
                lambda: self.quick_thinker.rank_papers(
                    self.quick_thinker.papers_queue,
                    expand_depth=current_step,
                    all_rank=True,
                ),
            )
        self._needs_rescore = False
        added = await loop.run_in_executor(
            None,
            self.quick_thinker.process_step,
            extra_queries,
            current_step,
        )
        logger.info(
            f"📚 [quick_thinker] step completed, {added} new papers added, "
            f"current total: {len(self.quick_thinker.papers_queue)}"
        )
        return added

    # ========== top-k summary for the Reasoner / self-check ==========

    def _get_top_papers_for_reasoner(self, k: int = 10) -> List[Dict[str, Any]]:
        """
        Take the top k highest-scoring papers from quick_thinker and convert them into simple dicts,
        for easy injection into the prompt of the Reasoner or quick_thinker's self-check LLM.
        """
        # TODO: should we include the top of each expansion?
        top_nodes = sorted(
            self.quick_thinker.papers_queue,
            key=lambda p: getattr(p, "select_score", -1),
            reverse=True,
        )[:k]

        result = []
        for p in top_nodes:
            abstract_raw = getattr(p, "abstract", "") or ""
            
            result.append(
                {
                    "title": getattr(p, "title", ""),
                    "abstract": abstract_raw,
                    "score": getattr(p, "select_score", -1),
                    "reason": getattr(p, "reason", ""),
                }
            )
        return result


    def _build_reasoner_decision_prompt(
        self,
        round_idx: int,
        top_papers: List[Dict[str, Any]],
    ) -> str:
        """
        After each round of quick_thinker retrieval: hand the top papers + quick_thinker state to the reasoner,
        so it can decide whether to continue retrieving based on "existing retrieval results + this round's top5".
        """
        # quick_thinker's current checklist / already-used queries
        qt_checklist = getattr(self.quick_thinker, "checklist", []) or []
        qt_checklist_str = ";".join([str(x).strip() for x in qt_checklist if str(x).strip()])
        qt_queries = sorted(getattr(self.quick_thinker, "searched_queries", set()) or [])
        qt_queries_str = ";".join([str(x).strip() for x in qt_queries if str(x).strip()])

        if top_papers:
            lines = []
            for i, p in enumerate(top_papers, 1):
                title = p.get("title", "")
                abstract = p.get("abstract", "") or ""
                lines.append(
                    f"{i}. Title: {title}\n"
                    f"   Abstract: {abstract}\n"
                    f"   Relevance score: {p.get('score', -1)}\n"
                    f"   Assessment: {p.get('reason', '')}"
                )
            papers_block = "\n\n".join(lines)
        else:
            papers_block = "(No candidate papers yet.)"

        # Round 1: explicitly give quick_thinker's uncertain info to the reasoner, asking it to clarify in parsed_query
        uncertain_block = ""
        if round_idx == 1:
            uncertain_content = (
                getattr(self.quick_thinker, "uncertain_content", "") or ""
            ).strip()
            if uncertain_content:
                uncertain_block = (
                    "Uncertain points (QuickThinker is not confident about the following):\n"
                    f"{uncertain_content}\n\n"
                    "Please clarify these points and make the clarification explicit in <parsed_query> and <checklist>.\n\n"
                )
            else:
                uncertain_block = (
                    "If any part of the user query is ambiguous or underspecified, clarify it and make the clarification explicit in <parsed_query>.\n\n"
                )

        prompt = (
            f"[Round {round_idx}] QuickThinker has finished searching.\n\n"
            f"User query:\n{self.shared_mind.user_query}\n\n"
            f"{uncertain_block}"
            f"QuickThinker checklist:\n{qt_checklist_str}\n\n"
            f"QuickThinker search_query used so far:\n{qt_queries_str}\n\n"
            f"Top-{len(top_papers)} papers (title+abstract):\n{papers_block}\n\n"
            "Decide whether QuickThinker should continue searching.\n"
            "- If the current queries and all of the top papers already cover the user's needs with no obvious gaps, output <state>stop</state>.\n"
            "- Otherwise, output <state>continue</state> and propose improved/extra queries in <query> (use ';' to separate).\n"
            "- For both 'stop' and 'continue' decisions, you MUST provide your detailed reasoning in <reasoning> tags to explain why you made this judgment.\n\n"
            "When generating the XML tags (<parsed_query>, <keywords>, <checkpoints>, <query>):\n"
            "- If the current ones already fully satisfy the user's query, output them exactly as their current values.\n"
            "- If they are missing some content or can be improved based on the latest search results, refine and complete them.\n"
            "- Use the QuickThinker checklist above as the primary reference for <checkpoints> (keep ';' separated).\n\n"
            "Output ONLY the following XML tags (no extra text):\n"
            "<parsed_query>...</parsed_query>\n"
            "<keywords>kw1,kw2,...</keywords>\n"
            "<checkpoints>cp1;cp2;cp3;...</checkpoints>\n"
            "<query>q1;q2;q3;...</query>\n"
            "<reasoning>your detailed reasoning for the decision...</reasoning>\n"
            "<state>continue|stop</state>\n"
        )
        return prompt

    def _sync_user_query_from_parsed_query(self, parsed_query: str) -> None:
        """
        After the reasoner provides parsed_query, uniformly build the
        "[Origin Query] + [Parsed Query]" concatenated query,
        and synchronously update quick_thinker.user_query and shared_mind.user_query.
        """
        pq = (parsed_query or "").strip()
        if not pq:
            return

        new_query = (
            f"[Origin Query]: {self.origin_query}\n"
            f"[Parsed Query]: {pq}"
        )
        self.quick_thinker.user_query = new_query
        self.shared_mind.user_query = new_query

    def _record_query_fixkey(self, round_idx: int, query_value: str) -> None:
        """Save the origin+parsed concatenated query for each round."""
        if not query_value:
            return

        snapshot = {"round": round_idx, "query_fix": query_value}
        for idx, item in enumerate(self.query_fixkey):
            if item.get("round") == round_idx:
                self.query_fixkey[idx] = snapshot
                return
        self.query_fixkey.append(snapshot)

    def _sync_quick_thinker_from_shared_mind(self, round_idx: int) -> None:
        """
        Each time SharedMind receives a new reasoner result,
        sync the latest parsed_query / checklist / search_query back to quick_thinker.
        """
        reason_result = self.shared_mind.get_reason_result()
        parsed_query = (reason_result.get("parsed_query") or "").strip()
        checklist = reason_result.get("checklist") or []
        search_query = reason_result.get("search_query") or []
        old_context = (
            self.quick_thinker.user_query,
            tuple(self.quick_thinker.checklist or []),
        )

        if parsed_query:
            self._sync_user_query_from_parsed_query(parsed_query)
            self._record_query_fixkey(round_idx, self.quick_thinker.user_query)
            logger.info(f"📝 [Manager] Combined query after sync:\n{self.quick_thinker.user_query}")

        if checklist:
            self.quick_thinker.checklist = list(checklist)
            logger.info(f"📋 [Manager] Synced Reasoner checklist: {checklist}")

        self.quick_thinker.parsed_query = parsed_query
        self.quick_thinker.reasoner_search_query = list(search_query)
        new_context = (
            self.quick_thinker.user_query,
            tuple(self.quick_thinker.checklist or []),
        )
        if new_context != old_context:
            self._needs_rescore = True

    def _new_search_queries(self, queries):
        seen = {str(q).strip() for q in self.quick_thinker.searched_queries}
        fresh = []
        for query in queries:
            query = str(query).strip()
            if query and query not in seen:
                fresh.append(query)
                seen.add(query)
        return fresh


    # ========== Persist a SharedMind snapshot each round ==========

    def _save_sharedmind_round_log(
        self,
        round_idx: int,
        top_papers_for_reasoner: List[Dict[str, Any]],
        reasoner_result: Optional[Dict[str, Any]],
        quick_state: str,
        reasoner_state: str,
    ) -> None:
        """
        After each round, save the SharedMind state + this round's top-k + reasoner output (if any)
        together to a file for easier analysis.
        """
        try:
            log_dir = os.path.join(self.quick_thinker.output_dir, "shared_mind_logs")
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(
                log_dir, f"{self.quick_thinker.task_id}_round{round_idx}.json"
            )

            payload = {
                "task_id": self.quick_thinker.task_id,
                "round": round_idx,
                "user_query": self.shared_mind.user_query,
                "use_reasoner": self.use_reasoner,
                "use_chunk": use_chunk,
                "chunk_query": getattr(self.quick_thinker, "chunk_query", []),
                "quick_state": quick_state,
                "reasoner_state": reasoner_state,
                "top_papers_for_reasoner": top_papers_for_reasoner,
                "reasoner_result": reasoner_result or {},
                "shared_mind_reason_result": self.shared_mind.get_reason_result(),
                "shared_mind_round_logs": self.shared_mind.get_round_logs(),
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)

            logger.info(
                f"📝 [Manager] Saved SharedMind round {round_idx} snapshot to: {log_path}"
            )
        except Exception as e:
            logger.exception(f"❌ [Manager] Failed to save SharedMind snapshot: {e}")

    # ========== 2. Build the fused candidate paper list ==========

    @staticmethod
    def _maybe_extract_arxiv_id(url: str) -> Optional[str]:
        """
        Try to extract arxiv_id from the URL; a simple implementation.
        Supports forms like:
        - https://arxiv.org/abs/2501.12345
        - https://arxiv.org/pdf/2501.12345.pdf
        """
        if not url:
            return None
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url)
        if m:
            return m.group(1)
        return None

    def _build_fusion_candidates(self) -> List[PaperNode]:
        """
        Merge quick_thinker.papers_queue and reasoner.paper_list into a single PaperNode list.
        - Deduplicate by title
        - Convert the reasoner's paper_list into PaperNode
        - If a reasoner paper is already found by quick_thinker, add a description key to it
        """
        fusion_nodes: List[PaperNode] = []
        seen_titles = set()

        # 1) Papers already found by quick_thinker
        for p in self.quick_thinker.papers_queue:
            title_norm = (p.title or "").strip().lower()
            if not title_norm:
                continue
            if title_norm in seen_titles:
                continue
            seen_titles.add(title_norm)
            fusion_nodes.append(p)

        # 2) Papers found by the reasoner (dict -> PaperNode)
        reason_papers = self.shared_mind.get_reason_papers()
        logger.info(f"reasoner find {len(reason_papers)} papers")
        reason_titles_sn = [
            (rp.get("title") or "").strip().replace("[PDF]", "") for rp in reason_papers
        ]
        paper_info_list_sn = sn_get_paper_info(reason_titles_sn)
        for paper_info, reason_paper in zip(paper_info_list_sn, reason_papers):
            if paper_info is None:
                continue
            reason_paper["abstract"] = paper_info.get("abstract", "")
            reason_paper["authors"] = paper_info.get("authors", "")
            reason_paper["year"] = paper_info.get("year", "")
            reason_paper["url"] = paper_info.get("url", "")
            reason_paper["journal"] = paper_info.get("journal", "")
            reason_paper["title"] = paper_info.get("title", "")
            reason_paper["doi"] = paper_info.get("doi", "")
            reason_paper["source"] = paper_info.get("source", "")

        count = 0
        for rp in reason_papers:
            title = (rp.get("title") or "").strip()
            if not title:
                continue
            title_norm = title.lower()
            if title_norm in seen_titles:
                logger.info(
                    f"reasoner paper {title} already in quick_thinker, add description"
                )
                for paper in fusion_nodes:
                    if paper.title.strip().lower() == title_norm:
                        paper.description = rp.get(
                            "description", ""
                        )  # add description
                        break
                continue
            seen_titles.add(title_norm)

            url = rp.get("url", "")
            arxiv_id = self._maybe_extract_arxiv_id(url)

            paper_node = PaperNode(
                {
                    "title": title,
                    "arxiv_id": arxiv_id or url,  # use url as a fallback when there is no arxiv_id
                    "depth": 0,
                    "abstract": "",
                    "description": rp.get("description", ""),
                    "sections": "",  # the reasoner has no sections, leave empty for now
                    "journal": rp.get("journal", ""),
                    "authors": rp.get("authors", ""),
                    "year": rp.get("year", ""),
                    "h5": -1,
                    "IF": -1,
                    "CCF": "",
                    "citations": -1,
                    "source": "reasoner",
                    "select_score": -1,
                    "extra": {"from_reasoner": True},
                    "relevant_section": {},
                    "relevant_section_ref": {},
                }
            )
            logger.info(f"reasoner paper node: {title}")
            fusion_nodes.append(paper_node)
            count += 1
            # the reasoner finds at most 100 papers
            if count > 100:
                break

        logger.info(f"fusion nodes: {len(fusion_nodes)}")
        return fusion_nodes

    # ========== 3. Final fused scoring ==========

    def _final_rerank(
        self,
        fusion_nodes: List[PaperNode],
        expand_depth: int,
    ):
        """
        Use quick_thinker's built-in ranker to uniformly score the fused paper list.

        checklist selection logic:
        - If the reasoner provided a checklist, use it
        - Otherwise fall back to quick_thinker's own checklist

        Also record:
        - papers whose score / reason changed before vs. after rerank
        - the top-N paper lists before vs. after top_rank
        """
        # 1) checklist selection logic
        reason_checklist = self.shared_mind.get_reason_checklist()
        base_checklist = getattr(self.quick_thinker, "checklist", [])
        if not reason_checklist:
            logger.info("Reasoner did not provide a checklist, using quick_thinker's checklist")
        final_checklist = reason_checklist if reason_checklist else base_checklist
        logger.info(
            f"final checklist: {final_checklist}, base checklist: {self.origin_checklist}"
        )

        # 2) Record a snapshot of scores / reasons before rank_papers
        pre_info = {}
        for p in fusion_nodes:
            pre_info[id(p)] = {
                "title": getattr(p, "title", ""),
                "abstract": getattr(p, "abstract", ""),
                "score": getattr(p, "select_score", -1),
                "reason": getattr(p, "reason", ""),
                "source": getattr(p, "source", ""),
                "depth": getattr(p, "depth", None),
            }
        # 3) Call rank_papers with the final checklist
        old_checklist = getattr(self.quick_thinker, "checklist", [])
        self.quick_thinker.checklist = final_checklist
        ranked = self.quick_thinker.rank_papers(
            fusion_nodes, expand_depth=expand_depth, filter=False, batch_size=50,
            all_rank=True,
        )

        # 4) Collect papers that changed before vs. after rank_papers
        rerank_changes: List[Dict[str, Any]] = []
        for p in fusion_nodes:
            pid = id(p)
            if pid not in pre_info:
                continue
            before = pre_info[pid]
            after_score = getattr(p, "select_score", -1)
            after_reason = getattr(p, "reason", "")
            if after_score != before["score"]:
                rerank_changes.append(
                    {
                        "title": getattr(p, "title", ""),
                        "abstract": getattr(p, "abstract", ""),
                        "score_before": before["score"],
                        "score_after": after_score,
                        "reason_before": before["reason"],
                        "reason_after": after_reason,
                        "source": getattr(p, "source", ""),
                        "depth": getattr(p, "depth", None),
                    }
                )

        TOP_N = 20
        top_before: List[Dict[str, Any]] = []
        for p in ranked[:TOP_N]:
            top_before.append(
                {
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                    "score": getattr(p, "select_score", -1),
                    "reason": getattr(p, "reason", ""),
                    "source": getattr(p, "source", ""),
                    "depth": getattr(p, "depth", None),
                }
            )

        # First let quick_thinker do a "fine ranking"
        self.quick_thinker.top_rank(ranked)

        ranked_after_top = sorted(
            ranked,
            key=lambda p: getattr(p, "select_score", -1),
            reverse=True,
        )

        top_after: List[Dict[str, Any]] = []
        for p in ranked_after_top[:TOP_N]:
            top_after.append(
                {
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                    "score": getattr(p, "select_score", -1),
                    "reason": getattr(p, "reason", ""),
                    "source": getattr(p, "source", ""),
                    "depth": getattr(p, "depth", None),
                }
            )

        # Restore the original checklist (if you want)
        self.quick_thinker.checklist = old_checklist

        return ranked_after_top, rerank_changes, top_before, top_after, final_checklist

    # ========== 4. Unified external entry point ==========

    def _do_chunk_splitting(self) -> None:
        """Shared logic for use_chunk splitting of the retrieval query."""
        logger.info("🔍 [Manager] Starting to split the query into chunk retrieval queries...")
        try:
            if not self.quick_thinker.checklist:
                self.quick_thinker.extract_keywords_and_queries()
            chunk_queries = self.quick_thinker._split_query_keywords_with_llm(
                user_query=self.quick_thinker.user_query,
                checklist=self.quick_thinker.checklist,
            )
            if not chunk_queries:
                chunk_queries = [self.quick_thinker.user_query]
            self.quick_thinker.chunk_query = chunk_queries
            logger.info(f"📝 [Manager] Chunk queries: {chunk_queries}")
        except Exception as e:
            logger.exception(f"⚠️ [Manager] Error while splitting chunk query: {e}")
            self.quick_thinker.chunk_query = []

    async def run(
        self,
        layers_per_round: int = 1,
        max_rounds: int = 3,
        topk_for_reasoner: int = 10,
    ) -> Dict[str, Any]:
        """
        Multi-round coordination logic.

        When use_reasoner=True:
          1. First call the reasoner to analyze the query -> parsed_query / checklist / search_query
          2. Use the reasoner's output to guide quick_thinker
             (query = "[Origin Query]: ... \\n[Parsed Query]: ...")
          3. quick_thinker's results are handed back to the reasoner to decide whether to continue, iterating up to max_rounds times
          4. use_chunk splitting is performed after the reasoner finishes analysis

        When use_reasoner=False:
          Only quick_thinker runs directly
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        start_time = time.time()
        coop_logs: List[Dict[str, Any]] = []
        expand_depth = 0
        round_idx = 0
        self.stop_reason = ""

        # ===== 0. Translation phase =====
        logger.info("🚀 [Manager] Starting translation and initial retrieval planning ...")
        begin = time.time()
        try:
            translated = self.quick_thinker.translate(self.quick_thinker.user_query)
            if isinstance(translated, str):
                self.quick_thinker.user_query = translated
                self.shared_mind.user_query = translated
        except Exception as e:
            logger.exception(f"⚠️ [Manager] Error in quick_thinker.translate: {e}")
        end = time.time()
        logger.info(
            f"⏱️\033[92m [Manager] Translation and initial retrieval planning completed, took {end - begin:.2f} s.\033[0m"
        )
        
        # date_suffix = f"current date is {datetime.now().strftime('%Y-%m-%d')}."
        # self.quick_thinker.user_query += date_suffix
        # self.shared_mind.user_query += date_suffix
        self.origin_query = self.quick_thinker.user_query
        logger.info(
            f"📝 [Manager] Translated user query: {self.quick_thinker.user_query}"
        )
        logger.info(f"🧠 [Manager] use_reasoner={self.use_reasoner}")

        if self.use_reasoner:
            round_idx = await self._run_with_reasoner(
                max_rounds, topk_for_reasoner, expand_depth, coop_logs,
            )
        else:
            round_idx = await self._run_without_reasoner(
                max_rounds, expand_depth, coop_logs,
            )
        expand_depth = round_idx  # expand_depth += 1 each round, ultimately equal to round_idx

        # ===== Coordination finished: do a single unified rerank =====
        begin = time.time()
        fusion_nodes = self._build_fusion_candidates()
        (ranked_nodes, rerank_changes, top_before, top_after, final_checklist) = (
            self._final_rerank(fusion_nodes, expand_depth)
        )
        end = time.time()
        logger.info(f"⏱️ rerank runtime: {end - begin:.2f} s")

        final_papers: List[Dict[str, Any]] = []
        for p in ranked_nodes:
            arxiv_id = getattr(p, "arxiv_id", None)
            link = None
            if isinstance(arxiv_id, str) and arxiv_id and "://" not in arxiv_id:
                link = f"https://arxiv.org/pdf/{arxiv_id}"
            elif isinstance(arxiv_id, str) and "://" in arxiv_id:
                link = arxiv_id

            final_papers.append(
                {
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                    "authors": getattr(p, "authors", ""),
                    "journal": getattr(p, "journal", ""),
                    "year": getattr(p, "year", ""),
                    "score": getattr(p, "select_score", -1),
                    "depth": getattr(p, "depth", None),
                    "source": getattr(p, "source", ""),
                    "reason": getattr(p, "reason", ""),
                    "url": link,
                }
            )

        final_results = {
            "expand_depth": expand_depth,
            "max_rounds": max_rounds,
            "stop_reason": self.stop_reason,
            "cooperation_rounds": coop_logs,
            "search_queries": self.quick_thinker.queries,
            "final_checklist": final_checklist,
            "base_checklist": self.origin_checklist,
            "chunk_query": getattr(self.quick_thinker, "chunk_query", []),
            "all_papers_title": [(p.title, p.select_score) for p in ranked_nodes],
            "all_papers": [paper.todic() for paper in ranked_nodes],
            "rerank_changes": rerank_changes,
            "top_rank_before": top_before,
            "top_rank_after": top_after,
        }

        end_time = time.time()
        logger.info(
            f"\n🏁 \033[92m [Manager] Coordinated run finished, total time {end_time - start_time:.2f} s,"
            f"{round_idx} rounds executed, use_reasoner={self.use_reasoner}.\033[0m"
        )

        return {
            "query": self.original_user_query,
            "query_fixkey": list(self.query_fixkey),
            "extra": final_results,
            "final_papers": final_papers,
            "reason_trace": self.reason_agent.trace if self.use_reasoner else [],
            "reason_papers": self.reason_agent.paper_list if self.use_reasoner else [],
        }

    # ------------------------------------------------------------------
    # Path A: use_reasoner=True — first the Reasoner analyzes, then serially guides QuickThinker
    # ------------------------------------------------------------------

    async def _run_with_reasoner(
        self,
        max_rounds: int,
        topk_for_reasoner: int,
        expand_depth: int,
        coop_logs: List[Dict[str, Any]],
    ) -> int:
        """Return the number of rounds actually executed (also equals the final expand_depth)."""
        reasoner_state = "stop"
        pending_extra_queries: List[str] = []

        # ---- A1: Reasoner analyzes the query first ----
        logger.info("🧠 [Manager] Calling Reasoner to analyze query first ...")
        begin = time.time()
        try:
            initial_result = await self.reason_agent.run(
                self.quick_thinker.user_query
            )
        except Exception as e:
            logger.exception(f"⚠️ [Manager] Reasoner initial analysis error: {e}")
            initial_result = None
        end = time.time()
        logger.info(f"⏱️ [Manager] Reasoner initial analysis done, took {end - begin:.2f} s")

        if initial_result:
            self.shared_mind.update_from_reasoner(initial_result)
            self._sync_quick_thinker_from_shared_mind(round_idx=0)

        # ---- A2: Use Reasoner output to guide QuickThinker ----
        pending_extra_queries = self.shared_mind.get_reason_search_query()
        logger.info(f"🔍 [Manager] Search queries provided by Reasoner: {pending_extra_queries}")

        # ---- A3: use_chunk splitting (performed after Reasoner analysis) ----
        if use_chunk:
            self._do_chunk_splitting()
        # ---- A4: Iterative loop: QuickThinker -> Reasoner decision, up to max_rounds rounds ----
        round_idx = 0
        quick_state = "continue"

        for round_idx in range(1, max_rounds + 1):
            logger.info(
                f"\n🔁 \033[92m [Manager] ===== Cooperation round {round_idx} begins =====\033[0m"
            )

            # QuickThinker step
            queries_this_round = list(pending_extra_queries)
            expand_depth += 1
            begin = time.time()
            added = await self._run_quick_step(pending_extra_queries, expand_depth)
            end = time.time()
            logger.info(
                f"⏱️ QuickThinker round {round_idx} runtime: {end - begin:.2f} s"
            )
            pending_extra_queries = []

            if round_idx == 1:
                self.origin_checklist = (
                    getattr(self.quick_thinker, "checklist", []) or []
                )

            # Take top papers for the Reasoner to review
            top_papers = self._get_top_papers_for_reasoner(topk_for_reasoner)
            self.shared_mind.update_quick_round(round_idx, top_papers)

            # Reasoner decision: whether to keep searching
            decision_prompt = self._build_reasoner_decision_prompt(
                round_idx, top_papers
            )
            reasoner_result: Optional[Dict[str, Any]] = None
            try:
                begin = time.time()
                reasoner_result = await self.reason_agent.run(decision_prompt)
                end = time.time()
                logger.info(
                    f"⏱️ Reasoner round {round_idx} decision time: {end - begin:.2f} s"
                )
            except Exception as e:
                logger.exception(f"⚠️ [Manager] Reasoner decision error: {e}")
                reasoner_result = None
            reasoner_result = reasoner_result or {}
            if reasoner_result:
                self.shared_mind.update_from_reasoner(reasoner_result)
                self._sync_quick_thinker_from_shared_mind(round_idx=round_idx)
                pending_extra_queries = self._new_search_queries(
                    self.shared_mind.get_reason_search_query()
                )
                quick_state = self.shared_mind.get_reason_quick_state()
            else:
                quick_state = "stop"
                pending_extra_queries = []

            logger.info(
                f"🧠 [Round {round_idx}] added={added}, state={quick_state}, "
                f"extra_queries={pending_extra_queries}"
            )

            coop_logs.append(
                {
                    "round": round_idx,
                    "added_papers": added,
                    "search_queries": queries_this_round,
                    "next_search_queries": list(pending_extra_queries),
                    "top_papers": top_papers,
                    "reasoner_output": reasoner_result,
                    "quick_state": quick_state,
                    "reasoner_state": reasoner_state,
                    "use_reasoner": True,
                }
            )
            self._save_sharedmind_round_log(
                round_idx, top_papers, reasoner_result,
                quick_state, reasoner_state,
            )

            if quick_state == "stop":
                self.stop_reason = "navigator_stop"
                logger.info("🛑 [Manager] quick_state=stop, ending cooperation loop.")
                break
            if added == 0 and not pending_extra_queries:
                self.stop_reason = "no_progress"
                logger.info("📭 [Manager] No new papers or new search directions, ending cooperation loop.")
                break
        else:
            self.stop_reason = "max_rounds"

        return round_idx if round_idx > 0 else 1

    # ------------------------------------------------------------------
    # Path B: use_reasoner=False — run QuickThinker directly only
    # ------------------------------------------------------------------

    async def _run_without_reasoner(
        self,
        max_rounds: int,
        expand_depth: int,
        coop_logs: List[Dict[str, Any]],
    ) -> int:
        """Return the number of rounds actually executed (also equals the final expand_depth)."""
        logger.info("⚡ [Manager] use_reasoner=False, running QuickThinker directly only.")

        # chunk splitting (does not depend on Reasoner)
        if use_chunk:
            self._do_chunk_splitting()

        round_idx = 0
        for round_idx in range(1, max_rounds + 1):
            logger.info(
                f"\n🔁 \033[92m [Manager] ===== QuickThinker round {round_idx} =====\033[0m"
            )
            expand_depth += 1
            begin = time.time()
            added = await self._run_quick_step([], expand_depth)
            end = time.time()
            logger.info(
                f"⏱️ QuickThinker round {round_idx} runtime: {end - begin:.2f} s"
            )

            if round_idx == 1:
                self.origin_checklist = (
                    getattr(self.quick_thinker, "checklist", []) or []
                )

            coop_logs.append(
                {
                    "round": round_idx,
                    "added_papers": added,
                    "top_papers": [],
                    "reasoner_output": {},
                    "quick_state": "continue" if round_idx < max_rounds else "stop",
                    "reasoner_state": "stop",
                    "use_reasoner": False,
                }
            )

            if added == 0:
                self.stop_reason = "no_progress"
                logger.info(
                    "📭 [Manager] No new papers expanded this round, ending early."
                )
                break
        else:
            self.stop_reason = "max_rounds"

        return round_idx if round_idx > 0 else 1


async def main(
    idx,
    user_query: str,
    answer: str,
    output_folder: str,
    end_date=datetime.now().strftime("%Y%m%d"),
    expand_layers=1,
    use_reasoner=False,
    max_rounds=3,
):
    start = time.time()
    # 1) Build quick_thinker
    keyminder = Agent("deepseek-v3.2")
    top_ranker = Agent("reason_rank")
    ranker = Agent("qwen-30b-train")
    logger.info("begin to init quick_thinker...")
    quick_thinker = PaperAgent(
        task_id=f"query_{idx}",
        user_query=user_query,
        keyminder=keyminder,
        top_ranker=top_ranker,
        ranker=ranker,
        prompts_path=f"{current_path}/prompt/papermaster_prompt.json",
        expand_layers=expand_layers,
        search_queries=5,
        search_papers=10,
        expand_papers=80,
        threads_num=50,
        end_date=end_date,
        output_dir=output_folder,
    )

    logger.info("begin to init reasoner...")
    # 2) Build reasoner
    reasoner = ReasonAgent(origin_query=user_query)
    reasoner.set_stream_file(os.path.join(output_folder, f"reasoner_stream_{idx}.log"))

    logger.info("begin to run manager...")
    # 3) Build and run Manager
    coordinator = Manager(
        quick_thinker=quick_thinker,
        reason_agent_inst=reasoner,
        use_reasoner=use_reasoner,
    )

    # 4) Run the whole pipeline
    begin = time.time()
    try:
        result = await coordinator.run(max_rounds=max_rounds)
    finally:
        stream_file = getattr(reasoner, "stream_file", None)
        if stream_file is not None:
            stream_file.close()
    end = time.time()
    result["execution_time"] = {
        "paper_agent_run_duration": end - begin,
        "total_batch_time": end - start,
    }

    result["user_query"] = user_query
    result["extra"]["answer"] = answer

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{idx}.json")
    tmp_path = None
    try:
        import tempfile

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{idx}.", suffix=".json.tmp", dir=output_folder
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, save_path)  # atomic on POSIX
        tmp_path = None
        print(f"Final ranked papers saved to {save_path}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import time

    from model_vllm import Agent  # the class you already have
    import os, sys

    current_dir = os.path.dirname(os.path.abspath(__file__))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max_rounds", type=_positive_int, default=3,
                        help="Maximum retrieval/feedback rounds (default: 3; may stop early).")
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"{current_dir}/data/pasamaster_bench.jsonl",
        help="Input benchmark JSONL (default: data/pasamaster_bench.jsonl)",
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        default="",
        help="Output directory for resuming from a checkpoint; if empty, a new timestamped directory is created. If {idx}.json already exists in the directory, the corresponding task is skipped.",
    )
    parser.add_argument(
        "--use_reasoner",
        type=_parse_bool,
        default=use_reasoner,
        help="Whether to use the reasoner",
    )

    args = parser.parse_args()
    use_reasoner = args.use_reasoner
    MAX_WORKERS = args.max_workers
    max_num = -1
    data_path = args.data_path
    continue_path = (args.continue_path or "").strip()
    if continue_path:
        output_folder = os.path.abspath(continue_path)
    else:
        output_folder = f"{current_dir}/result/{args.data_path.split('/')[-1].split('.')[0]}_{time.strftime('%Y%m%d_%H%M')}"

    data_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    # data_list = [
    #     {"question": "electrolyte formulation"}
    # ]
    # data_list = [{"question":"help me find articles similar to websailor that top the hle leaderboard"}]
    # data_list = [{"question": "qwen deepresearch paper"}]
    begin = time.time()
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime

    os.makedirs(output_folder, exist_ok=True)
    setup_logging(os.path.join(output_folder, "detail_run.log"))

    def run_one_task(idx, data, output_folder):
        user_query = data["question"]
        end_date = data["source_meta"]["published_time"] or datetime.now().strftime("%Y%m%d")
        answer = data.get("answer", "")

        print(f"Processing task_id: {idx}, user_query: {user_query}")
        asyncio.run(
            main(
                idx,
                user_query=user_query,
                answer=answer,
                output_folder=output_folder,
                end_date=end_date,
                use_reasoner=use_reasoner,
                max_rounds=args.max_rounds,
            )
        )

    def _iter_tasks(data_list, max_num):
        if max_num is None or int(max_num) < 0:
            return enumerate(data_list)
        return enumerate(data_list[: int(max_num)])

    def _is_task_done(output_folder: str, idx: int) -> bool:
        p = os.path.join(output_folder, f"{idx}.json")
        try:
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                return False
            # Validate the JSON to avoid a half-written file from a previous interruption causing a "false skip"
            with open(p, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except OSError:
            return False
        except Exception:
            return False

    def run_all(data_list, max_num, output_folder):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, data in _iter_tasks(data_list, max_num):
                if continue_path and _is_task_done(output_folder, idx):
                    print(f"Skip done task_id: {idx}, result exists: {os.path.join(output_folder, f'{idx}.json')}")
                    continue
                futures.append(executor.submit(run_one_task, idx, data, output_folder))

            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"Task error: {e}")

    run_all(data_list, max_num, output_folder)
    end = time.time()
    print(f"\nTotal Time elapsed: {end - begin:.2f}s")
