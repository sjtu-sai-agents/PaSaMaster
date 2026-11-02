import re
import ast
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Callable, Optional, Awaitable, Any
import sys
import os

import httpx
from openai import APITimeoutError, APIConnectionError, APIError

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path))

from prompt.reasoner_prompt import prompt as SYSTEM_PROMPT
from tools.search import Search
from tools.visit import Visit
from tools.llm_call import model_config
import openai
from log import reason_logger as logger
from log import setup_logging
from difflib import SequenceMatcher
from utils import is_pdf_url


# ============================================================
# Tools Schema (LLM function-calling definitions; defined as module-level constants since they are class-independent)
# ============================================================

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Performs batched web searches: supply an array 'query'; "
                "the tool retrieves the top 10 results for each query in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of query strings. Include multiple complementary "
                            "search queries in a single call. "
                            "The number of queries should less than 5"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Parse webpage(s) and return the summary of the content according to the goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Provide an array of URLs.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit for webpage(s).",
                    },
                },
                "required": ["url", "goal"],
            },
        },
    },
]

_FINALIZE_MSG = (
    "<|FINALIZE|>: You must give your response immediately "
    "following the required format and based on your current "
    "understanding without any further tool calls."
)


class ReasonAgent:
    """
    Reasoner Agent: completes reasoning tasks through an LLM + search/visit tool loop.

    Public entry point: async run(user_query) -> Dict
    """

    # ============================================================
    # 1. Initialization
    # ============================================================

    def __init__(
        self,
        origin_query: str = "",
        system_prompt: Optional[str] = None,
        max_tool_calls: int = 10,
        model_chat_func: Optional[
            Callable[
                [List[Dict[str, Any]], List[Dict[str, Any]]],
                Awaitable[Dict[str, Any]],
            ]
        ] = None,
        search_func: Optional[Callable[[List[str]], Awaitable[str]]] = None,
        visit_func: Optional[
            Callable[[List[Tuple[str, str]]], Awaitable[str]]
        ] = None,
    ) -> None:
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_tool_calls = max_tool_calls
        self.cooperate_times = 0
        self.origin_query = origin_query

        self.call_model = model_chat_func or self._default_call_model
        self.call_search = search_func or self._default_call_search
        self.call_visit = visit_func or self._default_call_visit

        self.paper_list: List[Dict[str, str]] = []
        self._seen_pdf_urls: set[str] = set()

        self.trace: List[Dict[str, Any]] = []
        self.message_history: List[Dict[str, Any]] = []

        self.stream_file = None
        self.stream_file_path = None
        self.interrupt_prompt = None
        self.response_now = False

        self.tools: List[Dict[str, Any]] = TOOLS_SCHEMA

    # ============================================================
    # 2. Streaming output
    # ============================================================

    def set_stream_file(self, file_path: str, mode: str = "a", encoding: str = "utf-8"):
        """Set the real-time output file; both the model's streaming output and tool results are written to it."""
        if getattr(self, "stream_file", None):
            try:
                self.stream_file.close()
            except Exception:
                pass
        self.stream_file_path = file_path
        self.stream_file = open(file_path, mode, encoding=encoding, buffering=1)

    def _stream_write(
        self,
        text: str,
        end: str = "",
        flush: bool = True,
        *,
        color_ansi: str | None = None,
    ):
        """Unified streaming output: stdout + optional file write."""
        if color_ansi:
            print(f"{color_ansi}{text}\033[0m", end=end, flush=flush)
        else:
            print(text, end=end, flush=flush)

        if self.stream_file:
            try:
                self.stream_file.write(text + end)
                if flush:
                    self.stream_file.flush()
            except Exception as e:
                logger.warning(f"write stream_file error: {e}")

    # ============================================================
    # 3. Message handling (sanitization & context pruning)
    # ============================================================

    @staticmethod
    def _sanitize_messages_for_api(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build the messages to send to the API: drop irrelevant fields; keep the assistant's tool_calls and
        reasoning_content (compatible with multi-turn interfaces that support chain-of-thought).
        """
        sanitized: List[Dict[str, Any]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip()

            if role in ("system", "user"):
                sanitized.append({"role": role, "content": m.get("content") or ""})

            elif role == "assistant":
                nm: Dict[str, Any] = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls") is not None:
                    calls = []
                    for call in m["tool_calls"]:
                        name, call_id, raw = ReasonAgent._decode_tool_call(call)
                        try:
                            parsed = json.loads(raw) if isinstance(raw, str) else raw
                            if not isinstance(parsed, dict):
                                parsed = {}
                        except (ValueError, TypeError):
                            # The matching tool result explains the parse error.
                            # Keep the next API request valid so the model can retry.
                            parsed = {}
                        calls.append({
                            "id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(parsed, ensure_ascii=False)},
                        })
                    nm["tool_calls"] = calls
                rc = m.get("reasoning_content")
                if isinstance(rc, str) and rc.strip():
                    nm["reasoning_content"] = rc
                sanitized.append(nm)

            elif role == "tool":
                nm = {
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id") or "",
                    "content": m.get("content") or "",
                }
                if m.get("name"):
                    nm["name"] = m["name"]
                sanitized.append(nm)

        return sanitized

    @staticmethod
    def _prune_tool_results_in_messages(
        messages: List[Dict[str, Any]],
        *,
        keep_last_tool_rounds: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Context window management: keep only the body of the tool results (the tool role's content) from the last N rounds,
        and blank out the tool content of earlier rounds; do not modify the content,
        reasoning_content, or tool_calls of any assistant turn (to satisfy the API's requirements on tool message ordering).

        "1 round" = one assistant message (with tool_calls) + the consecutive tool messages that immediately follow it.
        """
        keep_last_tool_rounds = max(keep_last_tool_rounds, 0)

        tool_rounds: List[List[int]] = []
        i, n = 0, len(messages)
        while i < n:
            m = messages[i]
            role = (m.get("role") or "").strip() if isinstance(m, dict) else ""
            if (
                role == "assistant"
                and isinstance(m.get("tool_calls"), list)
                and m["tool_calls"]
            ):
                j = i + 1
                idxs: List[int] = []
                while j < n:
                    rj = (
                        (messages[j].get("role") or "").strip()
                        if isinstance(messages[j], dict)
                        else ""
                    )
                    if rj != "tool":
                        break
                    idxs.append(j)
                    j += 1
                if idxs:
                    tool_rounds.append(idxs)
                i = j
                continue
            i += 1

        if not tool_rounds:
            return messages

        keep_set: set[int] = set()
        if keep_last_tool_rounds > 0:
            for round_idxs in tool_rounds[-keep_last_tool_rounds:]:
                keep_set.update(round_idxs)

        pruned: List[Dict[str, Any]] = []
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            if (m.get("role") or "").strip() == "tool" and idx not in keep_set:
                pruned.append({**m, "content": ""})
            else:
                pruned.append(m)
        return pruned

    # ============================================================
    # 4. LLM call
    # ============================================================

    async def _default_call_model(self, messages, tools):
        """Streaming LLM call based on the OpenAI-compatible interface; returns a unified structure."""
        model_name = "deepseek-v3.2"
        conf = model_config[model_name]
        client = openai.OpenAI(api_key=conf["api_key"], base_url=conf["base_url"])

        enable_thinking = False
        api_messages = self._sanitize_messages_for_api(messages)

        stream = client.chat.completions.create(
            model=conf.get("model", model_name),
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            timeout=30,
            extra_body={"enable_thinking": enable_thinking},
        )

        full_content = ""
        reasoning_content = ""
        tool_calls_dict: Dict[int, Dict[str, Any]] = {}

        self._stream_write("\n--- LLM STREAM START ---\n", end="")
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    self._stream_write(delta.content, end="")
                    full_content += delta.content

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._stream_write(delta.reasoning_content, end="")
                    reasoning_content += delta.reasoning_content

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", None)
                        if idx is None:
                            continue
                        entry = tool_calls_dict.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc_id := getattr(tc, "id", None):
                            entry["id"] = tc_id
                        if fn := getattr(tc, "function", None):
                            if fn_name := getattr(fn, "name", None):
                                entry["function"]["name"] = fn_name
                            if fn_args := getattr(fn, "arguments", None):
                                entry["function"]["arguments"] += fn_args
                        self._stream_write(f"\n[TOOL CALL DELTA] {tc}\n")

        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            self._stream_write(f"\n[ERROR] HTTP streaming error: {e}\n", end="")
        except (APITimeoutError, APIConnectionError) as e:
            self._stream_write(
                f"\n[ERROR] OpenAI API timeout/connection error: {e}\n", end=""
            )
        except APIError as e:
            self._stream_write(f"\n[ERROR] OpenAI API error: {e}\n", end="")
        except Exception as e:
            self._stream_write(
                f"\n[ERROR] Unexpected streaming error: {e}\n", end=""
            )
        finally:
            self._stream_write("\n--- LLM STREAM END ---\n", end="")

        collected = [tool_calls_dict[i] for i in sorted(tool_calls_dict)]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": full_content or None,
                        "reasoning_content": reasoning_content or None,
                        "tool_calls": collected or None,
                    }
                }
            ]
        }

    # ============================================================
    # 5. Tool execution (parse args → dispatch → build tool message)
    # ============================================================

    @staticmethod
    def _decode_tool_call(tc: Any) -> tuple[str, str, str]:
        """
        Extract (tool_name, tool_call_id, raw_args) from a tool_call.
        Compatible with both dicts and OpenAI SDK objects.
        """
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            return fn.get("name", ""), tc.get("id", ""), fn.get("arguments", "{}")
        fn = getattr(tc, "function", None)
        return (
            getattr(fn, "name", "") if fn else "",
            getattr(tc, "id", ""),
            getattr(fn, "arguments", "{}") if fn else "{}",
        )

    @staticmethod
    def _parse_tool_args(raw_args: Any) -> Optional[Dict[str, Any]]:
        """Parse tool arguments; return a dict on success, None on failure."""
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                logger.warning(f"Failed to parse tool arguments: {raw_args}")
                return None
        logger.warning(f"Unexpected type for tool arguments: {type(raw_args)}")
        return None

    @staticmethod
    def _make_tool_msg(tc_id: str, tool_name: str, content: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": tool_name,
            "content": content,
        }

    async def _execute_tool_call(
        self, tool_name: str, tc_id: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch execution based on tool_name and return the tool message."""
        if tool_name == "search":
            queries = args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                queries = [str(queries)]
            result_text = await self.call_search(queries)
            self._stream_write(
                f"\n--- TOOL RESULT({tool_name}) ---\n{result_text}\n",
                color_ansi="\033[34m",
            )

        elif tool_name == "visit":
            url = args.get("url", [])
            if not isinstance(url, list):
                url = [str(url)]
            goal = args.get("goal", "")
            result_text = await self.call_visit(url, goal)
            self._stream_write(
                f"\n--- TOOL RESULT({tool_name}) ---\n{result_text}\n",
                color_ansi="\033[35m",
            )

        else:
            logger.warning(f"Unknown tool name: {tool_name}")
            result_text = f"[Unknown tool] {tool_name}"

        return self._make_tool_msg(tc_id, tool_name, result_text)

    # --- Default tool implementations ---

    async def _default_call_search(self, query: List[str]) -> str:
        async def handle_one(q: str):
            if not q.strip():
                return f"Query: {q}\n[empty query]"
            try:
                search_result = await asyncio.to_thread(
                    Search().call, {"query": q}
                )
            except Exception as e:
                search_result = f"[search error] query={q}, error={e}"
                self.trace.append(
                    {
                        "type": "tool",
                        "tool_name": "search",
                        "input": {"query": [q]},
                        "error": str(e),
                        "output": str(search_result),
                    }
                )
                return f"Query: {q}\n{search_result}"
            self._collect_papers_from_search_result(search_result)
            return f"Query: {q}\n{search_result}"

        results = await asyncio.gather(*(handle_one(q) for q in query))
        return "\n\n".join(results)

    async def _default_call_visit(self, url: List[str], goal: str) -> str:
        async def handle_one(u: str, g: str):
            try:
                parse_result = await asyncio.to_thread(
                    Visit().call, {"url": u, "goal": g}
                )
            except Exception as e:
                return f"URL: {u}\nGoal: {g}\n[web_parse error] url={u}, goal={g}, error={e}"
            self._collect_papers_from_parse_result(parse_result, u)
            return f"URL: {u}\nGoal: {g}\n{parse_result}"

        urls = url if isinstance(url, list) else [str(url)]
        results = await asyncio.gather(*(handle_one(u, goal) for u in urls))
        return "\n\n".join(results)

    # ============================================================
    # 6. Paper collection and deduplication
    # ============================================================

    # --- 6.1 Basic utility functions ---

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize a title: lowercase → strip prefix tags → keep only alphanumerics → collapse whitespace."""
        if not title:
            return ""
        t = title.strip().lower()
        t = re.sub(r"^\[[^\]]*\]\s*", "", t)
        t = re.sub(r"[^a-z0-9]+", " ", t)
        return " ".join(t.split())

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        if not date_str:
            return ""
        s = date_str.strip()
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        return s

    @staticmethod
    def _extract_arxiv_id(url: str) -> str:
        m = re.search(r"arxiv\.org/(pdf|abs)/([0-9]{4}\.[0-9]{4,5})", url)
        return m.group(2) if m else ""

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    # --- 6.2 Paper field operations ---

    @staticmethod
    def _get_paper_urls(paper: Dict[str, Any]) -> set[str]:
        """Get all URLs associated with a paper (compatible with the url / urls fields)."""
        urls: set[str] = set()
        if main := (paper.get("url") or "").strip():
            urls.add(main)
        extra = paper.get("urls")
        if isinstance(extra, list):
            urls.update(u.strip() for u in extra if isinstance(u, str) and u.strip())
        elif isinstance(extra, str) and extra.strip():
            urls.add(extra.strip())
        return urls

    @staticmethod
    def _ensure_paper_urls_field(paper: Dict[str, Any]) -> None:
        """Ensure the paper has a urls: list field that includes the main url."""
        if not isinstance(paper.get("urls"), list):
            paper["urls"] = []
        if (main := (paper.get("url") or "").strip()) and main not in paper["urls"]:
            paper["urls"].append(main)

    @staticmethod
    def _append_description(paper: Dict[str, Any], description: str) -> None:
        """Incrementally append the description (with deduplication)."""
        desc = (description or "").strip()
        if not desc:
            return
        cur = str(paper.get("description") or "")
        if desc in cur:
            return
        paper["description"] = (cur + ("\n\n" if cur else "") + desc).strip()

    @staticmethod
    def _maybe_update_year(paper: Dict[str, Any], year: str) -> None:
        if (y := (year or "").strip()) and not (paper.get("year") or "").strip():
            paper["year"] = y

    # --- 6.3 Matching and merging ---

    def _find_matching_paper(self, title: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Match an existing paper using multiple strategies (in decreasing priority):
        exact URL → arXiv ID → exact title → fuzzy/substring title (>0.80).
        """
        t = (title or "").strip()
        u = (url or "").strip()
        if not self.paper_list:
            return None

        if u:
            for p in self.paper_list:
                if u in self._get_paper_urls(p):
                    return p

        if new_arxiv := (self._extract_arxiv_id(u) if u else ""):
            for p in self.paper_list:
                for pu in self._get_paper_urls(p):
                    if (old_arxiv := self._extract_arxiv_id(pu)) and old_arxiv == new_arxiv:
                        return p

        if t:
            for p in self.paper_list:
                if (p.get("title") or "").strip() == t:
                    return p

        new_norm = self._normalize_title(t)
        if not new_norm:
            return None
        for p in self.paper_list:
            old_norm = self._normalize_title(p.get("title") or "")
            if not old_norm:
                continue
            if new_norm in old_norm or old_norm in new_norm:
                return p
            if self._similar(new_norm, old_norm) > 0.80:
                return p

        return None

    def _merge_paper(
        self,
        existing: Dict[str, Any],
        *,
        title: str,
        url: str,
        description: str,
        year: str = "",
    ) -> None:
        """Merge newly found information into an existing paper (incremental description, deduplicated url append, year backfill)."""
        self._ensure_paper_urls_field(existing)
        if u := (url or "").strip():
            if u not in existing["urls"]:
                existing["urls"].append(u)
            self._seen_pdf_urls.add(u)
        self._append_description(existing, description)
        self._maybe_update_year(existing, year)

    def _try_add_or_merge_paper(
        self,
        title: str,
        url: str,
        description: str,
        year: str = "",
        source: str = "",
    ) -> None:
        """
        Unified paper ingestion logic:
        1. Filter out non-paper URLs
        2. Matched an existing paper → merge
        3. URL already seen → skip
        4. Otherwise create a new paper
        """
        if not title or not url:
            return
        if not is_pdf_url(url, title):
            return

        existing = self._find_matching_paper(title, url)
        if existing is not None:
            self._merge_paper(
                existing, title=title, url=url, description=description, year=year
            )
            return

        if url in self._seen_pdf_urls:
            return
        self._seen_pdf_urls.add(url)

        tag = f" from {source}" if source else ""
        logger.info(f"📰 [Paper Extracted{tag}] {title}, url:{url}")
        self.paper_list.append(
            {
                "title": title,
                "description": description,
                "authors": "",
                "year": year,
                "url": url,
                "urls": [url],
            }
        )

    # --- 6.4 Collect papers from search/parse results ---

    def _collect_papers_from_search_result(self, search_result: Any) -> None:
        """Extract papers from search results, supporting both dict (organic) and markdown formats."""
        data = self._try_parse_structured(search_result)
        if data is None and isinstance(search_result, str):
            self._collect_papers_from_markdown(search_result)
            return
        if data is None:
            return
        self._collect_papers_from_organic(data)

    def _collect_papers_from_parse_result(
        self, parse_result: Any, parsed_url: str
    ) -> None:
        """Update/add paper information from webpage parse results."""
        data = self._try_parse_structured(parse_result)

        if data is None:
            if isinstance(parse_result, str):
                self._attach_text_to_existing_paper(str(parse_result), parsed_url)
            return

        # Attach the parsed content to an existing paper
        parsed_info = data.get("content", {})
        is_failed = (
            isinstance(parsed_info, str)
            and "failed to parse web content" in parsed_info.lower()
        )
        if not is_failed:
            for paper in self.paper_list:
                if parsed_url in self._get_paper_urls(paper):
                    logger.info(
                        f"📰 [Paper Updated by web_parse] "
                        f"{paper.get('title')}, url:{parsed_url}"
                    )
                    self._append_description(paper, str(parsed_info))
                    self._link_url_to_paper(paper, parsed_url)
                    break

        # Extract new papers from data["urls"]
        for item in data.get("urls") or []:
            if not isinstance(item, dict):
                continue
            self._try_add_or_merge_paper(
                title=(item.get("title") or "").strip(),
                url=(item.get("url") or "").strip(),
                description=(item.get("description") or "").strip(),
                source="web_parse",
            )

    # --- 6.5 Parsing / collection helpers ---

    @staticmethod
    def _try_parse_structured(result: Any) -> Optional[Dict]:
        """Try to parse the result into a dict (compatible with dict / repr string)."""
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            s = result.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    return ast.literal_eval(result)
                except Exception:
                    return None
        return None

    def _attach_text_to_existing_paper(self, text: str, url: str) -> None:
        """Attach a plain-text result to an existing paper (matched by URL)."""
        for paper in self.paper_list:
            if url in self._get_paper_urls(paper):
                self._append_description(paper, text)
                self._link_url_to_paper(paper, url)
                return

    def _link_url_to_paper(self, paper: Dict[str, Any], url: str) -> None:
        """Associate a url with a paper (ensuring the urls field contains this url)."""
        if not url:
            return
        self._ensure_paper_urls_field(paper)
        if url not in paper["urls"]:
            paper["urls"].append(url)
        self._seen_pdf_urls.add(url)

    def _collect_papers_from_markdown(self, text: str) -> None:
        """Extract papers from markdown-format (1. [title](url)) search results."""
        pat = re.compile(
            r"^\s*(\d+)\.\s+\[(?P<title>[^\]]+)\]\((?P<link>https?://[^\)]+)\)\s*$",
            re.MULTILINE,
        )
        matches = list(pat.finditer(text))
        if not matches:
            return

        for i, m in enumerate(matches):
            title = (m.group("title") or "").strip().replace("[PDF]", "").strip()
            url = (m.group("link") or "").strip()
            if not title or not url:
                continue

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()

            year = ""
            if dm := re.search(r"Date published:\s*(.+)", chunk):
                year = self._normalize_date(dm.group(1).strip())
            description = re.sub(
                r"^\s*(Date published|Source):.*$", "", chunk, flags=re.MULTILINE
            ).strip()

            self._try_add_or_merge_paper(title, url, description, year, source="search")

    def _collect_papers_from_organic(self, data: Dict) -> None:
        """Extract papers from organic-structured search results."""
        for item in data.get("organic") or []:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip().replace("[PDF]", "")
            url = (item.get("link") or "").strip()
            description = (item.get("snippet") or "").strip()
            year = self._normalize_date((item.get("date") or "").strip())

            self._try_add_or_merge_paper(title, url, description, year, source="search")

    # ============================================================
    # 7. Main loop
    # ============================================================

    async def run(self, user_query: str) -> Dict[str, Any]:
        """
        Core entry point:
        1. Build the initial conversation (system + user)
        2. Loop between the LLM and tools until the model no longer produces tool_calls
        3. Parse the final XML output and return a structured result
        """
        messages: List[Dict[str, Any]] = []

        # if self.cooperate_times == 0:
        #     messages.append({"role": "system", "content": self.system_prompt})
        #     self.trace.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "system", "content": self.system_prompt})
        self.trace.append({"role": "system", "content": self.system_prompt})
        self.cooperate_times += 1
        messages.append({"role": "user", "content": user_query})
        self.trace.append({"role": "user", "content": user_query})

        tool_calls_used = 0
        limit_reached = False

        while True:
            # Inject the interrupt prompt
            if self.response_now:
                messages.append({"role": "user", "content": self.interrupt_prompt})
                self.trace.append({"role": "user", "content": self.interrupt_prompt})

            # Prune context: keep full tool results only for the latest round; historical rounds keep only placeholder structure
            messages_for_llm = self._prune_tool_results_in_messages(messages)
            response = await self.call_model(messages_for_llm, self.tools)
            self.response_now = False

            # Parse the LLM response
            msg = response["choices"][0]["message"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            reasoning = msg.get("reasoning_content") or ""

            assistant_msg = {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)
            self.trace.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                }
            )

            # Termination condition
            if not tool_calls or limit_reached:
                self.message_history.extend(messages)
                return self._parse_final_answer(content or "")

            # Execute each tool_call one by one
            for tc in tool_calls:
                tool_name, tc_id, raw_args = self._decode_tool_call(tc)
                if not tool_name:
                    logger.warning(f"Empty tool_name in tool_call: {tc}")
                    continue

                args = self._parse_tool_args(raw_args)
                tool_calls_used += 1
                if args is None:
                    err = f"[tool args parse error] tool={tool_name}, raw_args={raw_args!r}"
                    tool_msg = self._make_tool_msg(tc_id, tool_name, err)
                else:
                    tool_msg = await self._execute_tool_call(tool_name, tc_id, args)
                messages.append(tool_msg)
                self.trace.append(tool_msg)

                if tool_calls_used >= self.max_tool_calls:
                    logger.warning("Max tool calls reached, forcing finalize.")
                    finalize = {"role": "user", "content": _FINALIZE_MSG}
                    messages.append(finalize)
                    self.trace.append(finalize)
                    limit_reached = True
                    break

    # ============================================================
    # 8. Final output parsing
    # ============================================================

    def _parse_final_answer(self, text: str) -> Dict[str, Any]:
        """Parse the final XML-tag output, attaching paper_list and trace."""

        def extract_tag(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def split_by(raw: str, sep: str) -> List[str]:
            return [x.strip() for x in raw.split(sep) if x.strip()] if raw else []

        parsed_query = extract_tag("parsed_query")
        checklist = split_by(extract_tag("checkpoints"), ";")
        keywords = split_by(extract_tag("keywords"), ",")
        search_query = split_by(extract_tag("query"), ";")
        quick_state = extract_tag("state").lower() or "stop"

        return {
            "parsed_query": parsed_query,
            "checklist": checklist,
            "search_query": search_query,
            "keywords": keywords,
            "quick_state": quick_state,
            "paper_list": self.paper_list,
            "trace": self.trace,
        }


# Backward compatibility: external code references it via `from reason_agent import reason_agent`
reason_agent = ReasonAgent


# ============================================================
# Test entry point
# ============================================================

if __name__ == "__main__":
    import time

    output_dir = os.path.join(current_path, "result", "reasoner_demo")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "reasoner.log")
    setup_logging(log_path)
    save_trace_path = os.path.join(output_dir, "reasoner_trace.json")

    async def main():
        user_question = "Help me find articles similar to WebSailor that top the HLE leaderboard"
        agent = ReasonAgent(origin_query=user_question)
        agent.set_stream_file(os.path.join(output_dir, "reasoner_stream.txt"))
        result = await agent.run(user_question)

        print("\n=== Final Parsed Result ===")
        print("parsed_query:", result["parsed_query"])
        print("keywords:", result["keywords"])
        print("checklist:", result["checklist"])
        print("search_query:", result["search_query"])

        for i, p in enumerate(result["paper_list"], 1):
            print(
                f"{i}. {p['title']} ({p.get('year', 'N/A')}) "
                f"- {p['url']} - {p['description']}"
            )

        with open(save_trace_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Trace saved to {save_trace_path}")

    begin = time.time()
    asyncio.run(main())
    print("Time elapsed: ", time.time() - begin)
