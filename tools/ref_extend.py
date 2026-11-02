from config_loader import get_config_path
import re, sys
import json
import os
from rapidfuzz import fuzz, process
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import time,tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_path}")
from web_search import title_search_rag
from loguru import logger
from sn_server import sn_paper_search

with open(get_config_path(), "r") as f:
    config = json.load(f)
    use_sn = config.get("use_sn", True)
    use_reference_rag = config.get("use_reference_rag", True)
model_config = config["model_config"]
with open(f"{current_path}/../prompt/papermaster_prompt.json", "r") as f:
    prompts = json.load(f)
    
def get_sn_titles(qs):
    if use_sn:
        titles = []
        if not qs:
            return titles

        max_w = len(qs) if len(qs) > 0 else 1

        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = [executor.submit(sn_paper_search, q, 1000) for q in qs]
            for future in futures:
                try:
                    papers = future.result()
                    # sn_paper_search returns a list[dict]; iterate to extract title
                    if isinstance(papers, list):
                        if len(papers) == 0:
                            continue
                        titles.extend(
                            [
                                p.get("title", "")
                                for p in papers
                                if isinstance(p, dict)
                            ]
                        )
                except Exception:
                    pass
        return titles
    
    return []


def normalize_title(text):
    """Convert the title to plain characters"""
    text = unicodedata.normalize('NFKC', text) # Unicode normalization
    text = text.lower() # convert to lowercase
    text = re.sub(r'[^a-z0-9]', '', text) # keep only letters and digits
    return text


def embedding_retrieval(
    queries,
    candidates,
    expand_papers=-1,
    threshold=80,
    embedding_paper_list=None,
):
    """
    Find paper titles in the title pool that fuzzy-match the candidates.

    Logic:
    1) First check the "intersection" of embedding_paper_list (cache pool) with candidates
       (here the intersection is defined via fuzzy matching).
       If the number of hits already satisfies expand_papers, return directly.
    2) If fewer than expand_papers, call retrieval (RAG + SN) to add new titles, update
       embedding_paper_list, then repeat matching until expand_papers is satisfied or no
       more new titles can be obtained.

    Note:
    - embedding_paper_list is updated in-place, used to cache retrieval results across rounds.
    """
    try:
        begin = time.time()
        print(
            f"[Embedding Retrieval] length of queries: {len(queries)}, "
            f"length of candidates: {len(candidates)},queries: {queries}"
        )

        # Prevent a mutable default argument from causing cross-call pollution
        if embedding_paper_list is None:
            embedding_paper_list = []

        def _clean_titles(items):
            if not items:
                return []
            out = []
            for x in items:
                if not x:
                    continue
                s = str(x).replace("\n", " ").strip()
                if s:
                    out.append(s)
            # Order-preserving dedup (by normalize_title)
            seen_norm = set()
            uniq = []
            for t in out:
                nt = normalize_title(t)
                if not nt or nt in seen_norm:
                    continue
                seen_norm.add(nt)
                uniq.append(t)
            return uniq

        def _fuzzy_match(cands, title_pool, limit=-1):
            if not cands or not title_pool:
                return []
            matches = []
            for cand in tqdm.tqdm(cands):
                cleaned_candidate = cand.replace("\n", " ").strip()
                if not cleaned_candidate:
                    continue
                best = process.extractOne(
                    cleaned_candidate, title_pool, scorer=fuzz.ratio
                )
                if not best:
                    continue
                _best_match, score, idx = best
                if score >= threshold:
                    hit = title_pool[idx]
                    if hit not in matches:
                        matches.append(hit)
                        if limit != -1 and len(matches) >= limit:
                            break
            return matches

        # First match using the cache pool (intersection check)
        candidates_clean = _clean_titles(candidates)
        embedding_paper_list[:] = _clean_titles(embedding_paper_list)

        if expand_papers != -1 and int(expand_papers) <= 0:
            return []

        matches = _fuzzy_match(
            candidates_clean,
            embedding_paper_list,
            limit=expand_papers if expand_papers != -1 else -1,
        )
        if expand_papers != -1 and len(matches) >= expand_papers:
            logger.info(f"🔍 [Embedding Retrieval] Matched {len(matches)} papers directly in the existing pool, reaching the limit {expand_papers}")
            return matches[:expand_papers]
        # expand_papers == -1 with existing matches: keep the fast path, avoid repeated retrieval
        if expand_papers == -1 and matches:
            return matches

        # If insufficient, add retrieval results in rounds and recompute matches until expand_papers is reached
        # Gradually increase RAG topk to avoid fetching too much at once
        rag_topk_rounds = [8192]

        pool_norm = {normalize_title(t) for t in embedding_paper_list if t}

        for round_idx, rag_topk in enumerate(rag_topk_rounds, 1):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_rag = executor.submit(title_search_rag, queries, rag_topk) if use_reference_rag else None
                future_sn = executor.submit(get_sn_titles, queries)

                try:
                    rag_titles = future_rag.result() if future_rag is not None else []
                except Exception as e:
                    logger.error(f"RAG search failed: {e}")
                    rag_titles = []

                try:
                    sn_titles = future_sn.result()
                except Exception as e:
                    logger.error(f"SN search failed: {e}")
                    sn_titles = []

            # Merge, dedup and clean, incrementally writing into the cache pool
            new_titles = []
            for t in _clean_titles((rag_titles or []) + (sn_titles or [])):
                nt = normalize_title(t)
                if not nt or nt in pool_norm:
                    continue
                pool_norm.add(nt)
                new_titles.append(t)

            if new_titles:
                embedding_paper_list.extend(new_titles)

            end = time.time()
            logger.info(
                f"🔍 [Embedding Retrieval] Round {round_idx}: rag_topk={rag_topk}, "
                f"new_titles={len(new_titles)}, pool={len(embedding_paper_list)}, time={end-begin:.2f}s"
            )

            matches = _fuzzy_match(
                candidates_clean,
                embedding_paper_list,
                limit=expand_papers if expand_papers != -1 else -1,
            )

            if expand_papers != -1 and len(matches) >= expand_papers:
                logger.info(
                    f"🔍 [Embedding Retrieval] Matched {len(matches)} papers, reaching the limit {expand_papers}"
                )
                return matches[:expand_papers]

            # No new titles added, so stop supplementing
            if not new_titles:
                break

        end = time.time()
        logger.info(
            f"🔍 [Embedding Retrieval] Fuzzy matching complete, matches={len(matches)}, time={end-begin:.2f}s"
        )
        return matches[:expand_papers] if (expand_papers != -1) else matches
    except Exception as e:
        logger.error(f"rag retrieval failed,{e}")
        return []



# ========= Usage example =========
if __name__ == "__main__":
    # import time
    begin = time.time()
    # arxiv_id = "2503.10410"
    # citations = search_info_by_local_db(arxiv_id)
    query = ["nerf and 3dgs","autonomous driving","llm agent"]
    extract_ref = embedding_retrieval(query, [])
    print(extract_ref)
    end = time.time()
    print(f"Time cost: {end-begin:.2f}s")
    # print(llm_call("hello"))
