from __future__ import annotations

from config_loader import get_config_path
import asyncio
import os
from typing import Any, Dict, List, Optional

import requests,json

current_path = os.path.dirname(os.path.abspath(__file__))
with open(get_config_path(), "r") as f:
    config = json.load(f)
    chunk_url = config["chunk_url"]

def _extract_chunks(obj: Any, *, arxiv_id: str) -> List[Any]:
    """
    Best-effort extractor for chunk server responses.
    We don't assume a strict schema; we search common nesting patterns.
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # direct common keys
        for k in ("chunks", "chunk_list", "results", "result", "data", "papers"):
            if k in obj:
                out = _extract_chunks(obj.get(k), arxiv_id=arxiv_id)
                if out:
                    return out
        # sometimes keyed by paper id
        if arxiv_id in obj:
            out = _extract_chunks(obj.get(arxiv_id), arxiv_id=arxiv_id)
            if out:
                return out
        # or keyed by normalized arxiv id variants
        for k, v in obj.items():
            if isinstance(k, str) and (k.endswith(arxiv_id) or k.replace("arxiv:", "") == arxiv_id):
                out = _extract_chunks(v, arxiv_id=arxiv_id)
                if out:
                    return out
        # last resort: scan values
        for v in obj.values():
            out = _extract_chunks(v, arxiv_id=arxiv_id)
            if out:
                return out
    return []


def get_paper_chunk(
    arxiv_id: str,
    query: str,
    topk: int = 3,
    server_url: str = chunk_url,
    timeout: int = 60,
    return_scores: bool = True,
    proxies: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Call chunk server (`/paper_retrieve`) and return relevant chunks for a paper.

    - **Input**: `arxiv_id` like "2509.13313"
    - **Output**: a list of chunk dicts (best-effort normalized)
    - **Default**: returns top 3 chunks
    """
    arxiv_id = str(arxiv_id).strip()
    if not arxiv_id:
        raise ValueError("arxiv_id is empty")

    url = server_url.rstrip("/") + "/paper_retrieve"
    payload: Dict[str, Any] = {
        "queries": [str(query)],
        "paper_ids": [arxiv_id],
        "topk": int(topk),
        "return_scores": bool(return_scores),
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    chunks_raw = _extract_chunks(data, arxiv_id=arxiv_id)
    out: List[Dict[str, Any]] = []
    for c in chunks_raw:
        docs = []
        for doc in c[:topk]:
            if doc['score'] > 0.6:
                docs.append([doc['document']['contents'], doc['score']])
        out.append(docs)
    return out


async def get_paper_chunk_async(
    arxiv_id: str,
    query: str,
    topk: int = 3,
    server_url: str = chunk_url,
    timeout: int = 60,
    return_scores: bool = True,
    proxies: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Async version of get_paper_chunk"""
    return await asyncio.to_thread(
        get_paper_chunk,
        arxiv_id,
        query,
        topk,
        server_url,
        timeout,
        return_scores,
        proxies,
    )


async def test_concurrent(num_tasks: int = 100):
    """Concurrency test function"""
    arxiv_id = "2510.24694"
    query = "websailor baseline"
    
    print(f">> Starting {num_tasks} concurrent tests...")
    start_time = asyncio.get_event_loop().time()
    
    # Create concurrent tasks
    tasks = [
        get_paper_chunk_async(arxiv_id, query, topk=3)
        for _ in range(num_tasks)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    # Aggregate results
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    error_count = num_tasks - success_count
    
    print(f">> Completed {num_tasks} tasks in {elapsed:.2f}s")
    print(f">> Success: {success_count}, Errors: {error_count}")
    print(f">> Average time per task: {elapsed/num_tasks:.3f}s")
    print(f">> Throughput: {num_tasks/elapsed:.2f} tasks/sec")
    
    # Print the first successful result as an example
    for i, result in enumerate(results):
        if not isinstance(result, Exception):
            print(f"\n>> Example result (task {i+1}):")
            print(result)
            break
    
    # If there are errors, print the first error
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\n>> Example error (task {i+1}):")
            print(f"  {type(result).__name__}: {result}")
            break


if __name__ == "__main__":
    # Async concurrency test
    asyncio.run(test_concurrent(num_tasks=1))