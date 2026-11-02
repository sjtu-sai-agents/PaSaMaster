"""Check setup before loading the research corpora or starting a benchmark."""

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from config_loader import ROOT, get_config_path


MODELS = ("deepseek-v3.2", "qwen-30b-train", "reason_rank", "qwen-flash")
CORPUS_FILES = (
    "paper_database/id2paper.json",
    "paper_database/cs_paper_2nd.zip",
    "arxiv_dataset/arxiv_idx_aid2pos.json",
    "arxiv_dataset/arxiv_idx_sim2pos.json",
    "arxiv_dataset/arxiv-metadata-oai-snapshot.json",
)
IMPORTS = (
    "arxiv", "aiohttp", "bs4", "httpx", "litellm", "numpy", "openai",
    "qwen_agent", "soundfile", "rapidfuzz", "requests", "tenacity", "tiktoken", "tqdm",
    "lxml", "loguru",
)


def is_placeholder(value):
    return not isinstance(value, str) or not value.strip() or any(
        marker in value.lower() for marker in ("your ", "<", "/path/to/")
    )


def probe_model(item):
    name, model = item
    request = Request(
        model["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model.get("model", name),
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 2,
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + model["api_key"]},
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = json.load(response)
        return name, None if body.get("choices") else "response has no choices"
    except HTTPError as error:
        return name, f"HTTP {error.code}"
    except Exception as error:
        # Exception text can contain credential-bearing URLs. Report types only.
        return name, type(error).__name__


def service_probes(config):
    """Small requests matching the retrieval clients; never print credentials."""
    common = config["common_config"]
    title = "Attention Is All You Need"
    headers = {"accessKey": config["sn_access_key"]}
    titles = {"titles": [{"enName": title}]}
    jobs = [
        ("Serper", "https://google.serper.dev/search",
         {"X-API-KEY": common["google_api_key"]}, {"q": title, "num": 1}, "organic"),
        ("Jina", "https://r.jina.ai/https://example.com",
         {"Authorization": "Bearer " + common["JINA_API_KEYS"]}, None, "text"),
        ("SN keyword search", config["sn_search_paper_url"], headers,
         {"type": 1, "words": [], "areaIds": [], "question": title,
          "startTime": "", "endTime": "2024-01-01", "pageSize": 5}, "data"),
        ("SN paper metadata", config["sn_get_paper_info_url"], headers, titles, "data"),
        ("SN references", config["sn_get_ref_url"] + config["sn_access_key"],
         {}, titles, "data"),
    ]
    # Reference expansion calls RAG independently of the direct-search switch.
    if config.get("use_local_search") or config.get("use_reference_rag", True):
        jobs.append(("Reference RAG", common["arxiv_search_url"], {},
                     {"questions": [title], "topk": 1}, "results"))
    if config.get("use_chunk"):
        jobs.append(("Chunk retrieval", config["chunk_url"].rstrip("/") + "/paper_retrieve", {},
                     {"queries": ["attention"], "paper_ids": ["1706.03762"],
                      "topk": 1, "return_scores": True}, "chunk"))
    return jobs


def probe_service(item, proxy=None):
    name, url, headers, payload, expected = item
    request = Request(
        url, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        open_request = urlopen
        if name == "Jina" and proxy and proxy.get("http_proxy"):
            proxy_url = proxy["http_proxy"]
            open_request = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url})).open
        with open_request(request, timeout=20) as response:
            body = response.read()
        if expected == "text":
            return name, None if body.strip() else "empty response"
        result = json.loads(body)
        if name.startswith("SN") and isinstance(result, dict) and result.get("code") not in (None, 0, "0", 200, "200"):
            return name, "service rejected request"
        if expected == "chunk":
            valid = isinstance(result, list) or (
                isinstance(result, dict) and any(
                    key in result for key in ("chunks", "chunk_list", "results", "result", "data", "papers", "1706.03762")
                )
            )
        else:
            valid = isinstance(result, dict) and isinstance(result.get(expected), (list, dict))
        if valid and expected in ("organic", "data", "results") and not result[expected]:
            return name, "no results for the test query"
        return name, None if valid else "unexpected response schema"
    except HTTPError as error:
        return name, f"HTTP {error.code}"
    except Exception as error:
        return name, type(error).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, default=ROOT / "data" / "pasamaster_bench.jsonl",
                        help="Input benchmark JSONL (default: data/pasamaster_bench.jsonl)")
    parser.add_argument("--check-models", action="store_true", help="Send one tiny request to each model used by the main pipeline.")
    parser.add_argument("--check-services", action="store_true", help="Probe Serper, Jina, SN, reference RAG, and the chunk service when enabled.")
    args = parser.parse_args()
    errors = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required")
    missing = [name for name in IMPORTS if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Missing Python modules: " + ", ".join(missing))
    try:
        config = json.loads(get_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("FAIL: configuration file is missing or invalid JSON")
        return 1
    if config.get("use_local_cache", True):
        for name in CORPUS_FILES:
            if not (ROOT / "data" / name).is_file():
                errors.append("Missing data/" + name)
    meta = config.get("meta_info_path", "")
    if meta and (is_placeholder(meta) or not Path(meta).is_file()):
        errors.append("meta_info_path must be empty or point to an existing metadata JSONL")
    model_configs = config.get("model_config", {})
    ready = []
    for name in MODELS:
        model = model_configs.get(name, {})
        if any(is_placeholder(model.get(key)) for key in ("base_url", "api_key")):
            errors.append("Configure model_config." + name + " (base_url and api_key)")
        else:
            ready.append((name, model))
    common = config.get("common_config", {})
    for key in ("google_api_key", "JINA_API_KEYS"):
        if is_placeholder(common.get(key)):
            errors.append("Configure common_config." + key)
    # Main retrieval still uses SN metadata and reference lookup when use_sn=false.
    for key in ("sn_access_key", "sn_get_ref_url", "sn_search_paper_url", "sn_get_paper_info_url"):
        if is_placeholder(config.get(key)):
            errors.append("Configure " + key)
    if (config.get("use_local_search") or config.get("use_reference_rag", True)) and is_placeholder(common.get("arxiv_search_url")):
        errors.append("Configure common_config.arxiv_search_url for enabled RAG retrieval")
    if config.get("use_chunk") and is_placeholder(config.get("chunk_url")):
        errors.append("Configure chunk_url, or disable use_chunk")
    for key, value in common.get("proxy", {}).items():
        if value and is_placeholder(value):
            errors.append("Configure common_config.proxy." + key + ", or use an empty string for direct access")
    try:
        count = 0
        with args.data_path.open(encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                if not isinstance(sample["question"], str) or not sample["question"].strip():
                    raise ValueError("empty question")
                published_time = sample["source_meta"]["published_time"] or datetime.now().strftime("%Y%m%d")
                datetime.strptime(published_time, "%Y%m%d")
                count += 1
        if not count:
            raise ValueError("empty dataset")
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("Input JSONL is missing or invalid; each row needs question and source_meta.published_time (YYYYMMDD)")
    if args.check_services and not errors:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for name, error in pool.map(lambda item: probe_service(item, common.get("proxy")), service_probes(config)):
                if error:
                    errors.append("Service " + name + ": " + error)
                else:
                    print("PASS: service " + name)
    if args.check_models:
        with ThreadPoolExecutor(max_workers=5) as pool:
            for name, error in pool.map(probe_model, ready):
                if error:
                    errors.append("Model " + name + ": " + error)
                else:
                    print("PASS: model " + name)
    for error in errors:
        print("FAIL: " + error)
    if not errors:
        print("PASS: dependencies, configuration, corpus files, and benchmark format")
        if not args.check_models:
            print("Model connectivity not checked; add --check-models to test endpoints.")
        if not args.check_services:
            print("Retrieval services not checked; add --check-services to probe them.")
        print("Small probes do not validate a complete benchmark run or reproduce the paper's metrics.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
