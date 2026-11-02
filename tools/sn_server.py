from config_loader import get_config_path
import requests
import json
import os, sys
from datetime import datetime
import aiohttp
import asyncio
from loguru import logger
from rapidfuzz import fuzz, process
from tenacity import retry, retry_if_exception_type, wait_random, stop_after_attempt

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(get_config_path(), "r") as f:
    conf = json.load(f)
    access_key = conf["sn_access_key"]
    sn_get_ref_url = conf["sn_get_ref_url"]
    sn_search_paper_url = conf["sn_search_paper_url"]
    sn_get_paper_info_url = conf["sn_get_paper_info_url"]


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random(0.5, 10),
    stop=stop_after_attempt(0),
    reraise=True,
)
def _make_request_with_retry(url, headers, payload, timeout=30, is_json=False):
    try:
        if is_json:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        else:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout)
    except requests.RequestException as error:
        # Requests errors may include a credential-bearing URL. Keep only the type.
        raise RuntimeError(f"SN request failed ({type(error).__name__})") from None

    if response.status_code != 200:
        raise Exception(f"HTTP Error: {response.status_code}")
    return response


def sn_get_paper_info(titles, batch_size=20):
    url = f"{sn_get_paper_info_url}"
    all_paper_info_list = []
    headers = {
        "accessKey": access_key,
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i : i + batch_size]
        payload = json.dumps({"titles": [{"enName": title} for title in batch_titles]})
        try:
            response = _make_request_with_retry(url, headers, payload)
            result = response.json()
            data = result.get("data", {}) if isinstance(result, dict) else {}
            for i in range(len(batch_titles)):
                value = data.get(batch_titles[i], {})
                try:
                    paper_info = {
                        "title": value.get("enName", ""),
                        "year": value.get("coverDateStart", ""),
                        "authors": value.get("authors", ""),
                        "journal": value.get("publicationEnName", ""),
                        "abstract": value.get("enAbstract", ""),
                        "doi": value.get("doi", ""),
                        "source": "SearchFrom:sn",
                        "paper_id": value.get("paperId", ""),
                    }
                    all_paper_info_list.append(paper_info)
                except Exception as e:
                    logger.error(
                        f"\033[91m SN Error during sn_get_paper_info processing: {e}\033[0m"
                    )
                    all_paper_info_list.append({})
        except Exception as e:
            logger.error(
                f"SN Error during sn_get_paper_info when request: {e}, batch_titles: {batch_titles}"
            )
            for _ in range(len(batch_titles) - len(all_paper_info_list) % batch_size):
                all_paper_info_list.append({})
    return all_paper_info_list


# def sn_get_paper_info(titles, batch_size=20):
#     url = f"{sn_get_ref_url}{access_key}"
#     all_paper_info_list = []
#     headers = {
#         "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
#         "Content-Type": "application/json",
#         "Accept": "*/*",
#         "Host": "openapi.dp.tech",
#         "Connection": "keep-alive",
#         "Cookie": "your cookie",
#     }

#     for i in range(0, len(titles), batch_size):
#         batch_titles = titles[i : i + batch_size]
#         payload = json.dumps({"titles": [{"enName": title} for title in batch_titles]})

#         try:
#             response = _make_request_with_retry(url, headers, payload)

#             result = response.json()
#             data = result.get("data", {}) if isinstance(result, dict) else {}
#             for key, value in data.items():
#                 try:
#                     paper_info = {
#                         "title": key,  # value.get("enName", "") is the journal name, key is the paper name
#                         "year": value.get("coverDateStart", ""),
#                         "authors": value.get("authors", ""),
#                         "journal": value.get("publicationEnName", ""),
#                         "abstract": value.get("enAbstract", ""),
#                         "doi": value.get("doi", ""),
#                         "source": "SearchFrom:sn",
#                     }
#                     all_paper_info_list.append(paper_info)
#                 except Exception as e:
#                     logger.error(
#                         f"\033[91m Error during sn_get_paper_info processing: {e}\033[0m"
#                     )
#                     all_paper_info_list.append({})
#         except Exception as e:
#             logger.error(
#                 f"Error during sn_get_paper_info when request: {e}, batch_titles: {batch_titles}"
#             )
#             for _ in range(len(batch_titles) - len(all_paper_info_list) % batch_size):
#                 all_paper_info_list.append({})
#     return all_paper_info_list


def sn_get_ref(titles, batch_size=20, threshold=80):
    """
    :param titles: input list of paper titles (each element is a title string)
    :param batch_size: batch size per request
    :param threshold: fuzzy match score threshold (0-100); larger is stricter
    :return: all_refs_list, exactly the same length as titles and in corresponding order
    """
    original_titles = titles
    valid_indices = [i for i, title in enumerate(titles) if isinstance(title, str) and title.strip()]
    titles = [titles[i].strip() for i in valid_indices]
    url = f"{sn_get_ref_url}{access_key}"
    all_refs_list = []

    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Cookie": "your cookie",
    }

    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i : i + batch_size]
        payload = json.dumps({"titles": [{"enName": title} for title in batch_titles]})

        try:
            response = _make_request_with_retry(url, headers, payload)
            result = response.json()

            data = result.get("data", {}) if isinstance(result, dict) else {}
            if not isinstance(data, dict):
                data = {}
            response_title_list = list(data.keys())

            for original_title in batch_titles:
                refs_info = {}

                try:
                    cleaned_title = original_title.replace("\n", " ").strip()

                    matched_key = None

                    if cleaned_title in data:
                        matched_key = cleaned_title
                    elif response_title_list:
                        best_match, score, idx = process.extractOne(
                            cleaned_title, response_title_list, scorer=fuzz.ratio
                        )
                        if score >= threshold:
                            matched_key = response_title_list[idx]

                    if matched_key is not None:
                        value = data.get(matched_key, {}) or {}
                        refs = value.get("refs", []) or []

                        refs_info = {
                            str(idx + 1): paper.get("enName", "")
                            for idx, paper in enumerate(refs)
                        }

                except Exception as e:
                    logger.error(
                        f"\033[91m SN Error during sn_get_ref processing title: {original_title!r}, error: {e} \033[0m"
                    )

                all_refs_list.append(refs_info)

        except Exception as e:
            logger.error(
                f"\033[91m SN Error during sn_get_ref when request: {e}, batch_titles: {batch_titles} \033[0m"
            )
            for _ in batch_titles:
                all_refs_list.append({})

    # Preserve one output per input while excluding empty titles from API batches.
    aligned_refs = [{} for _ in original_titles]
    for index, refs in zip(valid_indices, all_refs_list):
        aligned_refs[index] = refs
    return aligned_refs


def sn_paper_search(query: str, num=-1, end_date=datetime.now().strftime("%Y-%m-%d"), words=[]):
    if num == -1:
        page_size = 50
    else:
        page_size = 200
    payload = {
        "type": 1,
        "words": words,
        "areaIds": [],
        "question": query,
        "startTime": "",
        "endTime": end_date,
        "pageSize": page_size,
    }
    headers = {
        "accessKey": access_key,
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    try:
        response = _make_request_with_retry(
            sn_search_paper_url, headers, payload, is_json=True
        )

        result = response.json()
        data = result.get("data", [])
        paper_list = []
        for idx, paper in enumerate(data):
            paper_list.append(
                {
                    "title": paper.get("enName", ""),
                    "year": paper.get("coverDateStart", ""),
                    "authors": paper.get("authors", ""),
                    "journal": paper.get("publicationEnName", ""),
                    "abstract": paper.get("enAbstract", ""),
                    "doi": paper.get("doi", ""),
                    "sections": "",
                    "source": "SearchFrom:sn",
                }
            )
            if num != -1 and idx >= num:
                break
        return paper_list
    except Exception as e:
        logger.error(f"\033[91m SN Error during sn_paper_search: {e} \033[0m")
        return []


def sn_search_chunk(query: str, paper_ids: list, page_size=50):
    payload = json.dumps({
    "type": 3,
    "rerank": 0,
    "words": [
      query
    ],
    "paperIds": paper_ids,
    "question": query,
    "pageSize": page_size
    })
    headers = {
    'accessKey': access_key,
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Connection': 'keep-alive',
    'Cookie': 'your cookie'
    }

    response = requests.request("POST", sn_search_paper_url, headers=headers, data=payload)
    result = response.json()
    data = result.get("data", [])
    if data==[] or 'pieces' not in data[0].keys():
        return ""
    return data[0]['pieces']

if __name__ == "__main__":
    # print(
    #     sn_get_ref(
    #         [
    #             "The Effect of Trichophytin and Candidin on Neutrophil Chemotaxis/Die Wirkung von Trichophytin und Candidin auf die Neutrophilen-Chemotaxis",
    #             "Agricultural climatology of the Upper Murrumbidgee River Valley, New South Wales",
    #         ]
    #     )
    # )
    import time
    from concurrent.futures import ThreadPoolExecutor

    # Title list for testing
    # test_titles = [
    #     "End-to-End Urban Autonomous Driving With Safety Constraints"
    # ]
    test_titles = [
        "Enabling late-stage drug diversification by high-throughput experimentation with geometric deep learning",
    ]
    print(len(test_titles))
    test = "Graph Construction"

    def test_worker(worker_id):
        logger.info(f"Worker {worker_id} is attempting to send a request...")
        start_time = time.time()

        # Call the rate-limit-protected function 
        # result = sn_get_paper_info(test_titles)
        # result = sn_get_ref(test_titles)
        result = sn_search_chunk(test, ["1175137425740005382"])
        # result = sn_paper_search(test)
        print(result)

        duration = time.time() - start_time
        logger.info(
            f"Worker {worker_id} request completed ✅ (elapsed: {duration:.2f}s, got {len(result)} results)"
        )

    with ThreadPoolExecutor(max_workers=100) as executor:
        for i in range(1):
            executor.submit(test_worker, i)

    logger.info("All test tasks completed")
