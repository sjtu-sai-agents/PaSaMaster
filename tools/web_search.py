from config_loader import get_config_path
import json
import os
import re
import sys
import time
import threading
import warnings
from datetime import datetime
import requests
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_pdf_url
from loguru import logger
import http.client
import asyncio
import aiohttp
import traceback


# Read the Google API key and proxy settings from the config file
def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config/config.json"
    )
    try:
        with open(get_config_path(), "r") as f:
            config = json.load(f)
        common_config = config.get("common_config", {})
        google_key = common_config.get(
            "google_api_key", "your serper api key"
        )
        proxy_config = common_config.get("proxy", {})
        proxy_url = proxy_config.get("http_proxy", "http://127.0.0.1:7899").replace(
            "http://", ""
        )
        arxiv_search_url = common_config.get(
            "arxiv_search_url", "http://<your-arxiv-search-host>:<port>/search"
        )
        return google_key, proxy_url, arxiv_search_url
    except Exception as e:
        print(f"⚠️ Failed to load config from config file: {e}")
        return (
            "your serper api key",
            "127.0.0.1:7899",
            "http://<your-arxiv-search-host>:<port>/search",
        )


GOOGLE_KEY, USE_PROXY, ARXIV_SEARCH_URL = load_config()
print("🔧 Search configuration loaded")
current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, "..", "config", "web_agent.json"), "r") as f:
    tools_config = json.load(f)


async def web_search(query: str, top_k=10, depth=0):
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": top_k,
        "gl": tools_config["search_region"],
        "hl": tools_config["search_lang"],
        "location": "United States",
    }

    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    raise Exception(f"API Error: {response.status}")
                data = await response.json()

                data.pop("searchParameters", None)
                data.pop("credits", None)

                if not data:
                    raise Exception(
                        "The google search API is temporarily unavailable, please try again later."
                    )
                return data

    except Exception as e:
        if depth < 3:
            await asyncio.sleep(1)
            return await web_search(query, top_k, depth=depth + 1)
        print(f"search failed: {e}")
        print(traceback.format_exc())
        return []


def google_search_arxiv_id(query, num=10, end_date=None):
    query = re.sub(r'\\"|"', "", query)

    start_time = time.time()
    conn = http.client.HTTPSConnection("google.serper.dev")
    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    payload = json.dumps(
        {
            "q": search_query,
            "num": num * 3,
            "page": 1,
        }
    )
    headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
    assert headers["X-API-KEY"] != "your google keys", "add your google search key!!!"
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        results = json.loads(data.decode("utf-8"))
        arxiv_id_list = []
        for paper in results["organic"]:
            if re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", paper["link"]):
                arxiv_id = re.search(
                    r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", paper["link"]
                ).group(1)
                arxiv_id_list.append(arxiv_id)
        return list(dict.fromkeys(arxiv_id_list))[:num]
    except:
        warnings.warn(f"google search failed, query: {query}")
    return []


def google_search_scholar(query, num = 20,end_date=None):
    query = re.sub(r'\\"|"', "", query)
    
    # Compute how many pages are needed (10 results per page)
    num_pages = (num + 9) // 10  # round up
    
    def _search_single_page(page: int):
        """Perform a single-page search"""
        conn = http.client.HTTPSConnection("google.serper.dev")
        try:
            search_query = query
            if end_date:
                try:
                    end_date_str = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
                    search_query = f"{query} before:{end_date_str}"
                except:
                    search_query = f"{query}"

            payload = json.dumps(
                {
                    "q": search_query,
                    "page": page,
                }
            )
            headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
            assert headers["X-API-KEY"] != "your google keys", "add your google search key!!!"
            conn.request("POST", "/scholar", payload, headers)
            res = conn.getresponse()
            data = res.read()
            try:
                results = json.loads(data.decode("utf-8"))
                arxiv_id_list = []
                title_list = []
                
                for paper in results.get("organic", []):
                    link = paper.get("link", "")
                    title = paper.get("title", "")
                    if re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", link):
                        arxiv_id = re.search(
                            r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", link
                        ).group(1)
                        arxiv_id_list.append(arxiv_id)
                    else:
                        clean_title = title.strip()
                        title_list.append(clean_title)

                return arxiv_id_list, title_list
            except Exception as e:
                warnings.warn(f"google search failed, query: {query}, page: {page}, error: {e}")
                return [], []
        finally:
            conn.close()
    
    # If there is only 1 page, request directly
    if num_pages == 1:
        arxiv_id_list, title_list = _search_single_page(1)
        return list(dict.fromkeys(arxiv_id_list)), list(dict.fromkeys(title_list))
    
    # Concurrent requests for multiple pages
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total_arxiv_id_list = []
    total_title_list = []
    
    with ThreadPoolExecutor(max_workers=num_pages) as executor:
        futures = {executor.submit(_search_single_page, page): page for page in range(1, num_pages + 1)}
        for future in as_completed(futures):
            page = futures[future]
            try:
                arxiv_id_list, title_list = future.result()
                total_arxiv_id_list.extend(arxiv_id_list)
                total_title_list.extend(title_list)
            except Exception as e:
                warnings.warn(f"google search failed for page {page}, query: {query}, error: {e}")
    
    return list(dict.fromkeys(total_arxiv_id_list)), list(dict.fromkeys(total_title_list))

def google_search_website(query, num = 20,end_date=None):
    query = re.sub(r'\\"|"', "", query)
    
    # Compute how many pages are needed (10 results per page)
    num_pages = (num + 9) // 10  # round up
    
    def _search_single_page(page: int):
        """Perform a single-page search"""
        conn = http.client.HTTPSConnection("google.serper.dev")
        try:
            paper_site = "(   site:arxiv.org OR   site:openreview.net OR   site:sciencedirect.com OR   site:ieee.org OR   site:springer.com OR   site:nature.com OR   site:wiley.com OR   site:elsevier.com OR   site:mdpi.com OR   site:researchgate.net OR   site:semanticscholar.org OR   site:wolfram.com )" # serper syntax: site:+website, separated by OR
            search_query = f"{query} {paper_site}"
            if end_date:
                try:
                    end_date_str = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
                    search_query = f"{query} before:{end_date_str} {paper_site}"
                except:
                    search_query = f"{query}{paper_site}"
            payload = json.dumps(
                {
                    "q": search_query,
                    "page": page,
                }
            )
            headers = {"X-API-KEY": GOOGLE_KEY, "Content-Type": "application/json"}
            assert headers["X-API-KEY"] != "your google keys", "add your google search key!!!"
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            try:
                results = json.loads(data.decode("utf-8"))
                arxiv_id_list = []
                title_list = []
                for paper in results.get("organic", []):
                    link = paper.get("link", "")
                    title = paper.get("title", "")

                    if re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", link):
                        arxiv_id = re.search(
                            r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", link
                        ).group(1)
                        arxiv_id_list.append(arxiv_id)
                    elif is_pdf_url(link, title):
                        clean_title = title.replace("[PDF]", "").strip()
                        title_list.append(clean_title)

                return arxiv_id_list, title_list
            except Exception as e:
                warnings.warn(f"google search failed, query: {query}, page: {page}, error: {e}")
                return [], []
        finally:
            conn.close()
    
    # If there is only 1 page, request directly
    if num_pages == 1:
        arxiv_id_list, title_list = _search_single_page(1)
        return list(dict.fromkeys(arxiv_id_list)), list(dict.fromkeys(title_list))
    
    # Concurrent requests for multiple pages
    from concurrent.futures import ThreadPoolExecutor, as_completed
    total_arxiv_id_list = []
    total_title_list = []
    
    with ThreadPoolExecutor(max_workers=num_pages) as executor:
        futures = {executor.submit(_search_single_page, page): page for page in range(1, num_pages + 1)}
        for future in as_completed(futures):
            page = futures[future]
            try:
                arxiv_id_list, title_list = future.result()
                total_arxiv_id_list.extend(arxiv_id_list)
                total_title_list.extend(title_list)
            except Exception as e:
                warnings.warn(f"google search failed for page {page}, query: {query}, error: {e}")
    
    return list(dict.fromkeys(total_arxiv_id_list)), list(dict.fromkeys(total_title_list))

def google_search_paper(query, num = 20,end_date=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(google_search_website, query, num, end_date=end_date), executor.submit(google_search_scholar, query, num, end_date=end_date)]
        total_arxiv_ids,total_title_list = [],[]
        for idx, future in enumerate(as_completed(futures)):
            arxiv_ids,title_list = future.result()
            
            if arxiv_ids:
                total_arxiv_ids.extend(arxiv_ids)
            if title_list:
                total_title_list.extend(title_list)
            logger.info(f"google search paper[{idx}]: {query}, arxiv_ids: {len(arxiv_ids)}, title_list: {len(title_list)}")
    logger.info(f"google search paper: {query}, total_arxiv_ids: {len(list(dict.fromkeys(total_arxiv_ids)))}, total_title_list: {len(list(dict.fromkeys(total_title_list)))}")
    return list(dict.fromkeys(total_arxiv_ids)), list(dict.fromkeys(total_title_list))


def arxiv_search_id(query, num=100, end_date=None, threshold=0.7):
    url = ARXIV_SEARCH_URL
    payload = {"questions": query, "topk": num}
    response = requests.post(url, json=payload)
    try:
        if response.status_code == 200:
            results = response.json()["results"][0]
            scores = response.json()["scores"][0]
            result = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:num]
            # Keep all papers above the threshold
            arxiv_id_list = [r["arxiv_id"] for s, r in result if s > threshold]
            return list(set(arxiv_id_list))
        else:
            print(f"\033[31m {url} request failed, status code: {response.status_code}")
            return []
    finally:
        response.close()


def title_search_rag(query, num=2048, threshold=0.7):
    url = ARXIV_SEARCH_URL
    payload = {"questions": query, "topk": num}
    response = requests.post(url, json=payload)
    try:
        if response.status_code == 200:
            scores_list = []
            results_list = []
            response_data = response.json()
            scores = response_data["scores"]
            results = response_data["results"]
            for i in range(len(scores)):
                scores_list.extend(scores[i])
            for i in range(len(results)):
                results_list.extend(results[i])
            result = sorted(
                zip(scores_list, results_list), key=lambda x: x[0], reverse=True
            )
            # Keep all papers above the threshold
            title_list = list(set([r for s, r in result if s > threshold]))
            logger.info(
                f"\033[32m Total search results: query: {query}, num: {len(title_list)}\033[0m"
            )
            return list(set(title_list))
        else:
            logger.exception(f"\033[31m {url} request failed, status code: {response.status_code}")
            return []
    finally:
        response.close()


if __name__ == "__main__":
    # query = ["scimaster"]
    # num = 5
    # end_date = "20251201"
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(arxiv_search_id, query, num) for _ in range(1)]
    #     for future in as_completed(futures):
    #         results = future.result()
    #         print(results)

    # exit(1)
    # question_list =  ['data curation AND large language model AND pre-training AND performance', 'data filtering AND data efficiency AND LLM pre-training', 'quality over quantity AND language model pre-training AND dataset size']*30
    # save_data = []
    # for question in question_list:
    #     results = title_search_rag([question])
    #     save_config = {
    #         "question": question,
    #         "results": results
    #     }
    #     save_data.append(save_config)
    # begin_time = time.time()
    # result = title_search_rag(question_list)
    # end_time = time.time()
    # print(f"time cost: {end_time - begin_time}s")
    # print(asyncio.run(web_search("scimaster")))
    import time
    begin_time = time.time()
    query = "context management"
    # result = google_search_paper(query,num=30,end_date="20231201")
    result =  google_search_website(query,num=10)
    print(result)
    end_time = time.time()
    print(f"time cost: {end_time - begin_time}s")
