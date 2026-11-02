"""Regression checks for configured service boundaries, without network access."""

import ast
import json
import re
import threading
import time
import types
import unicodedata
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ServiceSetupTests(unittest.TestCase):
    def test_sn_search_uses_configured_url_without_overriding_host(self):
        request = Mock(return_value=types.SimpleNamespace(json=lambda: {"data": []}))
        namespace = {
            "datetime": datetime, "access_key": "private-test-key",
            "sn_search_paper_url": "https://configured.example/search",
            "_make_request_with_retry": request, "logger": Mock(),
        }
        search = load_function("tools/sn_server.py", "sn_paper_search", namespace)
        self.assertEqual(search("query"), [])
        self.assertEqual(request.call_args.args[0], "https://configured.example/search")
        headers = request.call_args.args[1]
        self.assertNotIn("Host", headers)
        self.assertEqual(headers["accessKey"], "private-test-key")

    def run_reference_expansion(self, local_enabled, reference_rag=True):
        rag = Mock(return_value=["Target"])
        sn = Mock(return_value=["Target"])
        namespace = {
            "time": time, "re": re, "unicodedata": unicodedata,
            "ThreadPoolExecutor": ThreadPoolExecutor, "logger": Mock(),
            "use_local_search": local_enabled, "use_reference_rag": reference_rag,
            "title_search_rag": rag, "get_sn_titles": sn,
            "tqdm": types.SimpleNamespace(tqdm=lambda items: items),
            "fuzz": types.SimpleNamespace(ratio=Mock()),
            "process": types.SimpleNamespace(extractOne=lambda query, pool, scorer: (pool[0], 100, 0)),
        }
        load_function("tools/ref_extend.py", "normalize_title", namespace)
        expand = load_function("tools/ref_extend.py", "embedding_retrieval", namespace)
        self.assertEqual(expand(["query"], ["Target"], expand_papers=1), ["Target"])
        sn.assert_called_once_with(["query"])
        namespace["logger"].error.assert_not_called()
        return rag

    def test_reference_rag_is_preserved_when_direct_local_search_is_disabled(self):
        self.run_reference_expansion(False).assert_called_once_with(["query"], 8192)

    def test_disabling_reference_rag_keeps_sn_expansion(self):
        self.run_reference_expansion(False, reference_rag=False).assert_not_called()

    def test_enabling_local_search_calls_rag(self):
        self.run_reference_expansion(True).assert_called_once_with(["query"], 8192)

    def test_empty_local_search_does_not_reference_undefined_paper_titles(self):
        local = Mock(return_value=[])
        google = Mock()
        namespace = {
            "threading": threading, "time": time, "USE_LOCAL_SEARCH": True,
            "arxiv_search_id": local, "google_search_paper": google,
        }
        worker = load_function("quick_thinker.py", "search_paper_worker", namespace, "PaperAgent")
        quick = types.SimpleNamespace(lock=threading.Lock(), search_papers=5,
                                      end_date="20240101", logger=Mock())
        worker(quick, ["query"])
        local.assert_called_once_with("query", 5, "20240101")
        google.assert_not_called()

    def test_service_probe_reports_http_status_without_credential_url(self):
        url = "https://configured.example/?accessKey=private-test-key"
        namespace = {
            "Request": Request, "json": json, "HTTPError": HTTPError,
            "urlopen": Mock(side_effect=HTTPError(url, 403, "Forbidden", {}, None)),
        }
        probe = load_function("check_setup.py", "probe_service", namespace)
        self.assertEqual(probe(("SN", url, {}, {}, "data")), ("SN", "HTTP 403"))

    def test_service_probe_rejects_successful_http_with_error_body(self):
        response = Mock()
        response.__enter__ = Mock(return_value=types.SimpleNamespace(read=lambda: b'{"error":"unavailable"}'))
        response.__exit__ = Mock(return_value=False)
        namespace = {"Request": Request, "json": json, "HTTPError": HTTPError,
                     "urlopen": Mock(return_value=response)}
        probe = load_function("check_setup.py", "probe_service", namespace)
        self.assertEqual(probe(("SN", "https://configured.example/", {}, {}, "data")),
                         ("SN", "unexpected response schema"))

    def test_disabled_rag_and_chunk_services_are_not_probed(self):
        probes = load_function("check_setup.py", "service_probes", {})
        config = {
            "common_config": {"google_api_key": "test", "JINA_API_KEYS": "test"},
            "sn_access_key": "test", "sn_search_paper_url": "https://example.test/search",
            "sn_get_paper_info_url": "https://example.test/metadata",
            "sn_get_ref_url": "https://example.test/refs?accessKey=",
            "use_local_search": False, "use_reference_rag": False, "use_chunk": False,
        }
        self.assertEqual({job[0] for job in probes(config)},
                         {"Serper", "Jina", "SN keyword search", "SN paper metadata", "SN references"})

    def test_reference_rag_is_probed_even_when_direct_local_search_is_disabled(self):
        probes = load_function("check_setup.py", "service_probes", {})
        config = {
            "common_config": {"google_api_key": "test", "JINA_API_KEYS": "test",
                              "arxiv_search_url": "https://example.test/rag"},
            "sn_access_key": "test", "sn_search_paper_url": "https://example.test/search",
            "sn_get_paper_info_url": "https://example.test/metadata",
            "sn_get_ref_url": "https://example.test/refs?accessKey=",
        }
        jobs = probes(config)
        self.assertEqual({job[0] for job in jobs},
                         {"Serper", "Jina", "SN keyword search", "SN paper metadata", "SN references", "Reference RAG"})


if __name__ == "__main__":
    unittest.main()
