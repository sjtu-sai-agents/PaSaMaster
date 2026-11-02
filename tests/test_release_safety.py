"""Offline regressions for optional local data and credential-safe errors."""

import ast
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class ReleaseSafetyTests(unittest.TestCase):
    def test_sn_transport_failure_does_not_expose_url_credentials(self):
        class TransportError(Exception):
            pass

        marker = "audit-only-credential"
        url = "https://configured.example/refs?accessKey=" + marker
        post = Mock(side_effect=TransportError("Connection failed: " + url))
        namespace = {"requests": types.SimpleNamespace(post=post, RequestException=TransportError)}
        request = load_function("tools/sn_server.py", "_make_request_with_retry", namespace)
        for is_json in (False, True):
            with self.subTest(is_json=is_json):
                with self.assertRaises(RuntimeError) as caught:
                    request(url, {}, {}, is_json=is_json)
                self.assertNotIn(marker, str(caught.exception))
                self.assertNotIn(url, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_empty_reference_titles_do_not_poison_valid_batch_entries(self):
        request = Mock(return_value=types.SimpleNamespace(json=lambda: {
            "data": {"Valid paper": {"refs": [{"enName": "Reference"}]}}
        }))
        namespace = {"sn_get_ref_url": "https://example.test/refs?accessKey=",
                     "access_key": "test", "json": json, "logger": Mock(),
                     "_make_request_with_retry": request}
        refs = load_function("tools/sn_server.py", "sn_get_ref", namespace)
        self.assertEqual(refs(["Valid paper", "", None, "  "]), [{"1": "Reference"}, {}, {}, {}])
        self.assertEqual(json.loads(request.call_args.args[2]),
                         {"titles": [{"enName": "Valid paper"}]})
        namespace["logger"].error.assert_not_called()

    def test_empty_optional_paper_metadata_uses_online_lookup(self):
        tree = ast.parse((ROOT / "tools/search_info.py").read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetaInfoClient")
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "load_dataset")
        namespace = {"Dict": dict, "Any": object, "json": json}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "tools/search_info.py", "exec"), namespace)
        self.assertEqual(namespace["load_dataset"](types.SimpleNamespace(data_path="")), {})

    def test_journal_enrichment_is_optional_but_still_loads_supplied_data(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "os": os, "json": json, "current_path": directory, "logger": Mock(),
                "JOURNAL_METAINFO_CACHE": {}, "JOURNAL_METAINFO_LIST": [],
                "JOURNAL_METAINFO_LOADED": False,
            }
            load = load_function("quick_thinker.py", "_load_journal_metainfo", namespace)
            load()
            self.assertTrue(namespace["JOURNAL_METAINFO_LOADED"])
            self.assertEqual(namespace["JOURNAL_METAINFO_CACHE"], {})
            namespace["logger"].error.assert_not_called()
            config = Path(directory) / "config"
            config.mkdir()
            record = {"journal_full": "Sample Journal", "journal_alias": ["SJ"], "h5_index": 10}
            (config / "journal_metainfo.jsonl").write_text(json.dumps(record) + "\n")
            namespace["JOURNAL_METAINFO_LOADED"] = False
            load()
            self.assertEqual(namespace["JOURNAL_METAINFO_CACHE"]["sample journal"], record)
            self.assertEqual(namespace["JOURNAL_METAINFO_CACHE"]["sj"], record)
            self.assertEqual(namespace["JOURNAL_METAINFO_LIST"], ["Sample Journal"])


if __name__ == "__main__":
    unittest.main()
