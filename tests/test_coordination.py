"""Coordination regressions; no API access or research corpora are required."""

import ast
import asyncio
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_coordinator():
    """Replace only service/import boundaries; execute the real coordinator."""
    stubs = {}
    for name, attrs in {
        "quick_thinker": {"PaperAgent": object},
        "reason_agent": {"reason_agent": object},
        "paper_node": {"PaperNode": object},
        "log": {"manager_logger": Mock(), "setup_logging": Mock()},
        "tools.sn_server": {"sn_get_paper_info": Mock(return_value=[])},
    }.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        stubs[name] = stub
    spec = importlib.util.spec_from_file_location("tested_coordinator", ROOT / "PasaMaster.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    module.use_chunk = False
    return module


class Paper:
    def __init__(self, title, depth=0):
        self.title = title
        self.abstract = "Retrieved evidence for " + title
        self.select_score = 80
        self.reason = "Matches the current checklist"
        self.depth = depth
        self.arxiv_id = "1706.03762"
        self.authors = "Test author"
        self.year = "2017"
        self.source = "test retrieval boundary"

    def todic(self):
        return dict(self.__dict__)


class QuickThinker:
    def __init__(self, output, added):
        self.user_query = "Research question"
        self.checklist = None
        self.papers_queue = [Paper("Initial evidence")]
        self.searched_queries = set()
        self.queries = {}
        self.uncertain_content = ""
        self.expand_layers = 1
        self.output_dir = str(output)
        self.task_id = "unit"
        self.steps = []
        self.scoring_contexts = []
        self.added = added

    def translate(self, query):
        return query

    def process_step(self, queries, step):
        self.steps.append((list(queries), list(self.checklist or [])))
        self.searched_queries.update(queries)
        self.queries[str(step)] = list(queries)
        added = self.added[step - 1]
        for index in range(added):
            self.papers_queue.append(Paper(f"Evidence {step}-{index}", step))
        return added

    def rank_papers(self, papers, **kwargs):
        self.scoring_contexts.append((list(self.checklist or []), kwargs.get("all_rank")))
        for paper in papers:
            paper.select_score = 90
        return sorted(papers, key=lambda paper: paper.select_score, reverse=True)

    def top_rank(self, papers):
        pass


def plan(query, checklist="Initial constraint", state="continue"):
    return {
        "parsed_query": "Intent: " + checklist,
        "checklist": [checklist],
        "search_query": [query] if query else [],
        "quick_state": state,
    }


class Navigator:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.prompts = []
        self.trace = []
        self.paper_list = []

    async def run(self, prompt):
        self.prompts.append(prompt)
        return next(self.plans)


class CoordinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_coordinator()

    def run_case(self, plans, added, max_rounds=3, use_reasoner=True):
        with tempfile.TemporaryDirectory() as output:
            quick = QuickThinker(Path(output), added)
            navigator = Navigator(plans)
            manager = self.module.Manager(quick, navigator, use_reasoner=use_reasoner)
            result = asyncio.run(manager.run(max_rounds=max_rounds))
            return quick, navigator, manager, result

    def test_three_rounds_transfer_directions_feedback_and_refined_checklists(self):
        quick, navigator, manager, result = self.run_case([
            plan("direction one"),
            plan("direction two", "Refined constraint"),
            plan("direction three", "Final constraint"),
            plan("", "Final constraint", "stop"),
        ], [0, 1, 1])
        self.assertEqual([step[0] for step in quick.steps], [
            ["direction one"], ["direction two"], ["direction three"],
        ])
        self.assertEqual([step[1] for step in quick.steps], [
            ["Initial constraint"], ["Refined constraint"], ["Final constraint"],
        ])
        self.assertEqual(len(navigator.prompts), 4)
        self.assertIn("Initial evidence", navigator.prompts[1])
        self.assertIn("Relevance score: 90", navigator.prompts[1])
        self.assertIn("direction one", navigator.prompts[1])
        self.assertIn("Evidence 2-0", navigator.prompts[2])
        self.assertIn((["Refined constraint"], True), quick.scoring_contexts)
        self.assertEqual(quick.scoring_contexts[-1], (["Final constraint"], True))
        self.assertEqual(result["extra"]["expand_depth"], 3)
        self.assertEqual(len(result["extra"]["cooperation_rounds"]), 3)
        self.assertEqual(result["extra"]["stop_reason"], "navigator_stop")
        self.assertEqual(manager.shared_mind.get_reason_search_query(), [])

    def test_navigator_can_stop_early(self):
        quick, _, _, result = self.run_case([plan("first"), plan("", state="stop")], [1])
        self.assertEqual(len(quick.steps), 1)
        self.assertEqual(result["extra"]["stop_reason"], "navigator_stop")

    def test_no_progress_and_no_fresh_direction_stops(self):
        quick, _, _, result = self.run_case([plan("first"), plan("first")], [0])
        self.assertEqual(len(quick.steps), 1)
        self.assertEqual(result["extra"]["stop_reason"], "no_progress")
        self.assertEqual(result["extra"]["cooperation_rounds"][0]["next_search_queries"], [])

    def test_round_limit_is_enforced(self):
        quick, _, _, result = self.run_case([
            plan("first"), plan("second"), plan("third"),
        ], [1, 1], max_rounds=2)
        self.assertEqual(len(quick.steps), 2)
        self.assertEqual(result["extra"]["stop_reason"], "max_rounds")

    def test_disabled_navigator_is_not_called(self):
        quick, navigator, _, result = self.run_case([], [1, 1], max_rounds=2, use_reasoner=False)
        self.assertEqual(len(quick.steps), 2)
        self.assertEqual(navigator.prompts, [])
        self.assertEqual(result["extra"]["stop_reason"], "max_rounds")

    def test_cli_boolean_and_round_validation(self):
        self.assertFalse(self.module._parse_bool("False"))
        self.assertTrue(self.module._parse_bool("True"))
        self.assertEqual(self.module._positive_int("3"), 3)
        with self.assertRaises(ValueError):
            self.module._parse_bool("invalid")
        with self.assertRaises(ValueError):
            self.module._positive_int("0")

    def test_quick_thinker_does_not_repeat_searched_directions(self):
        tree = ast.parse((ROOT / "quick_thinker.py").read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PaperAgent")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "process_step")
        namespace = {"time": time, "logger": Mock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "quick_thinker.py", "exec"), namespace)
        quick = types.SimpleNamespace(
            coop_initialized=True, papers_queue=[], searched_queries={"old direction"},
            logger=Mock(), current_depth_processed=0, expand_layers=1, queries={},
            search_papers_with_queries=Mock(),
        )
        namespace["process_step"](quick, ["old direction", "new direction", "new direction"], 0)
        quick.search_papers_with_queries.assert_called_once_with(["new direction"], current_depth=0)


if __name__ == "__main__":
    unittest.main()
