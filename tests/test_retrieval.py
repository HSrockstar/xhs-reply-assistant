"""检索模块的合成数据测试，不加载真实模型或真实用户数据。"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.retrieval import Retriever


def synthetic_examples() -> list[dict]:
    return [
        {
            "id": "math_fraction",
            "question_text": "初三数学分数方程怎么约分，分子分母怎样同时处理",
            "context_text": "代数基础练习",
            "answer_text": "先找分子和分母的公因数，再同时约分。",
            "grade": "初三",
            "subject": "数学",
            "conversation_id": "math-session-1",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "math_equation",
            "question_text": "初三数学一次方程移项后怎样检查结果",
            "context_text": "一次方程练习",
            "answer_text": "把求得的数代回原方程，两边相等就说明计算正确。",
            "grade": "九年级",
            "subject": "数学",
            "conversation_id": "math-session-2",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "physics_force",
            "question_text": "初二物理力的方向和作用点怎么画",
            "context_text": "力学作图",
            "answer_text": "先确定受力物体，再从作用点沿力的方向画带箭头的线段。",
            "grade": "初二",
            "subject": "物理",
            "conversation_id": "physics-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "english_words",
            "question_text": "初二英语单词总记不住，有什么记忆办法",
            "context_text": "英语词汇",
            "answer_text": "可以按词根词缀和主题分组，每天复习旧词再学少量新词。",
            "grade": "初二",
            "subject": "英语",
            "conversation_id": "english-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "chinese_essay",
            "question_text": "高一语文作文立意怎样避免跑题",
            "context_text": "议论文写作",
            "answer_text": "先圈出题目中的限制词，再用一句话写清中心观点。",
            "grade": "高一",
            "subject": "语文",
            "conversation_id": "essay-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "emotion_support",
            "question_text": "我最近焦虑难过，遇到挫折后总是很自责",
            "context_text": "希望得到安慰和鼓励",
            "answer_text": "抱抱你，遇到挫折时难受很正常，可以先给自己一点时间。",
            "conversation_id": "emotion-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "clarify_term",
            "question_text": "这个词是什么意思，能不能解释得具体一点",
            "context_text": "对方在追问概念",
            "answer_text": "你说的是哪一处用法？发一下前后句，我可以结合语境说明。",
            "conversation_id": "clarify-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "thanks_reply",
            "question_text": "谢谢老师的帮助，想回复一句祝福的话",
            "context_text": "节日问候",
            "answer_text": "也祝你一切顺利，保持好奇心，慢慢进步！",
            "conversation_id": "thanks-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "python_sort",
            "question_text": "Python 列表按数字排序的代码怎么写",
            "context_text": "编程入门",
            "answer_text": "直接使用 sorted(values) 会返回排好序的新列表。",
            "conversation_id": "python-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "math_ratio",
            "question_text": "初三数学比例尺和实际距离怎么换算",
            "context_text": "比例与应用题",
            "answer_text": "先统一图上距离和实际距离的单位，再按比例列式。",
            "grade": "九年级",
            "subject": "数学",
            "conversation_id": "math-session-1",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "disabled_secret",
            "question_text": "初三数学稀有合金题的独特代换技巧",
            "context_text": "这条记录已停用",
            "answer_text": "停用内容不可返回",
            "grade": "初三",
            "subject": "数学",
            "conversation_id": "disabled-session",
            "active": False,
            "ready": True,
            "style_enabled": True,
        },
        {
            "id": "wrong_grade",
            "question_text": "高二数学复合函数调换定理的特殊证明",
            "context_text": "只用于高二课程",
            "answer_text": "这是高二专用的合成函数解法。",
            "grade": "高二",
            "subject": "数学",
            "conversation_id": "wrong-grade-session",
            "ready": True,
            "style_enabled": False,
        },
        {
            "id": "style_ready",
            "question_text": "朋友遇到失败时怎样表达支持",
            "context_text": "目标作者的情绪回应样例",
            "answer_text": "抱抱你，已经很努力了。先缓一缓，再决定下一步也来得及。",
            "conversation_id": "style-session",
            "scene": "情绪回应",
            "length": "自然",
            "ready": True,
            "style_enabled": True,
        },
        {
            "id": "style_not_ready",
            "question_text": "同学压力很大时怎样安慰",
            "context_text": "尚未审核的风格样例",
            "answer_text": "你别难过。",
            "conversation_id": "unready-style-session",
            "scene": "情绪回应",
            "length": "短",
            "ready": False,
            "style_enabled": True,
        },
    ]


class NoCudaRetriever(Retriever):
    def _probe_cuda(self):
        return False, "测试模拟：无 CUDA"


class FakeVectorRetriever(Retriever):
    """跳过模型加载，用显式合成分数验证融合与变更触发重建。"""

    def __init__(self, cache_dir: Path, settings: dict):
        super().__init__(cache_dir, settings)
        self.rebuild_calls = 0

    def _probe_cuda(self):
        return True, None

    def rebuild(self, examples: list[dict]) -> dict:
        self.rebuild_calls += 1
        records = self._prepare_records(examples)
        fingerprint = self._fingerprint(records)
        self._set_index(object(), [], records, fingerprint)
        return self._rebuild_result(True, len(records), [])

    def _vector_scores(self, query_text, records, limit):
        if "python" in query_text.lower():
            preferences = {"python_sort": 0.99}
        else:
            preferences = {
                "math_equation": 0.97,
                "math_fraction": 0.82,
                "physics_force": 0.12,
            }
        active_ids = {record["_rid"] for record in records}
        scores = {rid: score for rid, score in preferences.items() if rid in active_ids}
        order = sorted(scores, key=lambda rid: (-scores[rid], rid))[:limit]
        return scores, order


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        values = list(text)
        return [ord(char) for char in values] + ([0, 0] if add_special_tokens else [])

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token) for token in ids if token)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeBatch:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, selection):
        return FakeBatch(len(range(*selection.indices(self.rows))))


class FakeTorch:
    @staticmethod
    def cat(items, dim=0):
        return FakeBatch(sum(item.rows for item in items))


class OomRetriever(Retriever):
    def __init__(self):
        super().__init__(Path("."), {"embedding_batch_size": 4})
        self._torch = FakeTorch()
        self._tokenizer = CharTokenizer()
        self._model = SimpleNamespace(max_seq_length=100)
        self.calls = []

    def _encode_batch(self, texts):
        self.calls.append(len(texts))
        if len(texts) > 2:
            raise RuntimeError("CUDA out of memory")
        return FakeBatch(len(texts))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.examples = synthetic_examples()

    def test_12_synthetic_expected_candidates(self):
        cases = [
            ("初三数学分数方程约分", "math_fraction"),
            ("初三数学一次方程移项检查", "math_equation"),
            ("初二物理力的方向作用点", "physics_force"),
            ("初二英语单词记忆办法", "english_words"),
            ("高一语文作文立意跑题", "chinese_essay"),
            ("焦虑难过挫折后自责安慰", "emotion_support"),
            ("这个词是什么意思具体解释", "clarify_term"),
            ("谢谢老师祝福的话", "thanks_reply"),
            ("Python 列表数字排序代码", "python_sort"),
            ("初三数学比例尺实际距离换算", "math_ratio"),
            ("初三数学稀有合金题代换技巧", "math_ratio"),
            ("九年级数学复合函数调换定理证明", "math_fraction"),
        ]
        retriever = NoCudaRetriever(Path("."), {})
        for query, expected_id in cases:
            with self.subTest(query=query):
                result = retriever.search(query, "", self.examples)
                found_ids = [item["example"]["id"] for item in result["content"]]
                self.assertNotIn("disabled_secret", found_ids)
                if "复合函数调换定理" in query:
                    self.assertNotIn("wrong_grade", found_ids)
                    self.assertEqual(found_ids[0], expected_id)
                    continue
                if "稀有合金题" in query:
                    self.assertNotIn("disabled_secret", found_ids)
                    self.assertEqual(found_ids[0], expected_id)
                    continue
                self.assertTrue(found_ids, result["warnings"])
                self.assertEqual(found_ids[0], expected_id)
                self.assertIn("GPU 向量检索不可用，当前仅使用关键词检索", result["warnings"][0])

    def test_style_only_uses_ready_author_answer(self):
        retriever = NoCudaRetriever(Path("."), {})
        result = retriever.search(
            "我最近很焦虑，遇到挫折后难受",
            "",
            self.examples,
            scene="auto",
            length="自然",
        )
        style_ids = [item["id"] for item in result["style"]]
        merged_style_ids = style_ids + [item["example"]["id"] for item in result["content"] if "style" in item["uses"]]
        self.assertIn("style_ready", merged_style_ids)
        self.assertNotIn("style_not_ready", style_ids)
        for item in result["style"]:
            self.assertNotIn("question_text", item)
            self.assertIn("answer_text", item)

    def test_vector_and_keyword_rank_fuse_without_score_scale_addition(self):
        retriever = FakeVectorRetriever(Path("."), {})
        result = retriever.search("初三数学一次方程", "向量召回优先", self.examples)
        first = result["content"][0]
        self.assertEqual(first["example"]["id"], "math_equation")
        self.assertEqual(first["scores"]["keyword_rank"], 1)
        self.assertEqual(first["scores"]["vector_rank"], 1)
        self.assertAlmostEqual(first["scores"]["rrf"], 2 / 61)

    def test_changed_or_disabled_example_triggers_rebuild_and_live_filter(self):
        retriever = FakeVectorRetriever(Path("."), {})
        retriever.search("Python 列表排序", "", self.examples)
        self.assertEqual(retriever.rebuild_calls, 1)
        self.assertEqual(
            retriever.search("Python 列表排序", "", self.examples)["content"][0]["example"]["id"], "python_sort"
        )
        self.assertEqual(retriever.rebuild_calls, 1)
        changed = copy.deepcopy(self.examples)
        next(item for item in changed if item["id"] == "python_sort")["active"] = False
        after = retriever.search("Python 列表排序", "", changed)
        self.assertEqual(retriever.rebuild_calls, 2)
        self.assertNotIn("python_sort", [item["example"]["id"] for item in after["content"]])

    def test_sentence_windows_cover_long_single_sentence_without_truncation(self):
        text = "超长单句必须拆成多个完整token窗口并保留最后一段内容"
        tokenizer = CharTokenizer()
        windows = Retriever._split_text(text, tokenizer, max_seq_length=10)
        self.assertGreater(len(windows), 1)
        self.assertEqual("".join(windows), text)
        self.assertTrue(all(len(tokenizer.encode(window, add_special_tokens=True)) <= 10 for window in windows))

    def test_sentence_boundaries_are_kept_when_they_fit(self):
        text = "第一句。第二句！第三句"
        windows = Retriever._split_text(text, CharTokenizer(), max_seq_length=40)
        self.assertEqual(windows, [text])

    def test_oom_reduces_batch_finitely_and_keeps_cuda_path(self):
        retriever = OomRetriever()
        encoded = retriever._encode_texts(["一", "二", "三", "四", "五"])
        self.assertEqual(retriever.calls, [4, 2, 2, 1])
        self.assertEqual(encoded.rows, 5)
        self.assertEqual(retriever._effective_batch_size, 2)

    def test_cpu_serialization_is_atomic_and_cache_metadata_is_checked(self):
        try:
            import torch
        except ImportError:
            self.skipTest("环境未安装 PyTorch")
        with tempfile.TemporaryDirectory() as folder:
            retriever = Retriever(Path(folder), {})
            retriever._torch = torch
            vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            metadata = {"revision": "rev-a", "text_hash": "hash-a"}
            retriever._save_cache(metadata, vectors, ["a", "b"])
            cache_path = retriever._cache_path()
            self.assertTrue(cache_path.is_file())
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["metadata"], metadata)
            self.assertEqual(payload["vectors"].device.type, "cpu")
            self.assertEqual(list(Path(folder).glob(".retrieval-index-*.tmp")), [])
            self.assertIsNone(retriever._load_cache({"revision": "rev-b"}))

    def test_fingerprint_changes_with_text_or_model_revision(self):
        original = Retriever(Path("."), {})
        prepared = original._prepare_records(self.examples)
        before = original._fingerprint(prepared)
        changed = copy.deepcopy(self.examples)
        changed[0]["question_text"] += " 新条件"
        after_text = original._fingerprint(original._prepare_records(changed))
        revised = Retriever(Path("."), {"embedding_revision": "other-revision"})
        after_revision = revised._fingerprint(revised._prepare_records(self.examples))
        self.assertNotEqual(before, after_text)
        self.assertNotEqual(before, after_revision)
        self.assertEqual(len(prepared), len(self.examples) - 1)

    def test_conditions_normalize_ninth_grade_and_filter_subject_conflicts(self):
        ninth = Retriever._extract_conditions("九年级数学方程")
        junior = Retriever._extract_conditions("初三数学方程")
        physics = Retriever._extract_conditions("初三物理方程")
        self.assertEqual(ninth["grade"], junior["grade"])
        self.assertIn("subject", Retriever._condition_conflicts(ninth, physics))

    def test_content_results_preserve_full_example_and_rank_metadata(self):
        retriever = NoCudaRetriever(Path("."), {})
        result = retriever.search("初三数学分数方程约分", "", self.examples)
        item = result["content"][0]
        self.assertIn("conversation_id", item["example"])
        self.assertIn("question_text", item["example"])
        self.assertEqual(item["uses"], ["content"])
        self.assertEqual(item["rank"], 1)
        self.assertIn("rrf", item["scores"])
        self.assertEqual(result["runtime"]["model_name"], "BAAI/bge-small-zh-v1.5")
        self.assertIn("gpu_name", result["runtime"])
        self.assertIn("model_parameter_device", result["runtime"])
        self.assertIn("model_output_device", result["runtime"])

    def test_short_style_request_prefers_short_scene_match(self):
        retriever = NoCudaRetriever(Path("."), {})
        result = retriever.search(
            "我最近焦虑难过",
            "",
            self.examples,
            scene="情绪回应",
            length="短",
        )
        self.assertTrue(result["style"])
        self.assertNotIn("style_not_ready", [item["id"] for item in result["style"]])


if __name__ == "__main__":
    unittest.main()
