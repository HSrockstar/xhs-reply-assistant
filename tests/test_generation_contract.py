from pathlib import Path

from test_retrieval import NoCudaRetriever

from app import generation
from app.config import DEFAULTS


def test_persisted_ready_records_and_answer_only_style_references_build_prompt(tmp_path):
    examples = [
        {
            "id": "short-reply",
            "status": "ready",
            "question_text": "谢谢你的帮助",
            "context_text": "",
            "answer_text": "不客气，祝你顺利！",
            "source_thread_key": "thread-one",
            "source_conversation_id": "thread-one",
            "content_enabled": False,
            "style_enabled": True,
        },
        {
            "id": "held-out",
            "status": "ready",
            "question_text": "谢谢",
            "context_text": "",
            "answer_text": "留出语料不得参与",
            "source_thread_key": "thread-two",
            "content_enabled": True,
            "style_enabled": True,
            "evaluation_holdout": True,
        },
    ]
    retriever = NoCudaRetriever(Path(tmp_path), {})
    references = retriever.search("感谢你的解释", "", examples, scene="thanks", length="短")
    assert references["style"]
    assert {item["id"] for item in references["style"]} == {"short-reply"}
    prompt, mapping, selected = generation.make_input({"question": "感谢你的解释"}, references, None, DEFAULTS)
    assert prompt["references"][0]["answer"] == "不客气，祝你顺利！"
    assert prompt["references"][0]["purposes"] == ["style"]
    assert mapping == {"e1": {"id": "short-reply", "purposes": ["style"]}}
    assert selected["style"][0]["id"] == "short-reply"


def test_content_reference_that_also_supports_style_keeps_both_declared_purposes():
    source = {"id": "dual", "question_text": "怎样复习", "context_text": "", "answer_text": "先拆小问题。"}
    references = {"content": [{"example": source, "uses": ["content", "style"]}], "style": []}
    prompt, mapping, selected = generation.make_input({"question": "怎样复习"}, references, None, DEFAULTS)
    assert prompt["references"][0]["purposes"] == ["content", "style"]
    assert selected["content"][0]["id"] == selected["style"][0]["id"] == "dual"
    result = {
        "reply_text": "先拆小问题。",
        "reference_usage": [{"example_id": "e1", "purpose": "style", "note": "表达参考"}],
        "supplement_summary": "",
        "clarification_needed": False,
        "review_notes": [],
    }
    assert generation.validate_reply(result, mapping)["reference_usage"][0]["example_id"] == "dual"
