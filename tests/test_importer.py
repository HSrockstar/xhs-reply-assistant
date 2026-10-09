"""xhs_qa 导入器的合成回归用例。"""

import json

import pytest

from app.importers.xhs_qa import parse_upload


def msg(comment_id, content, *, target, parent=None, root=None, user_id=None, context_only=False, time=1):
    return {
        "note_id": "note-1",
        "comment_id": comment_id,
        "parent_comment_id": parent,
        "root_comment_id": root or parent or comment_id,
        "content": content,
        "nickname": "作者" if target else "提问者",
        "user_id": user_id or ("author-1" if target else "asker-1"),
        "target": target,
        "context_only": context_only,
        "platform_time": time,
        "missing_parent": False,
    }


def group(kind, messages, warnings=None):
    return {"kind": kind, "messages": messages, "warnings": warnings or []}


def upload(groups, *, status="complete"):
    return {
        "id": "collection-1",
        "status": status,
        "nickname": "作者",
        "user_id": "author-1",
        "groups": groups,
        "warnings": [],
    }


def parse(value):
    return parse_upload(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def test_same_file_is_stable_and_has_hashes():
    source = upload(
        [group("qa", [msg("q", "怎么保存？", target=False), msg("a", "点击收藏。", target=True, parent="q")])],
        status="partial",
    )
    raw = json.dumps(source, ensure_ascii=False).encode()
    first = parse_upload(raw)
    second = parse_upload(raw)
    assert first == second
    assert first["file_hash"]
    assert first["stats"]["ready"] == 1
    assert first["warnings"]
    assert first["units"][0]["id"] == second["units"][0]["id"]
    assert first["units"][0]["content_hash"]


def test_multiturn_reply_chain_merges_and_splits_new_question():
    source_group = group(
        "qa",
        [
            msg("q1", "这个怎么用？", target=False, root="q1", time=1),
            msg("a1", "先打开页面。", target=True, parent="q1", root="q1", time=2),
            msg("a2", "然后点右上角。", target=True, parent="a1", root="q1", time=3),
            msg("q2", "保存后在哪里看？", target=False, parent="a2", root="q1", time=4),
            msg("a3", "在个人页的收藏里。", target=True, parent="q2", root="q1", time=5),
        ],
    )
    result = parse(upload([source_group]))
    assert len(result["units"]) == 2
    first, second = result["units"]
    assert first["status"] == "ready"
    assert first["question_text"] == "这个怎么用？"
    assert first["answer_text"] == "先打开页面。\n然后点右上角。"
    assert first["core_ids"] == ["note-1:a1", "note-1:a2"]
    assert second["question_text"] == "保存后在哪里看？"
    assert second["answer_text"] == "在个人页的收藏里。"
    assert second["context_text"] == "这个怎么用？\n先打开页面。\n然后点右上角。"
    assert second["core_ids"] == ["note-1:a3"]
    assert result["stats"]["core_target_comments"] == 3


def test_sibling_target_replies_are_separate_and_need_review():
    result = parse(
        upload(
            [
                group(
                    "qa",
                    [
                        msg("q", "怎么设置？", target=False, root="q", time=1),
                        msg("a1", "先进入设置。", target=True, parent="q", root="q", time=2),
                        msg("a2", "也可以从菜单进入。", target=True, parent="q", root="q", time=3),
                    ],
                )
            ]
        )
    )
    assert len(result["units"]) == 2
    assert {unit["status"] for unit in result["units"]} == {"needs_review"}
    assert not any(unit["content_enabled"] or unit["style_enabled"] for unit in result["units"])
    assert {tuple(unit["core_ids"]) for unit in result["units"]} == {
        ("note-1:a1",),
        ("note-1:a2",),
    }


def test_self_question_is_not_misused_as_non_target_question():
    result = parse(
        upload(
            [
                group(
                    "qa",
                    [
                        msg("self-q", "具体要怎么做？", target=True, root="self-q", time=1),
                        msg("self-a", "按页面提示逐步操作。", target=True, parent="self-q", root="self-q", time=2),
                    ],
                )
            ]
        )
    )
    assert len(result["units"]) == 1
    unit = result["units"][0]
    assert unit["status"] == "needs_review"
    assert unit["question_text"] == "具体要怎么做？"
    assert unit["answer_text"] == "按页面提示逐步操作。"
    assert unit["core_ids"] == ["note-1:self-a"]
    assert not unit["content_enabled"] and not unit["style_enabled"]
    assert result["stats"]["context_target_comments"] == 1
    assert result["stats"]["target_disposition_counts"] == {
        "core": 1,
        "context_only": 1,
        "empty": 0,
        "unclassified": 0,
    }
    assert sum(result["stats"]["target_disposition_counts"].values()) == result["stats"]["target_comments"]


def test_missing_parent_is_reviewed_without_unrelated_question():
    result = parse(
        upload(
            [
                group(
                    "qa",
                    [
                        msg("unrelated", "普通说明。", target=False, root="unrelated"),
                        msg("answer", "这样就可以了。", target=True, parent="absent", root="unrelated"),
                    ],
                )
            ]
        )
    )
    unit = result["units"][0]
    assert unit["status"] == "needs_review"
    assert unit["question_text"] == ""
    assert not unit["content_enabled"] and not unit["style_enabled"]
    assert any("缺少父评论" in reason for reason in unit["review_reasons"])


def test_context_only_target_is_counted_but_not_duplicated_as_core():
    source = upload(
        [
            group(
                "qa",
                [
                    msg("old-answer", "之前的回复。", target=True, root="q", context_only=True),
                    msg("q", "还有后续吗？", target=False, root="q"),
                    msg("new-answer", "这里补充说明。", target=True, parent="q", root="q", time=2),
                ],
            )
        ]
    )
    result = parse(source)
    assert result["stats"]["target_comments"] == 2
    assert result["stats"]["core_target_comments"] == 1
    assert result["stats"]["context_target_comments"] == 1
    assert result["stats"]["target_disposition_counts"] == {
        "core": 1,
        "context_only": 1,
        "empty": 0,
        "unclassified": 0,
    }
    assert sum(result["stats"]["target_disposition_counts"].values()) == result["stats"]["target_comments"]
    assert result["units"][0]["core_ids"] == ["note-1:new-answer"]


def test_context_copy_and_core_copy_of_same_comment_do_not_conflict():
    context_copy = msg("same", "原文保留。", target=True, root="same", context_only=True)
    core_copy = dict(context_copy)
    core_copy["context_only"] = False
    result = parse(
        upload(
            [
                group("qa", [context_copy]),
                group("standalone", [core_copy]),
            ]
        )
    )
    assert len(result["units"]) == 1
    assert result["units"][0]["status"] == "ready"
    assert result["units"][0]["core_ids"] == ["note-1:same"]
    assert result["stats"]["core_target_comments"] == 1
    assert result["stats"]["context_target_comments"] == 1
    assert result["stats"]["target_disposition_counts"] == {
        "core": 1,
        "context_only": 0,
        "empty": 0,
        "unclassified": 0,
    }
    assert sum(result["stats"]["target_disposition_counts"].values()) == result["stats"]["target_comments"]


def test_content_hash_detects_text_and_relationship_changes():
    base = upload(
        [
            group(
                "qa",
                [
                    msg("q", "如何操作？", target=False, root="q"),
                    msg("a", "打开页面。", target=True, parent="q", root="q", time=2),
                ],
            )
        ]
    )
    changed = upload(
        [
            group(
                "qa",
                [
                    msg("q", "如何操作？", target=False, root="q"),
                    msg("a", "打开设置页。", target=True, parent="missing", root="q", time=2),
                ],
            )
        ]
    )
    first = parse(base)["units"][0]
    second = parse(changed)["units"][0]
    assert first["source_fingerprint"] == second["source_fingerprint"]
    assert first["content_hash"] != second["content_hash"]
    assert first["original_messages"][1]["content"] == "打开页面。"


def test_empty_and_emoji_only_target_comments_are_excluded():
    result = parse(
        upload(
            [
                group(
                    "standalone",
                    [
                        msg("empty", "   ", target=True, root="empty"),
                        msg("emoji", "👍🏻❤️", target=True, root="emoji", time=2),
                    ],
                )
            ]
        )
    )
    assert result["stats"]["empty_target_comments"] == 2
    assert result["stats"]["excluded"] == 2
    assert {unit["status"] for unit in result["units"]} == {"excluded"}
    assert not any(unit["content_enabled"] or unit["style_enabled"] for unit in result["units"])


@pytest.mark.parametrize("kind", ["unresolved", "future-kind"])
def test_unresolved_and_unknown_groups_are_retained_but_disabled(kind):
    result = parse(upload([group(kind, [msg("a", "保留原文。", target=True, root="a")])]))
    unit = result["units"][0]
    assert unit["status"] == "needs_review"
    assert unit["original_messages"][0]["content"] == "保留原文。"
    assert not unit["content_enabled"] and not unit["style_enabled"]
    assert result["stats"]["group_kind_counts"] == {kind: 1}


def test_partial_collection_keeps_good_group_and_reviews_unresolved():
    result = parse(
        upload(
            [
                group(
                    "qa",
                    [
                        msg("q", "怎么处理？", target=False, root="q"),
                        msg("a", "按步骤提交。", target=True, parent="q", root="q", time=2),
                    ],
                ),
                group("unresolved", [msg("u", "暂存原文。", target=True, root="u")]),
            ],
            status="partial",
        )
    )
    assert [unit["status"] for unit in result["units"]] == ["ready", "needs_review"]
    assert result["metadata"]["status"] == "partial"
    assert result["stats"]["ready"] == 1
    assert result["stats"]["needs_review"] == 1


def test_duplicate_core_comment_across_groups_is_only_imported_once():
    repeated = msg("same", "同一条回复。", target=True, root="same")
    result = parse(upload([group("standalone", [repeated]), group("standalone", [dict(repeated)])]))
    assert result["stats"]["groups"] == 2
    assert result["stats"]["target_comments"] == 1
    assert result["stats"]["core_target_comments"] == 1
    assert len(result["units"]) == 1
    assert any("未重复生成核心回复" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    "raw,max_bytes",
    [
        (b"{}", 1),
        (b"\xff", 10),
        (b"[]", 10),
        (b'{"id":"x","id":"y"}', 100),
    ],
)
def test_invalid_file_shapes_and_size_raise_chinese_value_error(raw, max_bytes):
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        parse_upload(raw, max_bytes=max_bytes)


def test_invalid_message_field_types_raise_value_error():
    value = upload([group("qa", [msg("a", "正文", target="true")])])
    with pytest.raises(ValueError, match="target"):
        parse(value)


def test_invalid_size_limit_raise_value_error():
    with pytest.raises(ValueError, match="大小上限"):
        parse_upload(b"{}", max_bytes=0)
