"""纯 Python 的小红书 QAService JSON 导入解析器。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MESSAGES = 50_000
MAX_TOTAL_CONTENT_CHARS = 500_000
MAX_CONTENT_CHARS = 100_000
MAX_ID_CHARS = 256
MAX_WARNING_CHARS = 2_048

_QUESTION_RE = re.compile(
    r"[?？]|(?:吗|呢|什么|怎么|如何|为何|为什么|多少|几种|几个|哪个|哪些|哪里|哪儿|谁|"
    r"能否|可否|是不是|是否|有没有|有吗|请问|请教|求问|想问|问下|求推荐|推荐一下)"
)
_KEYCAP_RE = re.compile(r"[0-9#*]\ufe0f?\u20e3")


class _DuplicateKeyError(ValueError):
    pass


@dataclass
class _Message:
    group_index: int
    ordinal: int
    raw: dict[str, Any]
    note_id: str
    comment_id: str
    target: bool
    context_only: bool
    content: str
    user_id: str
    parent_ref: tuple[str, str] | None
    explicit_missing_parent: bool
    time_key: tuple[float, str] = (0.0, "")
    missing_parent: bool = False
    cycle: bool = False

    @property
    def ref(self) -> tuple[str, str]:
        return self.note_id, self.comment_id


@dataclass
class _Group:
    index: int
    raw: dict[str, Any]
    kind: str
    messages: list[_Message]
    warnings: list[str]
    by_ref: dict[tuple[str, str], _Message] = field(default_factory=dict)
    order: list[_Message] = field(default_factory=list)
    duplicate_refs: set[tuple[str, str]] = field(default_factory=set)
    conflicting_refs: set[tuple[str, str]] = field(default_factory=set)
    roots: set[tuple[str, str]] = field(default_factory=set)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON 中包含无效数值：{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"JSON 中包含重复字段：{key}")
        result[key] = value
    return result


def _require_string(value: Any, label: str, *, allow_empty: bool = False, limit: int = MAX_ID_CHARS) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        adjective = "可为空的" if allow_empty else "非空的"
        raise ValueError(f"{label}必须是{adjective}字符串。")
    if len(value) > limit:
        raise ValueError(f"{label}超过长度限制。")
    return value


def _optional_string(value: Any, label: str, *, limit: int = MAX_ID_CHARS) -> str:
    if value is None:
        return ""
    return _require_string(value, label, allow_empty=True, limit=limit)


def _time_value(value: Any, comment_id: str) -> tuple[float, str]:
    try:
        if isinstance(value, bool) or value is None:
            raise ValueError
        number = float(value)
        if math.isfinite(number):
            if number > 100_000_000_000:
                number /= 1000
            return number, comment_id
    except (TypeError, ValueError, OverflowError):
        pass

    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.timestamp(), comment_id
        except (ValueError, OverflowError):
            pass
    return 0.0, comment_id


def _is_emoji_codepoint(char: str) -> bool:
    code = ord(char)
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0x2300 <= code <= 0x23FF
        or 0x1F1E6 <= code <= 0x1F1FF
        or 0x1F3FB <= code <= 0x1F3FF
        or 0xE0020 <= code <= 0xE007F
        or code in {0x200D, 0xFE0E, 0xFE0F, 0x20E3}
    )


def _is_empty_or_emoji(content: str) -> bool:
    compact = "".join(char for char in content if not char.isspace())
    if not compact:
        return True
    compact = re.sub(r"[0-9#*]\ufe0f?\u20e3", "", compact)
    remainder = "".join(char for char in compact if not _is_emoji_codepoint(char))
    return not remainder.strip()


def _looks_question(content: str) -> bool:
    return bool(content.strip() and _QUESTION_RE.search(content))


def _source_view(message: _Message) -> dict[str, Any]:
    raw = message.raw
    return {
        "note_id": message.note_id,
        "comment_id": message.comment_id,
        "parent_comment_id": raw.get("parent_comment_id"),
        "root_comment_id": raw.get("root_comment_id"),
        "content": message.content,
        "user_id": raw.get("user_id"),
        "platform_time": raw.get("platform_time"),
        "target": message.target,
    }


def _message_signature(message: _Message) -> str:
    return _canonical_hash(_source_view(message))


def _parent_ref(raw: dict[str, Any], note_id: str, comment_id: str) -> tuple[str, str] | None:
    parent = raw.get("parent_comment_id")
    root = raw.get("root_comment_id")
    if isinstance(parent, str) and parent:
        return None if parent == comment_id else (note_id, parent)
    if isinstance(root, str) and root and root != comment_id:
        return note_id, root
    return None


def _normalize_group(index: int, raw: dict[str, Any]) -> _Group:
    kind = _require_string(raw.get("kind"), f"第{index}组的 kind", limit=64)
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(f"第{index}组的 messages 必须是数组。")
    warnings_value = raw.get("warnings", [])
    if not isinstance(warnings_value, list) or any(not isinstance(item, str) for item in warnings_value):
        raise ValueError(f"第{index}组的 warnings 必须是字符串数组。")
    if any(len(item) > MAX_WARNING_CHARS for item in warnings_value):
        raise ValueError(f"第{index}组的警告文本超过长度限制。")

    messages: list[_Message] = []
    for ordinal, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"第{index}组第{ordinal + 1}条消息必须是对象。")
        note_id = _require_string(message.get("note_id"), f"第{index}组笔记 ID")
        comment_id = _require_string(message.get("comment_id"), f"第{index}组评论 ID")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"第{index}组评论正文必须是字符串。")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"第{index}组评论正文超过长度限制。")
        target = message.get("target")
        if not isinstance(target, bool):
            raise ValueError(f"第{index}组 target 必须是布尔值。")
        context_only = message.get("context_only", False)
        if not isinstance(context_only, bool):
            raise ValueError(f"第{index}组 context_only 必须是布尔值。")

        parent = message.get("parent_comment_id")
        root = message.get("root_comment_id")
        if parent is not None:
            _optional_string(parent, f"第{index}组 parent_comment_id")
        if root is not None:
            _optional_string(root, f"第{index}组 root_comment_id")
        user_id = _optional_string(message.get("user_id"), f"第{index}组 user_id")
        if "nickname" in message and message["nickname"] is not None:
            _optional_string(message["nickname"], f"第{index}组 nickname")
        if "platform_time" in message and message["platform_time"] is not None:
            timestamp = message["platform_time"]
            if not isinstance(timestamp, (str, int, float)) or isinstance(timestamp, bool):
                raise ValueError(f"第{index}组 platform_time 类型无效。")

        messages.append(
            _Message(
                group_index=index,
                ordinal=ordinal,
                raw=message,
                note_id=note_id,
                comment_id=comment_id,
                target=target,
                context_only=context_only,
                content=content,
                user_id=user_id,
                parent_ref=_parent_ref(message, note_id, comment_id),
                explicit_missing_parent=message.get("missing_parent") is True,
                time_key=_time_value(message.get("platform_time"), comment_id),
            )
        )
    return _Group(index, raw, kind, messages, list(warnings_value))


def _prepare_group(group: _Group) -> None:
    occurrences: dict[tuple[str, str], list[_Message]] = {}
    for node in group.messages:
        occurrences.setdefault(node.ref, []).append(node)
    for ref, rows in occurrences.items():
        if len(rows) > 1:
            group.duplicate_refs.add(ref)
        canonical = next(
            (row for row in rows if row.target and not row.context_only),
            next((row for row in rows if row.target), rows[0]),
        )
        group.by_ref[ref] = canonical
        if len({_message_signature(row) for row in rows}) > 1:
            group.conflicting_refs.add(ref)

    for node in group.by_ref.values():
        if node.parent_ref is not None:
            node.missing_parent = node.explicit_missing_parent or node.parent_ref not in group.by_ref
        else:
            node.missing_parent = node.explicit_missing_parent
        if node.context_only and not node.target:
            group.conflicting_refs.add(node.ref)

    children: dict[tuple[str, str], list[_Message]] = {}
    roots: list[_Message] = []
    for node in group.by_ref.values():
        if node.parent_ref in group.by_ref and node.parent_ref != node.ref:
            children.setdefault(node.parent_ref, []).append(node)
        else:
            roots.append(node)
        root_id = node.raw.get("root_comment_id")
        root = root_id if isinstance(root_id, str) and root_id else node.comment_id
        group.roots.add((node.note_id, root))

    for rows in children.values():
        rows.sort(key=lambda item: item.time_key)
    roots.sort(key=lambda item: item.time_key)
    state: dict[tuple[str, str], int] = {}
    ordered: list[_Message] = []

    def visit(node: _Message) -> None:
        mark = state.get(node.ref, 0)
        if mark == 1:
            node.cycle = True
            return
        if mark == 2:
            return
        state[node.ref] = 1
        ordered.append(node)
        for child in children.get(node.ref, []):
            if state.get(child.ref) == 1:
                node.cycle = True
                child.cycle = True
                continue
            visit(child)
        state[node.ref] = 2

    for node in roots:
        visit(node)
    for node in sorted(group.by_ref.values(), key=lambda item: item.time_key):
        if state.get(node.ref, 0) == 0:
            node.cycle = True
            visit(node)
    group.order = ordered


def _path_to_root(group: _Group, node: _Message) -> tuple[list[_Message], bool, bool]:
    reverse_path: list[_Message] = []
    seen: set[tuple[str, str]] = set()
    current = node
    missing = False
    cycle = False
    while True:
        if current.ref in seen:
            cycle = True
            break
        seen.add(current.ref)
        reverse_path.append(current)
        cycle = cycle or current.cycle
        parent_ref = current.parent_ref
        if parent_ref is None:
            missing = missing or current.missing_parent
            break
        parent = group.by_ref.get(parent_ref)
        if parent is None:
            missing = True
            break
        current = parent
    reverse_path.reverse()
    return reverse_path, missing, cycle


def _descendant_targets(group: _Group, root: _Message) -> list[_Message]:
    children: dict[tuple[str, str], list[_Message]] = {}
    for node in group.by_ref.values():
        if node.parent_ref in group.by_ref and node.parent_ref != node.ref:
            children.setdefault(node.parent_ref, []).append(node)
    found: list[_Message] = []
    seen: set[tuple[str, str]] = set()
    stack = list(children.get(root.ref, []))
    while stack:
        node = stack.pop()
        if node.ref in seen:
            continue
        seen.add(node.ref)
        if node.target and not node.context_only:
            found.append(node)
        stack.extend(children.get(node.ref, []))
    return found


def _ordered_nodes(group: _Group, nodes: list[_Message]) -> list[_Message]:
    positions = {node.ref: index for index, node in enumerate(group.order)}
    unique = {node.ref: node for node in nodes}
    return sorted(unique.values(), key=lambda node: positions.get(node.ref, node.ordinal))


def _append_unique(target: list[str], values: list[str] | tuple[str, ...] | set[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _thread_key(group: _Group) -> str:
    roots = sorted(group.roots)
    if len(roots) == 1:
        return f"xhs:{roots[0][0]}:{roots[0][1]}"
    return "xhs-thread:" + _canonical_hash(roots)


def _author_for(core_nodes: list[_Message], metadata_user_id: str | None) -> tuple[str, bool]:
    authors = list(dict.fromkeys(node.user_id for node in core_nodes if node.user_id))
    if len(authors) == 1:
        author = authors[0]
        return author, bool(metadata_user_id and metadata_user_id != author)
    if len(authors) > 1:
        return authors[0], True
    return metadata_user_id or "", False


def _group_source_refs(group: _Group) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in group.messages:
        if node.ref not in seen:
            seen.add(node.ref)
            refs.append({"note_id": node.note_id, "comment_id": node.comment_id})
    return refs


def _content_hash_view(group: _Group) -> list[dict[str, Any]]:
    views = [_source_view(node) for node in group.messages]
    return sorted(
        views,
        key=lambda item: (
            item["note_id"],
            item["comment_id"],
            str(item.get("parent_comment_id") or ""),
            str(item.get("content") or ""),
        ),
    )


def _make_unit(
    group: _Group,
    *,
    metadata_user_id: str | None,
    answer_nodes: list[_Message],
    core_nodes: list[_Message],
    question_node: _Message | None = None,
    context_nodes: list[_Message] | None = None,
    reasons: list[str] | None = None,
    empty_core: bool = False,
    question_without_answer: bool = False,
) -> dict[str, Any]:
    context_nodes = context_nodes or []
    reasons = list(dict.fromkeys(reason for reason in (reasons or []) if reason))
    ordered_core = _ordered_nodes(group, core_nodes)
    ordered_answer = _ordered_nodes(group, answer_nodes)
    answer_text = "\n".join(
        node.content for node in ordered_answer if node.content and not _is_empty_or_emoji(node.content)
    )
    question_text = question_node.content if question_node is not None else ""
    context_text = "\n".join(node.content for node in _ordered_nodes(group, context_nodes) if node.content)
    if question_node is not None and question_node.target:
        _append_unique(reasons, ["自问自答关系需要人工确认。"])

    author_id, author_conflict = _author_for(ordered_core or ordered_answer, metadata_user_id)
    if author_conflict:
        _append_unique(reasons, ["目标作者与合集元数据不一致或核心评论存在多个作者。"])

    if empty_core:
        _append_unique(reasons, ["目标评论正文为空或仅含表情。"])
        unit_status = "excluded"
    elif question_without_answer:
        _append_unique(reasons, ["目标评论提出问题，但未找到明确的目标回复。"])
        unit_status = "needs_review"
    elif reasons:
        unit_status = "needs_review"
    else:
        unit_status = "ready"

    core_ids = [f"{node.note_id}:{node.comment_id}" for node in ordered_core]
    source_fingerprint = _canonical_hash({"author_id": author_id, "core_ids": sorted(set(core_ids))})
    content_hash = _canonical_hash(
        {
            "kind": group.kind,
            "source_thread_key": _thread_key(group),
            "core_ids": sorted(set(core_ids)),
            "question_text": question_text,
            "context_text": context_text,
            "answer_text": answer_text,
            "messages": _content_hash_view(group),
        }
    )
    return {
        "id": "xhsqa_" + _canonical_hash({"version": 1, "source_fingerprint": source_fingerprint})[:32],
        "version": 1,
        "source_type": "xhs_qa",
        "source_refs": _group_source_refs(group),
        "source_thread_key": _thread_key(group),
        "original_messages": copy.deepcopy([node.raw for node in group.messages]),
        "question_text": question_text,
        "context_text": context_text,
        "answer_text": answer_text,
        "topic_tags": [],
        "scene_tags": [],
        "status": unit_status,
        "review_reasons": reasons,
        "content_enabled": unit_status == "ready" and group.kind == "qa",
        "style_enabled": unit_status == "ready" and bool(answer_text),
        "content_hash": content_hash,
        "source_fingerprint": source_fingerprint,
        "core_ids": core_ids,
        "origin": "imported",
        "author_id": author_id,
        "group_index": group.index,
    }


def _question_for_root(group: _Group, root: _Message) -> tuple[_Message | None, list[_Message], list[str], bool]:
    path, missing, cycle = _path_to_root(group, root)
    ancestors = path[:-1]
    question_candidates = [node for node in ancestors if _looks_question(node.content)]
    question_node = question_candidates[-1] if question_candidates else None
    reasons: list[str] = []
    if missing:
        reasons.append("回复链缺少父评论，问题关系无法确认。")
    if cycle:
        reasons.append("评论父子关系存在循环，需人工确认。")
    if question_node is None:
        reasons.append("未能从父子关系中确认非目标提问。")
        return None, ancestors, reasons, False

    question_index = ancestors.index(question_node)
    context_nodes = ancestors[:question_index]
    between = ancestors[question_index + 1 :]
    if question_node.target:
        reasons.append("目标作者的提问与回复构成自问自答，需人工确认。")
    if any(not node.target for node in between):
        reasons.append("提问与目标回复之间存在其他用户评论，属于旁支或关系含糊。")
    return question_node, context_nodes, reasons, question_node.target


def _build_target_components(group: _Group, candidates: dict[tuple[str, str], _Message]) -> list[list[_Message]]:
    children: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for ref, node in candidates.items():
        if node.parent_ref in candidates and node.parent_ref != ref:
            children.setdefault(node.parent_ref, []).append(ref)
    for refs in children.values():
        refs.sort(key=lambda ref: candidates[ref].time_key)

    positions = {node.ref: index for index, node in enumerate(group.order)}
    roots = [ref for ref, node in candidates.items() if node.parent_ref not in candidates or node.parent_ref == ref]
    roots.sort(key=lambda ref: positions.get(ref, 0))
    visited: set[tuple[str, str]] = set()
    components: list[list[_Message]] = []

    def collect(ref: tuple[str, str], component: list[_Message]) -> None:
        if ref in visited:
            return
        visited.add(ref)
        component.append(candidates[ref])
        for child in children.get(ref, []):
            collect(child, component)

    for ref in roots:
        component: list[_Message] = []
        collect(ref, component)
        if component:
            components.append(_ordered_nodes(group, component))
    for ref in sorted(candidates, key=lambda key: positions.get(key, 0)):
        if ref not in visited:
            component = []
            collect(ref, component)
            if component:
                components.append(_ordered_nodes(group, component))
    return components


def _process_group(
    group: _Group,
    *,
    metadata_user_id: str | None,
    claimed_core_refs: set[tuple[str, str]],
    global_conflicts: set[tuple[str, str]],
    context_target_refs: set[tuple[str, str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    group_reasons: list[str] = []
    if group.duplicate_refs:
        _append_unique(warnings, [f"第{group.index}组含重复评论 ID，已按来源 ID 去重。"])
        group_reasons.append("分组中存在重复评论 ID。")
    if group.conflicting_refs:
        group_reasons.append("同一评论 ID 的正文或关系字段不一致。")
    if any(row.context_only and not row.target for row in group.by_ref.values()):
        group_reasons.append("存在 context_only=true 但 target=false 的消息。")
    if len(group.roots) > 1:
        group_reasons.append("分组包含多个笔记或评论楼。")
    if group.warnings:
        _append_unique(warnings, group.warnings)
        group_reasons.append("来源分组带有警告，需人工检查。")

    supported_kind = group.kind in {"qa", "standalone"}
    if not supported_kind:
        label = "unresolved" if group.kind == "unresolved" else "未知类型"
        group_reasons.append(f"来源分组类型为 {label}，不能自动启用。")
        _append_unique(warnings, [f"第{group.index}组类型为 {label}，已关闭内容和风格用途。"])

    target_rows = [node for node in group.by_ref.values() if node.target]
    if not target_rows:
        _append_unique(warnings, [f"第{group.index}组不含目标评论，未生成回复单元。"])
        return units

    for node in target_rows:
        if node.context_only:
            context_target_refs.add(node.ref)
        if node.ref in global_conflicts:
            group_reasons.append("同一来源评论在文件内存在正文或关系冲突。")

    candidates: dict[tuple[str, str], _Message] = {}
    duplicate_refs: set[tuple[str, str]] = set()
    for node in target_rows:
        if node.context_only:
            continue
        if node.ref in claimed_core_refs:
            duplicate_refs.add(node.ref)
            context_target_refs.add(node.ref)
            continue
        claimed_core_refs.add(node.ref)
        candidates[node.ref] = node

    if duplicate_refs:
        group_reasons.append("目标评论与前序分组核心内容重复，已避免重复计入。")
        _append_unique(warnings, [f"第{group.index}组包含已计入的目标评论，未重复生成核心回复。"])
    if not candidates:
        if any(row.context_only for row in target_rows):
            _append_unique(warnings, [f"第{group.index}组的目标评论仅作为上下文计数，不生成独立回复单元。"])
        return units

    empty_refs = {ref for ref, node in candidates.items() if _is_empty_or_emoji(node.content)}
    for ref in empty_refs:
        node = candidates[ref]
        path, _, _ = _path_to_root(group, node)
        if group.kind == "qa":
            qnode, context_nodes, q_reasons, _ = _question_for_root(group, node)
        else:
            qnode, context_nodes, q_reasons = None, path[:-1], []
        if qnode is not None and qnode.target:
            context_target_refs.add(qnode.ref)
        units.append(
            _make_unit(
                group,
                metadata_user_id=metadata_user_id,
                answer_nodes=[],
                core_nodes=[node],
                question_node=qnode,
                context_nodes=context_nodes,
                reasons=[*group_reasons, *q_reasons],
                empty_core=True,
            )
        )

    nonempty = {ref: node for ref, node in candidates.items() if ref not in empty_refs}
    question_targets: set[tuple[str, str]] = set()
    for ref, node in nonempty.items():
        if _looks_question(node.content) and _descendant_targets(group, node):
            question_targets.add(ref)
            context_target_refs.add(ref)

    for ref, node in nonempty.items():
        if ref in question_targets or not _looks_question(node.content):
            continue
        if not _descendant_targets(group, node):
            units.append(
                _make_unit(
                    group,
                    metadata_user_id=metadata_user_id,
                    answer_nodes=[],
                    core_nodes=[node],
                    question_node=node,
                    context_nodes=_path_to_root(group, node)[0][:-1],
                    reasons=[*group_reasons, "目标评论提出问题，但未找到明确的目标回复。"],
                    question_without_answer=True,
                )
            )
            question_targets.add(ref)

    answer_candidates = {ref: node for ref, node in nonempty.items() if ref not in question_targets}
    if not supported_kind:
        for node in _ordered_nodes(group, list(answer_candidates.values())):
            path, missing, cycle = _path_to_root(group, node)
            reasons = list(group_reasons)
            if missing:
                reasons.append("回复链缺少父评论，需人工确认。")
            if cycle:
                reasons.append("评论父子关系存在循环，需人工确认。")
            units.append(
                _make_unit(
                    group,
                    metadata_user_id=metadata_user_id,
                    answer_nodes=[node],
                    core_nodes=[node],
                    context_nodes=path[:-1],
                    reasons=reasons,
                )
            )
        return units

    components = _build_target_components(group, answer_candidates)
    component_questions: list[tuple[list[_Message], _Message | None, list[_Message], list[str], bool]] = []
    for component in components:
        root = next(
            (node for node in component if node.parent_ref not in answer_candidates or node.parent_ref == node.ref),
            component[0],
        )
        if group.kind == "qa":
            qnode, context_nodes, q_reasons, self_qa = _question_for_root(group, root)
        else:
            path, missing, cycle = _path_to_root(group, root)
            target_questions = [node for node in path[:-1] if node.target and _looks_question(node.content)]
            if target_questions:
                qnode = target_questions[-1]
                context_nodes = path[: path.index(qnode)]
                q_reasons = ["目标作者的提问与回复构成自问自答，需人工确认。"]
                self_qa = True
            else:
                qnode, context_nodes, q_reasons, self_qa = None, path[:-1], [], False
            if missing:
                q_reasons.append("回复链缺少父评论，需人工确认。")
            if cycle:
                q_reasons.append("评论父子关系存在循环，需人工确认。")
        component_questions.append((component, qnode, context_nodes, q_reasons, self_qa))

    question_component_counts: dict[tuple[str, str], int] = {}
    for _, qnode, _, _, _ in component_questions:
        if qnode is not None:
            question_component_counts[qnode.ref] = question_component_counts.get(qnode.ref, 0) + 1

    for component, qnode, context_nodes, q_reasons, self_qa in component_questions:
        reasons = [*group_reasons, *q_reasons]
        if qnode is not None and question_component_counts.get(qnode.ref, 0) > 1:
            reasons.append("同一提问下存在多个目标回复分支，需人工选择。")
        component_refs = {node.ref for node in component}
        child_counts: dict[tuple[str, str], int] = {}
        for node in component:
            if node.parent_ref in component_refs:
                child_counts[node.parent_ref] = child_counts.get(node.parent_ref, 0) + 1
        if any(count > 1 for count in child_counts.values()):
            reasons.append("目标回复链出现多个分支，需人工确认。")
        if group.kind == "qa" and qnode is None:
            reasons.append("QA 分组中未能确认提问文本，不能自动启用。")
        if self_qa:
            reasons.append("目标作者先提问再自答，需人工确认。")
        if qnode is not None and qnode.target:
            context_target_refs.add(qnode.ref)
        for node in context_nodes:
            if node.target:
                context_target_refs.add(node.ref)

        answer_nodes = [node for node in component if not _looks_question(node.content)]
        units.append(
            _make_unit(
                group,
                metadata_user_id=metadata_user_id,
                answer_nodes=answer_nodes,
                core_nodes=component,
                question_node=qnode,
                context_nodes=context_nodes,
                reasons=reasons,
            )
        )

    context_target_refs.update(question_targets)
    return units


def parse_upload(raw: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> dict[str, Any]:
    try:
        return _parse_upload(raw, max_bytes)
    except RecursionError:
        raise ValueError("材料嵌套或回复链过深，请拆分文件后导入。") from None


def _parse_upload(raw: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> dict[str, Any]:
    """解析 QAService 保存的 JSON，并生成待入库的回复单元。"""
    if not isinstance(raw, bytes):
        raise ValueError("上传内容必须是 bytes。")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("文件大小上限必须是正整数。")
    if len(raw) > max_bytes:
        raise ValueError(f"上传文件超过 {max_bytes} 字节限制。")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("上传文件必须使用 UTF-8 编码。") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as exc:
        raise ValueError(str(exc)) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("上传文件不是有效的 JSON。") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象。")

    collection_id = _require_string(data.get("id"), "合集 ID")
    status = _require_string(data.get("status"), "合集状态", limit=64)
    nickname = _require_string(data.get("nickname"), "合集昵称")
    user_id_value = data.get("user_id")
    user_id = _require_string(user_id_value, "合集用户 ID", allow_empty=True) if user_id_value is not None else None

    raw_groups = data.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError("合集 groups 必须是数组。")
    if len(raw_groups) > MAX_MESSAGES:
        raise ValueError("分组数量超过限制。")
    top_warnings = data.get("warnings", [])
    if not isinstance(top_warnings, list) or any(not isinstance(item, str) for item in top_warnings):
        raise ValueError("合集 warnings 必须是字符串数组。")
    if any(len(item) > MAX_WARNING_CHARS for item in top_warnings):
        raise ValueError("合集警告文本超过长度限制。")

    groups: list[_Group] = []
    message_count = 0
    for index, raw_group in enumerate(raw_groups, 1):
        if not isinstance(raw_group, dict):
            raise ValueError(f"第{index}组必须是对象。")
        group = _normalize_group(index, raw_group)
        message_count += len(group.messages)
        if message_count > MAX_MESSAGES:
            raise ValueError("评论总数超过限制。")
        _prepare_group(group)
        groups.append(group)

    unique_content: dict[tuple[str, str], int] = {}
    target_rows: dict[tuple[str, str], _Message] = {}
    source_signatures: dict[tuple[str, str], set[str]] = {}
    context_target_refs: set[tuple[str, str]] = set()
    empty_target_refs: set[tuple[str, str]] = set()
    for group in groups:
        for node in group.messages:
            unique_content.setdefault(node.ref, len(node.content))
            if node.target:
                target_rows.setdefault(node.ref, node)
                source_signatures.setdefault(node.ref, set()).add(_message_signature(node))
                if node.context_only:
                    context_target_refs.add(node.ref)
                if _is_empty_or_emoji(node.content):
                    empty_target_refs.add(node.ref)
    if sum(unique_content.values()) > MAX_TOTAL_CONTENT_CHARS:
        raise ValueError("评论正文总长度超过限制。")
    global_conflicts = {ref for ref, signatures in source_signatures.items() if len(signatures) > 1}

    warnings: list[str] = []
    _append_unique(warnings, list(top_warnings))
    if status == "partial":
        _append_unique(
            warnings,
            ["来源合集处于 partial 状态；成功分组照常解析，未完成分组保持待审。"],
        )
    elif status not in {"complete", "partial", "running", "interrupted", "failed"}:
        _append_unique(warnings, ["来源合集状态未知，请人工核对。"])

    units: list[dict[str, Any]] = []
    claimed_core_refs: set[tuple[str, str]] = set()
    for group in groups:
        units.extend(
            _process_group(
                group,
                metadata_user_id=user_id,
                claimed_core_refs=claimed_core_refs,
                global_conflicts=global_conflicts,
                context_target_refs=context_target_refs,
                warnings=warnings,
            )
        )
    if global_conflicts:
        _append_unique(
            warnings,
            ["文件中同一评论 ID 对应多个正文或关系版本，相关回复单元已待审。"],
        )

    core_id_values = {core_id for unit in units for core_id in unit["core_ids"]}
    core_refs = {ref for ref in target_rows if f"{ref[0]}:{ref[1]}" in core_id_values}
    context_target_refs.update(ref for ref in target_rows if ref not in core_refs)
    group_kind_counts: dict[str, int] = {}
    for group in groups:
        group_kind_counts[group.kind] = group_kind_counts.get(group.kind, 0) + 1

    # 互斥去向按 source ref 去重，优先级为空正文、核心、上下文/非核心。
    disposition_core = core_refs - empty_target_refs
    disposition_context = context_target_refs - core_refs - empty_target_refs
    disposition_unclassified = set(target_rows) - disposition_core - disposition_context - empty_target_refs
    stats = {
        "groups": len(groups),
        "group_kind_counts": group_kind_counts,
        "target_comments": len(target_rows),
        "ready": sum(unit["status"] == "ready" for unit in units),
        "needs_review": sum(unit["status"] == "needs_review" for unit in units),
        "excluded": sum(unit["status"] == "excluded" for unit in units),
        "core_target_comments": len(core_refs),
        "context_target_comments": len(context_target_refs),
        "empty_target_comments": len(empty_target_refs),
        "target_disposition_counts": {
            "core": len(disposition_core),
            "context_only": len(disposition_context),
            "empty": len(empty_target_refs),
            "unclassified": len(disposition_unclassified),
        },
        "target_disposition_rule": (
            "按唯一 (note_id, comment_id) 计数，互斥优先级为 empty > core > context_only > unclassified；"
            "旧版 core/context_target_comments/empty_target_comments 字段可因同一评论多用途而重叠。"
        ),
    }
    return {
        "file_hash": hashlib.sha256(raw).hexdigest(),
        "metadata": {
            "collection_id": collection_id,
            "nickname": nickname,
            "user_id": user_id,
            "status": status,
        },
        "units": units,
        "stats": stats,
        "warnings": warnings,
    }
