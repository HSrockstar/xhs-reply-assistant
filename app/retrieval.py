"""个人回复助手的本地混合检索与 CUDA 向量索引。"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
PREPROCESS_VERSION = "zh-sentence-windows-v1"
_CACHE_VERSION = 1
_GPU_LOCK = threading.RLock()
_RRF_K = 60
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_SCENES = ("方法解释", "情绪回应", "澄清追问", "观点讨论", "感谢祝福")
_SUBJECTS = ("语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治", "科学")
_GRADE_PATTERNS = (
    (re.compile(r"小学\s*([一二三四五六1-6])\s*年级"), "小学"),
    (re.compile(r"初中\s*([一二三])\s*年级"), "初中"),
    (re.compile(r"高中\s*([一二三])\s*年级"), "高中"),
    (re.compile(r"([初高])([一二三])"), None),
    (re.compile(r"大([一二三四1-4])"), "大学"),
    (re.compile(r"(10|11|12|[一二三四五六七八九1-9])\s*年级"), None),
)
_DIGITS = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六"}


class RetrievalError(RuntimeError):
    """检索环境或向量编码错误。"""


class Retriever:
    """按完整问答检索内容，并独立选择目标作者的风格示例。"""

    def __init__(self, cache_dir: Path, settings: dict):
        self.cache_dir = Path(cache_dir)
        supplied = dict(settings or {})
        self.settings = {
            "embedding_model": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
            "embedding_device": "cuda:0",
            "embedding_dtype": "float16",
            "embedding_batch_size": 32,
            **supplied,
        }
        self.settings["embedding_batch_size"] = max(1, int(self.settings.get("embedding_batch_size", 32)))
        self._device = str(self.settings.get("embedding_device", "cuda:0"))
        self._dtype = str(self.settings.get("embedding_dtype", "float16")).lower()
        self._revision = str(self.settings.get("embedding_revision", MODEL_REVISION))
        self._model_name = str(self.settings.get("embedding_model", MODEL_NAME))
        self._preprocess_version = PREPROCESS_VERSION
        self._lock = _GPU_LOCK
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._index_vectors = None
        self._window_ids: list[str] = []
        self._records: dict[str, dict] = {}
        self._indexed_record_ids: list[str] = []
        self._index_fingerprint: str | None = None
        self._failed_fingerprint: str | None = None
        self._failure_retryable = False
        self._last_error: str | None = None
        self._last_error_kind: str | None = None
        self._gpu_name: str | None = None
        self._model_parameter_device: str | None = None
        self._model_output_device: str | None = None
        self._model_output_dtype: str | None = None
        self._effective_batch_size = self.settings["embedding_batch_size"]
        self._runtime_data = {
            "index_ms": 0.0,
            "query_ms": 0.0,
            "cold_load_ms": 0.0,
            "peak_memory_mb": 0.0,
        }

    def runtime(self) -> dict:
        """返回当前运行状态；只探测框架与 CUDA，不解析或下载模型。"""
        with self._lock:
            available, reason = self._probe_cuda()
            return {
                "available": available,
                "vector_ready": self._index_vectors is not None,
                "device": self._device if available else None,
                "gpu_name": self._gpu_name,
                "model_name": self._model_name,
                "model_source": str(self.settings.get("local_model_path") or self._model_name),
                "model_parameter_device": self._model_parameter_device,
                "model_output_device": self._model_output_device,
                "model_output_dtype": self._model_output_dtype,
                "dtype": self._dtype,
                "revision": self._revision,
                "batch_size": self._effective_batch_size,
                "peak_memory_mb": self._runtime_data["peak_memory_mb"],
                "index_ms": self._runtime_data["index_ms"],
                "query_ms": self._runtime_data["query_ms"],
                "cold_load_ms": self._runtime_data["cold_load_ms"],
                "error": self._last_error or reason,
                "error_kind": self._last_error_kind,
            }

    def rebuild(self, examples: list[dict]) -> dict:
        """重建或读取与当前示例快照对应的向量索引。"""
        with self._lock:
            started = time.perf_counter()
            records = self._prepare_records(examples)
            fingerprint = self._fingerprint(records)
            self._index_vectors = None
            self._window_ids = []
            self._records = {record["_rid"]: record for record in records}
            self._indexed_record_ids = list(self._records)
            self._index_fingerprint = None
            self._failed_fingerprint = None
            self._last_error = None
            self._last_error_kind = None

            if not records:
                self._index_fingerprint = fingerprint
                self._runtime_data["index_ms"] = 0.0
                return self._rebuild_result(True, 0, [])

            available, reason = self._probe_cuda()
            if not available:
                return self._fail_rebuild(fingerprint, "cuda", reason, started, retryable=True)

            try:
                self._ensure_model()
                index_started = time.perf_counter()
                metadata = self._cache_metadata(records, fingerprint)
                cached = self._load_cache(metadata)
                if cached is not None:
                    vectors, window_ids = cached
                    self._set_index(vectors, window_ids, records, fingerprint)
                    elapsed = (time.perf_counter() - index_started) * 1000
                    self._runtime_data["index_ms"] = elapsed
                    self._refresh_peak_memory()
                    return self._rebuild_result(True, len(records), [])

                torch = self._torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self._device)
                window_texts: list[str] = []
                window_ids: list[str] = []
                for record in records:
                    raw = self._document_text(record)
                    if not raw.strip():
                        continue
                    chunks = self._split_text(
                        raw,
                        self._tokenizer,
                        int(self._model.max_seq_length),
                    )
                    for chunk in chunks:
                        window_texts.append(chunk)
                        window_ids.append(record["_rid"])

                if window_texts:
                    vectors = self._encode_texts(window_texts)
                else:
                    dimension = int(self._model.get_sentence_embedding_dimension())
                    vectors = torch.empty((0, dimension), device=self._device, dtype=torch.float32)
                self._set_index(vectors, window_ids, records, fingerprint)
                self._save_cache(metadata, vectors, window_ids)
                elapsed = (time.perf_counter() - index_started) * 1000
                self._runtime_data["index_ms"] = elapsed
                self._refresh_peak_memory()
                return self._rebuild_result(True, len(records), [])
            except Exception as exc:
                kind = "oom" if self._is_oom(exc) else "model"
                return self._fail_rebuild(
                    fingerprint,
                    kind,
                    self._format_error(exc),
                    started,
                    retryable=False,
                )

    def search(
        self,
        question: str,
        background: str,
        examples: list[dict],
        scene: str = "auto",
        length: str = "自然",
    ) -> dict:
        """同步检索内容与风格示例；异步服务应通过 asyncio.to_thread 调用。"""
        with self._lock:
            warnings: list[str] = []
            safe_question = self._text(question)
            safe_background = self._text(background)
            query_text = self._join_text(safe_question, safe_background)
            active = self._prepare_records(examples)
            content_records = [record for record in active if self._is_content_ready(record)]
            fingerprint = self._fingerprint(active)

            if self._index_fingerprint != fingerprint:
                if self._failed_fingerprint == fingerprint and not self._failure_retryable:
                    pass
                elif self._failed_fingerprint == fingerprint:
                    available, _ = self._probe_cuda()
                    if available:
                        self.rebuild(examples)
                else:
                    self.rebuild(examples)

            keyword_scores = self._keyword_scores(query_text, content_records)
            keyword_order = self._top_order(keyword_scores, 10)
            vector_scores: dict[str, float] = {}
            vector_order: list[str] = []
            if self._index_vectors is not None and self._index_fingerprint == fingerprint:
                try:
                    vector_scores, vector_order = self._vector_scores(query_text, content_records, limit=10)
                except Exception as exc:
                    self._last_error = self._format_error(exc)
                    self._last_error_kind = "oom" if self._is_oom(exc) else "query"
                    warnings.append(self._fallback_warning(self._last_error))
                    self._index_vectors = None
                    self._index_fingerprint = None
                    self._failed_fingerprint = fingerprint
                    self._failure_retryable = False
            else:
                reason = self._last_error or self.runtime().get("error")
                warnings.append(self._fallback_warning(reason))

            content = self._rank_content(
                content_records,
                query_text,
                keyword_scores,
                keyword_order,
                vector_scores,
                vector_order,
            )
            inferred_scene = self._normalize_scene(scene, query_text)
            wanted_length = self._normalize_length(length)
            style = self._rank_style(
                active,
                query_text,
                inferred_scene,
                wanted_length,
                content,
                keyword_scores,
                vector_scores,
            )
            return {
                "content": content,
                "style": style,
                "runtime": self.runtime(),
                "warnings": list(dict.fromkeys(warnings)),
            }

    def _probe_cuda(self) -> tuple[bool, str | None]:
        self._gpu_name = None
        if not self._device.startswith("cuda:"):
            return False, "仅支持显式 CUDA 设备，未启用 CPU 回退"
        try:
            torch = self._load_torch()
        except Exception as exc:
            return False, f"PyTorch 不可用：{self._format_error(exc)}"
        try:
            if not torch.cuda.is_available():
                return False, "PyTorch 未检测到可用 CUDA 设备"
            index = int(self._device.partition(":")[2])
            if index < 0 or index >= torch.cuda.device_count():
                return False, f"CUDA 设备不存在：{self._device}"
            self._gpu_name = str(torch.cuda.get_device_name(index))
            return True, None
        except Exception as exc:
            return False, f"CUDA 探测失败：{self._format_error(exc)}"

    def _load_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _ensure_model(self) -> None:
        if self._model is not None:
            self._verify_model_device()
            return
        self._load_torch()
        available, reason = self._probe_cuda()
        if not available:
            raise RetrievalError(reason or "CUDA 不可用")
        if self._dtype not in {"float16", "float32"}:
            raise RetrievalError(f"向量精度不受支持：{self._dtype}")

        started = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            local_path = self.settings.get("local_model_path")
            source = str(local_path) if local_path else self._model_name
            kwargs = {
                "device": self._device,
                "local_files_only": True,
            }
            if not local_path:
                kwargs["revision"] = self._revision
            model = SentenceTransformer(source, **kwargs)
            model.to(self._device)
            model.half() if self._dtype == "float16" else model.float()
            model.eval()
            self._model = model
            self._tokenizer = getattr(model, "tokenizer", None)
            if self._tokenizer is None:
                raise RetrievalError("SentenceTransformer 未提供 tokenizer")
            self._verify_model_device()
            if not getattr(model, "max_seq_length", None):
                raise RetrievalError("SentenceTransformer 未提供有效 max_seq_length")
        finally:
            self._runtime_data["cold_load_ms"] = (time.perf_counter() - started) * 1000

    def _verify_model_device(self) -> None:
        expected = self._device
        try:
            devices = {str(parameter.device) for parameter in self._model.parameters()}
        except Exception as exc:
            raise RetrievalError(f"无法检查模型参数设备：{self._format_error(exc)}") from exc
        if not devices or devices != {expected}:
            raise RetrievalError(f"模型参数设备为 {sorted(devices) if devices else '空'}，期望 {expected}")
        self._model_parameter_device = expected

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower()

    @staticmethod
    def _format_error(exc: BaseException | str | None) -> str:
        if exc is None:
            return "未知错误"
        message = str(exc).strip()
        return message or type(exc).__name__

    @staticmethod
    def _fallback_warning(reason: str | None) -> str:
        base = "GPU 向量检索不可用，当前仅使用关键词检索"
        return f"{base}；原因：{reason}" if reason else base

    def _fail_rebuild(
        self,
        fingerprint: str,
        kind: str,
        reason: str,
        started: float,
        retryable: bool,
    ) -> dict:
        self._index_vectors = None
        self._index_fingerprint = None
        self._failed_fingerprint = fingerprint
        self._failure_retryable = retryable
        self._last_error = reason
        self._last_error_kind = kind
        self._runtime_data["index_ms"] = (time.perf_counter() - started) * 1000
        return self._rebuild_result(False, 0, [self._fallback_warning(reason)])

    def _rebuild_result(self, ok: bool, count: int, warnings: list[str]) -> dict:
        return {
            "ok": ok,
            "indexed": count,
            "runtime": self.runtime(),
            "warnings": warnings,
        }

    def _refresh_peak_memory(self) -> None:
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                peak = self._torch.cuda.max_memory_allocated(self._device)
                self._runtime_data["peak_memory_mb"] = round(float(peak) / (1024 * 1024), 3)
        except Exception:
            pass

    def _cache_path(self) -> Path:
        return self.cache_dir / "retrieval-index-v1.pt"

    def _cache_metadata(self, records: list[dict], fingerprint: str) -> dict:
        doc_hashes = {
            record["_rid"]: hashlib.sha256(self._document_text(record).encode("utf-8")).hexdigest()
            for record in records
        }
        return {
            "cache_version": _CACHE_VERSION,
            "model": self._model_name,
            "model_source": str(self.settings.get("local_model_path") or self._model_name),
            "revision": self._revision,
            "device": self._device,
            "dtype": self._dtype,
            "preprocess_version": self._preprocess_version,
            "fingerprint": fingerprint,
            "document_hashes": doc_hashes,
        }

    def _load_cache(self, metadata: dict):
        path = self._cache_path()
        if not path.is_file():
            return None
        try:
            torch = self._torch
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("metadata") != metadata:
                return None
            vectors = payload["vectors"]
            window_ids = payload["window_ids"]
            if len(window_ids) != int(vectors.shape[0]):
                return None
            vectors = vectors.to(device=self._device, dtype=torch.float32)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            return vectors, list(window_ids)
        except Exception:
            return None

    def _save_cache(self, metadata: dict, vectors, window_ids: list[str]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch = self._torch
            payload = {
                "metadata": metadata,
                "window_ids": list(window_ids),
                "vectors": vectors.detach().to(device="cpu", dtype=torch.float32),
            }
            fd, temp_name = tempfile.mkstemp(prefix=".retrieval-index-", suffix=".tmp", dir=str(self.cache_dir))
            os.close(fd)
            try:
                torch.save(payload, temp_name)
                with open(temp_name, "rb+") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp_name, self._cache_path())
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        except Exception as exc:
            self._last_error = f"索引缓存未能保存：{self._format_error(exc)}"

    def _set_index(self, vectors, window_ids: list[str], records: list[dict], fingerprint: str):
        self._index_vectors = vectors
        self._window_ids = list(window_ids)
        self._records = {record["_rid"]: record for record in records}
        self._indexed_record_ids = list(self._records)
        self._index_fingerprint = fingerprint
        self._failed_fingerprint = None
        self._failure_retryable = False
        self._last_error = None
        self._last_error_kind = None

    def _encode_texts(self, texts: Sequence[str]):
        torch = self._torch
        if not texts:
            return torch.empty((0, 0), device=self._device, dtype=torch.float32)
        token_counts = [self._token_count(text) for text in texts]
        if any(count <= 0 for count in token_counts):
            raise RetrievalError("发现空编码窗口")
        if any(count + self._special_tokens() > int(self._model.max_seq_length) for count in token_counts):
            raise RetrievalError("窗口超过模型长度限制；拒绝静默截断")

        order = sorted(range(len(texts)), key=lambda index: token_counts[index])
        encoded: list[Any] = [None] * len(texts)
        batch_size = min(self._effective_batch_size, self.settings["embedding_batch_size"])
        cursor = 0
        while cursor < len(order):
            current_batch = min(batch_size, len(order) - cursor)
            chosen = order[cursor : cursor + current_batch]
            try:
                batch_vectors = self._encode_batch([texts[index] for index in chosen])
                for offset, index in enumerate(chosen):
                    encoded[index] = batch_vectors[offset : offset + 1]
                cursor += current_batch
                batch_size = min(batch_size, self._effective_batch_size)
            except Exception as exc:
                if not self._is_oom(exc):
                    raise
                batch_vectors = None
                del exc
                self._release_cuda_temporaries()
                if current_batch <= 1:
                    raise RetrievalError("单条编码仍因 CUDA 显存不足而失败") from None
                batch_size = max(1, current_batch // 2)
                self._effective_batch_size = min(self._effective_batch_size, batch_size)

        return torch.cat(encoded, dim=0)

    def _encode_batch(self, texts: Sequence[str]):
        torch = self._torch
        features = self._model.tokenize(list(texts))
        for key, value in list(features.items()):
            if hasattr(value, "to"):
                features[key] = value.to(self._device)
        tensor_inputs = [value for value in features.values() if hasattr(value, "device")]
        if not tensor_inputs or any(str(value.device) != self._device for value in tensor_inputs):
            raise RetrievalError(f"模型输入张量未位于 {self._device}")
        input_ids = features.get("input_ids")
        if input_ids is None or input_ids.shape[1] > int(self._model.max_seq_length):
            raise RetrievalError("分窗编码超出模型长度限制；拒绝静默截断")
        with torch.inference_mode():
            output = self._model(features)
        embeddings = output.get("sentence_embedding") if isinstance(output, dict) else None
        if embeddings is None or not hasattr(embeddings, "device"):
            raise RetrievalError("SentenceTransformer 未返回 sentence_embedding 张量")
        if str(embeddings.device) != self._device:
            raise RetrievalError(f"模型输出张量位于 {embeddings.device}，期望 {self._device}")
        self._model_output_device = str(embeddings.device)
        self._model_output_dtype = str(embeddings.dtype).removeprefix("torch.")
        embeddings = embeddings.to(dtype=torch.float32)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _release_cuda_temporaries(self) -> None:
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass

    def _vector_scores(
        self,
        query_text: str,
        records: list[dict],
        limit: int,
    ) -> tuple[dict[str, float], list[str]]:
        if not query_text.strip() or self._index_vectors is None or not self._window_ids:
            return {}, []
        torch = self._torch
        query_windows = self._split_text(
            query_text,
            self._tokenizer,
            int(self._model.max_seq_length),
            prefix=QUERY_INSTRUCTION,
        )
        if not query_windows:
            return {}, []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device)
        started = time.perf_counter()
        query_vectors = self._encode_texts(query_windows)
        similarities = query_vectors @ self._index_vectors.T

        active_ids = {record["_rid"] for record in records}
        ids = [rid for rid in self._indexed_record_ids if rid in active_ids]
        per_record = []
        retained_ids = []
        for rid in ids:
            window_positions = [index for index, indexed_id in enumerate(self._window_ids) if indexed_id == rid]
            if not window_positions:
                continue
            positions = torch.tensor(window_positions, device=self._device, dtype=torch.long)
            score = similarities.index_select(1, positions).max()
            per_record.append(score.reshape(1))
            retained_ids.append(rid)
        if not per_record:
            return {}, []
        scores_tensor = torch.cat(per_record, dim=0)
        top_count = min(max(0, int(limit)), int(scores_tensor.shape[0]))
        if top_count == 0:
            return {}, []
        values, indices = torch.topk(scores_tensor, k=top_count, largest=True, sorted=True)
        selected_values = values.detach().to(device="cpu").tolist()
        selected_indices = indices.detach().to(device="cpu").tolist()
        order = [retained_ids[index] for index in selected_indices]
        score_map = {rid: float(value) for rid, value in zip(order, selected_values, strict=False)}
        self._runtime_data["query_ms"] = (time.perf_counter() - started) * 1000
        self._refresh_peak_memory()
        return score_map, order

    def _prepare_records(self, examples: Iterable[dict] | None) -> list[dict]:
        prepared: list[dict] = []
        seen_ids: set[str] = set()
        for position, original in enumerate(examples or []):
            if not isinstance(original, dict) or not self._is_active(original):
                continue
            record = dict(original)
            rid = self._record_id(record)
            if rid in seen_ids:
                rid = f"{rid}#{position}"
            seen_ids.add(rid)
            record["_rid"] = rid
            record["_original"] = original
            question = self._first_text(record, "question_text", "question", "prompt_text")
            context = self._first_text(record, "context_text", "context", "background")
            record["_question"] = question
            record["_context"] = context
            record["_answer"] = self._first_text(
                record,
                "answer_text",
                "target_answer",
                "reply_text",
                "answer",
                "final_text",
                "text",
            )
            record["_session"] = self._session_id(record, rid)
            if not record["_answer"]:
                continue
            prepared.append(record)
        return prepared

    @staticmethod
    def _first_text(record: dict, *keys: str) -> str:
        for key in keys:
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _join_text(*parts: str) -> str:
        return "\n".join(part.strip() for part in parts if part and part.strip())

    def _document_text(self, record: dict) -> str:
        return self._join_text(record.get("_question", ""), record.get("_context", ""))

    def _record_id(self, record: dict) -> str:
        for key in ("id", "example_id", "qa_id", "record_id"):
            if record.get(key) is not None and str(record[key]).strip():
                return str(record[key]).strip()
        raw = self._canonical_json(record)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _session_id(record: dict, fallback: str) -> str:
        for key in ("source_conversation_id", "conversation_id", "thread_id", "session_id"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    @staticmethod
    def _bool_value(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"false", "0", "no", "off", "disabled", "inactive"}:
                return False
            if value in {"true", "1", "yes", "on", "enabled", "active"}:
                return True
        return bool(value)

    def _is_active(self, record: dict) -> bool:
        if record.get("evaluation_holdout"):
            return False
        for key in ("active", "enabled", "is_active", "usable", "target", "is_target"):
            if key in record and not self._bool_value(record.get(key)):
                return False
        if self._bool_value(record.get("context_only"), default=False):
            return False
        status = str(record.get("status", "")).strip().lower()
        if status and status not in {"ready", "active", "enabled", "已就绪", "启用"}:
            return False
        if "content_enabled" in record and "style_enabled" in record:
            if not record["content_enabled"] and not record["style_enabled"]:
                return False
        return status not in {
            "disabled",
            "inactive",
            "deleted",
            "unresolved",
            "rejected",
            "draft",
            "停用",
            "未解决",
            "草稿",
        }

    def _is_style_ready(self, record: dict) -> bool:
        if not self._is_active(record):
            return False
        if not self._bool_value(record.get("style_enabled"), default=False):
            return False
        if not self._bool_value(record.get("ready"), default=record.get("status") == "ready"):
            return False
        status = str(record.get("status", "")).strip().lower()
        if status and status not in {"ready", "active", "enabled", "已就绪", "启用"}:
            return False
        return bool(record.get("_answer", "").strip())

    def _is_content_ready(self, record: dict) -> bool:
        if not self._is_active(record) or not record.get("_answer", "").strip():
            return False
        if "content_enabled" in record and not self._bool_value(record.get("content_enabled")):
            return False
        if "ready" in record and not self._bool_value(record.get("ready")):
            return False
        return bool(self._document_text(record).strip())

    def _fingerprint(self, records: list[dict]) -> str:
        relevant = []
        for record in records:
            original = record.get("_original", record)
            relevant.append(
                {
                    "id": record["_rid"],
                    "record": original,
                    "document": self._document_text(record),
                    "answer": record.get("_answer", ""),
                }
            )
        payload = {
            "model": self._model_name,
            "model_source": str(self.settings.get("local_model_path") or self._model_name),
            "revision": self._revision,
            "dtype": self._dtype,
            "preprocess": self._preprocess_version,
            "records": relevant,
        }
        return hashlib.sha256(self._canonical_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: str(item),
        )

    def _keyword_scores(self, query: str, records: list[dict]) -> dict[str, float]:
        query_terms = self._terms(query)
        if not query_terms or not records:
            return {}
        documents = [self._terms(self._document_text(record)) for record in records]
        lengths = [len(document) for document in documents]
        average = sum(lengths) / max(1, len(lengths))
        document_frequency: Counter[str] = Counter()
        term_counts = []
        for document in documents:
            counts = Counter(document)
            term_counts.append(counts)
            document_frequency.update(counts.keys())
        total = len(documents)
        result = {}
        for record, counts, doc_length in zip(records, term_counts, lengths, strict=False):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * doc_length / max(1.0, average))
                score += idf * (frequency * (1.2 + 1.0) / denominator)
            if score > 0:
                result[record["_rid"]] = score
        return result

    @staticmethod
    def _terms(text: str) -> list[str]:
        normalized = (text or "").lower()
        terms: list[str] = []
        cjk_run: list[str] = []

        def flush_cjk() -> None:
            nonlocal cjk_run
            if cjk_run:
                terms.extend(cjk_run)
                terms.extend(cjk_run[index] + cjk_run[index + 1] for index in range(len(cjk_run) - 1))
                cjk_run = []

        for char in normalized:
            if _CJK.fullmatch(char):
                cjk_run.append(char)
            else:
                flush_cjk()
        flush_cjk()
        terms.extend(match.group(0) for match in _ASCII_WORD.finditer(normalized))
        return terms

    @staticmethod
    def _top_order(scores: dict[str, float], limit: int) -> list[str]:
        return [key for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit] if score > 0]

    def _rank_content(
        self,
        records: list[dict],
        query_text: str,
        keyword_scores: dict[str, float],
        keyword_order: list[str],
        vector_scores: dict[str, float],
        vector_order: list[str],
    ) -> list[dict]:
        by_id = {record["_rid"]: record for record in records}
        keyword_rank = {rid: rank for rank, rid in enumerate(keyword_order, start=1)}
        vector_rank = {rid: rank for rank, rid in enumerate(vector_order, start=1)}
        candidate_ids = list(dict.fromkeys(keyword_order + vector_order))
        query_conditions = self._conditions_from_query(query_text)
        ranked = []
        for rid in candidate_ids:
            record = by_id.get(rid)
            if record is None or not self._is_content_ready(record):
                continue
            conflicts = self._condition_conflicts(
                query_conditions,
                self._conditions_from_record(record),
            )
            if conflicts:
                continue
            kr = keyword_rank.get(rid)
            vr = vector_rank.get(rid)
            rrf = (1.0 / (_RRF_K + kr) if kr else 0.0) + (1.0 / (_RRF_K + vr) if vr else 0.0)
            ranked.append(
                (
                    rrf,
                    rid,
                    {
                        "keyword": keyword_scores.get(rid),
                        "keyword_rank": kr,
                        "vector": vector_scores.get(rid),
                        "vector_rank": vr,
                        "rrf": rrf,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))

        results: list[dict] = []
        seen_sessions: set[str] = set()
        seen_topics: set[tuple] = set()
        for _, rid, scores in ranked:
            record = by_id[rid]
            session = record["_session"]
            topic_key = self._topic_dedupe_key(record)
            if session in seen_sessions or (topic_key is not None and topic_key in seen_topics):
                continue
            seen_sessions.add(session)
            if topic_key is not None:
                seen_topics.add(topic_key)
            results.append(
                {
                    "example": dict(record["_original"]),
                    "rank": len(results) + 1,
                    "scores": scores,
                    "uses": ["content"],
                    "_rid": rid,
                }
            )
            if len(results) >= 3:
                break
        return results

    @staticmethod
    def _conditions_from_query(text: str) -> dict:
        return Retriever._extract_conditions(text)

    @staticmethod
    def _conditions_from_record(record: dict) -> dict:
        parts = [Retriever._text(record.get("_question")), Retriever._text(record.get("_context"))]
        for key in ("grade", "school_level", "subject", "course", "question_type", "problem_type"):
            if record.get(key):
                value = str(record[key])
                if key == "grade" and value.strip().isdigit():
                    value = f"{value.strip()}年级"
                parts.append(value)
        return Retriever._extract_conditions(" ".join(parts))

    @staticmethod
    def _extract_conditions(text: str) -> dict:
        value = (text or "").lower().replace(" ", "")
        grade = None
        for pattern, level in _GRADE_PATTERNS:
            match = pattern.search(value)
            if not match:
                continue
            token = match.group(1)
            token = _DIGITS.get(token, token)
            if pattern is _GRADE_PATTERNS[3][0]:
                level = "初中" if match.group(1) == "初" else "高中"
                grade_token = match.group(2)
            elif pattern is _GRADE_PATTERNS[4][0]:
                level = "大学"
                grade_token = token
            elif pattern is _GRADE_PATTERNS[0][0]:
                grade_token = token
            elif pattern is _GRADE_PATTERNS[1][0]:
                grade_token = token
            elif pattern is _GRADE_PATTERNS[2][0]:
                grade_token = token
            else:
                grade_token = token
                if pattern is _GRADE_PATTERNS[5][0]:
                    ordinal = {
                        "七": "一",
                        "八": "二",
                        "九": "三",
                        "十": "一",
                        "十一": "二",
                        "十二": "三",
                        "7": "一",
                        "8": "二",
                        "9": "三",
                        "10": "一",
                        "11": "二",
                        "12": "三",
                    }
                    if token in {"七", "八", "九", "7", "8", "9"}:
                        level, grade_token = "初中", ordinal[token]
                    elif token in {"十", "十一", "十二", "10", "11", "12"}:
                        level, grade_token = "高中", ordinal[token]
            grade = (level or "年级", grade_token)
            break
        subjects = set()
        for subject in _SUBJECTS:
            if subject in value:
                subjects.add(subject)
        question_type = None
        for label in ("选择题", "填空题", "计算题", "应用题", "作文", "阅读理解", "实验题", "编程题"):
            if label in value:
                question_type = label
                break
        return {"grade": grade, "subjects": subjects, "question_type": question_type}

    @staticmethod
    def _condition_conflicts(query: dict, candidate: dict) -> list[str]:
        conflicts = []
        if query["grade"] and candidate["grade"] and query["grade"] != candidate["grade"]:
            conflicts.append("grade")
        if query["subjects"] and candidate["subjects"] and not (query["subjects"] & candidate["subjects"]):
            conflicts.append("subject")
        if (
            query["question_type"]
            and candidate["question_type"]
            and query["question_type"] != candidate["question_type"]
        ):
            conflicts.append("question_type")
        return conflicts

    def _normalize_scene(self, scene: str, query: str) -> str:
        value = str(scene or "auto").strip().lower()
        aliases = {
            "方法": "方法解释",
            "方法解释": "方法解释",
            "explain": "方法解释",
            "advice": "方法解释",
            "情绪": "情绪回应",
            "情绪回应": "情绪回应",
            "emotion": "情绪回应",
            "empathy": "情绪回应",
            "澄清": "澄清追问",
            "澄清追问": "澄清追问",
            "clarify": "澄清追问",
            "question": "澄清追问",
            "观点": "观点讨论",
            "观点讨论": "观点讨论",
            "opinion": "观点讨论",
            "discussion": "观点讨论",
            "感谢": "感谢祝福",
            "感谢祝福": "感谢祝福",
            "thanks": "感谢祝福",
            "greeting": "感谢祝福",
        }
        if value != "auto":
            return aliases.get(value, scene if scene in _SCENES else "观点讨论")
        lowered = query.lower()
        checks = (
            ("感谢祝福", ("谢谢", "感谢", "恭喜", "祝你", "生日快乐", "节日快乐")),
            ("情绪回应", ("难过", "焦虑", "委屈", "崩溃", "压力", "害怕", "失望", "心情不好")),
            ("澄清追问", ("什么意思", "具体是", "能否", "请问", "还是", "你说的")),
            ("方法解释", ("怎么做", "如何", "步骤", "方法", "怎么办", "怎么解决", "建议")),
        )
        for selected, markers in checks:
            if any(marker in lowered for marker in markers):
                return selected
        return "观点讨论"

    @staticmethod
    def _normalize_length(length: str) -> str:
        value = str(length or "自然").strip().lower()
        if value in {"短", "简短", "short"}:
            return "短"
        if value in {"长", "详细", "long"}:
            return "长"
        return "自然"

    def _rank_style(
        self,
        records: list[dict],
        query_text: str,
        scene: str,
        length: str,
        content: list[dict],
        keyword_scores: dict[str, float],
        vector_scores: dict[str, float],
    ) -> list[dict]:
        used_content_ids = {item["_rid"] for item in content}
        by_id = {record["_rid"]: record for record in records}
        used_content_sessions = {by_id[rid]["_session"] for rid in used_content_ids if rid in by_id}
        keyword_rank = {rid: rank for rank, rid in enumerate(self._top_order(keyword_scores, 10), start=1)}
        vector_order = sorted(
            vector_scores,
            key=lambda rid: (-vector_scores[rid], rid),
        )[:10]
        vector_rank = {rid: rank for rank, rid in enumerate(vector_order, start=1)}
        candidates = []
        for record in records:
            if not self._is_style_ready(record):
                continue
            inferred_scene = self._record_scene(record)
            inferred_length = self._record_length(record)
            structure = self._record_structure(record["_answer"])
            scene_score = 1.0 if inferred_scene == scene else 0.0
            length_score = 1.0 if inferred_length == length else 0.0
            query_structure = self._query_structure(query_text, scene)
            structure_score = 1.0 if structure == query_structure else 0.0
            keyword_score = keyword_scores.get(record["_rid"], 0.0)
            vector_score = vector_scores.get(record["_rid"])
            kr = keyword_rank.get(record["_rid"])
            vr = vector_rank.get(record["_rid"])
            theme_rrf = (1.0 / (_RRF_K + kr) if kr else 0.0) + (1.0 / (_RRF_K + vr) if vr else 0.0)
            candidates.append(
                {
                    "record": record,
                    "scene": inferred_scene,
                    "length": inferred_length,
                    "structure": structure,
                    "scene_score": scene_score,
                    "length_score": length_score,
                    "structure_score": structure_score,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                    "keyword_rank": kr,
                    "vector_rank": vr,
                    "theme_rrf": theme_rrf,
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["scene_score"],
                -item["length_score"],
                -item["structure_score"],
                -item["theme_rrf"],
                item["record"]["_rid"],
            )
        )

        results = []
        seen_sessions: set[str] = set(used_content_sessions)
        for candidate in candidates:
            record = candidate["record"]
            if record["_rid"] in used_content_ids:
                for item in content:
                    if item["_rid"] == record["_rid"] and "style" not in item["uses"]:
                        item["uses"].append("style")
                continue
            if record["_session"] in seen_sessions:
                continue
            seen_sessions.add(record["_session"])
            results.append(
                {
                    "id": record["_rid"],
                    "answer_text": record["_answer"],
                    "scene": candidate["scene"],
                    "length": candidate["length"],
                    "structure": candidate["structure"],
                    "source_conversation_id": record["_session"],
                    "rank": len(results) + 1,
                    "scores": {
                        "scene": candidate["scene_score"],
                        "length": candidate["length_score"],
                        "structure": candidate["structure_score"],
                        "keyword": candidate["keyword_score"],
                        "keyword_rank": candidate["keyword_rank"],
                        "vector": candidate["vector_score"],
                        "vector_rank": candidate["vector_rank"],
                        "theme_rrf": candidate["theme_rrf"],
                    },
                    "uses": ["style"],
                }
            )
            if len(results) >= 3:
                break
        for item in content:
            item.pop("_rid", None)
        return results

    def _record_scene(self, record: dict) -> str:
        value = self._first_text(record, "scene", "reply_scene", "response_scene")
        if value:
            return self._normalize_scene(value, "")
        return self._normalize_scene("auto", record.get("_answer", ""))

    def _topic_dedupe_key(self, record: dict) -> tuple | None:
        conditions = self._conditions_from_record(record)
        condition_key = (
            conditions["grade"],
            tuple(sorted(conditions["subjects"])),
            conditions["question_type"],
        )
        for key in ("topic_id", "theme_id", "topic"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return (key, str(value).strip().lower(), condition_key)
        normalized_question = re.sub(r"[\W_]+", "", record.get("_question", "").lower())
        if not normalized_question:
            return None
        return ("question", normalized_question, condition_key)

    @staticmethod
    def _record_length(record: dict) -> str:
        value = Retriever._first_text(record, "length", "reply_length", "length_class")
        if value:
            return Retriever._normalize_length(value)
        answer_length = len(record.get("_answer", ""))
        if answer_length < 55:
            return "短"
        if answer_length > 180:
            return "长"
        return "自然"

    @staticmethod
    def _record_structure(answer: str) -> str:
        value = answer or ""
        if re.search(r"(^|\n)\s*(?:\d+[.、)]|[-*•])", value):
            return "分点"
        if re.search(r"[？?]\s*$", value):
            return "追问"
        if any(token in value for token in ("抱抱", "辛苦了", "理解你", "能感受到", "别着急")):
            return "共情"
        if any(token in value for token in ("首先", "其次", "因为", "建议", "可以先")):
            return "解释建议"
        if any(token in value for token in ("谢谢", "感谢", "祝你", "恭喜")):
            return "祝福感谢"
        return "直接回应"

    @staticmethod
    def _query_structure(query: str, scene: str) -> str:
        if scene == "感谢祝福":
            return "祝福感谢"
        if scene == "情绪回应":
            return "共情"
        if scene == "澄清追问":
            return "追问"
        if any(marker in query for marker in ("步骤", "怎么做", "如何", "建议")):
            return "解释建议"
        return "直接回应"

    @staticmethod
    def _split_text(
        text: str,
        tokenizer: Any,
        max_seq_length: int,
        prefix: str = "",
    ) -> list[str]:
        """按完整句子打包；单句超长时按 token 窗切开，拒绝静默截断。"""
        text = (text or "").strip()
        if not text:
            return []
        specials = int(tokenizer.num_special_tokens_to_add(pair=False))
        body_limit = max_seq_length - specials
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
        content_limit = body_limit - prefix_tokens
        if content_limit < 1:
            raise RetrievalError("查询指令占满模型窗口")

        segments = [part for part in _SENTENCE_SPLIT.split(text) if part]
        if not segments:
            segments = [text]
        windows: list[str] = []
        current: list[str] = []
        current_count = 0

        def flush() -> None:
            nonlocal current, current_count
            if current:
                windows.append(prefix + "".join(current))
                current = []
                current_count = 0

        for segment in segments:
            ids = tokenizer.encode(segment, add_special_tokens=False)
            if not ids:
                continue
            if len(ids) <= content_limit:
                if current and current_count + len(ids) > content_limit:
                    flush()
                current.append(segment)
                current_count += len(ids)
                continue

            flush()
            offset = 0
            while offset < len(ids):
                end = min(offset + content_limit, len(ids))
                chunk = ids[offset:end]
                decoded = tokenizer.decode(
                    chunk,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                while (
                    decoded
                    and len(tokenizer.encode(prefix + decoded, add_special_tokens=True)) > max_seq_length
                    and end - offset > 1
                ):
                    end -= 1
                    chunk = ids[offset:end]
                    decoded = tokenizer.decode(
                        chunk,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                if not decoded or len(tokenizer.encode(prefix + decoded, add_special_tokens=True)) > max_seq_length:
                    raise RetrievalError("无法在模型窗口内无损切分超长句子")
                windows.append(prefix + decoded)
                offset = end
        flush()

        for window in windows:
            if len(tokenizer.encode(window, add_special_tokens=True)) > max_seq_length:
                raise RetrievalError("分窗结果超过模型长度限制；拒绝静默截断")
        return windows

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _special_tokens(self) -> int:
        return int(self._tokenizer.num_special_tokens_to_add(pair=False))
