"""在本机 CUDA 上核验合成中文检索与 FP16/FP32 top-3 一致性。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from app.retrieval import MODEL_NAME, MODEL_REVISION, QUERY_INSTRUCTION, Retriever

DOCUMENTS = [
    ("math_function", "初三数学一次函数的图像由斜率和截距决定，先找两个点再画直线。"),
    ("math_equation", "解一元一次方程后，把结果代回原式检查等式两边是否相等。"),
    ("physics_force", "力的示意图需要标出作用点、方向和大小，箭头沿力的方向。"),
    ("chinese_essay", "写作文先审题，圈出限制条件，再用一句话确定中心观点。"),
    ("python_list", "Python 列表排序可以使用 sorted 返回新列表，也可以用 sort 原地排序。"),
    ("emotion_reply", "遇到挫折后感到焦虑很正常，可以先休息一下，再慢慢想下一步。"),
]
QUERIES = [
    ("function_query", QUERY_INSTRUCTION + "怎么根据斜率和截距画出一次函数图像？"),
    ("equation_query", QUERY_INSTRUCTION + "解完一元一次方程后怎样检验结果？"),
]


def _sync(torch, device: str) -> None:
    torch.cuda.synchronize(device)


def _parameter_report(model) -> dict:
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()})
    return {"devices": devices, "parameter_dtypes": dtypes}


def _top3(torch, queries, documents, document_ids) -> list[dict]:
    similarities = queries @ documents.T
    count = min(3, int(similarities.shape[1]))
    values, indices = torch.topk(similarities, k=count, dim=1, largest=True, sorted=True)
    values = values.detach().to(device="cpu").tolist()
    indices = indices.detach().to(device="cpu").tolist()
    return [
        {
            "query_index": index,
            "ids": [document_ids[position] for position in indices[index]],
            "scores": [float(score) for score in values[index]],
        }
        for index in range(len(indices))
    ]


def _write_json(path: Path, value: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".verify-gpu-", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(temp_name, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def verify(output: Path, local_model_path: str | None, batch_size: int) -> dict:
    evidence = {
        "status": "failed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": MODEL_NAME,
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "device": "cuda:0",
        "dtype": "float16",
        "batch_size": batch_size,
        "peak_memory_mb": None,
        "index_ms": None,
        "query_ms": None,
        "cold_load_ms": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "sentence_transformers_version": None,
        "gpu_name": None,
        "gpu_total_memory_mb": None,
        "local_model_path": local_model_path,
        "synthetic_document_ids": [item[0] for item in DOCUMENTS],
        "synthetic_queries": [item[0] for item in QUERIES],
    }
    try:
        import importlib.metadata

        import torch

        evidence["torch_version"] = torch.__version__
        evidence["torch_cuda_version"] = torch.version.cuda
        evidence["sentence_transformers_version"] = importlib.metadata.version("sentence-transformers")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 未检测到 CUDA；核验不切换到 CPU。")
        device = "cuda:0"
        if torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA:0 不存在。")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        evidence["gpu_name"] = properties.name
        evidence["gpu_total_memory_mb"] = round(properties.total_memory / (1024 * 1024), 3)

        settings = {
            "embedding_model": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
            "embedding_device": device,
            "embedding_dtype": "float16",
            "embedding_batch_size": batch_size,
        }
        if local_model_path:
            settings["local_model_path"] = local_model_path
        retriever = Retriever(output.parent / ".verify-gpu-cache", settings)
        load_started = time.perf_counter()
        retriever._ensure_model()
        _sync(torch, device)
        evidence["cold_load_ms"] = round((time.perf_counter() - load_started) * 1000, 3)
        evidence["model_parameters_fp16"] = _parameter_report(retriever._model)
        if evidence["model_parameters_fp16"]["devices"] != [device]:
            raise RuntimeError("模型参数没有全部加载到 CUDA:0。")
        if evidence["model_parameters_fp16"]["parameter_dtypes"] != ["float16"]:
            raise RuntimeError("模型参数没有全部使用 FP16。")

        docs = []
        document_ids = []
        for document_id, text in DOCUMENTS:
            docs.extend(
                Retriever._split_text(
                    text,
                    retriever._tokenizer,
                    int(retriever._model.max_seq_length),
                )
            )
            document_ids.append(document_id)
        query_texts = []
        for _, text in QUERIES:
            query_texts.extend(
                Retriever._split_text(
                    text[len(QUERY_INSTRUCTION) :],
                    retriever._tokenizer,
                    int(retriever._model.max_seq_length),
                    prefix=QUERY_INSTRUCTION,
                )
            )
        if len(docs) != len(DOCUMENTS) or len(query_texts) != len(QUERIES):
            raise RuntimeError("核验文本分窗数量异常。")

        torch.cuda.reset_peak_memory_stats(device)
        index_started = time.perf_counter()
        doc_fp16 = retriever._encode_texts(docs)
        _sync(torch, device)
        evidence["index_ms"] = round((time.perf_counter() - index_started) * 1000, 3)
        query_started = time.perf_counter()
        query_fp16 = retriever._encode_texts(query_texts)
        top_fp16 = _top3(torch, query_fp16, doc_fp16, document_ids)
        _sync(torch, device)
        evidence["query_ms"] = round((time.perf_counter() - query_started) * 1000, 3)
        evidence["fp16_output_device"] = str(doc_fp16.device)
        evidence["fp16_embedding_dtype_after_normalization"] = str(doc_fp16.dtype).removeprefix("torch.")
        evidence["fp16_top3"] = top_fp16
        peak_fp16 = torch.cuda.max_memory_allocated(device)

        retriever._model.float()
        retriever._verify_model_device()
        evidence["model_parameters_fp32"] = _parameter_report(retriever._model)
        if evidence["model_parameters_fp32"]["devices"] != [device]:
            raise RuntimeError("FP32 基线模型参数离开了 CUDA:0。")
        torch.cuda.reset_peak_memory_stats(device)
        doc_fp32 = retriever._encode_texts(docs)
        query_fp32 = retriever._encode_texts(query_texts)
        top_fp32 = _top3(torch, query_fp32, doc_fp32, document_ids)
        _sync(torch, device)
        evidence["fp32_top3"] = top_fp32

        comparisons = []
        for fp16_result, fp32_result in zip(top_fp16, top_fp32, strict=False):
            fp16_ids = fp16_result["ids"]
            fp32_ids = fp32_result["ids"]
            overlap = len(set(fp16_ids) & set(fp32_ids))
            comparisons.append(
                {
                    "query_id": QUERIES[fp16_result["query_index"]][0],
                    "same_order": fp16_ids == fp32_ids,
                    "top3_overlap": overlap,
                    "top3_jaccard": round(overlap / max(1, len(set(fp16_ids) | set(fp32_ids))), 4),
                }
            )
        evidence["top3_comparison"] = comparisons
        evidence["effective_batch_size"] = retriever._effective_batch_size
        peak_fp32 = torch.cuda.max_memory_allocated(device)
        evidence["peak_memory_mb"] = round(max(peak_fp16, peak_fp32) / (1024 * 1024), 3)
        evidence["precision_recommendation"] = (
            "fp16_top3_matches_fp32"
            if all(item["same_order"] for item in comparisons)
            else "review_fp32_top3_difference"
        )
        evidence["status"] = "completed"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    _write_json(output, evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行合成中文 CUDA 检索精度核验并保存 JSON 证据。")
    parser.add_argument("--output", required=True, type=Path, help="JSON 证据输出路径")
    parser.add_argument("--local-model-path", help="本地模型目录；未指定时只读取 HF 本地缓存")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    evidence = verify(args.output, args.local_model_path, max(1, args.batch_size))
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
