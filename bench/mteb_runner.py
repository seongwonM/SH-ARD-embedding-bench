"""
MTEB 기반 한국어 임베딩 벤치마크 러너.

한국어 corpus + 한국어 쿼리 태스크 6개의 문서/쿼리를 전부 합쳐
단일 combined corpus 에서 retrieval 평가를 수행한다.

usage:
  # 단일 모델
  python -m bench.mteb_runner --model BAAI/bge-m3

  # 복수 모델 순차 실행 (RunPod 권장)
  python -m bench.mteb_runner --models BAAI/bge-m3 Qwen/Qwen3-0.6B

  # 특정 태스크만 (MIRACL 제외 등)
  python -m bench.mteb_runner --model BAAI/bge-m3 --tasks AutoRAGRetrieval Ko-StrategyQA
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings

import numpy as np

_DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

_KO_RETRIEVAL_TASKS = [
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "LawIRKo",
    "SQuADKorV1Retrieval",
    "PublicHealthQA",
    "MIRACLRetrieval",
]

_DTYPE_MAP = {"auto": "auto", "fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


# ──────────────────────────────────────────────────────────────────────────────
# 메모리 리포트 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _mem(label: str = "") -> None:
    parts = []
    try:
        import psutil
        parts.append(f"CPU={psutil.Process().memory_info().rss / 1e9:.1f}GB")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    except ImportError:
        pass
    if parts:
        tag = f"[MEM:{label}] " if label else "[MEM] "
        print(tag + " ".join(parts), flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────────────────────

def _load_task_data(task) -> tuple[dict, dict, dict]:
    """
    MTEB 2.15 새 포맷 / 구 포맷 모두 지원.
    반환: corpus={id: {title, text}}, queries={id: text}, qrels={qid: {did: score}}
    """
    split = task.metadata.eval_splits[0]

    try:
        task._data_loaded = False
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            task.load_data(eval_splits=[split])
        except TypeError:
            task.load_data()

    # ── 새 포맷 (MTEB 2.15+) ─────────────────────────────────────────────────
    ds = getattr(task, "dataset", None)
    if ds is not None and hasattr(ds, "keys"):
        config   = list(ds.keys())[0]
        cfg_data = ds[config]
        inner    = cfg_data.get(split, {}) if hasattr(cfg_data, "get") else cfg_data[split]
        if isinstance(inner, dict) and "corpus" in inner:
            corpus  = {r["id"]: {"title": r.get("title", ""), "text": r["text"]}
                       for r in inner["corpus"]}
            queries = {r["id"]: r["text"] for r in inner["queries"]}
            qrels   = dict(inner.get("relevant_docs", {}))
            return corpus, queries, qrels

    # ── 구 포맷 (BEIR — PublicHealthQA) ──────────────────────────────────────
    corpus_attr = getattr(task, "corpus", None)
    if corpus_attr:
        lang_key  = list(corpus_attr.keys())[0]
        split_map = corpus_attr[lang_key]
        split_key = split if split in split_map else list(split_map.keys())[0]

        corpus_raw = split_map[split_key]
        corpus = (corpus_raw if isinstance(corpus_raw, dict)
                  else {r["_id"]: {"title": r.get("title", ""), "text": r["text"]}
                        for r in corpus_raw})

        queries_raw = (getattr(task, "queries", {}) or {}).get(lang_key, {}).get(split_key, {})
        queries = (queries_raw if isinstance(queries_raw, dict)
                   else {r["_id"]: r["text"] for r in queries_raw})

        qrels = ((getattr(task, "relevant_docs", {}) or {})
                 .get(lang_key, {}).get(split_key, {})) or {}

        return corpus, queries, qrels

    raise ValueError(f"{task.metadata.name}: 데이터 로드 실패")


def _build_combined_corpus(tasks: list) -> tuple[dict, dict, dict, list[str]]:
    """
    모든 태스크의 corpus/queries/qrels를 '<태스크명>__' 접두어로 합산.
    반환: combined_corpus, combined_queries, combined_qrels, task_names
    """
    combined_corpus:  dict = {}
    combined_queries: dict = {}
    combined_qrels:   dict = {}

    for task in tasks:
        name   = task.metadata.name
        prefix = name + "__"
        corpus, queries, qrels = _load_task_data(task)
        print(f"  [로딩] {name}: corpus={len(corpus):,}  queries={len(queries):,}", flush=True)

        for did, doc in corpus.items():
            combined_corpus[prefix + did] = doc
        for qid, text in queries.items():
            combined_queries[prefix + qid] = text
        for qid, rels in qrels.items():
            combined_qrels[prefix + qid] = {prefix + did: score for did, score in rels.items()}

    print(
        f"  [합산] corpus {len(combined_corpus):,}건 · "
        f"queries {len(combined_queries):,}건",
        flush=True,
    )
    return combined_corpus, combined_queries, combined_qrels, [t.metadata.name for t in tasks]


# ──────────────────────────────────────────────────────────────────────────────
# 인코딩
# ──────────────────────────────────────────────────────────────────────────────

_ENCODE_CHUNK = 50_000  # sentence-transformers 선(先)할당 OOM 방지용 청크 크기


def _encode(model, texts: list[str], batch_size: int, show_progress: bool = True) -> np.ndarray:
    """
    모델 encode 후 L2 정규화.
    texts 수가 _ENCODE_CHUNK 초과 시 청크 단위로 나눠 인코딩.
    청크 간 cuda.empty_cache() 호출로 GPU reserved pool 조각화 방지.
    """
    inner = getattr(model, "model", model)
    kw = {"batch_size": batch_size, "show_progress_bar": show_progress}

    if len(texts) <= _ENCODE_CHUNK:
        embs = np.array(inner.encode(texts, **kw))
    else:
        import torch as _torch
        chunks = []
        n_chunks = math.ceil(len(texts) / _ENCODE_CHUNK)
        _log_every = max(1, n_chunks // 4)
        for ci in range(n_chunks):
            s, e = ci * _ENCODE_CHUNK, min((ci + 1) * _ENCODE_CHUNK, len(texts))
            chunks.append(np.array(inner.encode(texts[s:e], **kw)))
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            if (ci + 1) % _log_every == 0 or ci + 1 == n_chunks:
                print(f"    encode {e:,}/{len(texts):,}", flush=True)
        embs = np.concatenate(chunks, axis=0)
        del chunks
        gc.collect()

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval 평가
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_retrieval(
    model,
    corpus:  dict,
    queries: dict,
    qrels:   dict,
    batch_size: int,
    corpus_batch_size: int | None = None,
    top_k: int = 100,
) -> dict:
    """
    corpus 전체 인코딩 → query별 top-k cosine similarity → pytrec_eval 지표 계산.
    반환: {ndcg_at_10, mrr_at_10, recall_at_1, recall_at_5, recall_at_10, map_at_10}
    """
    import pytrec_eval

    corp_bs = corpus_batch_size if corpus_batch_size is not None else batch_size

    corp_ids   = list(corpus.keys())
    corp_texts = [f"{corpus[d].get('title', '')} {corpus[d]['text']}".strip() for d in corp_ids]
    print(f"  corpus 인코딩 ({len(corp_ids):,}건)...", flush=True)
    corp_embs = _encode(model, corp_texts, corp_bs, show_progress=True)
    del corp_texts
    gc.collect()
    _mem("corpus")

    q_ids   = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    print(f"  query 인코딩 ({len(q_ids):,}건)...", flush=True)
    q_embs = _encode(model, q_texts, batch_size, show_progress=False)
    del q_texts
    gc.collect()

    print(f"  top-{top_k} 검색...", flush=True)
    run: dict = {}
    for start in range(0, len(q_ids), 256):
        end    = min(start + 256, len(q_ids))
        scores = np.dot(q_embs[start:end], corp_embs.T)
        top_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        for i, qid in enumerate(q_ids[start:end]):
            run[qid] = {corp_ids[j]: float(scores[i, j]) for j in top_idx[i]}
        del scores

    del corp_embs, q_embs
    gc.collect()

    binary_qrels = {
        qid: {did: 1 for did, s in rels.items() if s >= 1}
        for qid, rels in qrels.items()
        if any(s >= 1 for s in rels.values())
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        binary_qrels,
        {"ndcg_cut.10", "recip_rank", "recall.1", "recall.5", "recall.10", "map_cut.10"},
    )
    per_query = evaluator.evaluate(run)
    del run, binary_qrels
    gc.collect()

    def _avg(key: str) -> float | None:
        vals = [v.get(key, 0.0) for v in per_query.values()]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "ndcg_at_10":   _avg("ndcg_cut_10"),
        "mrr_at_10":    _avg("recip_rank"),
        "recall_at_1":  _avg("recall_1"),
        "recall_at_5":  _avg("recall_5"),
        "recall_at_10": _avg("recall_10"),
        "map_at_10":    _avg("map_cut_10"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 모델별 실행
# ──────────────────────────────────────────────────────────────────────────────

def _run_model(
    model_id: str,
    combined_corpus:  dict,
    combined_queries: dict,
    combined_qrels:   dict,
    task_names: list[str],
    batch_size: int,
    corpus_batch_size: int | None = None,
    model_dtype: str = "auto",
) -> dict:
    import mteb

    print(f"\n{'='*64}")
    print(f"  모델: {model_id}  (dtype={model_dtype})")
    print(f"{'='*64}")

    try:
        model = mteb.get_model(model_id, model_kwargs={"torch_dtype": _DTYPE_MAP[model_dtype]})
    except TypeError:
        model = mteb.get_model(model_id)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return {}
    _mem("로드")

    t0 = time.time()
    metrics = _evaluate_retrieval(
        model, combined_corpus, combined_queries, combined_qrels,
        batch_size, corpus_batch_size,
    )
    elapsed = time.time() - t0

    print(
        f"\n  NDCG@10={metrics.get('ndcg_at_10')}  "
        f"MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  "
        f"({elapsed:.0f}s)"
    )

    return {
        "model":  model_id,
        "task":   "CombinedKoreanRetrieval",
        "tasks":  task_names,
        **metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="MTEB 한국어 벤치마크 (combined corpus)")
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument("--model",  help="단일 모델 HuggingFace ID")
    model_group.add_argument("--models", nargs="+",
                             help=f"복수 모델 순차 실행 (기본: {' '.join(_DEFAULT_MODELS)})")
    ap.add_argument("--tasks", nargs="*", default=_KO_RETRIEVAL_TASKS,
                    help="평가 태스크 목록 (기본: 한국어 6개)")
    ap.add_argument("--out",               default="reports", help="결과 저장 루트 경로")
    ap.add_argument("--batch-size",        type=int, default=32,
                    help="query encode 배치 크기 (GPU: 32~64, CPU: 16)")
    ap.add_argument("--corpus-batch-size", type=int, default=None,
                    help="corpus encode 배치 크기 (미지정 시 --batch-size 값 사용)")
    ap.add_argument("--model-dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    args = ap.parse_args()

    try:
        import mteb
    except ImportError:
        sys.exit("mteb 패키지가 필요합니다: pip install mteb")

    import torch

    os.makedirs(args.out, exist_ok=True)
    model_ids = [args.model] if args.model else (args.models or _DEFAULT_MODELS)

    tasks = mteb.get_tasks(
        tasks=args.tasks,
        languages=["kor"],
        modalities=["text"],
        exclusive_modality_filter=True,
    )
    print(f"[태스크] {len(tasks)}개: {[t.metadata.name for t in tasks]}")
    print(f"[모델]   {len(model_ids)}개: {', '.join(model_ids)}")

    # corpus는 모든 모델이 공유 → 1회만 로딩
    print(f"\n[corpus 병합] {len(tasks)}개 태스크 (1회 로딩)...", flush=True)
    t0_merge = time.time()
    combined_corpus, combined_queries, combined_qrels, task_names = _build_combined_corpus(tasks)
    print(f"  병합 완료 ({time.time() - t0_merge:.0f}s)", flush=True)

    def _json_default(obj):
        try:
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return str(obj)

    all_results = []
    t0_total = time.time()

    for model_id in model_ids:
        # 체크포인트: 이미 완료된 모델은 스킵
        ckpt_path = os.path.join(args.out, model_id.replace("/", "_") + ".json")
        if os.path.exists(ckpt_path):
            print(f"\n[스킵] {model_id} — 결과 이미 존재: {ckpt_path}", flush=True)
            with open(ckpt_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        result = _run_model(
            model_id,
            combined_corpus, combined_queries, combined_qrels, task_names,
            args.batch_size, args.corpus_batch_size, args.model_dtype,
        )
        if result:
            all_results.append(result)
            # 모델 완료 즉시 체크포인트 저장
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"  [체크포인트] {ckpt_path}", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_default)

    total = time.time() - t0_total
    print(f"\n{'='*64}")
    print(f"  전체 완료: {len(all_results)}개 모델  ({total:.0f}s)")
    print(f"  저장: {summary_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
