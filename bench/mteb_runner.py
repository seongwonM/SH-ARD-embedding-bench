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

# 한국어 corpus + 한국어 쿼리 태스크
_KO_RETRIEVAL_TASKS = [
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "LawIRKo",
    "SQuADKorV1Retrieval",
    "PublicHealthQA",
    "MIRACLRetrieval",   # 한국어 Wikipedia corpus (1.5M docs) + 한국어 쿼리
]

_DTYPE_MAP = {"auto": "auto", "fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────────────────────

def _load_task_data(task) -> tuple[dict, dict, dict]:
    """
    MTEB 2.15 새 포맷 / 구 포맷 모두 지원.

    새 포맷 (대부분의 태스크):
      task.dataset[config][split] = {
        'corpus':        HF Dataset (id, text, title)
        'queries':       HF Dataset (id, text)
        'relevant_docs': dict {qid: {did: score}}
      }

    구 포맷 (PublicHealthQA):
      task.corpus[lang][split]  = {id: {text, title, ...}}
      task.queries[lang][split] = {id: text}
      task.relevant_docs[lang][split] = {qid: {did: score}}

    반환: corpus={id: {title, text}}, queries={id: text}, qrels={qid: {did: score}}
    """
    split = task.metadata.eval_splits[0]

    # data_loaded 강제 리셋 (MTEB 2.15 에서 초기값이 True인 경우 대비)
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

    # ── 새 포맷 ──────────────────────────────────────────────────────────────
    ds = getattr(task, "dataset", None)
    if ds is not None and hasattr(ds, "keys"):
        config   = list(ds.keys())[0]
        cfg_data = ds[config]
        inner    = cfg_data.get(split, {}) if hasattr(cfg_data, "get") else cfg_data[split]

        if isinstance(inner, dict) and "corpus" in inner:
            corpus_ds  = inner["corpus"]
            queries_ds = inner["queries"]
            qrels_raw  = inner.get("relevant_docs", {})

            corpus  = {r["id"]: {"title": r.get("title", ""), "text": r["text"]}
                       for r in corpus_ds}
            queries = {r["id"]: r["text"] for r in queries_ds}
            qrels   = dict(qrels_raw) if qrels_raw else {}
            return corpus, queries, qrels

    # ── 구 포맷 ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Combined corpus 구축
# ──────────────────────────────────────────────────────────────────────────────

def _build_combined_corpus(tasks: list) -> tuple[dict, dict, dict]:
    """
    모든 태스크의 corpus/queries/qrels를 '<태스크명>__' 접두어로 합산.
    반환: combined_corpus, combined_queries, combined_qrels
    """
    combined_corpus:  dict = {}
    combined_queries: dict = {}
    combined_qrels:   dict = {}

    for task in tasks:
        name   = task.metadata.name
        prefix = name + "__"
        print(f"  [로딩] {name}")
        corpus, queries, qrels = _load_task_data(task)

        for did, doc in corpus.items():
            combined_corpus[prefix + did] = doc
        for qid, text in queries.items():
            combined_queries[prefix + qid] = text
        for qid, rels in qrels.items():
            combined_qrels[prefix + qid] = {
                prefix + did: score for did, score in rels.items()
            }

    print(
        f"  [합산] corpus {len(combined_corpus):,}건 · "
        f"queries {len(combined_queries):,}건 · "
        f"qrel pairs {sum(len(v) for v in combined_qrels.values()):,}건"
    )
    return combined_corpus, combined_queries, combined_qrels


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval 평가
# ──────────────────────────────────────────────────────────────────────────────

def _encode(model, texts: list[str], batch_size: int, show_progress: bool = True) -> np.ndarray:
    """
    모델 encode 후 L2 정규화.
    MTEB 2.15 SentenceTransformerEncoderWrapper (task_metadata/hf_split/hf_subset 필수)
    와 일반 SentenceTransformer 모두 지원.
    """
    kw = {"batch_size": batch_size, "show_progress_bar": show_progress}
    # MTEB 2.15 SentenceTransformerEncoderWrapper는 task_metadata 필수라 우회.
    # .model 속성으로 내부 SentenceTransformer에 직접 위임.
    inner = getattr(model, "model", model)
    embs = inner.encode(texts, **kw)
    embs = np.array(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


def _evaluate_retrieval(
    model,
    corpus:  dict,
    queries: dict,
    qrels:   dict,
    batch_size: int,
    top_k: int = 100,
) -> dict:
    """
    corpus 전체 인코딩 → query별 top-k cosine similarity → pytrec_eval 지표 계산.
    반환: {ndcg_at_10, mrr_at_10, recall_at_1, recall_at_5, recall_at_10, map_at_10}
    """
    import pytrec_eval

    # corpus 인코딩
    corp_ids   = list(corpus.keys())
    corp_texts = [f"{corpus[d].get('title', '')} {corpus[d]['text']}".strip()
                  for d in corp_ids]
    print(f"  corpus 인코딩 ({len(corp_ids):,}건)...")
    corp_embs = _encode(model, corp_texts, batch_size, show_progress=True)

    # query 인코딩
    q_ids   = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    print(f"  query 인코딩 ({len(q_ids):,}건)...")
    q_embs = _encode(model, q_texts, batch_size, show_progress=False)

    # top-k 검색 (query chunk 단위로 처리해 메모리 절약)
    print(f"  top-{top_k} 검색 ({len(q_ids):,} queries × {len(corp_ids):,} docs)...")
    chunk = 256
    run: dict = {}
    for start in range(0, len(q_ids), chunk):
        end    = min(start + chunk, len(q_ids))
        scores = np.dot(q_embs[start:end], corp_embs.T)              # (chunk, N_corp)
        top_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        for i, qid in enumerate(q_ids[start:end]):
            run[qid] = {corp_ids[j]: float(scores[i, j]) for j in top_idx[i]}

    # score>=1 인 항목만 relevant 로 처리 (MIRACL은 0점 hard negative 포함)
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
    tasks: list,
    out_dir: str,
    batch_size: int,
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

    print(f"\n[corpus 병합] {len(tasks)}개 태스크...")
    t0_merge = time.time()
    combined_corpus, combined_queries, combined_qrels = _build_combined_corpus(tasks)
    print(f"  병합 완료 ({time.time() - t0_merge:.0f}s)")

    t0_eval = time.time()
    metrics = _evaluate_retrieval(
        model, combined_corpus, combined_queries, combined_qrels, batch_size
    )
    elapsed = time.time() - t0_eval

    print(
        f"\n  NDCG@10={metrics.get('ndcg_at_10')}  "
        f"MRR@10={metrics.get('mrr_at_10')}  "
        f"Recall@10={metrics.get('recall_at_10')}  "
        f"({elapsed:.0f}s)"
    )

    return {
        "model":  model_id,
        "task":   "CombinedKoreanRetrieval",
        "tasks":  [t.metadata.name for t in tasks],
        **metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="MTEB 한국어 벤치마크 (combined corpus)")
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument("--model",  help="단일 모델 HuggingFace ID")
    model_group.add_argument(
        "--models", nargs="+",
        help=f"복수 모델 순차 실행 (기본: {' '.join(_DEFAULT_MODELS)})",
    )
    ap.add_argument(
        "--tasks", nargs="*", default=_KO_RETRIEVAL_TASKS,
        help="평가 태스크 목록 (기본: 한국어 6개)",
    )
    ap.add_argument("--out",          default="reports",
                    help="결과 저장 루트 경로")
    ap.add_argument("--batch-size",   type=int, default=256,
                    help="encode 배치 크기 (GPU: 256, CPU: 32)")
    ap.add_argument("--model-dtype",  default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
    args = ap.parse_args()

    try:
        import mteb
    except ImportError:
        sys.exit("mteb 패키지가 필요합니다: pip install mteb")

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

    all_results = []
    t0_total = time.time()

    for model_id in model_ids:
        result = _run_model(model_id, tasks, args.out, args.batch_size, args.model_dtype)
        if result:
            all_results.append(result)

    def _json_default(obj):
        try:
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return str(obj)

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
