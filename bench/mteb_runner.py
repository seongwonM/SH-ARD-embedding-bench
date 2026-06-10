"""
MTEB 기반 한국어 임베딩 벤치마크 러너.

usage:
  # 단일 모델
  python -m bench.mteb_runner --model BAAI/bge-m3

  # 복수 모델 순차 실행 (RunPod 권장)
  python -m bench.mteb_runner --models BAAI/bge-m3 Qwen/Qwen3-0.6B Qwen/Qwen3-4B Qwen/Qwen3-8B

  # 태스크 타입 필터
  python -m bench.mteb_runner --models BAAI/bge-m3 --task-types Retrieval

  # 단일 태스크 테스트
  python -m bench.mteb_runner --model BAAI/bge-m3 --tasks Ko-StrategyQA --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time


_DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def _print_tasks(tasks) -> None:
    by_type: dict[str, list] = {}
    for t in tasks:
        tp = t.metadata.type
        by_type.setdefault(tp, []).append(t.metadata.name)
    for tp, names in sorted(by_type.items()):
        print(f"  [{tp}] {', '.join(names)}")


def _extract_metrics(res) -> tuple[dict, dict]:
    """
    반환: (key_metrics, raw_scores)
      key_metrics — NDCG@10·MRR@10·Recall@1/5/10 평균값 (보기 쉬운 요약)
      raw_scores  — res.scores 원본 전체 (split → list of score dicts)
    다중 언어 태스크는 key_metrics만 평균 처리, raw_scores는 있는 그대로 유지.
    """
    raw_scores = res.scores  # 원본 그대로

    if not res.scores:
        return {}, raw_scores

    flat: list[dict] = []
    for split_scores in res.scores.values():
        for score in split_scores:
            if not isinstance(score, dict):
                continue
            if "ndcg_at_10" in score:
                flat.append(score)
            else:
                for v in score.values():
                    if isinstance(v, dict) and "ndcg_at_10" in v:
                        flat.append(v)

    target = ["ndcg_at_10", "mrr_at_10", "recall_at_1", "recall_at_5", "recall_at_10", "map_at_10"]
    key_metrics: dict = {}
    for k in target:
        vals = [m[k] for m in flat if k in m and m[k] is not None]
        if vals:
            key_metrics[k] = round(sum(vals) / len(vals), 4)

    return key_metrics, raw_scores


_DTYPE_MAP = {"auto": "auto", "fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _load_task_data(task) -> tuple[dict, dict, dict]:
    """
    HF datasets에서 직접 corpus/queries/qrels를 로드해 dict로 반환.
    MTEB 1.34+에서 load_data()가 self.corpus를 설정하지 않으므로 직접 로드.
    """
    from datasets import load_dataset

    meta     = task.metadata
    path     = meta.dataset["path"]
    revision = meta.dataset.get("revision")
    split    = next((s for s in ("test", "dev") if s in meta.eval_splits), meta.eval_splits[0])

    hf_kw: dict = {"trust_remote_code": True}
    if revision:
        hf_kw["revision"] = revision

    corpus_ds  = load_dataset(path, "corpus",  split="corpus",  **hf_kw)
    queries_ds = load_dataset(path, "queries", split="queries", **hf_kw)
    qrels_ds   = load_dataset(path, "qrels",   split=split,     **hf_kw)

    corpus = {r["_id"]: {"title": r.get("title", ""), "text": r["text"]} for r in corpus_ds}
    queries = {r["_id"]: r["text"] for r in queries_ds}
    qrels: dict = {}
    for r in qrels_ds:
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r.get("score", 1))

    return corpus, queries, qrels


def _build_combined_retrieval_task(tasks: list):
    """
    여러 Retrieval 태스크의 corpus/queries/qrels를 합쳐서 단일 태스크 객체 반환.
    doc_id/query_id 충돌 방지를 위해 "<태스크명>__<원본id>" 형태로 prefix 부여.
    """
    import copy
    import dataclasses

    combined_corpus: dict = {}
    combined_queries: dict = {}
    combined_qrels: dict = {}

    for task in tasks:
        name = task.metadata.name
        print(f"  [로딩] {name}")
        corpus, queries, qrels = _load_task_data(task)

        prefix = name + "__"
        for doc_id, doc in corpus.items():
            combined_corpus[prefix + doc_id] = doc
        for q_id, q in queries.items():
            combined_queries[prefix + q_id] = q
        for q_id, rels in qrels.items():
            combined_qrels[prefix + q_id] = {
                prefix + did: score for did, score in rels.items()
            }

    print(
        f"  [합산] corpus {len(combined_corpus):,}건 · "
        f"queries {len(combined_queries):,}건 · "
        f"qrel pairs {sum(len(v) for v in combined_qrels.values()):,}건"
    )

    # 첫 번째 태스크를 template으로 shallow copy 후 데이터 교체
    combined = copy.copy(tasks[0])
    combined.corpus         = {"test": combined_corpus}
    combined.queries        = {"test": combined_queries}
    combined.relevant_docs  = {"test": combined_qrels}
    combined.data_loaded    = True
    combined.load_data      = lambda **kwargs: None

    try:
        combined.metadata = copy.copy(combined.metadata)
        combined.metadata.name = "CombinedKoreanRetrieval"
    except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
        pass

    return combined


def _run_model(model_id: str, tasks: list, out_dir: str, batch_size: int, model_dtype: str = "auto") -> list[dict]:
    """단일 모델 평가 실행. summary 리스트 반환."""
    import mteb
    import warnings

    print(f"\n{'='*64}")
    print(f"  모델: {model_id}  (dtype={model_dtype})")
    print(f"{'='*64}")

    try:
        model = mteb.get_model(model_id, model_kwargs={"torch_dtype": _DTYPE_MAP[model_dtype]})
    except TypeError:
        # 구버전 MTEB: model_kwargs 미지원
        model = mteb.get_model(model_id)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return []

    # 모든 태스크가 Retrieval이면 corpus를 합쳐서 단일 평가
    all_retrieval = all(t.metadata.type == "Retrieval" for t in tasks)
    if all_retrieval and len(tasks) > 1:
        print(f"\n[합산 모드] {len(tasks)}개 Retrieval 태스크 corpus 병합 중...")
        eval_tasks = [_build_combined_retrieval_task(tasks)]
    else:
        eval_tasks = tasks

    summary = []
    t0_model = time.time()

    for task in eval_tasks:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                evaluation = mteb.MTEB(tasks=[task])
            results = evaluation.run(
                model,
                output_folder=out_dir,
                encode_kwargs={"batch_size": batch_size},
            )
            for res in results:
                m, raw = _extract_metrics(res)
                elapsed = time.time() - t0
                print(
                    f"  {res.task_name}: "
                    f"NDCG@10={m.get('ndcg_at_10')}  "
                    f"MRR@10={m.get('mrr_at_10')}  "
                    f"Recall@1={m.get('recall_at_1')}  "
                    f"Recall@10={m.get('recall_at_10')}  "
                    f"({elapsed:.0f}s)"
                )
                summary.append({
                    "model":        model_id,
                    "task":         res.task_name,
                    "ndcg_at_10":  m.get("ndcg_at_10"),
                    "mrr_at_10":   m.get("mrr_at_10"),
                    "recall_at_1": m.get("recall_at_1"),
                    "recall_at_5": m.get("recall_at_5"),
                    "recall_at_10":m.get("recall_at_10"),
                    "map_at_10":   m.get("map_at_10"),
                    "raw_scores":  raw,
                })
        except Exception as e:
            print(f"  [SKIP] {task.metadata.name}: {e}")

    total = time.time() - t0_model
    print(f"\n  → {len(summary)}개 완료 ({total:.0f}s)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="MTEB 한국어 벤치마크")
    model_group = ap.add_mutually_exclusive_group()
    model_group.add_argument("--model",  help="단일 모델 HuggingFace ID")
    model_group.add_argument(
        "--models", nargs="+",
        help=f"복수 모델 순차 실행 (기본: {' '.join(_DEFAULT_MODELS)})",
    )
    ap.add_argument("--tasks",      nargs="*", help="특정 태스크 이름 목록")
    ap.add_argument("--task-types", nargs="*", help="태스크 타입 필터 (Retrieval 등)")
    ap.add_argument("--languages",  nargs="*", default=["kor"], help="언어 필터, 기본 kor")
    ap.add_argument("--out",        default="reports",
                    help="결과 저장 루트 경로 (RunPod: /workspace/reports  k8s: /reports)")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="encode 배치 크기 (GPU 기본: 256 / CPU 검증: 32)")
    ap.add_argument("--model-dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"],
                    help="모델 로드 dtype. auto=모델 기본값(GPU권장), fp32=CPU 안전")
    args = ap.parse_args()

    try:
        import mteb
    except ImportError:
        sys.exit("mteb 패키지가 필요합니다: pip install mteb")

    os.makedirs(args.out, exist_ok=True)

    # 모델 목록 결정
    if args.model:
        model_ids = [args.model]
    elif args.models:
        model_ids = args.models
    else:
        model_ids = _DEFAULT_MODELS

    # 태스크 목록 (모든 모델 공통)
    kwargs: dict = dict(
        languages=args.languages,
        modalities=["text"],
        exclusive_modality_filter=True,
    )
    if args.tasks:
        kwargs["tasks"] = args.tasks
    if args.task_types:
        kwargs["task_types"] = args.task_types
    tasks = mteb.get_tasks(**kwargs)

    print(f"[태스크] {len(tasks)}개:")
    _print_tasks(tasks)
    print(f"\n[모델] {len(model_ids)}개: {', '.join(model_ids)}")

    all_summary = []
    t0_total = time.time()

    for model_id in model_ids:
        results = _run_model(model_id, tasks, args.out, args.batch_size, args.model_dtype)
        all_summary.extend(results)

    # 통합 summary 저장 (numpy float64 / NaN → Python float 변환)
    def _json_default(obj):
        try:
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            return str(obj)

    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2, default=_json_default)

    total = time.time() - t0_total
    print(f"\n\n{'='*64}")
    print(f"  전체 완료: {len(all_summary)}개 결과  ({total:.0f}s)")
    print(f"  저장: {summary_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
