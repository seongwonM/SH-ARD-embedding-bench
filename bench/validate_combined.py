"""
Combined corpus 파이프라인 end-to-end 검증 (k8s CPU 용).

검증 내용:
  0. 환경 확인     — MTEB 버전, pytrec_eval 임포트
  1. 데이터 로딩   — 새 포맷(AutoRAGRetrieval) + 구 포맷(PublicHealthQA) 모두 확인
  2. corpus 합산   — _build_combined_corpus() prefix·건수 검증
  3. 평가 실행     — _evaluate_retrieval() 로 ndcg_at_10 계산 확인

MIRACL(1.5M docs)은 CPU 시간 초과 방지를 위해 제외.
사용 모델: paraphrase-multilingual-MiniLM-L12-v2 (CPU 가능)
"""
from __future__ import annotations
import sys
import warnings
import mteb

from bench.mteb_runner import (
    _load_task_data,
    _build_combined_corpus,
    _evaluate_retrieval,
)

TASKS = ["AutoRAGRetrieval", "PublicHealthQA"]
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT   = "/reports/validate"

_pass = 0
_fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if cond:
        _pass += 1
    else:
        _fail += 1
        sys.exit(1)


def main() -> None:
    print("\n=== validate_combined ===\n")

    # ── Step 0: 환경 확인 ────────────────────────────────────────────────────
    print("[Step 0] 환경 확인")
    mteb_ver = getattr(mteb, "__version__", "unknown")
    print(f"  MTEB 버전: {mteb_ver}")

    try:
        import pytrec_eval  # noqa: F401
        check("pytrec_eval 임포트", True)
    except ImportError as e:
        check("pytrec_eval 임포트", False, str(e))

    # ── Step 1: 데이터 로딩 ──────────────────────────────────────────────────
    print(f"\n[Step 1] 데이터 로딩: {TASKS}")
    tasks = mteb.get_tasks(
        tasks=TASKS,
        languages=["kor"],
        modalities=["text"],
        exclusive_modality_filter=True,
    )
    check("태스크 로드", len(tasks) == len(TASKS),
          f"기대 {len(TASKS)}개, 실제 {len(tasks)}개: {[t.metadata.name for t in tasks]}")

    for task in tasks:
        name = task.metadata.name
        print(f"\n  [{name}] eval_splits={task.metadata.eval_splits}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corpus, queries, qrels = _load_task_data(task)

        check(f"{name} corpus 로드",  len(corpus)  > 0, f"{len(corpus):,}건")
        check(f"{name} queries 로드", len(queries) > 0, f"{len(queries):,}건")
        check(f"{name} qrels 로드",   len(qrels)   > 0, f"{len(qrels):,}건")

        # corpus 문서 포맷: {id: {title, text}}
        sample_doc = next(iter(corpus.values()))
        check(f"{name} doc에 'text' 키", "text" in sample_doc, str(sample_doc)[:80])

        # qrel doc_id → corpus에 실제로 존재하는지
        sample_qid = next(iter(qrels))
        sample_did = next(iter(qrels[sample_qid]))
        check(f"{name} qrel doc_id → corpus 존재",
              sample_did in corpus, f"qrel_did={sample_did!r}")

    # ── Step 2: corpus 합산 ──────────────────────────────────────────────────
    print(f"\n[Step 2] _build_combined_corpus()")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_corpus, combined_queries, combined_qrels = _build_combined_corpus(tasks)

    # 태스크별 원본 크기 합산과 일치 확인
    total_corpus  = sum(
        len(_load_task_data(t)[0]) for t in tasks
    )
    total_queries = sum(
        len(_load_task_data(t)[1]) for t in tasks
    )

    check("corpus 건수 일치",  len(combined_corpus)  == total_corpus,
          f"combined={len(combined_corpus):,} / 합계={total_corpus:,}")
    check("queries 건수 일치", len(combined_queries) == total_queries,
          f"combined={len(combined_queries):,} / 합계={total_queries:,}")
    check("corpus key에 '__' prefix",
          any("__" in k for k in combined_corpus),
          next(iter(combined_corpus)))
    check("combined qrel doc_id → combined corpus 존재", all(
        all(did in combined_corpus for did in rels)
        for rels in combined_qrels.values()
    ))

    # ── Step 3: 평가 실행 ────────────────────────────────────────────────────
    print(f"\n[Step 3] _evaluate_retrieval() — {MODEL}")
    import os
    os.makedirs(OUT, exist_ok=True)

    model = mteb.get_model(MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = _evaluate_retrieval(
            model, combined_corpus, combined_queries, combined_qrels,
            batch_size=32,
        )

    check("ndcg_at_10 존재",   "ndcg_at_10" in metrics, str(metrics))
    ndcg = metrics["ndcg_at_10"]
    check("ndcg_at_10 유효 범위 [0,1]",
          isinstance(ndcg, float) and 0.0 <= ndcg <= 1.0, f"{ndcg:.4f}")

    print(f"\n{'='*56}")
    print(f"  전체 PASS ({_pass}개) — ndcg_at_10={ndcg:.4f}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
