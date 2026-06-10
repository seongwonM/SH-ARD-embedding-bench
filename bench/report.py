"""
MTEB 결과 로컬 뷰어 — RunPod에서 받은 summary.json을 표로 출력.

usage:
  python -m bench.report /path/to/reports
  python -m bench.report /path/to/reports --metric mrr_at_10
  python -m bench.report /path/to/summary.json        # 직접 파일 지정도 가능
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


_METRICS = ["ndcg_at_10", "mrr_at_10", "recall_at_1", "recall_at_10"]
_METRIC_LABELS = {
    "ndcg_at_10":  "NDCG@10",
    "mrr_at_10":   "MRR@10",
    "recall_at_1": "R@1",
    "recall_at_10":"R@10",
}


def _load(path: Path) -> list[dict]:
    """summary.json 또는 디렉토리 내 summary.json 로드."""
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # 디렉토리면 summary.json 탐색
    for candidate in [
        path / "summary.json",
        *path.rglob("summary.json"),
    ]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)

    sys.exit(f"summary.json을 찾을 수 없음: {path}")


def _shorten(model_id: str, max_len: int = 30) -> str:
    name = model_id.split("/")[-1]
    return name[:max_len]


def _print_table(rows: list[dict], metric: str) -> None:
    # {task: {model_short: value}}
    by_task: dict[str, dict[str, float | None]] = defaultdict(dict)
    models: list[str] = []

    for row in rows:
        m = _shorten(row["model"])
        if m not in models:
            models.append(m)
        by_task[row["task"]][m] = row.get(metric)

    tasks = sorted(by_task)
    col = 14

    label = _METRIC_LABELS.get(metric, metric)
    header = f"{'Task':<32}" + "".join(f"{m:>{col}}" for m in models)
    sep = "-" * len(header)

    print(f"\n  [ {label} ]")
    print(sep)
    print(header)
    print(sep)

    for task in tasks:
        vals = by_task[task]
        row_str = f"{task:<32}"
        for m in models:
            v = vals.get(m)
            row_str += f"{'—':>{col}}" if v is None else f"{v:>{col}.4f}"
        print(row_str)

    print(sep)

    avg_row = f"{'AVERAGE':<32}"
    for m in models:
        vals_m = [by_task[t].get(m) for t in tasks if by_task[t].get(m) is not None]
        avg = sum(vals_m) / len(vals_m) if vals_m else None
        avg_row += f"{'—':>{col}}" if avg is None else f"{avg:>{col}.4f}"
    print(avg_row)
    print(sep)


def _print_raw(rows: list[dict]) -> None:
    """raw_scores가 있는 경우 태스크×모델별 전체 지표 출력."""
    has_raw = any(r.get("raw_scores") for r in rows)
    if not has_raw:
        return

    print("\n\n  [ 원본 전체 스코어 (raw_scores) ]")
    for row in rows:
        raw = row.get("raw_scores")
        if not raw:
            continue
        print(f"\n  ▶ {row['model']} / {row['task']}")
        for split, score_list in raw.items():
            print(f"    [{split}]")
            for entry in score_list if isinstance(score_list, list) else [score_list]:
                if isinstance(entry, dict):
                    for k, v in sorted(entry.items()):
                        if isinstance(v, (int, float)):
                            print(f"      {k}: {v:.4f}")
                        elif isinstance(v, dict):
                            print(f"      {k}:")
                            for kk, vv in sorted(v.items()):
                                if isinstance(vv, (int, float)):
                                    print(f"        {kk}: {vv:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MTEB 결과 뷰어")
    ap.add_argument("path", nargs="?", default="reports", help="결과 경로 (디렉토리 또는 summary.json)")
    ap.add_argument("--metric", choices=_METRICS, default=None, help="출력할 단일 지표")
    args = ap.parse_args()

    data = _load(Path(args.path))
    if not data:
        sys.exit("결과 데이터가 비어 있습니다.")

    models = sorted({r["model"] for r in data})
    tasks  = sorted({r["task"]  for r in data})
    print(f"\n모델 {len(models)}개 / 태스크 {len(tasks)}개")
    print(f"모델: {', '.join(models)}")

    metrics = [args.metric] if args.metric else _METRICS
    for m in metrics:
        _print_table(data, m)

    _print_raw(data)


if __name__ == "__main__":
    main()
