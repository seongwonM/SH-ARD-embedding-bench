# embedding_bench

임베딩 모델 × 벡터 DB **성능 테스트** 벤치마크.
대학원/연구실/논문에서 쓰는 표준 방식(BEIR 포맷, TREC 메트릭)을 차용하고,
각 구성요소는 공식 문서 권장 API로 구현했습니다.

> 부하테스트(동시성/QPS 스윕)는 제거했습니다. 먼저 **성능 테스트**(서빙 지연·처리량,
> 검색 정확도, 혼동행렬)에 집중합니다.

## 무엇을 비교하나

- **임베딩 모델**: Qwen3-Embedding-0.6B, BGE-M3 (CPU 8GB 노드 기준 가용 모델)
- **벡터 DB**: Qdrant, Milvus, Vespa — 동일 HNSW 조건으로 공정 비교
- **서빙**: **sentence-transformers 5.5.1**(2026-05 최신 안정판, Production/Stable).
  - Qwen3-0.6B / BGE-M3 모두 지원 (Qwen3 는 `transformers>=4.51` 필요)
  - 기본은 러너가 직접 로드(in-process). 모델 격리가 필요하면 `--remote` + k8s Pod.

## 입력 / 출력 (형태만 맞추면 동작)

**입력** = BEIR 표준 포맷 디렉터리 한 개. 내부 로직은 고정, 데이터셋만 교체합니다.
```
<data_dir>/
  corpus.jsonl     {"_id":"d1","title":"...","text":"..."}
  queries.jsonl    {"_id":"q1","text":"..."}
  qrels/<split>.tsv  query-id <TAB> corpus-id <TAB> score   (1행 헤더)
```
> Ko-miracl 은 `python -m data.prepare_ko_miracl` 로 위 형태로 변환합니다(아래 참고).
> 실데이터가 없으면 `--sample` 로 합성 데이터셋이 자동 생성되어 파이프라인이 돕니다.

**출력** = 성능 리포트 (`reports/*.json`, `*.csv`, 콘솔 요약):

| 범주 | 지표 |
|------|------|
| 서빙 (모델별) | 모델 로드 시간, 단일 텍스트 인코딩 p50/p95/p99, 배치별 처리량(texts/s) |
| 검색 정확도 (vs qrels) | nDCG@k, Recall@k, Precision@k, MAP, MRR — `pytrec_eval` |
| 혼동행렬 (top-k) | TP/FP/FN/TN, precision, recall, F1, accuracy |
| ANN 정확도 | recall@k (근사검색 vs 정확검색 brute-force) |
| 색인 | load_duration(적재+빌드 시간) |
| 지연 (단일 클라이언트) | 검색 / end-to-end p50·p95·p99 (ms) |

## 빠른 시작 (in-process, 기본)

```bash
# 0) 의존성 (torch CPU 휠 권장)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1) Ko-miracl 변환 (전체 corpus 1.49M)
python -m data.prepare_ko_miracl --out ./datasets/ko-miracl --split dev

# 2) 스모크 테스트 (합성 데이터)
python -m bench.runner --sample --models bge-m3 --dbs qdrant

# 3) 실제 벤치 (벡터 DB 가 클러스터/로컬에서 접근 가능해야 함)
python -m bench.runner --data ./datasets/ko-miracl --split dev \
       --models qwen3-0.6b bge-m3 --dbs qdrant milvus vespa
```

## MSA 서빙 (선택, 모델 격리)

모델을 Pod 단위로 격리해 서빙하려면 `serve/st_server.py`(FastAPI + SentenceTransformer)를
k8s 에 배포하고 `--remote` 로 붙입니다. 자세한 내용은 [`deploy/README.md`](deploy/README.md).

```bash
docker build -f deploy/Dockerfile.st -t <registry>/embedding-st:5.5.1 .
kubectl apply -f deploy/k8s/embedding-cpu.yaml
kubectl -n embedding port-forward svc/qwen3-0-6b 8000:8000 &
kubectl -n embedding port-forward svc/bge-m3     8003:8003 &
python -m bench.runner --remote --data ./datasets/ko-miracl-50k --split dev
```

## 구조

```
embedding_bench/
├── config.py              # 고정 내부 설정 (모델/DB 레지스트리, 성능 파라미터)
├── data/
│   ├── loader.py          # 입력: BEIR 로더 + 샘플 생성기
│   └── prepare_ko_miracl.py  # Ko-miracl → BeIR 디렉터리 변환
├── clients/               # 임베딩 클라이언트 (st_local: in-process, st_remote: HTTP)
├── vectordb/              # Qdrant / Milvus / Vespa 스토어 (공식 권장 API)
├── eval/metrics.py        # TREC 메트릭 + 혼동행렬 (pytrec_eval + 순수파이썬 fallback)
├── bench/
│   ├── timing.py          # 단일 클라이언트 지연/처리량 측정 (부하테스트 제거됨)
│   └── runner.py          # 오케스트레이션 + 리포트 (진입점)
├── serve/st_server.py     # MSA 서빙용 FastAPI 서버 (sentence-transformers)
└── deploy/                # Dockerfile.st + k8s 매니페스트 (CPU 모델 Pod)
```

## 설계 원칙

1. **공정 비교**: 세 벡터 DB 모두 동일 HNSW(M, efConstruction, efSearch)·동일
   거리 메트릭으로 색인. 파라미터는 `config.BENCH`에 고정.
2. **표준 방식**: 임베딩은 sentence-transformers 공식 사용법(Qwen3 쿼리 프롬프트 등),
   평가는 BEIR 포맷 + TREC(`pytrec_eval`) — 논문 재현성.
3. **혼동행렬**: 검색을 (쿼리, 문서) 이진 분류로 환원해 TP/FP/FN/TN 집계
   (corpus 전체가 후보 → TN 지배적이므로 precision/recall/F1 위주 해석).

## 참고 문헌 / 공식 문서

- sentence-transformers 5.5.1: <https://sbert.net/>, <https://pypi.org/project/sentence-transformers/>
- Qwen3-Embedding: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- BEIR: Thakur et al., *A Heterogeneous Benchmark for Zero-shot IR*, NeurIPS 2021
- pytrec_eval: Van Gysel & de Rijke, SIGIR 2018
- Qdrant / pymilvus / pyvespa 공식 문서 (각 스토어 docstring에 링크)
