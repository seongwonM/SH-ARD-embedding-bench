# 배포

## 구성

MTEB 라이브러리로 모델을 in-process 로드해 한국어 Retrieval 태스크를 평가합니다.  
모델 서빙 Pod 없음 — 러너 하나가 모델 로드 → 평가 → 결과 저장까지 담당합니다.

| 환경 | Dockerfile | 용도 |
|------|-----------|------|
| **RunPod (GPU)** | `Dockerfile.runpod` | 실제 벤치마크 (4개 모델 전체) |
| k8s / AKS (CPU) | `Dockerfile.mteb` | 파이프라인 검증용 (소형 모델 1개) |

## RunPod GPU 실행 (메인)

### 1. 이미지 빌드 & 푸시

빌드 컨텍스트는 프로젝트 루트(`embedding_bench/` 상위)에서 실행:

```bash
# embedding_bench/ 의 상위 디렉터리에서 실행
docker build -f embedding_bench/deploy/Dockerfile.runpod \
             -t <your-registry>/embedding-bench-gpu:latest \
             embedding_bench/
docker push <your-registry>/embedding-bench-gpu:latest
```

### 2. RunPod 설정

| 항목 | 값 |
|------|-----|
| Container Image | `<your-registry>/embedding-bench-gpu:latest` |
| GPU | A100 40G 이상 권장 (8B 모델 fp16: ~16GB VRAM 필요) |
| Volume Mount | `/workspace` — 모델 캐시·결과 모두 이곳에 저장됨 |
| Container Disk | 20GB 이상 (모델 캐시용) |

**환경변수 (선택):**

| 변수 | 설명 |
|------|------|
| `HF_TOKEN` | 비공개 모델 접근 시 (기본 모델 4개는 불필요) |

### 3. 결과 확인

벤치 완료 후 `/workspace/reports/` 에 저장됩니다:
- `summary.json` — 전체 모델×태스크 요약 (report.py 입력)
- `<model-name>/<hash>/<task>.json` — MTEB 원본 결과

로컬로 복사:
```bash
# RunPod SSH 접속 후
rsync -av <runpod-ssh>:/workspace/reports ./reports-local
# 또는 RunPod 웹 콘솔 → File Browser
```

로컬 리포트 뷰어:
```bash
python -m bench.report ./reports-local
python -m bench.report ./reports-local --metric ndcg_at_10
```

### 커스텀 실행 (CMD 오버라이드)

```bash
# 단일 모델만
python -m bench.mteb_runner --model BAAI/bge-m3 --out /workspace/reports

# 특정 태스크 타입만
python -m bench.mteb_runner --task-types Retrieval --out /workspace/reports

# dtype 명시 (기본 auto)
python -m bench.mteb_runner --model-dtype bf16 --out /workspace/reports
```

## 기본 모델 4개

| 모델 | VRAM (bf16) | HuggingFace |
|------|------------|-------------|
| BAAI/bge-m3 | ~2.3 GB | dense retrieval |
| Qwen/Qwen3-0.6B | ~1.2 GB | 소형 |
| Qwen/Qwen3-4B | ~8 GB | 중형 |
| Qwen/Qwen3-8B | ~16 GB | 대형 |

A100 40G 기준 4개 순차 실행 가능 (동시 실행 아님).

---

## k8s CPU 파이프라인 검증 (보조)

코드 변경 후 파이프라인 이상 없는지 확인할 때 사용합니다.  
소형 모델(`paraphrase-multilingual-MiniLM-L12-v2`)로 태스크가 정상 실행되는지만 확인합니다.

```bash
# ACR 빌드 (로컬 Docker 없을 때)
az acr build \
  --registry abspxdevcr \
  --image embedding-mteb:latest \
  --file embedding_bench/deploy/Dockerfile.mteb \
  embedding_bench/

# k8s 배포
kubectl apply -f deploy/k8s/mteb-bench-runner.yaml

# 로그 확인
kubectl -n embedding logs -l app=mteb-bench --follow

# 결과 복사
kubectl -n embedding exec <pod> -- find /reports -type f
```

k8s YAML(`mteb-bench-runner.yaml`)의 `command:` 에서 `--model` 과 `--batch-size 32` 를 조정해 테스트하세요.
