# 7단계 평가: 질의-문서 쌍(eval.jsonl)으로 검색 정확도를 잰다. 학습 전후 모두 이 파일로 잰다.
# 원문 README의 eval_ir.py에 장치 자동 선택(맥 mps / 보드 cuda), 차원별(768/512/256/128) 결과, 결과 저장(--out)을 더했다.
# eval.jsonl은 재기만 한다. 이 결과를 보고 학습 설정을 고르지 않는다(팀 규칙 8번).
import argparse
import json
import time

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

DIMS = [768, 512, 256, 128]


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_evaluator(path="data/eval.jsonl", name="classroom", truncate_dim=None, extra=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    corpus, queries, relevant, doc_id = {}, {}, {}, {}
    # extra: 정답은 아니지만 검색 대상에 섞을 문서 파일 (7단계 3차: 연습 시험을 어렵게)
    for l in (open(extra, encoding="utf-8") if extra else []):
        if l.strip():
            d = json.loads(l)["positive"]
            corpus[doc_id.setdefault(d, f"x{len(doc_id)}")] = d
    for i, r in enumerate(rows):
        d = doc_id.setdefault(r["positive"], f"d{len(doc_id)}")
        corpus[d] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        relevant.setdefault(f"q{i}", set()).add(d)
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name,
        query_prompt_name="SearchQuery", corpus_prompt_name="Document",
        accuracy_at_k=[1, 3, 5], truncate_dim=truncate_dim)


def load_model(path):
    return SentenceTransformer(path, device=device(),
        model_kwargs={"torch_dtype": torch.bfloat16},   # float16 금지
        config_kwargs={"vision_config": None, "audio_config": None})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="google/embeddinggemma-2")
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--extra", help="오답 후보로 섞을 문서 파일 (예: data/train2.jsonl)")
    ap.add_argument("--out", help="결과를 JSON으로 저장할 경로 (예: runs/eval-base.json)")
    a = ap.parse_args()

    model = load_model(a.model)
    print(f"모델 {a.model}  장치 {model.device}")
    result = {"model": a.model, "data": a.data, "extra": a.extra, "dims": {}}
    for dim in DIMS:
        t = time.perf_counter()
        res = build_evaluator(a.data, truncate_dim=None if dim == 768 else dim, extra=a.extra)(model)
        row = {k: round(res[f"classroom_cosine_{k}"], 3)
               for k in ["accuracy@1", "accuracy@3", "accuracy@5", "ndcg@10"]}
        result["dims"][dim] = row
        print(f"[dim={dim}] " + "  ".join(f"{k} {v}" for k, v in row.items())
              + f"  ({time.perf_counter() - t:.1f} s)")
    if a.out:
        json.dump(result, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("저장:", a.out)
