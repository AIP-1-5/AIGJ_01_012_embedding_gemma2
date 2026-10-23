# 7단계 미세조정: 강의실 질의-문서 쌍(train.jsonl)으로 텍스트 모델(270M)을 추가 학습한다. 맥에서만 돌린다(보드 학습 금지).
# 원문 README의 train_ft.py 설정(epoch 3, 배치 8x2, lr 2e-5, MatryoshkaLoss)을 그대로 쓰고,
# 장치 자동 선택(맥 mps), 끊겼을 때 이어서 하기(epoch마다 체크포인트), 학습 시간 기록만 더했다.
# epoch마다 찍히는 eval 점수는 지켜보기만 한다. 그 점수로 epoch·lr을 고르지 않고, 마지막 epoch 모델을 쓴다.
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from eval_ir import build_evaluator, device

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--eval", default="data/eval.jsonl", help="epoch마다 지켜볼 평가 파일. ''이면 안 봄 (2차)")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--run", default="runs/eg2-classroom", help="체크포인트 폴더")
ap.add_argument("--out", default="models/eg2-classroom", help="완성 모델 폴더")
a = ap.parse_args()

model = SentenceTransformer("google/embeddinggemma-2", device=device(),
    model_kwargs={"torch_dtype": torch.bfloat16},
    config_kwargs={"vision_config": None, "audio_config": None})

# gradient checkpointing은 Trainer 인자 대신 내부 HF 모델에 직접 켠다 (원문 Q9)
hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

train = load_dataset("json", data_files=a.train, split="train")
print(f"학습 쌍 {len(train)}개, 장치 {model.device}")

# MatryoshkaLoss로 감싸야 학습 후에도 256/128차원 절단이 유지된다
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model),
                      matryoshka_dims=[768, 512, 256, 128])

args = SentenceTransformerTrainingArguments(
    output_dir=a.run,
    num_train_epochs=a.epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=a.lr, warmup_steps=0.1,  # Transformers v5: 비율을 float로
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,   # 배치 안 중복 문서 방지
    prompts={"anchor": model.prompts["SearchQuery"],
             "positive": model.prompts["Document"]},
    eval_strategy="epoch" if a.eval else "no", save_strategy="epoch", save_total_limit=1,
    logging_steps=10, report_to="none",
)

trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                     loss=loss, evaluator=build_evaluator(a.eval) if a.eval else None)
resume = any(Path(a.run).glob("checkpoint-*"))
print("이어서 학습" if resume else "처음부터 학습")
t = time.perf_counter()
trainer.train(resume_from_checkpoint=True if resume else None)
minutes = (time.perf_counter() - t) / 60
model.save_pretrained(a.out)
Path(a.run).mkdir(parents=True, exist_ok=True)
json.dump({"train_minutes": round(minutes, 1), "resumed": resume, "pairs": len(train),
           "lr": a.lr, "epochs": a.epochs},
          open(Path(a.run) / "train_time.json", "w"), indent=2)
print(f"학습 {minutes:.1f}분, 저장: {a.out}")
