# 공통 로더 (3~5단계가 같이 쓴다). 필요한 인코더만 골라 GPU에 올린다.
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"


def _load_env(path=Path.home() / "eg2" / ".env"):
    # ~/eg2/.env의 HF_TOKEN 등을 환경 변수로 읽는다. 값은 출력하지 않는다.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def load(vision=False, audio=False):
    cfg = {}
    if not vision: cfg["vision_config"] = None   # 미사용 인코더는 로드하지 않음
    if not audio:  cfg["audio_config"] = None
    return SentenceTransformer(
        MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},  # float16 금지
        config_kwargs=cfg,
    )
