# 2단계 설치 확인: torch가 GPU 버전 그대로인지, 패키지 버전이 맞는지 본다.
import torch, transformers, sentence_transformers as st
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", st.__version__)   # 6.1.0 이상이어야 함
assert torch.cuda.is_available(), "Jetson torch가 덮어쓰여졌는지 확인"
