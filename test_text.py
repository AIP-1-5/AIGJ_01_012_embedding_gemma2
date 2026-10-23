# 3단계 텍스트 테스트: 한국어 질의가 알맞은 문서를 1위로 찾는지, 768/256/128차원에서 비교한다.
# 원문 README의 test_text.py에 1위 확인(PASS/FAIL), 점수 차이, 예열 1회, 프롬프트 없는 비교(변형 과제)를 더했다.
# 메모리 아끼기: 텍스트 인코더만 올리고, GPU 몫을 제한하고, 모델은 한 번만 올려 두 실험을 같이 돌린 뒤 바로 해제한다.
import gc
import time

import torch
from eg2 import load

torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)

docs = [
    "환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
    "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
    "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
    "Jetson Orin Nano는 67 TOPS의 AI 성능을 제공한다.",
]
queries = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]
expected = [0, 1]   # 원문 확인 포인트: 첫 질의 → CO2 문장, 둘째 질의 → 에어컨 문장


def run(model, q_prompt, d_prompt):
    enc = dict(batch_size=4, normalize_embeddings=True)
    d = model.encode(docs, prompt_name=d_prompt, **enc)
    model.encode(queries, prompt_name=q_prompt, **enc)   # 예열: 첫 호출의 CUDA 초기화 시간을 빼고 비교
    all_ok = True
    for dim in [768, 256, 128]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        q = model.encode(queries, prompt_name=q_prompt, truncate_dim=dim, **enc)
        ms = (time.perf_counter() - t) * 1000
        dd = d[:, :dim]
        dd = dd / (dd ** 2).sum(1, keepdims=True) ** 0.5   # 자른 뒤 재정규화
        sim = q @ dd.T
        print(f"\n[dim={dim}] {ms:.0f} ms")
        for qi, row, exp in zip(queries, sim, expected):
            order = row.argsort()[::-1]
            best, second = order[0], order[1]
            ok = best == exp
            all_ok &= bool(ok)
            print(f"  {qi} -> {docs[best][:25]}... ({row[best]:.3f})"
                  f"  2위와 차이 {row[best] - row[second]:.3f}  {'PASS' if ok else 'FAIL'}")
            print("    전체 점수: " + "  ".join(f"[{i}]{s:.3f}" for i, s in enumerate(row)))
    return all_ok


t = time.perf_counter()
model = load()  # 텍스트 전용
print(f"모델 로드 {time.perf_counter() - t:.1f} s")
for p in ["SearchQuery", "Document"]:
    assert p in model.prompts, f"프롬프트 {p!r} 없음. 있는 것: {list(model.prompts)}"

with torch.inference_mode():
    print("\n===== 본 실험: 질의=SearchQuery, 문서=Document (성공 기준) =====")
    ok = run(model, "SearchQuery", "Document")
    print("\n===== 변형 과제: 프롬프트 없음 (참고, 통과 조건 아님) =====")
    ok_np = run(model, None, None)

peak = torch.cuda.max_memory_allocated() / 2**20
del model
gc.collect()
torch.cuda.empty_cache()

print(f"\nGPU 메모리 최대 {peak:.0f} MB")
print("변형 과제:", "모든 차원 1위 정답" if ok_np else "1위가 틀린 차원이 있음")
print("결과:", "모든 차원에서 1위 정답 (통과)" if ok else "1위가 틀린 차원이 있음 (미통과)")
