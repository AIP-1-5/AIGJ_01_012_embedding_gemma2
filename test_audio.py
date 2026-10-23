# 5단계 음성 테스트: 텍스트 질의로 음성·소리 파일을 찾는다 (텍스트와 소리가 같은 벡터 공간에 있는지 확인).
# 원문 README의 test_audio.py에 1위 확인(PASS/FAIL), 768/512/128차원 비교, 마이크 녹음 비교를 더했다.
# 파일 이름 약속: media/speech.wav(합성 음성), beep.wav(경보음), noise.wav(잡음), mic.wav(마이크 녹음, 선택).
# 소리는 soundfile로 직접 읽어 배열로 넣는다 (torchcodec·torchaudio 없이 동작).
# 메모리 아끼기: 비전 인코더는 안 올리고, GPU 몫을 제한하고, 작은 배치로 나눠 넣고, 끝나면 바로 해제한다.
# 확장 과제(전체 모델, 이미지+소리 한 인덱스)는 `python test_audio.py all` 로 따로 돌린다.
import gc
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from eg2 import load

torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)
ALL = len(sys.argv) > 1 and sys.argv[1] == "all"

queries = ["덥다고 냉방을 요청하는 목소리", "날카로운 경보음", "웅웅거리는 배경 소음"]
expected = ["speech", "beep", "noise"]   # 질의마다 1위여야 하는 파일 이름 (원문 확인 포인트)
files = sorted(f for f in glob.glob("media/*.wav") if not os.path.basename(f).startswith("raw_"))
assert files, "media/*.wav 소리 파일이 없다. 안내 3-3을 먼저 한다"
names = [os.path.splitext(os.path.basename(p))[0] for p in files]
waves = []
for f in files:
    info = sf.info(f)
    assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님 → python make_audio.py"
    waves.append(sf.read(f, dtype="float32")[0])
print("소리:", [f"{n} {len(w) / 16000:.1f}s" for n, w in zip(names, waves)])
enc = dict(batch_size=2, normalize_embeddings=True)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ms(t):
    return (time.perf_counter() - t) * 1000


def audio(w):
    return {"audio": {"array": w, "sampling_rate": 16000}}


t = time.perf_counter()
model = load(vision=ALL, audio=True)   # 텍스트 + 오디오 (all이면 비전도)
print(f"모델 로드 {time.perf_counter() - t:.1f} s" + (" (전체 모델)" if ALL else ""))

with torch.inference_mode():
    model.encode([audio(waves[0])], **enc)   # 예열: 첫 호출의 CUDA 초기화 시간을 빼고 잰다
    torch.cuda.synchronize()
    t = time.perf_counter()
    aud_emb = model.encode([audio(w) for w in waves], **enc)
    torch.cuda.synchronize()
    print(f"오디오 {len(files)}개: {ms(t):.0f} ms")

    q_emb = model.encode(queries, prompt_name="SearchQuery", **enc)

    base = [i for i, n in enumerate(names) if n in expected]   # 판정은 원문의 세 파일로만 한다
    all_ok = len(base) == len(expected)
    for dim in [768, 512, 128]:
        sim = norm(q_emb[:, :dim]) @ norm(aud_emb[:, :dim]).T   # 자른 뒤 재정규화
        print(f"\n[dim={dim}]" + ("  (참고, 통과 조건 아님)" if dim == 128 else ""))
        for qi, row, exp in zip(queries, sim, expected):
            order = [i for i in row.argsort()[::-1] if i in base]
            if not order:
                print(f"  {qi} -> 판정할 파일이 없음 (speech·beep·noise.wav)")
                continue
            top3 = [(names[i], round(float(row[i]), 3)) for i in order[:3]]
            gap = row[order[0]] - row[order[1]] if len(order) > 1 else 0.0
            if exp in names:
                ok = names[order[0]] == exp
                if dim != 128:
                    all_ok &= bool(ok)
                mark = "PASS" if ok else "FAIL"
            else:
                mark = f"판정 못 함 ({exp}.wav 없음)"
            print(f"  {qi} -> {top3}  2위와 차이 {gap:.3f}  {mark}")

    print("\n[마이크 녹음 비교] (참고)")
    if "mic" in names and "speech" in names:
        s, m = names.index("speech"), names.index("mic")
        print(f"  '{queries[0]}' 점수: 합성 음성 {q_emb[0] @ aud_emb[s]:.3f} / 마이크 녹음 {q_emb[0] @ aud_emb[m]:.3f}")
        print(f"  합성 음성과 마이크 녹음끼리 유사도: {aud_emb[s] @ aud_emb[m]:.3f}")
    else:
        print("  mic.wav가 없어 건너뜀")

    if ALL:   # 확장 과제: 이미지와 소리를 한 인덱스에 넣고 하나의 질의로 검색
        print("\n[확장 과제] 이미지+소리 한 인덱스 (참고)")
        images = sorted(glob.glob("media/*.jpg"))
        if images:
            img_emb = model.encode([{"image": p} for p in images], **enc)
            idx = np.vstack([aud_emb, img_emb])
            labels = names + [os.path.splitext(os.path.basename(p))[0] + ".jpg" for p in images]
            for qi in queries + ["사람이 없는 빈 강의실"]:
                qv = model.encode([qi], prompt_name="SearchQuery", **enc)[0]
                row = idx @ qv
                top = [(labels[i], round(float(row[i]), 3)) for i in row.argsort()[::-1][:3]]
                print(f"  {qi} -> {top}")
        else:
            print("  media/*.jpg가 없어 건너뜀")

peak = torch.cuda.max_memory_allocated() / 2**20
del model
gc.collect()
torch.cuda.empty_cache()

print(f"\nGPU 메모리 최대 {peak:.0f} MB")
print("결과:", "768·512차원에서 질의 3개 모두 1위 정답 (통과)" if all_ok else "1위가 틀렸거나 판정 못 한 칸이 있음 (미통과)")
