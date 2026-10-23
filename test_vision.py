# 4단계 영상 테스트: 텍스트 질의로 사진과 동영상을 찾는다 (텍스트와 이미지가 같은 벡터 공간에 있는지 확인).
# 원문 README의 test_vision.py에 1위 확인(PASS/FAIL), 768/512/128차원 비교, 동영상 실패 시 프레임 검색 대체,
# 텍스트+이미지 혼합 입력(확장 과제)을 더했다.
# 사진 이름 약속: media/empty.jpg(빈 강의실), media/class.jpg(수업 중), media/board.jpg(화이트보드). 다른 사진은 이름 자유.
# 메모리 아끼기: 오디오 인코더는 안 올리고, GPU 몫을 제한하고, 작은 배치로 나눠 넣고, 끝나면 바로 해제한다.
import gc
import glob
import os
import time

import numpy as np
import torch
from eg2 import load

torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)

queries = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]
expected = ["empty", "class", "board"]   # 질의마다 1위여야 하는 사진 이름 (확장자 뺀 것)
images = sorted(glob.glob("media/*.jpg"))
frames = sorted(glob.glob("media/frames/*.jpg"))   # 동영상에서 2초마다 뽑은 프레임 (대체용)
VIDEO = "media/clip.mp4"
assert images, "media/*.jpg 사진이 없다. 안내 3-3을 먼저 한다"
names = [os.path.splitext(os.path.basename(p))[0] for p in images]
print("사진:", names)
enc = dict(batch_size=2, normalize_embeddings=True)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ms(t):
    return (time.perf_counter() - t) * 1000


t = time.perf_counter()
model = load(vision=True)   # 텍스트 + 비전 (오디오는 안 올림)
print(f"모델 로드 {time.perf_counter() - t:.1f} s")

with torch.inference_mode():
    model.encode([{"image": images[0]}], **enc)   # 예열: 첫 호출의 CUDA 초기화 시간을 빼고 잰다
    torch.cuda.synchronize()
    t = time.perf_counter()
    img_emb = model.encode([{"image": p} for p in images], **enc)
    torch.cuda.synchronize()
    print(f"이미지 {len(images)}장: {ms(t):.0f} ms")

    q_emb = model.encode(queries, prompt_name="SearchQuery", **enc)

    all_ok = True
    for dim in [768, 512, 128]:
        sim = norm(q_emb[:, :dim]) @ norm(img_emb[:, :dim]).T   # 자른 뒤 재정규화
        print(f"\n[dim={dim}]" + ("  (참고, 통과 조건 아님)" if dim == 128 else ""))
        for qi, row, exp in zip(queries, sim, expected):
            order = row.argsort()[::-1]
            top3 = [(names[i], round(float(row[i]), 3)) for i in order[:3]]
            gap = row[order[0]] - row[order[1]] if len(order) > 1 else 0.0
            if exp in names:
                ok = names[order[0]] == exp
                if dim != 128:
                    all_ok &= bool(ok)
                mark = "PASS" if ok else "FAIL"
            else:
                all_ok = False
                mark = f"판정 못 함 ({exp}.jpg 없음)"
            print(f"  {qi} -> {top3}  2위와 차이 {gap:.3f}  {mark}")

    print("\n[동영상]")
    vid_ok = False
    if os.path.exists(VIDEO):
        try:
            torch.cuda.synchronize()
            t = time.perf_counter()
            vid_emb = model.encode({"video": VIDEO}, normalize_embeddings=True)
            torch.cuda.synchronize()
            print(f"동영상 1개: {ms(t):.0f} ms")
            print("질의-동영상 유사도:", (q_emb @ vid_emb).round(3), " (순서: " + " / ".join(queries) + ")")
            vid_ok = True
        except Exception as e:   # 원문 확인 포인트: 디코딩 실패 시 프레임 검색으로 대체
            print(f"동영상 디코딩 실패: {type(e).__name__}: {str(e)[:200]}")
    else:
        print(f"{VIDEO} 없음")
    if not vid_ok and frames:
        t = time.perf_counter()
        fr_emb = model.encode([{"image": p} for p in frames], **enc)
        print(f"대체: 프레임 {len(frames)}장으로 검색 {ms(t):.0f} ms")
        sim = q_emb @ fr_emb.T
        for qi, row in zip(queries, sim):
            i = int(row.argmax())
            print(f"  {qi} -> {os.path.basename(frames[i])} ({row[i]:.3f}), 프레임 평균 {row.mean():.3f}")

    print("\n[확장 과제] 텍스트+이미지 혼합 입력 (참고)")
    if "board" in names:
        try:
            mix = model.encode({"text": "강의실 <|image|>", "image": [images[names.index("board")]]},
                               normalize_embeddings=True)
            print("  혼합 입력과 질의 유사도:", (q_emb @ mix).round(3))
            print("  board.jpg만 넣었을 때:  ", (q_emb @ img_emb[names.index("board")]).round(3))
        except Exception as e:
            print(f"  혼합 입력 실패: {type(e).__name__}: {str(e)[:200]}")
    else:
        print("  board.jpg가 없어 건너뜀")

peak = torch.cuda.max_memory_allocated() / 2**20
del model
gc.collect()
torch.cuda.empty_cache()

print(f"\nGPU 메모리 최대 {peak:.0f} MB")
print("동영상:", "통째로 임베딩 성공" if vid_ok else "실패 → 프레임 검색으로 대체" if frames else "실패, 대체 프레임도 없음")
print("결과:", "768·512차원에서 질의 3개 모두 1위 정답 (통과)" if all_ok else "1위가 틀렸거나 판정 못 한 칸이 있음 (미통과)")
