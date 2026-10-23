# 6단계 성능 측정: 원문 표의 5줄(텍스트 전용 / +비전 이미지 / +비전 동영상 / +오디오 / 전체 혼합)을 잰다.
# 사용: python bench.py text|vision|audio|all   구성마다 따로 실행한다 (모델을 매번 새로 올려야 "로드 후 RAM"이 깨끗하다)
#       python bench.py table                   지금까지 잰 값을 원문 표 모양으로 출력한다 (모델 안 올림)
# 재는 것: 로드 후 RAM(tegrastats), 지연(같은 입력 3번 중 두 번째 값, 원문 기준), GPU 사용률(같은 입력을 3초 이상
#         반복하는 동안 tegrastats GR3D_FREQ 평균·최대), torch GPU 메모리 최대.
# 메모리 아끼기: 필요한 인코더만 올리고, GPU 몫을 제한하고, 시작 전 available 3GB 미만이면 안 돌리고,
#             도는 중 1.2GB 아래로 내려가면 다음 측정을 하지 않고 모델을 내린 뒤 끝내고,
#             1GB 아래로 내려가면 그 자리에서 바로 프로세스를 끝낸다(메모리가 즉시 풀린다).
import gc
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

CONFIGS = {   # 구성: (비전, 오디오, 원문 로드 크기)
    "text": (False, False, "270M"),
    "vision": (True, False, "440M"),
    "audio": (False, True, "570M"),
    "all": (True, True, "740M"),
}
OUT = "media/bench.jsonl"   # 결과 기록 (media/는 커밋하지 않는다)
START_MIN_MB = 3072         # 시작 전 available 하한 (팀 메모리 원칙)
STOP_MIN_MB = 1200          # 도는 중 이 아래로 내려가면 다음 측정을 하지 않는다
KILL_MIN_MB = 1000          # 도는 중 이 아래로 내려가면 바로 끝낸다
BUSY_S = 3.0                # GPU 사용률을 볼 반복 시간


def mem_available_mb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return info["MemAvailable"] // 1024, (info["MemTotal"] - info["MemAvailable"]) // 1024


def gpu_load_sysfs():
    # tegrastats에 GR3D_FREQ가 안 나올 때 대신 읽는다 (0~1000 → %)
    for p in glob.glob("/sys/devices/platform/*/17000000.gpu*/load") + glob.glob("/sys/devices/platform/17000000.gpu*/load"):
        try:
            return int(open(p).read().strip()) / 10
        except (OSError, ValueError):
            pass
    return None


class Sampler:
    """tegrastats를 0.2초마다 읽어 (시각, RAM MB, GPU %, available MB)를 모은다. 없으면 /proc·sysfs로 대신한다."""

    def __init__(self):
        self.samples, self.min_avail, self.source = [], 10**9, "tegrastats"
        self.proc, self.got_tegra = None, False
        if shutil.which("tegrastats"):
            try:
                self.proc = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, text=True)
            except OSError:
                self.proc = None
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(1.5)
        if not self.got_tegra:   # tegrastats가 없거나 RAM 줄을 못 줬다
            self.source = "/proc/meminfo + sysfs (tegrastats 안 됨)"
            self.stop()

    def _record(self, ram, gpu):
        avail, used = mem_available_mb()
        self.min_avail = min(self.min_avail, avail)
        if avail < KILL_MIN_MB:
            print(f"\n!! available {avail} MB < {KILL_MIN_MB} MB → 바로 멈춘다 (이 구성은 기록 안 함)", flush=True)
            if self.proc:
                self.proc.terminate()
            os._exit(4)
        self.samples.append((time.perf_counter(), ram if ram is not None else used, gpu, avail))

    def _run(self):
        if self.proc:
            for line in self.proc.stdout:
                ram = re.search(r"RAM (\d+)/(\d+)MB", line)
                gpu = re.search(r"GR3D_FREQ (\d+)%", line)
                self.got_tegra |= bool(ram)
                self._record(int(ram.group(1)) if ram else None,
                             int(gpu.group(1)) if gpu else gpu_load_sysfs())
        while True:   # tegrastats가 없거나 꺼졌을 때
            self._record(None, gpu_load_sysfs())
            time.sleep(0.2)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def ram(self, t0, t1):
        w = [s[1] for s in self.window(t0, t1) if s[1] is not None]
        return round(statistics.median(w)) if w else None

    def gpu(self, t0, t1):
        w = [s[2] for s in self.window(t0, t1) if s[2] is not None]
        return (round(statistics.mean(w)), round(max(w)), len(w)) if w else (None, None, 0)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


def cycle(items, n):
    return [items[i % len(items)] for i in range(n)] if items else []


def table():
    rows = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            rows[r["row"]] = r   # 같은 줄을 여러 번 쟀으면 마지막 값
    print("| 구성 | 로드 크기 | 로드 후 RAM (MB) | 입력 | 지연 (ms) | GPU 사용률 (%) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key, name in [("text", "텍스트 전용"), ("vision_image", "+비전"), ("vision_video", "+비전"),
                      ("audio", "+오디오"), ("all", "전체")]:
        r = rows.get(key)
        if not r:
            print(f"| {name} | | (안 잼) | | | |")
            continue
        print(f"| {name} | {r['size']} | {r['ram_after_load']} (+{r['ram_delta']}) | {r['input']} | "
              f"{r['latency_ms'][1]:.0f} | {r['gpu_mean']} (최대 {r['gpu_max']}) |")
    print("\nRAM 괄호 = 로드 전보다 늘어난 양. 지연 = 3번 중 두 번째. GPU = 같은 입력을 3초 이상 반복하는 동안 평균(최대).")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in [*CONFIGS, "table"]:
        sys.exit("사용: python bench.py text|vision|audio|all|table")
    if sys.argv[1] == "table":
        return table()
    cfg = sys.argv[1]
    vision, audio, size = CONFIGS[cfg]

    avail, _ = mem_available_mb()
    print(f"[{cfg}] 시작 전 available {avail} MB")
    if avail < START_MIN_MB:
        sys.exit(f"available {avail} MB < {START_MIN_MB} MB → 돌리지 않는다. 다른 사람이 쓰는지 확인하고 나중에 다시")

    import numpy as np
    import torch
    from eg2 import load

    torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)

    # ---- 입력 준비 (원문 표의 "입력" 열) ----
    queries = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]   # 3단계 질의 2개
    images = sorted(glob.glob("media/*.jpg"))
    frames = sorted(glob.glob("media/frames/*.jpg"))
    waves_f = sorted(f for f in glob.glob("media/*.wav") if not os.path.basename(f).startswith("raw_"))
    waves_f = ([f for f in waves_f if os.path.basename(f)[:-4] in ("speech", "beep", "noise")] or waves_f)[:3]
    if vision:
        assert images, "media/*.jpg 사진이 없다 (4단계 사진). 사람이 안 찍힌 사진이면 무엇이든 된다"
    if audio:
        assert waves_f, "media/*.wav 소리가 없다 → python make_audio.py (5단계)"
    img5 = cycle(images, 5)   # 원문 "이미지 5장". 사진이 5장보다 적으면 있는 사진을 돌려 쓴다 (계산량은 같다)
    waves = []
    if audio:
        import soundfile as sf
        for f in waves_f:
            w, sr = sf.read(f, dtype="float32")
            assert sr == 16000 and w.ndim == 1, f"{f}: 16kHz 모노가 아님 → python make_audio.py"
            waves.append({"audio": {"array": w, "sampling_rate": 16000}})
    print("사진:", [os.path.basename(p) for p in img5] if vision else "-",
          "/ 소리:", [os.path.basename(p) for p in waves_f] if audio else "-")

    smp = Sampler()
    print("측정 도구:", smp.source)
    t = time.perf_counter()
    time.sleep(1.0)
    ram_before = smp.ram(t, time.perf_counter())

    t = time.perf_counter()
    model = load(vision=vision, audio=audio)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    time.sleep(1.5)   # 로드가 끝난 뒤 RAM이 자리 잡을 때까지
    t = time.perf_counter()
    time.sleep(1.0)
    ram_after = smp.ram(t, time.perf_counter())
    print(f"모델 로드 {load_s:.1f} s, RAM {ram_before} → {ram_after} MB (+{ram_after - ram_before})")

    def enc_text(items):
        return model.encode(items, prompt_name="SearchQuery", batch_size=4, normalize_embeddings=True)

    def enc(items):
        return model.encode(items, batch_size=2, normalize_embeddings=True)

    def video_input():
        # 원문: 동영상은 1초에 1장씩 처리 → 10초 = 10장. 통째 임베딩이 안 되면 원문 6단계 표대로 프레임 사진으로 대체
        clip = "media/clip.mp4"
        if os.path.exists(clip):
            try:
                with torch.inference_mode():
                    enc({"video": clip})
                return "동영상 10초 (clip.mp4 통째)", lambda: enc({"video": clip}), None
            except Exception as e:
                why = f"{type(e).__name__}: {str(e)[:150]}"
        else:
            why = "media/clip.mp4 없음"
        src = frames or images
        ten = cycle(src, 10)
        label = f"동영상 10초 대체: 프레임 사진 10장 ({'frames/' if frames else '사진 '}{len(src)}장에서)"
        return label, lambda: enc([{"image": p} for p in ten]), why

    jobs = []   # (원문 표의 줄, 입력 설명, 실행 함수, 비고)
    if cfg == "text":
        jobs.append(("text", "질의 2개", lambda: enc_text(queries), None))
    if cfg == "vision":
        jobs.append(("vision_image", f"이미지 5장 (사진 {len(images)}장)", lambda: enc([{"image": p} for p in img5]), None))
        jobs.append(("vision_video", *video_input()))
    if cfg == "audio":
        jobs.append(("audio", f"WAV {len(waves)}개", lambda: enc(waves), None))
    if cfg == "all":
        def mixed():   # 질의 2개 + 이미지 5장 + WAV 3개를 차례로 (한 번 = 세 encode의 합)
            enc_text(queries)
            enc([{"image": p} for p in img5])
            enc(waves)
        jobs.append(("all", f"혼합: 질의 2 + 이미지 5 + WAV {len(waves)}", mixed, None))

    results = []
    with torch.inference_mode():
        for row, label, fn, note in jobs:
            if smp.min_avail < STOP_MIN_MB:
                print(f"!! available이 {smp.min_avail} MB까지 내려가 '{label}' 측정을 하지 않고 멈춘다")
                break
            times = []
            for _ in range(3):   # 원문: 지연은 두 번째 실행 값 (첫 번째는 예열)
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) * 1000)
            t0 = time.perf_counter()   # GPU 사용률: 한 번이 짧아 tegrastats 간격(0.2초)에 안 잡히므로 반복해서 본다
            reps = 0
            while time.perf_counter() - t0 < BUSY_S or reps < 3:
                fn()
                reps += 1
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            g_mean, g_max, n = smp.gpu(t0, t1)
            peak_ram = max((s[1] for s in smp.window(t0, t1) if s[1] is not None), default=None)
            r = dict(row=row, cfg=cfg, size=size, input=label, note=note, load_s=round(load_s, 1),
                     ram_before=ram_before, ram_after_load=ram_after, ram_delta=ram_after - ram_before,
                     ram_peak_busy=peak_ram, latency_ms=[round(x) for x in times], gpu_mean=g_mean, gpu_max=g_max,
                     gpu_samples=n, busy_reps=reps, source=smp.source, time=time.strftime("%Y-%m-%d %H:%M:%S"))
            results.append(r)
            print(f"\n[{label}]" + (f"  (대체 이유: {note})" if note else ""))
            print(f"  지연 3번: {' / '.join(f'{x:.0f}' for x in times)} ms → 표에는 두 번째 {times[1]:.0f} ms")
            print(f"  GPU 사용률: 평균 {g_mean}% / 최대 {g_max}% ({reps}번 반복, 표본 {n}개), 반복 중 RAM 최대 {peak_ram} MB")

    peak_gpu = torch.cuda.max_memory_allocated() / 2**20
    del model
    gc.collect()
    torch.cuda.empty_cache()
    smp.stop()
    for r in results:
        r["torch_gpu_peak_mb"] = round(peak_gpu)
    os.makedirs("media", exist_ok=True)
    with open(OUT, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    avail, _ = mem_available_mb()
    print(f"\ntorch GPU 메모리 최대 {peak_gpu:.0f} MB, 도는 중 available 최저 {smp.min_avail} MB, 끝난 뒤 {avail} MB")
    print(f"기록: {OUT} ({len(results)}/{len(jobs)}줄)")
    if len(results) < len(jobs):
        sys.exit(3)


if __name__ == "__main__":
    main()
