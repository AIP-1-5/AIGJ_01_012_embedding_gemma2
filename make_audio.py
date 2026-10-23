# 5단계 소리 자료 만들기: media/에 16 kHz 모노 WAV(speech, beep, noise, 있으면 mic)를 만든다.
# 경보음과 잡음은 numpy로 직접 만들어서 ffmpeg(보드 것은 NVIDIA 판)·sudo 설치가 필요 없다.
# 합성 음성은 보드에 espeak-ng가 있으면 그것으로, 없으면 맥에서 만든 media/raw_speech.*를 변환한다.
# 마이크 녹음은 media/raw_mic.*로 넣어 두면 같이 변환한다 (선택).
# 사용: python make_audio.py
import glob
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf

SR = 16000
TEXT = "강의실이 너무 덥습니다. 에어컨을 켜 주세요."
os.makedirs("media", exist_ok=True)
rng = np.random.default_rng(0)   # 매번 같은 잡음이 나오게


def save(name, x):
    x = np.asarray(x, dtype=np.float32)
    x = x / max(1e-9, float(np.abs(x).max())) * 0.5   # 소리 크기를 최대 0.5로 맞춘다 (깨짐 방지)
    sf.write(f"media/{name}.wav", x, SR, subtype="PCM_16")
    print(f"저장: media/{name}.wav  {len(x) / SR:.1f}초")


# 경보음: 1 kHz 사인파 3초 (원문 ffmpeg sine=frequency=1000:duration=3 과 같은 소리)
t = np.arange(3 * SR) / SR
save("beep", np.sin(2 * np.pi * 1000 * t))

# 배경 잡음: 핑크 노이즈 5초 (원문 anoisesrc=color=pink). 흰 잡음의 높은 소리를 1/√f로 줄여 웅웅거리게 만든다
n = 5 * SR
spec = np.fft.rfft(rng.standard_normal(n))
f = np.fft.rfftfreq(n, 1 / SR)
spec[1:] /= np.sqrt(f[1:])
spec[0] = 0
save("noise", np.fft.irfft(spec, n))

# 합성 음성: 보드에 espeak-ng가 있고 raw_speech가 아직 없으면 만든다
if not glob.glob("media/raw_speech.*"):
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-v", "ko", TEXT, "-w", "media/raw_speech.wav"], check=True)
        print("espeak-ng로 media/raw_speech.wav 생성")
    else:
        print("espeak-ng 없음 → 맥에서 만든 raw_speech.aiff를 media/에 넣고 다시 실행 (안내 3-3)")

# raw_*를 16 kHz 모노로 변환 (wav, aiff, flac)
for p in sorted(glob.glob("media/raw_*")):
    name = os.path.splitext(os.path.basename(p))[0][4:]   # raw_speech.aiff → speech
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(axis=1)   # 여러 채널이면 평균으로 모노
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    print(f"변환: {p} ({sr} Hz) →", end=" ")
    save(name, x)

print("\n확인:")
for p in sorted(glob.glob("media/*.wav")):
    if "/raw_" in p:
        continue
    i = sf.info(p)
    print(f"  {os.path.basename(p)}: {i.samplerate} Hz, {i.channels}채널, {i.duration:.1f}초",
          "OK" if (i.samplerate, i.channels) == (SR, 1) else "← 16 kHz 모노 아님")
