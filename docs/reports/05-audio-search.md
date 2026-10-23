# 5단계 음성 테스트 (텍스트로 소리 찾기)

> 상태: **안내 작성, 보드 A 실행 전**
> 원문: 실습 README의 "5단계 음성 테스트"
> 앞 단계: [04-vision-search.md](04-vision-search.md) — 보드 A 미통과 5/6 (GPU 최대 1022 MB)
> 우리 보드: 보드 A, 모니터·스피커 없음 → 소리는 **노트북 브라우저**로 들어 본다.

## 0. 이 단계에서 무엇을 하나

4단계는 글로 사진을 찾았다. 5단계는 **오디오 인코더**(소리를 숫자로 바꾸는 부분, 약 570M)를 올려서 **글로 소리를 찾는다.**

1. 소리 파일 3개를 만든다: 사람 말소리(합성 음성) `speech`, 날카로운 경보음 `beep`, 웅웅거리는 잡음 `noise`.
2. 세 파일을 각각 숫자 768개(벡터)로 바꾼다.
3. "날카로운 경보음" 같은 **글** 질문도 벡터로 바꿔, 어느 소리와 가장 가까운지 본다.

글 벡터와 소리 벡터를 바로 비교할 수 있으면 **글과 소리도 같은 벡터 공간에 있다**는 뜻이다. 3·4·5단계를 거치면 텍스트·사진·소리를 한 가지 방법으로 찾을 수 있다는 것을 확인하게 된다.

- 모델은 소리를 **16 kHz 모노**(1초에 16000번 잰 값, 채널 1개)로만 제대로 이해한다. 그래서 파일을 꼭 이 형식으로 맞춘다. 형식이 다르면 결과가 무작위에 가까워진다(원문 6단계 표).
- 원문도 적어 두었듯 5단계는 다른 단계의 전제가 아니다. 그래도 6단계 표의 "+오디오" 행을 채우려고 한다.

## 1. 성공 기준 (먼저 적어 둔다)

원문 확인 포인트: "세 질의가 각각 speech, beep, noise 파일을 1위로 찾으면 성공".

| 질의 | 1위여야 하는 파일 | 내용 |
| --- | --- | --- |
| 덥다고 냉방을 요청하는 목소리 | `speech.wav` | 합성 음성 "강의실이 너무 덥습니다. 에어컨을 켜 주세요." |
| 날카로운 경보음 | `beep.wav` | 1 kHz 비프 3초 |
| 웅웅거리는 배경 소음 | `noise.wav` | 핑크 노이즈 5초 |

- **통과**: 4단계와 같은 규칙으로 768차원과 512차원(원문: "멀티모달 검색은 768 또는 512차원")에서 질의 3개 모두 1위가 정답, 즉 6칸 PASS. (원문 코드는 768만 본다. 768은 3칸 모두 맞았는데 512에서 틀리면 그 사실을 따로 적는다.)
- 128차원은 참고로만 본다(원문: 영상·음성 품질이 약 75%로 떨어짐).
- 판정은 위 세 파일로만 한다. 마이크 녹음 `mic.wav`(선택)는 "합성 음성과 사람 목소리 점수 비교"에만 쓴다(원문: "espeak-ng 합성음은 사람 목소리보다 점수가 낮게 나올 수 있다. 가능하면 마이크 녹음으로 비교해 본다").
- 결과를 본 뒤 문장·파일을 바꾸거나 다시 만들어 맞추지 않는다. 다시 만들었다면 그 사실과 이유를 적는다.
- 지연(ms)·메모리는 기록만 하고 6단계 표의 "+오디오" 행에 쓴다.

## 2. 우리 보드에서는 원문과 이렇게 다르게 한다

| 원문 | 우리 보드에서 | 이유 |
| --- | --- | --- |
| `sudo apt install -y espeak-ng alsa-utils` | **설치하지 않는다.** 3-1에서 이미 있는지만 본다 | sudo 일은 줄인다(팀 문서 7번). 아래 방법으로 sudo 없이 된다 |
| `ffmpeg -f lavfi`로 경보음·잡음 만들기, `ffmpeg -ar 16000 -ac 1`로 변환 | `make_audio.py`가 numpy로 같은 소리(1 kHz 사인 3초, 핑크 노이즈 5초)를 만들고, soundfile·librosa로 16 kHz 모노 변환 | 보드 ffmpeg는 NVIDIA Jetson 판(`n8.0.1`)이라 4단계에서 장치 입력이 없었다. 필터가 있어도 없어도 같은 결과가 나오게 ffmpeg에 기대지 않는다. soundfile·librosa는 2단계에서 깔았다 |
| `espeak-ng -v ko`로 합성 음성 | 보드에 espeak-ng가 **있으면** `make_audio.py`가 그대로 쓴다. **없으면 맥**에서 `say -v Yuna`(macOS 기본 한국어 음성)로 만들어 `scp`로 보낸다 | sudo 없이 하려고. 맥 음성은 espeak-ng보다 자연스러워 점수가 더 높게 나올 수 있다. 어느 쪽을 썼는지 결과에 적는다 |
| `arecord`로 USB 마이크 녹음 | 선택. 보드에 `arecord`와 소리 장치(웹캠 마이크 등)가 있으면 그것으로, 없으면 맥에서 녹음해 보낸다 | 3-1에서 확인 |
| `aplay`로 들어 보기 | 보드에서 `python3 -m http.server 8001`을 잠깐 켜고 **노트북 브라우저**로 듣는다 | 보드에 모니터·스피커가 없다. 4단계와 같은 방법 |
| `model.encode([{"audio": 파일경로}])` | soundfile로 읽은 배열을 `{"audio": {"array": ..., "sampling_rate": 16000}}`로 넣는다 | 경로를 넣으면 transformers가 `torchcodec`(4단계에서 없어서 동영상 실패)을 먼저 찾는다. 배열로 넣으면 그 도구 없이 확실히 된다 |
| 1위만 출력 | PASS/FAIL, 2위와 점수 차이, 768/512/128차원 비교, 마이크 비교, GPU 메모리 최대 | 기준 확인을 눈으로 세지 않게 |
| 확장 과제(전체 모델로 한 인덱스) | `python test_audio.py all`로 **따로**, 메모리가 넉넉할 때만 | 전체 모델(740M)은 비전까지 올려서 더 무겁다. 본 실험에는 필요 없다 |
| `python test_audio.py` | `~/practice.sh 재웅 python test_audio.py` | 자물쇠·메모리 제한 |
| (메모리 설정 없음) | 비전 인코더 안 올림, GPU 몫 `0.3`, `batch_size=2`, `inference_mode`, 끝나면 해제 | 팀 문서 11번, 메모리 아끼기 원칙 |

이번 단계는 **pip 설치가 없다.** 시작 전에 버전이 그대로인지만 본다.

## 3. 순서 (보드 A, 자기 계정으로 SSH)

SSH는 노트북에서 Tailscale 주소로 `ssh jaeung@100.113.228.46` (기본).

### 3-1. 시작 전 확인 (읽기만)

**무엇을, 왜:** 누가 보드를 쓰는 중인지, 메모리, torch 버전, 센서 서비스를 보고, 소리 도구(espeak-ng, arecord)와 소리 장치가 있는지 본다. 아무것도 설치하거나 바꾸지 않는다.

```bash
echo "== 누가 쓰는 중?"; cat /tmp/practice.who.* 2>/dev/null || echo "아무도 안 씀"
echo "== 메모리"; free -h                   # available 3GB 이상
source ~/eg2/bin/activate
echo "== 패키지"; pip freeze | grep -iE '^(torch|torchvision|torchaudio|torchcodec|numpy|soundfile|librosa)=='
echo "== 센서 서비스"; curl -s -m 3 localhost:8080/health || echo "응답 없음"
echo "== 소리 도구"; for c in espeak-ng arecord aplay ffmpeg; do printf '%-10s ' $c; command -v $c || echo "없음"; done
echo "== ffmpeg 소리 필터"; ffmpeg -hide_banner -filters 2>/dev/null | grep -E ' (sine|anoisesrc) ' || echo "없음"
echo "== 소리 장치"; cat /proc/asound/cards 2>/dev/null || echo "없음"; groups | grep -qw audio && echo "audio 그룹 OK" || echo "audio 그룹 아님"
```

- 기대: torch `2.14.1+cu132`, torchvision `0.29.1+cu132`, numpy `1.26.4`, soundfile·librosa 있음, torchaudio·torchcodec 없음.
- 소리 도구·장치 결과는 그대로 보내 준다. 3-3, 3-4에서 어느 방법을 쓸지 정한다.
- 센서 서비스 "응답 없음"이면 여기서 멈춘다.

### 3-2. 코드 받기

```bash
cd ~/eg2-재웅 && git pull && git log -1 --oneline   # 보고서에 적을 커밋
ls make_audio.py test_audio.py && mkdir -p media
```

### 3-3. 합성 음성 준비 (둘 중 하나)

**무엇을, 왜:** "덥다고 냉방을 요청하는" 말소리 파일을 만든다. 마이크가 없어도 되게 컴퓨터가 읽어 주는 목소리(합성 음성)를 쓴다.

**(가) 보드에 espeak-ng가 있으면:** 할 일 없음. 3-5에서 `make_audio.py`가 알아서 만든다.

**(나) 없으면 맥에서 만든다** (맥 터미널, 보드 아님):

```bash
say -v '?' | grep ko_KR                          # Yuna 줄이 보이면 한국어 음성 있음
say -v Yuna -o ~/raw_speech.aiff "강의실이 너무 덥습니다. 에어컨을 켜 주세요."
afplay ~/raw_speech.aiff                          # 맥에서 들어 보기
scp ~/raw_speech.aiff jaeung@100.113.228.46:~/eg2-재웅/media/
```

- `Yuna`가 없으면 맥 "시스템 설정 → 손쉬운 사용 → 읽기 및 말하기 → 시스템 목소리"에서 한국어 음성을 받는다. 다른 한국어 음성 이름이 보이면 그 이름을 쓰고 결과에 적는다.
- 문장은 원문과 똑같이 쓴다. 바꾸지 않는다.

### 3-4. 마이크 녹음 (선택)

**무엇을, 왜:** 같은 문장을 사람이 직접 말한 녹음이 합성 음성보다 점수가 높은지 비교한다(원문 확인 포인트). 판정에는 쓰지 않는다.

- 보드에 `arecord`가 있고 3-1 소리 장치에 웹캠·USB 마이크가 보이면 (번호 `N`은 `/proc/asound/cards`의 맨 앞 숫자):

  ```bash
  cd ~/eg2-재웅
  ~/practice.sh 재웅 arecord -D plughw:N,0 -f S16_LE -r 16000 -c 1 -d 5 media/raw_mic.wav   # 5초 동안 같은 문장을 말한다
  ```

- 보드에서 안 되면 맥에서 QuickTime Player → 파일 → 새로운 오디오 녹음으로 녹음해 `mic.m4a`로 저장한 뒤:

  ```bash
  afconvert -f WAVE -d LEI16@16000 -c 1 ~/Desktop/mic.m4a ~/raw_mic.wav
  scp ~/raw_mic.wav jaeung@100.113.228.46:~/eg2-재웅/media/
  ```

목소리 녹음은 개인 정보다. 보고서를 쓴 뒤 그날 지운다(3-8).

### 3-5. 소리 파일 만들기와 16 kHz 모노 확인

**무엇을, 왜:** 경보음·잡음을 만들고, 받아 둔 `raw_*` 파일을 모델이 원하는 16 kHz 모노로 바꾼다. 마지막에 파일마다 형식을 출력한다(원문 실행 절차 2번 `ffprobe` 대신).

```bash
cd ~/eg2-재웅 && source ~/eg2/bin/activate
~/practice.sh 재웅 python make_audio.py
```

기대 출력의 끝:

```text
확인:
  beep.wav: 16000 Hz, 1채널, 3.0초 OK
  noise.wav: 16000 Hz, 1채널, 5.0초 OK
  speech.wav: 16000 Hz, 1채널, x.x초 OK
```

`speech.wav`가 없으면 3-3을 다시 본다.

### 3-6. 노트북 브라우저로 들어 보기

**무엇을, 왜:** 파일이 이름대로 들리는지 사람 귀로 먼저 확인한다(검색을 돌리기 **전에**).

```bash
cd ~/eg2-재웅
~/practice.sh 재웅 python3 -m http.server 8001 -d media
```

노트북 브라우저에서 `http://100.113.228.46:8001` → `speech.wav`, `beep.wav`, `noise.wav`를 눌러 듣는다. 다 들었으면 **Ctrl+C로 끈다**(켜 둔 동안 자물쇠가 잠겨 있다).

### 3-7. 음성 검색 실행 (성공 기준)

**무엇을, 왜:** 소리 3~4개와 질의 3개를 벡터로 바꾸고, 질의마다 1위 파일을 찾는다. 768·512에서 `PASS`가 6번 나오면 통과다.

```bash
cd ~/eg2-재웅 && source ~/eg2/bin/activate
free -h | head -2                           # available 3GB 미만이면 실행하지 않고 화면을 보낸다
~/practice.sh 재웅 python test_audio.py 2>&1 | tee /tmp/eg2-05-audio.txt
```

출력 모양 (숫자는 자리만 표시):

```text
소리: ['beep 3.0s', 'noise 5.0s', 'speech x.xs']
모델 로드 x.x s
오디오 3개: xxx ms

[dim=768]
  덥다고 냉방을 요청하는 목소리 -> [('speech', 0.xxx), ...]  2위와 차이 0.xxx  PASS
  날카로운 경보음 -> [('beep', 0.xxx), ...]  2위와 차이 0.xxx  PASS
  웅웅거리는 배경 소음 -> [('noise', 0.xxx), ...]  2위와 차이 0.xxx  PASS
[dim=512]
  ...
[dim=128]  (참고, 통과 조건 아님)
  ...

[마이크 녹음 비교] (참고)
  mic.wav가 없어 건너뜀

GPU 메모리 최대 xxx MB
결과: 768·512차원에서 질의 3개 모두 1위 정답 (통과)
```

**확장 과제 (선택, 메모리가 넉넉할 때만):** 비전까지 올린 전체 모델로 4단계 사진과 소리를 한 인덱스에 넣고 검색한다. 위 실행이 끝난 뒤 `free -h`의 available이 3GB 이상일 때만.

```bash
free -h | head -2
~/practice.sh 재웅 python test_audio.py all 2>&1 | tee /tmp/eg2-05-audio-all.txt
```

### 3-8. 끝나고

```bash
curl -s localhost:8080/health; echo
free -h
rm -f ~/eg2-재웅/media/raw_mic.wav ~/eg2-재웅/media/mic.wav   # 목소리 녹음 지우기 (했을 때만)
```

`/tmp/eg2-05-audio.txt` 내용, 3-1 결과, 합성 음성을 어떻게 만들었는지(espeak-ng / 맥 Yuna), 실행 전후 `free -h`를 보내 주면 아래 결과 표를 채운다.

## 4. 자주 나는 오류

| 화면에 나온 것 | 원인 | 할 일 |
| --- | --- | --- |
| `16kHz 모노가 아님` | 변환 안 된 파일이 `media/`에 있음 | `python make_audio.py`를 다시 한다. 그래도 나오면 화면을 보낸다 |
| `media/*.wav 소리 파일이 없다` | 3-5를 안 했거나 다른 폴더에서 실행 | `cd ~/eg2-재웅` 후 `ls media` |
| `판정 못 함 (speech.wav 없음)` | 합성 음성을 못 만듦 | 3-3을 다시 본다 |
| `TorchAudio has CUDA version` / `AutoProcessor` | torchaudio가 다시 깔림 | `pip uninstall -y torchaudio` (2단계와 같음) |
| `OfflineModeIsEnabled` / `does not appear to have a file` | 오디오 인코더 파일이 캐시에 없음 | `HF_HUB_OFFLINE=0 hf download google/embeddinggemma-2`로 받고 재실행 |
| `CUDA out of memory`, `Killed` | 메모리 부족 | `free -h`, `cat /tmp/practice.who.*`를 보낸다. 센서 서비스는 끄지 않는다 |
| `say: ... voice not found` (맥) | 한국어 음성이 안 깔림 | 3-3 아래 안내대로 맥에서 음성을 받는다 |
| 점수가 `nan` | float16 사용 | `eg2.py`는 bfloat16이다. 코드를 고치지 않았는지 본다 |

## 5. 결과

| 항목 | 보드 A |
| --- | --- |
| 실행한 사람 / 날짜 | |
| 실행한 코드 | |
| 시작 전 확인 | |
| 소리 도구 (espeak-ng, arecord, ffmpeg 필터, 소리 장치) | |
| 합성 음성 만든 방법 | |
| 마이크 녹음 | |
| 모델 로드 | |
| 오디오 N개 지연 | |
| 768차원 | |
| 512차원 | |
| 128차원 (참고) | |
| 마이크 비교 (참고) | |
| 확장 과제 (참고) | |
| GPU 메모리 최대 | |
| 보드 메모리 (전 → 후) | |
| 센서 서비스 (`/health`) | |
| **성공 기준** | |

**왜 이렇게 나왔는지:** (실행 후 작성)

**다음에 할 일:** (실행 후 작성)
