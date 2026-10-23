# 2단계 패키지·모델 설치 (sentence-transformers · EmbeddingGemma 2 · 공통 로더)

> 상태: **보드 A 진행 중** (2026-10-23, 4-3까지: torch `2.14.1+cu132` 유지, sentence-transformers `6.2.0`, transformers `5.20.1`, 모델 gated 아님)
> 원문: 실습 README의 "2단계 패키지·모델 설치"
> 앞 단계: [01-environment.md](01-environment.md) — 보드 A 통과 (torch `2.14.1+cu132`, `~/eg2`, `~/eg2/torch-pin.txt`, `HF_HOME=~/.cache/huggingface`)
> 결정(2026-10-23): 재웅은 10단계까지 **보드 A**만 쓴다. 모델은 **계정마다 따로** 받는다(공용 폴더 아님).

## 0. 이 단계에서 무엇을 배우나

1단계에서 "GPU로 계산이 되는 파이썬 환경"을 만들었다. 2단계는 그 위에 **임베딩 모델**을 올린다.

- **임베딩**은 글(나중에는 사진·소리도)을 숫자 768개짜리 목록(벡터)으로 바꾸는 것이다. 뜻이 비슷한 글은 비슷한 숫자 목록이 된다. 3단계부터 하는 "검색"은 전부 이 숫자 목록끼리 얼마나 가까운지 재는 일이다.
- **sentence-transformers**는 이런 임베딩 모델을 한 줄(`model.encode(...)`)로 쓰게 해 주는 라이브러리다.
- **EmbeddingGemma 2**는 Google이 만든 임베딩 모델이다. 텍스트·비전·오디오 인코더가 들어 있는데, 이번에는 **텍스트만** 올린다. 안 쓰는 인코더를 안 올리면 메모리를 아낀다(보드는 8GB를 CPU·GPU·센서 서비스가 같이 쓴다).
- **`eg2.py`의 `load()`**는 3~5단계가 계속 쓰는 "모델 불러오는 함수"다. 한 번 잘 만들어 두면 다음 단계는 `from eg2 import load` 한 줄로 시작한다.

이 단계에서 가장 흔한 사고는 **패키지를 깔다가 1단계의 GPU용 torch가 CPU용으로 바뀌는 것**이다. 그래서 1단계에서 만든 `~/eg2/torch-pin.txt`(torch 버전 고정 파일)를 설치 명령에 꼭 붙인다.

## 1. 성공 기준 (먼저 적어 둔다)

아래 두 가지가 **모두** 나오면 통과다.

| 확인할 것 | 기대값 | 뜻 |
| --- | --- | --- |
| `python check.py` (4-4) | 세 줄이 나오고 첫 줄에 `cuda True`, 셋째 줄 sentence-transformers가 `6.1.0` 이상, 에러 없음 | 패키지를 깔고도 GPU용 torch가 그대로다 |
| `m.encode('test').shape` (4-5) | **`(768,)`** | 모델이 GPU에 올라가서 글 하나를 숫자 768개로 바꿨다 (원문 기대값) |

다음 중 하나라도 나오면 **미통과**다. 3단계로 넘어가지 않는다.

- `cuda False` 또는 `AssertionError: Jetson torch가 덮어쓰여졌는지 확인` → torch가 CPU 버전으로 바뀜 (6번 표)
- torch 버전이 `2.14.1+cu132`가 아님 → 고정이 안 먹음
- `(768,)`이 아닌 다른 모양, 또는 프로세스가 죽음

같이 적을 것(통과 조건은 아니지만 보고서에 쓴다): 모델 캐시 크기, 남은 디스크, 끝난 뒤 메모리, 센서 서비스 응답, 처음 로드에 걸린 시간.

## 2. 우리 보드에서는 원문과 이렇게 다르게 한다

| 원문 | 우리 보드에서 | 이유 |
| --- | --- | --- |
| `pip freeze ... > ~/eg2/torch-pin.txt`로 시작 | **다시 만들지 않는다.** 1단계에서 만든 파일 내용만 확인한다 | 이미 통과한 버전이 적혀 있다. 다시 만들면 혹시 바뀐 버전이 적힐 수 있다 |
| `~/eg2-lab` 폴더에서 `nano`로 코드 붙여넣기 | 코드 저장소를 `~/eg2-재웅`에 받고(`git clone`), 그 폴더에서 실행한다 | 팀 규칙 6번: 보드에서 코드를 직접 고치지 않는다. 코드는 저장소에 있다(`check.py`, `eg2.py`) |
| `python check.py` | `~/practice.sh 재웅 python check.py` | 팀 규칙 4번: 메모리 제한과 자물쇠 |
| 토큰은 `hf auth login`(선택) | 토큰을 쓴다면 `~/eg2/.env`에만 넣는다 (3번) | 팀 규칙 8번: 비밀값은 `.env`에만 |
| `HF_HUB_OFFLINE=1`을 `~/.bashrc`에 | 3단계에서 정한다 (이번엔 안 함) | 모델을 다 받고 3단계가 잘 돌면 그때 켠다 |

## 3. (선택) Hugging Face 동의와 토큰

> **보드 A 확인 결과(2026-10-23): `"gated":false`.** 동의·토큰 없이 받을 수 있어서 이 3번은 **하지 않아도 된다.** 다운로드가 속도 제한으로 자꾸 끊길 때만 한다.

### 3-1. 언제 필요한가

모델 파일은 Hugging Face라는 모델 저장소에서 받는다. 원문은 이 모델이 Apache 2.0(누구나 쓸 수 있는 라이선스)이라 **토큰 없이도 받을 수 있다**고 하지만, 모델에 따라 "사용 동의(Agree)"를 눌러야만 받을 수 있는 경우(gated 모델)가 있다. 그래서 4-2에서 먼저 확인하고, gated일 때만 아래가 꼭 필요하다. 토큰이 없어도 되더라도 넣어 두면 다운로드 속도 제한을 덜 받는다(원문 Q8).

### 3-2. 노트북 브라우저에서 (보드 아님)

1. https://huggingface.co 에 로그인한다(계정이 없으면 가입).
2. https://huggingface.co/google/embeddinggemma-2 에 들어간다. 위쪽에 **"Agree and access repository"** 같은 동의 버튼이 보이면 내용을 읽고 누른다. 버튼이 없고 파일 목록(Files and versions)이 바로 보이면 동의가 필요 없는 모델이다.
3. https://huggingface.co/settings/tokens → **Create new token** → 종류 **Read** → 이름은 `team5-a-eg2`처럼 → 만들기. `hf_`로 시작하는 긴 글자가 나온다.
4. 이 토큰은 **비밀번호와 같다.** 대화창·단톡방·코드·커밋·캡처에 붙이지 않는다. 바로 다음 3-3에서 보드에만 넣는다.

### 3-3. 보드 A에 토큰 넣기 (자기 계정, 화면에 안 보이게)

아래 덩어리를 붙여 넣으면 "HF 토큰 붙여넣기" 다음에 커서가 멈춘다. 거기서 토큰을 붙여 넣고(화면에 안 보이는 게 정상) Enter를 누른다. 토큰이 명령 기록(history)에도 남지 않는다.

```bash
read -rsp "HF 토큰 붙여넣기(화면에 안 보임): " T; echo
printf 'HF_TOKEN=%s\n' "$T" > ~/eg2/.env; unset T
chmod 600 ~/eg2/.env                 # 나만 읽을 수 있게
ls -l ~/eg2/.env                     # -rw------- 이면 OK (내용은 열어 보지 않는다)
```

확인 (토큰 값이 아니라 **내 HF 아이디**만 나온다):

```bash
source ~/eg2/bin/activate
set -a; . ~/eg2/.env; set +a
hf auth whoami
```

- 토큰을 잘못 넣었으면 위 첫 덩어리를 다시 실행하면 덮어써진다.
- `cat ~/eg2/.env`처럼 내용을 화면에 띄우지 않는다. 화면을 캡처해서 보낼 때 토큰이 같이 찍힌다.
- 토큰이 새어 나간 것 같으면 3-2의 토큰 페이지에서 그 토큰을 **Delete** 하고 새로 만든다.

`eg2.py`는 `~/eg2/.env`가 있으면 거기서 `HF_TOKEN`을 알아서 읽는다(값은 출력하지 않는다). `hf download` 같은 터미널 명령은 위처럼 `set -a; . ~/eg2/.env; set +a`를 먼저 해 줘야 토큰을 쓴다.

## 4. 순서 (보드 A, 자기 계정)

오래 걸리는 다운로드가 있으니 tmux 안에서 한다.

```bash
tmux new -s eg2-재웅     # 이미 있으면: tmux attach -t eg2-재웅
```

### 4-1. 시작 전 확인 (읽기만)

**무엇을, 왜:** 다른 사람이 보드를 쓰는 중인지, 메모리·디스크가 충분한지, 1단계 결과(torch 고정 파일)가 그대로인지 본다. 모델은 1.5~3GB라 디스크를, 모델 로드는 메모리를 쓴다.

```bash
echo "== 누가 쓰는 중?"; cat /tmp/practice.who.* 2>/dev/null || echo "아무도 안 씀"
echo "== 메모리"; free -h                   # available 3GB 이상
echo "== 디스크"; df -h /                   # 10GB 이상 남았는지
echo "== torch 고정"; cat ~/eg2/torch-pin.txt
echo "== HF_HOME"; echo "$HF_HOME"
echo "== 센서 서비스"; curl -s -m 3 localhost:8080/health || echo "응답 없음"
```

- `torch-pin.txt`에 `torch==2.14.1+cu132`, `torchvision==0.29.1+cu132` 두 줄이 있어야 한다.
- 센서 서비스 "응답 없음"이면 여기서 멈추고 재웅에게 알린다.

### 4-2. 모델이 동의(gated)가 필요한지 보기

**무엇을, 왜:** 토큰 없이 받을 수 있는 모델인지 확인한다. 이 결과에 따라 3번(토큰)이 꼭 필요한지 정해진다.

```bash
curl -s -m 10 https://huggingface.co/api/models/google/embeddinggemma-2 | grep -o '"gated":[^,]*'
```

| 나온 것 | 뜻 |
| --- | --- |
| `"gated":false` | 동의·토큰 없이 받을 수 있다. 3번은 건너뛴다 (보드 A 2026-10-23 결과가 이것) |
| `"gated":"auto"` 또는 `"gated":"manual"` | 3-2에서 **동의 버튼을 누르고 토큰을 넣어야** 받을 수 있다 |
| 아무것도 안 나옴 | 네트워크 문제일 수 있다. 화면을 그대로 보낸다 |

### 4-3. 패키지 설치 (torch 고정)

**무엇을, 왜:** 모델을 쓰는 데 필요한 라이브러리를 깐다. `-c ~/eg2/torch-pin.txt`가 "torch는 이 버전 말고는 깔지 마"라는 뜻이다. 이게 없으면 pip가 PyPI의 다른 torch로 바꿔 버릴 수 있다.

- `sentence-transformers[image,audio,video]`: 임베딩 라이브러리 + 이미지·음성·동영상 처리 부품 (4·5단계에서 쓴다)
- `transformers`, `accelerate`: 모델 본체를 돌리는 라이브러리
- `soundfile`, `librosa`, `av`, `pillow`: 소리·동영상·이미지 파일을 읽는 도구
- `huggingface_hub`: 모델 다운로드 도구(`hf` 명령)

```bash
source ~/eg2/bin/activate
pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub
pip uninstall -y torchaudio 2>/dev/null || true             # 딸려 깔린 torchaudio 지우기 (아래 설명)
pip freeze | grep -E '^(torch|torchvision|torchaudio)=='   # torch·torchvision 두 줄만, 2.14.1+cu132·0.29.1+cu132 그대로인지
```

- **torchaudio가 다른 패키지에 딸려서 같이 깔린다**(보드 A 2026-10-23: `torchaudio 2.11.0`). `torch-pin.txt`에는 torch·torchvision만 적혀 있어서 torchaudio는 막지 못한다. 이 torchaudio는 PyPI 것이라 CUDA 버전이 달라 `check.py`에서 원문 Q7 오류(`AutoProcessor`)를 낸다. 그래서 설치 바로 다음 줄에서 지운다. 소리 파일은 soundfile·librosa로 읽으니 없어도 된다.
- 마지막 줄에 `torchaudio`가 아직 보이면 `pip uninstall -y torchaudio`를 한 번 더 한다.
- "Cannot install ... conflicting dependencies"가 나오면 고정한 torch와 맞지 않는 패키지가 있다는 뜻이다. **고정을 빼고 다시 하지 말고** 화면을 그대로 보낸다.

### 4-4. 코드 받기와 설치 확인

**무엇을, 왜:** 코드 저장소를 보드에 받는다(보드에서는 받기만 하고 고치지 않는다). 그리고 `check.py`로 "패키지를 깔고도 GPU용 torch가 살아 있는지"를 확인한다. 성공 기준 첫째 줄이다.

```bash
cd ~ && [ -d ~/eg2-재웅 ] || git clone https://github.com/AIP-1-5/AIGJ_01_012_embedding_gemma2.git ~/eg2-재웅
cd ~/eg2-재웅 && git pull
source ~/eg2/bin/activate
~/practice.sh 재웅 python check.py
```

기대 출력 (버전 숫자는 설치 시점에 따라 다를 수 있다):

```
torch 2.14.1+cu132 cuda True
transformers 5.x.x
sentence-transformers 6.x.x
```

### 4-5. 모델 받기와 `load()` 확인 (성공 기준)

**무엇을, 왜:** 모델 파일(1.5~3GB)을 미리 받아 두고, `eg2.py`의 `load()`로 텍스트 인코더만 GPU에 올려서 `"test"`라는 글을 임베딩해 본다. 결과 모양 `(768,)`은 "숫자 768개짜리 벡터 하나"라는 뜻이다. 이게 나오면 3단계 검색을 할 준비가 끝난다.

```bash
source ~/eg2/bin/activate
set -a; [ -f ~/eg2/.env ] && . ~/eg2/.env; set +a
df -h /; time hf download google/embeddinggemma-2      # 진행률이 보인다. 수 분 걸릴 수 있다
du -sh "$HF_HOME/hub"; df -h /
```

```bash
cd ~/eg2-재웅
time ~/practice.sh 재웅 python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"
# 기대값: (768,)
```

- 첫 로드는 수십 초 걸린다. 멈춘 것 같아도 기다린다.
- 도는 동안 다른 터미널에서 `free -h`를 한 번 보면 6단계 측정에 참고가 된다(선택).

### 4-6. 끝나고

```bash
curl -s localhost:8080/health; echo
free -h; df -h /
```

응답이 없으면 재웅에게 바로 알린다. 다 끝났으면 tmux 세션을 닫는다(`exit`).

## 5. 커밋하지 않는 것

`.gitignore`에 `.env`, `models/`, `media/`, `*.safetensors`, `*.pt` 등을 넣어 두었다. 모델 파일은 `HF_HOME`(`~/.cache/huggingface`)에 쌓이고 저장소 폴더 밖이라 원래 커밋될 일이 없다.

## 6. 자주 나는 오류

| 화면에 나온 것 | 원인 | 할 일 |
| --- | --- | --- |
| `cuda False`, `AssertionError: Jetson torch가 덮어쓰여졌는지 확인` | 설치 중 torch가 CPU 버전으로 바뀜 | `pip freeze \| grep torch`로 확인 → 1단계 5-3으로 torch 다시 설치 → 4-3을 `-c`를 붙여 다시 |
| `PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0` / `Could not import module 'AutoProcessor'` | PyPI torchaudio가 같이 깔림 (원문 Q7) | `pip uninstall -y torchaudio` 후 `check.py` 다시 |
| `401 Unauthorized`, `GatedRepoError`, `Access to model ... is restricted` | 동의를 안 눌렀거나 토큰이 없음/틀림 | 3-2 동의 → 3-3 토큰 → `hf auth whoami` 확인 → 다시 |
| `You are sending unauthenticated requests to the HF Hub` 후 멈춘 듯함 | 경고일 뿐, 다운로드 중 (원문 Q8) | 다른 창에서 `watch -n 5 du -sh $HF_HOME/hub`로 커지는지 본다 |
| `hf: command not found` | `huggingface_hub`가 안 깔렸거나 venv가 꺼짐 | `source ~/eg2/bin/activate` 후 4-3 다시 |
| `prompt_name`/`config_kwargs` 관련 TypeError | sentence-transformers 구버전 | `pip install -c ~/eg2/torch-pin.txt -U "sentence-transformers>=6.1.0"` |
| `CUDA out of memory`, `Killed` | 메모리 부족 | 누가 앱을 켜 두었는지(`cat /tmp/practice.who.*`), `free -h` 확인. 센서 서비스는 끄지 않는다 |
| 임베딩에 `nan` | float16을 씀 | `eg2.py`는 bfloat16이다. 코드를 고치지 않았는지 본다 |
| `No space left on device` | 디스크 부족 | 멈추고 팀에 알린다. 센서 수집기가 DB에 못 쓰게 된다 |

여기서 해결이 안 되면 화면을 **그대로** 복사해서 보내 주세요(토큰이 찍혀 있지 않은지 한 번 보고).

## 7. 결과

| 항목 | 보드 A |
| --- | --- |
| 실행한 사람 / 날짜 | |
| 모델 동의 필요 여부 (4-2 `gated`) | |
| HF 토큰 (`~/eg2/.env`, 권한 600) | (넣었음/안 넣음만 적는다. 값은 절대 적지 않는다) |
| 설치 뒤 torch 버전 (`pip freeze`) | |
| `check.py` 출력 | |
| transformers / sentence-transformers 버전 | |
| 모델 캐시 크기 (`du -sh $HF_HOME/hub`) | |
| 남은 디스크 (시작 전 → 후) | |
| `m.encode('test').shape` | |
| 첫 로드 시간 (`time`) | |
| 끝난 뒤 메모리 (`free -h` available) | |
| 센서 서비스 (`curl localhost:8080/health`) | |
| **성공 기준** | |

**왜 이렇게 나왔는지:** (실행 후 채운다)

**다음에 할 일:** 3단계 텍스트 테스트 (`test_text.py`, 질의/문서 프롬프트로 한국어 검색, 768/256/128차원 비교).
