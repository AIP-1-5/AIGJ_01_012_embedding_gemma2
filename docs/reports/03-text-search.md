# 3단계 텍스트 테스트 (한국어 검색 · 768/256/128차원 비교)

> 상태: **보드 A 실행 전**
> 원문: 실습 README의 "3단계 텍스트 테스트"
> 앞 단계: [02-packages-model.md](02-packages-model.md) — 보드 A 통과 (`load()`로 `(768,)`, sentence-transformers 6.2.0, 모델 snapshot `914f7f89`)
> 결정(2026-10-23): 재웅은 10단계까지 **보드 A**만 쓴다.

## 0. 이 단계에서 무엇을 하나

2단계에서 글을 숫자 768개(벡터)로 바꾸는 모델을 올렸다. 3단계는 그 벡터로 **처음 진짜 검색**을 해 본다.

1. 문서 4개(CO2·에어컨·조명·Jetson 이야기)를 미리 벡터로 바꿔 둔다.
2. 한국어 질문 2개를 벡터로 바꾼다. 질문에는 문서에 있는 낱말이 거의 없다("졸리고 머리가 맑지 않아요" ↔ "CO2 농도 ... 집중력"). 그래서 **글자가 같은지가 아니라 뜻이 가까운지**로 찾아야 맞힐 수 있다.
3. 질문 벡터와 문서 벡터가 얼마나 같은 방향인지(코사인 유사도, 1에 가까울수록 비슷) 재서 1위 문서를 고른다.
4. 같은 일을 벡터 길이 **768 → 256 → 128**로 잘라서 다시 해 본다. EmbeddingGemma 2는 앞쪽 숫자에 중요한 뜻이 먼저 담기게 학습되어 있어서(Matryoshka), 잘라도 검색이 되는지가 이 단계의 두 번째 볼거리다. 짧으면 저장 공간과 비교 계산이 줄어든다.

질문에는 `SearchQuery`, 문서에는 `Document`라는 **프롬프트**(앞에 붙는 안내 문구)를 따로 붙인다. 모델이 "이건 찾는 질문", "이건 찾아질 문서"라고 구분하게 해 주는 것이라, 원문이 꼭 구분하라고 한다.

## 1. 성공 기준 (먼저 적어 둔다)

원문 "확인 포인트"를 그대로 기준으로 쓴다.

| 확인할 것 | 기대값 |
| --- | --- |
| 질의 1 "수업 중에 졸리고 머리가 맑지 않아요"의 1위 | **CO2 문장** (`환기가 부족하면 ...`) — 768·256·128 **모든 차원** |
| 질의 2 "여름에 냉방 몇 도로 맞춰야 하나요?"의 1위 | **에어컨 문장** (`에어컨 설정 온도는 ...`) — **모든 차원** |

- 6칸(질의 2개 × 차원 3개)이 모두 맞아야 **통과**다. 하나라도 틀리면 **미통과**로 적는다.
- 지연(ms)은 통과 조건이 아니다. 기록만 하고, 6단계 표의 "텍스트 전용" 행에 쓴다.
- 이 기준과 문서·질문은 원문 그대로다. 결과를 보고 문서·질문·기준을 바꾸지 않는다.

같이 적을 것(통과 조건 아님): 차원별 지연, 1위와 2위의 점수 차이, GPU 메모리 최대, 보드 메모리, 센서 서비스 응답, 변형 과제(`--no-prompt`) 결과.

## 2. 우리 보드에서는 원문과 이렇게 다르게 한다

| 원문 | 우리 보드에서 | 이유 |
| --- | --- | --- |
| `~/eg2-lab`에서 `nano test_text.py` | 저장소의 `test_text.py`를 `git pull`로 받는다 | 팀 규칙 6번: 보드에서 코드를 고치지 않는다 |
| `python test_text.py` | `~/practice.sh 재웅 python test_text.py` | 팀 규칙 4번: 자물쇠와 메모리 제한 |
| 모델을 온라인으로 확인하며 로드 | `HF_HUB_OFFLINE=1`을 켠다 (원문 Q&A의 `~/.bashrc` 한 줄) | 2단계에서 모델을 다 받았다. 켜면 로드 때 인터넷 확인을 안 해서 빠르고, `unauthenticated requests` 경고도 안 뜬다 |
| 첫 차원(768) 지연에 CUDA 초기화가 섞임 ("두 번째 차원부터 비교") | 루프 전에 **예열 1회**를 하고 세 차원 모두 비교한다 | 768도 공정하게 비교하려고. 예열은 결과(순위·점수)에 영향이 없다 |
| 1위 문서와 점수만 출력 | **PASS/FAIL**, 2위와의 점수 차이, 문서 4개 전체 점수, GPU 메모리 최대도 출력 | 기준 확인을 눈으로 세지 않게. 점수 차이가 작으면 "겨우 맞힌 것"이라 보고서에 쓸 거리가 된다 |
| 변형 과제: `prompt_name`을 빼고 돌려 보기 | `--no-prompt` 옵션으로 같은 코드에서 돌린다 | 코드를 고치지 않고 비교하려고 |
| `tegrastats` 켜 두기 | 선택. 켠다면 다른 창에서 `tegrastats --interval 1000` | 6단계 측정용. 이번 통과 여부와는 무관 |

이번 단계는 **pip 설치가 없다.** 그래서 torch 고정(`-c ~/eg2/torch-pin.txt`)이나 torchaudio 지우기는 할 일이 없고, 시작 전에 버전이 그대로인지만 본다.

## 3. 순서 (보드 A, 자기 계정)

### 3-1. 시작 전 확인 (읽기만)

**무엇을, 왜:** 누가 보드를 쓰는 중인지, 메모리가 넉넉한지, 2단계에서 맞춘 torch가 그대로인지, 센서 서비스가 살아 있는지 본다.

```bash
echo "== 누가 쓰는 중?"; cat /tmp/practice.who.* 2>/dev/null || echo "아무도 안 씀"
echo "== 메모리"; free -h                   # available 3GB 이상
source ~/eg2/bin/activate
echo "== torch"; pip freeze | grep -E '^(torch|torchvision|torchaudio)=='   # torch 2.14.1+cu132, torchvision 0.29.1+cu132 두 줄만
echo "== 센서 서비스"; curl -s -m 3 localhost:8080/health || echo "응답 없음"
```

- torchaudio 줄이 보이면 `pip uninstall -y torchaudio`를 하고 다시 본다.
- 센서 서비스 "응답 없음"이면 여기서 멈추고 재웅에게 알린다.

### 3-2. 오프라인 모드 켜기 (한 번만)

**무엇을, 왜:** 모델은 이미 `~/.cache/huggingface`에 있다. 이 설정을 켜면 앞으로 모델을 불러올 때 인터넷에 "새 버전 있나요?"를 묻지 않고 받아 둔 파일만 쓴다. 나중에 새 모델을 받아야 할 때만 `HF_HUB_OFFLINE=0 hf download ...`처럼 잠깐 끈다.

```bash
grep -q 'HF_HUB_OFFLINE=1' ~/.bashrc || echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
export HF_HUB_OFFLINE=1
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"      # 1 이면 OK
```

### 3-3. 코드 받기

```bash
cd ~/eg2-재웅 && git pull
git log -1 --oneline                       # 보고서에 적을 커밋
```

### 3-4. 텍스트 검색 실행 (성공 기준)

**무엇을, 왜:** 문서 4개와 질문 2개를 벡터로 바꿔, 768·256·128차원에서 각각 질문마다 1위 문서를 찾는다. 줄마다 `PASS`가 나오면 그 칸은 기준을 맞춘 것이다.

```bash
cd ~/eg2-재웅 && source ~/eg2/bin/activate
~/practice.sh 재웅 python test_text.py 2>&1 | tee /tmp/eg2-03-text.txt
```

출력 모양 (숫자는 예시가 아니라 자리만 표시):

```text
모델 로드 x.x s
프롬프트: 질의=SearchQuery, 문서=Document

[dim=768] xx ms
  수업 중에 졸리고 머리가 맑지 않아요 -> 환기가 부족하면 강의실 CO2 농도가 100... (0.xxx)  2위와 차이 0.xxx  PASS
    전체 점수: [0]0.xxx  [1]0.xxx  [2]0.xxx  [3]0.xxx
  여름에 냉방 몇 도로 맞춰야 하나요? -> 에어컨 설정 온도는 하계 26도, 동계 2... (0.xxx)  2위와 차이 0.xxx  PASS
    ...
[dim=256] xx ms
  ...
[dim=128] xx ms
  ...

GPU 메모리 최대 xxx MB
결과: 모든 차원에서 1위 정답 (통과)
```

- 도는 동안 다른 창에서 `free -h`를 한 번 보면 메모리 기록에 쓴다(선택).
- `프롬프트 'SearchQuery' 없음`이 나오면 화면을 그대로 보낸다(모델의 프롬프트 이름이 원문과 다르다는 뜻).

### 3-5. 변형 과제: 프롬프트 없이 (선택, 통과 조건 아님)

**무엇을, 왜:** `SearchQuery`/`Document` 안내 문구를 빼면 점수와 순위가 어떻게 달라지는지 본다. 프롬프트가 왜 필요한지 숫자로 확인하는 실험이다.

```bash
~/practice.sh 재웅 python test_text.py --no-prompt 2>&1 | tee /tmp/eg2-03-text-noprompt.txt
```

### 3-6. 끝나고

```bash
curl -s localhost:8080/health; echo
free -h
```

응답이 없으면 재웅에게 바로 알린다. `/tmp/eg2-03-text*.txt` 두 파일 내용을 그대로 보내 주면 아래 결과 표를 채운다.

## 4. 자주 나는 오류

| 화면에 나온 것 | 원인 | 할 일 |
| --- | --- | --- |
| `OfflineModeIsEnabled`, `couldn't connect` / `does not appear to have a file` | 오프라인인데 캐시에 필요한 파일이 없음 | `HF_HUB_OFFLINE=0 hf download google/embeddinggemma-2`로 다시 받고 재실행 |
| `프롬프트 'SearchQuery' 없음. 있는 것: [...]` | 모델의 프롬프트 이름이 원문과 다름 | 화면을 그대로 보낸다. 코드를 보드에서 고치지 않는다 |
| `prompt_name` 관련 TypeError | sentence-transformers 구버전 | 2단계 결과(6.2.0)와 같은지 `pip show sentence-transformers` |
| `cuda False`, CUDA 관련 오류 | torch가 바뀜 | 3-1의 `pip freeze` 화면을 보낸다 |
| `CUDA out of memory`, `Killed` | 메모리 부족 | `cat /tmp/practice.who.*`, `free -h` 확인. 센서 서비스는 끄지 않는다 |
| 점수가 `nan` | float16 사용 | `eg2.py`는 bfloat16이다. 코드를 고치지 않았는지 본다 |

## 5. 결과

_보드 A에서 실행한 뒤 채운다._

| 항목 | 보드 A |
| --- | --- |
| 실행한 사람 / 날짜 | |
| 실행한 코드 커밋 | |
| `HF_HUB_OFFLINE` | |
| 모델 로드 시간 | |
| 768차원: 질의1 1위 / 질의2 1위 / 지연 | |
| 256차원: 질의1 1위 / 질의2 1위 / 지연 | |
| 128차원: 질의1 1위 / 질의2 1위 / 지연 | |
| 1위와 2위 점수 차이 (가장 작은 칸) | |
| GPU 메모리 최대 | |
| 실행 중·후 보드 메모리 | |
| 센서 서비스 (`/health`) | |
| 변형 과제(`--no-prompt`) 1위 결과 | |
| **성공 기준** | |

**왜 이렇게 나왔는지:** _(실행 후)_

**다음에 할 일:** _(실행 후)_
