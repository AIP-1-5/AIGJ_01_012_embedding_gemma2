# 1단계 환경 준비 (SSD · 전원 모드 · 스왑 · `~/eg2` · Jetson용 PyTorch)

> 상태: **보드 A 통과 (2026-10-23)**, 보드 B 아직 안 함
> 원문: 실습 README의 "1단계 환경 준비"
> 팀 규칙: `AIP-1-5/docs_embedding_gemma2`의 README (특히 2·3·4·5·7·11번, 2026-10-23 `4901dbf` 기준)
> 우리 보드: A·B 모두 **JetPack 7.2.1 (Ubuntu 24.04, Python 3.12)**, 전원 모드 MAXN SUPER (팀 README 2번 "보드 A·B 정보")
> 보드 A 실제 확인(2026-10-23): `nv_tegra_release`가 **R39 (REVISION 2.1)**, `/usr/local/cuda` 없음, SSD로 부팅 중, `/swapfile` 8GB 있음, `~/practice.sh` 없어서 만듦. 결과는 7번 표

이 단계가 끝나면 각자 보드 계정에 GPU가 실제로 계산하는 파이썬 환경(`~/eg2`)이 생긴다. 2단계부터 모든 단계가 이 환경 위에서 돈다.

## 1. 성공 기준 (먼저 적어 둔다)

아래 GPU 확인 명령(5-4)을 실행했을 때, **출력(두 줄)에 다음 세 가지가 모두** 나오면 통과다.

| 확인할 것 | 기대값 | 뜻 |
| --- | --- | --- |
| `torch.cuda.get_device_capability()` | `(8, 7)` | Orin의 GPU(sm_87)를 torch가 알아봤다 |
| `torch.cuda.is_bf16_supported()` | `True` | 이 실습에서 쓰는 bfloat16 계산이 된다 |
| `(x@x).sum().item()` | 숫자 하나 (2억 6천만 근처, 예: `268400000.0`. 매번 조금씩 다르다) | GPU에서 실제 행렬 곱셈이 돌았다 |

다음 중 하나라도 나오면 **미통과**다. 2단계로 넘어가지 않는다.

- `no kernel image is available` 또는 `sm_87` 관련 경고 → 보드의 JetPack과 맞지 않는 torch가 깔린 것
- `False` 또는 `Torch not compiled with CUDA enabled` → CPU 전용 torch가 깔린 것
- 숫자가 안 나오고 프로세스가 죽음 → 메모리 부족 (다른 사람이 앱을 켜 두었는지 확인)

같이 확인할 것(통과 조건은 아니지만 보고서에 적는다):

- 실습 후 보드 A에서 `curl -s localhost:8080/health`가 응답한다 (공기 측정 서비스가 살아 있다)
- `free -h`의 Swap 줄에 추가한 스왑이 보인다

## 2. 우리 보드에서는 원문과 이렇게 다르게 한다

두 보드에는 공기 측정 프로젝트가 **11월 5일까지** 24시간 돌고 있다. 그래서 원문 1단계를 그대로 따라 하면 안 되는 곳이 있다.

| 원문 | 우리 보드에서 | 이유 |
| --- | --- | --- |
| SSD를 `parted`·`mkfs`로 포맷 | **먼저 확인만 한다.** 이미 쓰고 있는 SSD면 절대 포맷하지 않는다 | 포맷하면 SSD 안의 데이터가 전부 지워진다. 공기 측정 데이터가 있을 수 있다 |
| "재부팅 후 `df -h /ssd`로 확인" | 재부팅하지 않고 `sudo findmnt --verify`로 확인 | 보드 재부팅 금지 |
| `nvpmodel -m 0`, `jetson_clocks` | **바꾸지 않는다.** `nvpmodel -q`로 MAXN SUPER인지 확인만 한다 | 팀 규칙 11번. 이미 MAXN SUPER로 맞춰 두었고, 바꾸면 센서 서비스 측정값도 달라진다 |
| 스왑 8GB 추가 | 이미 있는지 먼저 보고, `/etc/fstab`에 같은 줄을 두 번 넣지 않는다 | 다른 사람이 이미 했을 수 있다 |
| sudo 명령을 각자 실행 | **보드마다 한 사람(sudo 담당)이 관리 계정(보드 A `team5`, 보드 B `team5-b`)으로 한 번만** 실행한다 | 팀 규칙 7번. 명령은 아래 4번에 덩어리로 정리했다. 관리 계정 비밀번호는 단톡방에서 받는다 |
| JetPack 6·7 둘 다 안내 | 우리 보드는 **JetPack 7 경로**를 쓴다 | 두 보드 모두 JetPack 7.2.1이다. 3번 확인에서 다르게 나오면 그때 JetPack 6 경로 |
| GPU 확인을 그냥 `python`으로 | `~/practice.sh 이름 python ...`로 실행 | 팀 규칙 4번. 메모리 제한과 자물쇠가 걸린다 |
| HF 캐시를 `/ssd/hf` 하나로 | 계정마다 `/ssd/계정이름/hf` (SSD가 없으면 홈 폴더). 받기 전에 `df -h`로 10GB 이상 남았는지 본다 | 한 폴더를 여러 계정이 같이 쓰면 권한 오류가 난다. 디스크가 차면 센서 수집기가 DB에 못 쓴다(팀 규칙 11번) |
| 아무 계정으로 접속 | **자기 개인 계정**으로 접속. 관리 계정(`team5`, `team5-b`)으로 하지 않는다 | 팀 규칙 5번. 홈 폴더(`~/eg2`)가 사람마다 따로 생긴다 |

하지 않는 것: `sudo apt upgrade`, `sudo apt autoremove`, 시스템 PyTorch 재설치, `sudo pip`·`pip install --user`, 전원 모드·클럭 바꾸기, 보드 재부팅, 센서 수집기·tmux `inference` 세션 건드리기.

## 3. 시작 전 확인 (누구나, 읽기만 한다)

보드에 **자기 개인 계정**으로 SSH 접속해서 아래를 **그대로 복사해 붙여 넣는다**. 아무것도 바꾸지 않는 명령만 있다. 출력 화면을 통째로 팀(또는 Claude)에게 보내 주세요.

```bash
echo "== 누가 쓰는 중?"; cat /tmp/practice.who.* 2>/dev/null || echo "아무도 안 씀"
echo "== 메모리";        free -h            # available 3GB 이상이어야 시작
echo "== 디스크";        lsblk -f; df -h / /ssd 2>/dev/null   # 10GB 아래면 팀에 먼저 말하기
echo "== fstab";         grep -v '^#' /etc/fstab
echo "== 스왑";          swapon --show
echo "== JetPack";       cat /etc/nv_tegra_release; lsb_release -rs; python3 -V
echo "== CUDA";          ls -d /usr/local/cuda* 2>/dev/null; nvcc --version 2>/dev/null | tail -1
echo "== 센서 서비스(보드 A)"; curl -s -m 3 localhost:8080/health || echo "응답 없음(보드 B면 정상)"
```

출력을 보고 아래 표로 갈 길을 정한다.

| 본 것 | 판단 |
| --- | --- |
| `nv_tegra_release`가 `R38` 또는 `R39`, `lsb_release`가 24.04, `python3 -V`가 3.12 | **JetPack 7** (우리 보드의 기대값. 보드 A는 R39) → 5-3에서 JetPack 7 명령 |
| `R36`, 22.04, 3.10 | JetPack 6 → 5-3에서 JetPack 6 명령. 팀 README의 보드 정보와 다르니 재웅에게 알린다 |
| `R` 번호가 처음 보는 값 | **`python3 -V`로 판단한다.** 3.12면 JetPack 7 경로, 3.10이면 JetPack 6 경로. 원문은 R36·R38만 적었지만 JetPack 7.2는 R39로 나온다 |
| `/usr/local/cuda`가 없고 `nvcc`도 없음 | 문제 아님. JetPack 7 torch 휠(cu132)은 필요한 CUDA 라이브러리를 pip로 같이 받는다. CUDA 툴킷 설치(`apt install nvidia-jetpack`, 수 GB)는 하지 않는다. 판단 기준은 5-4 GPU 확인 결과다 |
| `nvme0n1`의 MOUNTPOINT가 `/` | SSD로 부팅 중. 4-A(SSD)는 건너뛴다. 스왑은 `/swapfile`, 캐시는 홈 폴더 |
| `nvme0n1p1`이 `/ssd`에 마운트돼 있음 | 이미 준비됨. 4-A는 건너뛴다 |
| `nvme0n1p1`에 FSTYPE(ext4 등)이 있는데 마운트 안 됨 | **포맷 금지.** 4-A의 "마운트만" 명령을 쓴다. 안에 뭐가 있는지 먼저 본다 |
| `nvme0n1`에 파티션이 아예 없음 | 새 SSD. 재웅에게 확인받은 뒤에만 4-A의 포맷 명령을 쓴다 |
| `nvme0n1`이 아예 안 보임 | SSD 없음. 4-A 건너뛰고 스왑은 `/swapfile`, 캐시는 홈 폴더 |
| `swapon --show`에 8G짜리 파일이 이미 있음 | 4-C(스왑)는 건너뛴다 (`zram`만 있으면 아직 안 한 것) |
| 센서 서비스 "응답 없음"(보드 A) | **여기서 멈춘다.** 실습 전에 재웅에게 알린다 |
| `free -h`의 available이 3GB 아래 | 누가 앱을 켜 두었는지 본다. 5-4(GPU 확인)는 미룬다 |
| `df -h /`(또는 `/ssd`)의 남은 공간이 10GB 아래 | 5-3(torch 받기) 전에 팀에 말한다 |

## 4. sudo 담당이 한 번만 하는 일 (보드마다 한 사람)

> 이 부분은 sudo 비밀번호가 필요하다. Claude는 실행하지 않는다. sudo 담당이 관리 계정(보드 A `team5`, 보드 B `team5-b`)으로 접속해 직접 붙여 넣고, 끝나면 관리 계정에서 나온다(실습은 개인 계정으로).
> 시작 전에 팀에 "지금 보드 ○ sudo 작업 한다"고 말한다.

### 4-A. SSD 마운트 (3번 표에서 필요하다고 나온 경우만)

**마운트만** (이미 포맷된 SSD, 대부분 이쪽):

```bash
sudo mkdir -p /ssd
UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p1); echo "UUID=$UUID"
grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /ssd ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo findmnt --verify          # 'Success' 또는 경고 0개면 OK. 에러가 나오면 멈추고 화면을 보낸다
sudo mount -a
df -h /ssd
ls /ssd                        # 안에 원래 있던 파일이 보이면 절대 지우지 않는다
```

**새 SSD 포맷** (파티션이 아예 없고, 재웅이 OK한 경우만. SSD 내용이 전부 지워진다):

```bash
lsblk /dev/nvme0n1             # 다시 한 번: 파티션(nvme0n1p1)이 없어야 한다
sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L ssd /dev/nvme0n1p1
```

포맷한 뒤에는 바로 위 "마운트만" 덩어리를 실행한다.

**계정마다 SSD 폴더 만들기** (마운트가 끝났으면). `계정1 계정2`를 이 보드를 쓰는 두 사람의 보드 계정 이름으로 바꾼다(계정 이름은 `AIP-1-5/docs`의 `environment.md`에 있다):

```bash
for U in 계정1 계정2; do sudo install -d -o "$U" -g "$U" "/ssd/$U"; done
ls -l /ssd
```

### 4-B. 전원 모드 (확인만, 바꾸지 않는다)

팀 규칙 11번에 따라 전원 모드와 클럭은 **바꾸지 않는다**. 원문의 `nvpmodel -m 0`, `jetson_clocks`는 실행하지 않는다.

```bash
sudo nvpmodel -q               # 'MAXN_SUPER'가 나오면 OK
```

MAXN SUPER가 아니면 고치지 말고 출력 화면을 재웅에게 보낸다. 모드가 낮아도 실습은 되고 속도만 느리다. 보고서 7번 표에 적는다.

### 4-C. 스왑 8GB 추가 (3번 표에서 아직 없다고 나온 경우만)

SSD가 `/ssd`에 있으면 첫 줄 그대로, SSD가 없거나 SSD로 부팅 중이면 `SWAP=/swapfile`로 바꾼다.

```bash
SWAP=/ssd/swapfile
free -h; df -h "$(dirname $SWAP)"   # 빈 공간이 10GB 이상인지 확인
sudo fallocate -l 8G $SWAP && sudo chmod 600 $SWAP
sudo mkswap $SWAP && sudo swapon $SWAP
grep -q "^$SWAP " /etc/fstab || echo "$SWAP none swap sw 0 0" | sudo tee -a /etc/fstab
sudo findmnt --verify
swapon --show; free -h
```

### 4-D. 시스템 패키지

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-full ffmpeg libsndfile1 libopenblas-dev
```

`apt upgrade`나 `apt autoremove`는 하지 않는다. 설치 중 "다음 패키지를 제거합니다"가 나오면 `n`을 누르고 화면을 보낸다.

sudo 작업이 끝나면 3번 확인 덩어리를 한 번 더 실행해서 SSD·스왑이 보이는지, 보드 A면 센서 서비스가 응답하는지 확인한다.

## 5. 각자 자기 계정에서 하는 일 (sudo 없음)

이 부분은 보드를 쓰는 사람이 **각자** 자기 계정에서 한다. 내려받는 양이 있으니 tmux 안에서 한다.

```bash
tmux new -s eg2-재웅           # '재웅' 자리에 자기 이름. 끊기면: tmux attach -t eg2-재웅
```

> tmux 세션 이름에는 자기 이름을 붙인다(팀 규칙 11번). `inference` 세션에는 붙지도, 그 이름을 쓰지도 않는다.

### 5-1. 모델 캐시 위치 정하기

`/ssd/내계정`이 있으면 SSD에, 없으면 홈 폴더에 둔다. 아래 덩어리가 알아서 고른다.

```bash
if [ -d /ssd/$USER ]; then HF=/ssd/$USER/hf; else HF=$HOME/.cache/huggingface; fi
mkdir -p "$HF"
grep -q 'export HF_HOME=' ~/.bashrc || echo "export HF_HOME=$HF" >> ~/.bashrc
source ~/.bashrc; echo "HF_HOME=$HF_HOME"
```

### 5-2. 가상환경 `~/eg2`

```bash
python3 -m venv ~/eg2 && source ~/eg2/bin/activate
which pip                      # /home/내계정/eg2/bin/pip 이어야 한다
pip install -U pip wheel
```

공기 측정 프로젝트 폴더와 그 `.venv`에는 들어가지 않는다. 이 실습은 항상 `source ~/eg2/bin/activate`로 시작한다. pip는 venv를 켠 다음에만 쓴다(`sudo pip`, `pip install --user` 금지).

자기 코드 저장소는 venv와 따로 홈에 받는다(예: `~/eg2-재웅`). 1단계에서는 아직 필요 없다.

### 5-3. Jetson용 PyTorch 설치

3번에서 확인한 JetPack 버전에 맞는 **한 덩어리만** 실행한다. torch 휠은 수백 MB라 받기 전에 `df -h ~`를 한 번 더 본다. 그냥 `pip install torch`는 CPU 전용이라 안 된다.

**JetPack 7 (R38·R39, Python 3.12) — 우리 보드 A·B는 이것**:

```bash
source ~/eg2/bin/activate
python -m pip install --pre torch torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu132
```

- torchaudio는 따로 설치하지 않는다. PyPI의 torchaudio는 CUDA 버전이 달라서 2단계에서 `AutoProcessor` 오류를 낸다(원문 Q7). 오디오는 2단계에서 soundfile·librosa로 읽는다.

**JetPack 6 (R36, Python 3.10) — 3번 확인에서 R36이 나온 경우만**:

```bash
source ~/eg2/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

### 5-4. GPU 확인 (성공 기준)

`재웅` 자리에 자기 이름을 쓴다.

`~/practice.sh`가 있는지 먼저 본다: `ls -l ~/practice.sh`. 없다고 나오면(보드 A는 2026-10-23 기준 아직 없음) 팀 문서 저장소 README 4번의 "처음 한 번만 만들기" 덩어리를 **자기 개인 계정에서** 붙여 넣는다. 계정마다 홈이 따로라 사람마다 한 번씩 만든다. 자물쇠(`/tmp/practice.lock`)는 보드 전체가 같이 쓰니 계정이 달라도 서로 막힌다.

```bash
source ~/eg2/bin/activate
~/practice.sh 재웅 python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported()); x=torch.rand(1024,1024,device='cuda'); print((x@x).sum().item())"
```

- "지금 ○○ 사용 중이에요"가 나오면 그 사람이 끝날 때까지 기다렸다가 다시 한다.
- 결과는 두 줄이다. 첫 줄 `2.x.x (8, 7) True`처럼(앞의 torch 버전 숫자는 설치 시점에 따라 다르다), 둘째 줄에 숫자 하나. 1번 성공 기준과 비교한다.

### 5-5. torch 버전 고정 (통과했을 때만)

2단계에서 다른 패키지를 깔다가 torch가 CPU 버전으로 덮어써지는 일이 잦다. 통과한 지금의 버전을 파일로 적어 둔다. 2단계부터는 `pip install -c ~/eg2/torch-pin.txt ...`로 설치한다.

```bash
source ~/eg2/bin/activate
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > ~/eg2/torch-pin.txt
cat ~/eg2/torch-pin.txt
```

### 5-6. 끝나고 (보드 A)

```bash
curl -s localhost:8080/health; echo
free -h; df -h / /ssd 2>/dev/null
```

응답이 없으면 재웅에게 바로 알린다.

보드 A·B 모두, 다 끝났으면 자기 tmux 세션을 닫는다(`exit` 또는 `tmux kill-session -t eg2-재웅`).

## 6. 자주 나는 오류

| 화면에 나온 것 | 원인 | 할 일 |
| --- | --- | --- |
| `from versions: none` | Python 버전과 torch 인덱스가 안 맞음 (JetPack 6 인덱스는 3.10 전용이라 우리 보드의 3.12에서는 후보가 없다) | `python -V` 확인 후 맞는 5-3 덩어리로 다시 |
| `externally-managed-environment` | venv가 켜지지 않은 채 시스템 pip를 씀 | `source ~/eg2/bin/activate` 후 다시. `--break-system-packages`는 쓰지 않는다 |
| `python3 -m venv`가 실패 | `python3-venv`가 없음 | sudo 담당에게 4-D를 부탁 |
| `PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0` | PyPI torchaudio가 깔림 | `pip uninstall -y torchaudio` (원문 Q7) |
| `nvcc: command not found` | CUDA 툴킷이 PATH에 없거나 아예 없음 (보드 A는 `/usr/local/cuda` 자체가 없다) | torch는 nvcc 없이도 된다. 툴킷을 설치하지 않는다. JetPack 판단은 `python3 -V`와 `nv_tegra_release`로 한다 |
| `no kernel image` / `sm_87` 경고 | JetPack에 안 맞는 휠 | `pip uninstall -y torch torchvision torchaudio` 후 맞는 5-3 덩어리로 다시 |
| `curl -sI` 결과가 `302` | 인덱스 서버의 정상 넘겨주기 | 문제 아님. pip는 알아서 따라간다 |
| `Failed to connect to bus` (practice.sh) | SSH 세션에 사용자 systemd가 없음 | 화면을 그대로 보낸다. 그냥 python으로 우회하지 않는다 |

여기서 해결이 안 되면 화면을 **그대로** 복사해서 보내 주세요.

## 7. 결과

| 항목 | 보드 A | 보드 B |
| --- | --- | --- |
| 실행한 사람 / 날짜 | 재웅 / 2026-10-23 | |
| JetPack (`nv_tegra_release`) / OS / Python | R39 (REVISION 2.1) / Ubuntu 24.04 / 3.12.3 | |
| SSD | SSD로 부팅 중 (4-A 건너뜀) | |
| 전원 모드 (`nvpmodel -q`, 확인만) | **확인 못 함** (sudo 필요). 팀 README 기준 MAXN SUPER | |
| 스왑 (`swapon --show`) | `/swapfile` 8G 있음 (4-C 건너뜀) | |
| CUDA 툴킷 | `/usr/local/cuda` 없음 (설치하지 않음) | |
| `HF_HOME` | `~/.cache/huggingface` (SSD 부팅이라 홈 폴더) | |
| `~/practice.sh` | 없어서 팀 README 4번대로 만듦 | |
| torch 버전 | torch `2.14.1+cu132`, torchvision `0.29.1+cu132` (torchaudio 없음) | |
| GPU 확인 출력 (`practice.sh` 경유) | `2.14.1+cu132 (8, 7) True` / `268482720.0` | |
| `~/eg2/torch-pin.txt` | 작성함 | |
| 남은 디스크 (`df -h`, 시작 전 → 후) | 188G → 170G | |
| 끝난 뒤 메모리 (`free -h` available) | 5.3Gi | |
| 센서 서비스 (`curl localhost:8080/health`) | ok, 실패 0 | 해당 없음 |
| **성공 기준** | **통과** | |

**왜 이렇게 나왔는지:**

- `(8, 7)`과 `True`가 나오고 행렬 곱셈 결과 숫자가 나왔다. JetPack 7(R39, Python 3.12)에 맞는 cu132 torch 휠이 Orin GPU(sm_87)에서 실제로 돌았다는 뜻이다. `/usr/local/cuda`가 없어도 휠이 받아 온 CUDA 라이브러리로 충분했다.
- 숫자 `268482720.0`은 1024x1024 난수 행렬 곱의 합으로, 기대한 크기(약 2억 6천만)와 맞다.
- SSD 부팅과 8G 스왑이 이미 되어 있어서 sudo 작업(4-A, 4-C)은 필요 없었다. 전원 모드는 sudo가 있어야 볼 수 있어서 이번에는 확인하지 못했다.
- 디스크가 약 18G 줄었다. torch와 의존 라이브러리(CUDA 포함)가 큰 편이다. 2단계 모델(1.5~3GB)까지 받아도 여유는 충분하다.

**참고 (실습과 별개):** 보드 A가 09:45 KST에 재부팅된 뒤 추론 서비스가 자동으로 켜지지 않아 재웅이 직접 다시 켰다. 끝난 뒤 `/health`는 정상이었다. 재부팅 뒤 자동으로 켜지게 하는 건 공기 측정 프로젝트 쪽에서 볼 일이다.

**다음에 할 일:**

- 보드 B도 같은 순서로 하고 오른쪽 칸을 채운다.
- 관리 계정이 있는 사람이 `sudo nvpmodel -q`로 보드 A 전원 모드를 한 번 확인한다.
- 2단계(패키지·모델 설치)는 `-c ~/eg2/torch-pin.txt`를 붙여 설치한다. 모델을 계정마다 받을지, 한 사람이 공용 폴더(`/srv/team5/models/`)에 받을지 팀 규칙 11번대로 2단계 전에 정한다.
