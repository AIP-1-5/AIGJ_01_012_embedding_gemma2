# 8단계 웹캠 실시간 물체 추출과 웹 검색

> 상태: **보드 A 1차 미통과 (2026-10-23 16:29 KST)**. 기준 2(표 빈칸)를 못 채웠고 기준 5(relabel fps)는 숫자를 못 받아 판정 못 함. 라벨 개선(기준 3·4)은 충족. 2차 계획은 7번(실행 전에 적음)
> 원문: 실습 README의 "8단계 웹캠 실시간 물체 추출과 웹 검색"
> 앞 단계: [04-vision-search.md](04-vision-search.md)(글로 사진 찾기, 비전 인코더), [06-performance.md](06-performance.md)(텍스트+비전 도는 중 available 최저 2.2GB), [07-finetune.md](07-finetune.md)
> 우리 보드: 보드 A, USB 웹캠 1대, 모니터 없음 → 영상·검색은 **노트북 브라우저**(`http://100.113.228.46:8001`)로 본다.

## 0. 이 단계에서 무엇을 하나

4단계에서는 미리 찍어 둔 사진을 글로 찾았다. 8단계는 이걸 **실시간**으로 한다.

1. 웹캠 영상에서 **YOLO**(물체 검출기)가 물체마다 네모 박스를 친다.
2. 박스 안을 잘라 낸 작은 사진(크롭)을 EmbeddingGemma 2가 숫자 512개(벡터)로 바꿔 메모리에 쌓는다.
3. 브라우저에서 "건전지"처럼 글로 검색하면, 쌓인 크롭 중 가장 가까운 것을 보여 준다.

같은 물체를 세 가지 방식으로 비교한다.

| 순서 | 모드 | 무엇을 확인하나 |
| --- | --- | --- |
| 8-1 문제 확인 | `yolo` | YOLO는 배운 80종(COCO) 중 하나를 꼭 고른다. 그래서 세탁기를 `oven`, 전지 포장을 `cell phone`처럼 엉뚱하게 부른다 |
| 8-2 개선 1 | `relabel` | 박스 위치는 YOLO 것을 쓰고, **이름은 EmbeddingGemma 2가 다시 붙인다**(크롭 벡터와 "a photo of a 세탁기" 같은 문장 벡터의 유사도). 추가 학습 없음 |
| 8-3 개선 2 | `yoloe` | 검출기 자체를 **글로 찾을 물체를 정하는 YOLOE**로 바꾼다. YOLO가 박스조차 못 잡던 물체도 잡히는지 본다 |

핵심 개념: **이름(라벨)과 검색은 서로 다른 길이다.** 라벨은 검출기가 붙이고, 검색은 크롭 벡터끼리 비교한다. 그래서 `yolo` 모드에서 라벨이 틀려도 "세탁기" 검색은 맞을 수 있다. 반대로 박스가 아예 안 잡히면 크롭이 없어서 검색도 안 된다.

### 7단계 결과가 필요한가? → 필요 없다

원문 `app.py`에 `MODEL_ID = "google/embeddinggemma-2"  # 7단계 모델은 텍스트 전용이라 여기서는 원본 사용`이라고 되어 있다. 7단계 미세조정 모델은 텍스트 부분만 저장돼(542MB, [07-finetune.md](07-finetune.md)) 사진을 못 읽는다. 그래서 8단계는 7단계 통과 여부와 상관없이 원본 모델을 쓰고, 7단계와 **병렬로 준비**할 수 있다. 우리 `app.py`는 `EG2_MODEL` 환경 변수가 남아 있어도 원본을 쓰도록 고정했다.

보드 실행 순서만 겹친다: 보드에서는 한 번에 한 명·한 앱만 켜므로, 7단계 보드 평가가 끝난 뒤 8단계를 켠다.

## 1. 성공 기준 (실행 전에 적어 둔다)

원문 8단계에는 "성공 기준" 표가 따로 없다. 원문 8-1~8-4의 절차, "기대" 열, "확인 포인트"를 그대로 기준으로 옮긴다. 결과를 본 뒤 기준을 바꾸지 않는다.

| # | 기준 | 원문 근거 |
| --- | --- | --- |
| 1 | 세 모드 모두 켜지고, 브라우저 제목 옆에 `[mode: yolo/relabel/yoloe]`가 보이며 실시간 영상·텍스트 검색이 동작한다 | 8-1 2번, "세 모드 모두 … 텍스트 검색은 항상 동작" |
| 2 | 8-4의 두 표(물체 6개 × 3모드 라벨, 모드별 fps·임베딩 ms·RAM)를 빈칸 없이 채운다. 박스가 안 잡히면 "미검출"이라고 쓴다 | 8-1 3번, 8-4 |
| 3 | COCO 물체(컵·노트북·휴대폰)는 `relabel`에서도 맞는 이름이다 | 8-2 "COCO 물체는 여전히 맞는지" |
| 4 | COCO 밖 물체(드럼 세탁기·원통형 전지 포장·건전지) 중 `yolo`에서 틀렸거나 미검출인 것이 `relabel` 또는 `yoloe`에서 1개 이상 맞는 이름이 된다 | 테스트 물체 표 "yolo에서 오인식, 개선 후 정답" |
| 5 | `relabel` 모드 검출 fps 10 이상 | 확인 포인트 "fps가 10 미만이면 해상도를 낮춘다" |
| 6 | 메모리로 멈추지 않고(종료 코드 4 없음), 끝난 뒤 센서 서비스 `/health` ok | 팀 규칙 |

- 판정은 사람 눈으로 본 영상 박스의 이름(영상)과 웹 카드의 이름(점수)을 표에 적어서 한다.
- 정리(원문): 세 방식의 정확도·속도 차이와 강의실 서비스에 어떤 방식을 쓸지 한두 문장.
- 미통과면 1차 기록은 그대로 두고, 원인과 바꿀 것을 먼저 적은 뒤 2차를 한다. 예: fps 10 미만이면 원문대로 `SIZE=640x480`(또는 `REFRESH_S=60`)으로 2차. `LABEL_MIN` 조정과 `LABELS` 추가는 원문 8-2 3·4번이 하라고 한 절차라, 조정했으면 값과 이유를 적는다.

## 2. 우리 보드에서는 원문과 이렇게 다르게 한다

| 원문 | 우리 보드에서 | 이유 |
| --- | --- | --- |
| `~/eg2-lab`에서 `nano app.py` | 저장소 `app.py`를 `git pull`로 받는다 (`~/eg2-재웅`) | 팀 규칙 6번 (보드에서 코드 안 고침) |
| `python app.py --mode ...` | `~/practice.sh 재웅 python app.py --mode ...` | 자물쇠·2500MB 제한. 카메라는 한 번에 한 프로그램 |
| 포트 8000, `0.0.0.0`, `ufw allow 8000` | 포트 **8001**, **Tailscale 주소(100.113.228.46)에만** 연다. ufw 안 건드림 | 팀 문서 2번 표(보드 A 재웅 = 8001). 학교망 전체에 웹캠을 열지 않음(`live.py`와 같은 원칙) |
| `sudo apt install v4l-utils`로 웹캠 형식 확인 | 안 해도 된다(선택). 웹캠은 4~5단계에서 OpenCV로 이미 열었다 | sudo는 사용자가 직접. 꼭 보고 싶으면 사용자가 설치 |
| `pip install -c torch-pin ultralytics fastapi uvicorn` | ultralytics를 **`--no-deps`**로 깔고, 필요한 것만 numpy·opencv 고정과 같이 깐다(3-2) | ultralytics는 `opencv-python`(화면 있는 판)을 끌고 와서 이미 깔린 `opencv-python-headless 4.11`과 부딪히고, numpy 2를 끌어올 수 있다(2단계 보고서 숙제) |
| (자동 설치) | `YOLO_AUTOINSTALL=False`를 코드에 넣음 | ultralytics는 모자란 패키지를 실행 중에 몰래 `pip install`한다. 그러면 torch 고정이 깨질 수 있다 |
| 크롭 8개씩 임베딩 | 기본 **2개씩** (`BATCH=8`로 원문대로 바꿀 수 있음) | 6단계에서 텍스트+비전이 도는 중 2.2GB까지 내려갔고, 여기에 YOLO가 더 올라간다. 임베딩 ms/개 칸에 배치 크기를 같이 적는다 |
| (메모리 설정 없음) | 시작 전 available 3GB 미만이면 안 켬, 도는 중 1GB 미만이면 바로 끝냄, GPU 몫 0.3, 오디오 인코더 안 올림 | 팀 메모리 원칙, 6단계와 같은 기준 |
| 크롭을 `crops/<mode>/`에 계속 쌓음 | **끝날 때 그 모드의 `crops/`를 지운다**(`--keep-crops`로 남길 수 있음). `crops/`는 `.gitignore` | 사람이 찍힌 사진은 남기지 않는 팀 원칙. `person`은 원문대로 크롭 안 하지만, 다른 물체 크롭 뒤로 얼굴이 걸릴 수 있다 |
| RAM은 tegrastats로 | tegrastats 로그 + 화면의 available·최저 available | 6단계처럼 "도는 중 최저"가 더 중요해서 |
| 해상도 1280x720 고정 | 기본 1280x720(원문), `SIZE=640x480`으로 바꿀 수 있게 | 확인 포인트의 fps 대책을 코드 수정 없이 |

검출·재라벨·검색 로직, `LABELS`, `LABEL_MIN`, 모드별 conf, `REFRESH_S`, 웹 화면은 원문 그대로다.

## 3. 준비물

| 무엇 | 상태 / 할 일 |
| --- | --- |
| 웹캠 | 보드 A에 꽂힌 USB 웹캠(C920, `/dev/video0`). 팀원(yj, suyeon)과 번갈아 쓴다 → 시작 전 `cat /tmp/practice.who.*` |
| `live.py` | 같은 포트(8001)·같은 카메라를 쓴다. **꺼져 있어야 한다** |
| 테스트 물체 (재웅) | COCO 안: 컵, 노트북, 휴대폰. COCO 밖: 원통형 전지 포장(18650 등 포장), 건전지(AA), 드럼 세탁기 **사진**(종이로 뽑거나 태블릿에 띄움. 노트북·휴대폰 화면에 띄우면 그 자체가 "노트북·휴대폰"으로 잡혀 헷갈린다) |
| 놓는 법 | 세 모드 모두 **같은 위치·같은 조명**, 한 번에 하나씩 5초 이상. 책상 위에 놓고 사람은 화면 밖으로 빠진다(사람 얼굴이 크롭에 안 걸리게) |
| 새 패키지 | `ultralytics`, `fastapi`, `uvicorn[standard]` (+ ultralytics가 쓰는 작은 것들, 3-2) |
| 새로 받는 파일 | `yolo11n.pt`(약 5MB), `yoloe-11s-seg.pt`(수십 MB)와 YOLOE 텍스트 인코더. 첫 실행 때 GitHub에서 자동으로 받는다(보드 A github 200 확인됨). 커밋 안 됨(`*.pt`, `mobileclip*.ts`) |
| 메모리 | 시작 전 available **3GB 이상**. 6단계 기준 텍스트+비전만으로 도는 중 2.2GB까지 내려갔으니, YOLO(작음)·YOLOE(조금 더 큼)를 더하면 2GB 안팎까지 내려갈 수 있다(추정). 다른 계정(yj 등) 프로그램이 돌면 그만큼 줄어든다 |
| 노트북 | Tailscale 켜고 브라우저로 `http://100.113.228.46:8001` |

## 4. 순서 (보드 A, `ssh jaeung@100.113.228.46`)

### 4-1. 시작 전 확인 (읽기만)

**무엇을, 왜:** 카메라·자물쇠를 누가 쓰는지, 8001 포트가 비었는지, 메모리가 3GB 이상인지, 센서 서비스가 살아 있는지 본다.

```bash
echo "== 누가 쓰는 중?"; cat /tmp/practice.who.* 2>/dev/null || echo "아무도 안 씀"
echo "== 8001 포트"; ss -ltn | grep -q ':8001 ' && echo "8001 사용 중 (live.py가 켜져 있는지 확인)" || echo "8001 비어 있음"
echo "== 내 live.py"; pgrep -u "$USER" -af 'live.py|app.py' || echo "없음"
echo "== 메모리"; free -h                     # available 3GB 이상
echo "== 센서 서비스"; curl -s -m 3 localhost:8080/health || echo "응답 없음"
echo "== 웹캠"; ls /dev/video*
```

- 누가 쓰는 중이거나, 8001이 사용 중이거나, available 3GB 미만이거나, 센서 "응답 없음"이면 **여기서 멈추고** 화면을 보낸다.

### 4-2. 코드 받기와 설치 (한 번만)

**무엇을, 왜:** `app.py`를 받고, YOLO·웹 서버 패키지를 깐다. torch·numpy·opencv 버전이 바뀌지 않게 고정을 같이 건다.

```bash
cd ~/eg2-재웅 && git pull && git log -1 --oneline   # 보고서에 적을 커밋
source ~/eg2/bin/activate
pip install -c ~/eg2/torch-pin.txt --no-deps ultralytics ultralytics-thop
pip install -c ~/eg2/torch-pin.txt "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
  fastapi "uvicorn[standard]" matplotlib pyyaml requests scipy psutil polars pillow tqdm
pip uninstall -y opencv-python 2>/dev/null; pip freeze | grep -E '^(torch|torchvision|torchaudio|numpy|opencv.*|ultralytics)=='
python -c "import cv2, numpy, torch, ultralytics, fastapi; print(cv2.__version__, numpy.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

- 기대: torch `2.14.1+cu132`, torchvision `0.29.1+cu132`, numpy `1.26.4`, opencv-python-headless `4.11.0.86`, `cuda True`. torchaudio 줄이 보이면 `pip uninstall -y torchaudio`.
- `ImportError: ... No module named X`가 나오면 `pip install -c ~/eg2/torch-pin.txt "numpy==1.26.4" X`로 그것만 깔고 다시 확인한다(화면을 보내 주면 목록에 반영).
- YOLOE 텍스트 인코더가 `clip` 패키지를 요구하면(8-3 첫 실행 때 `No module named 'clip'`): `pip install -c ~/eg2/torch-pin.txt --no-deps "git+https://github.com/ultralytics/CLIP.git" ftfy regex` (추정, 보드에서 확인).

### 4-3. 8-1 기준선 (`yolo`)

**무엇을, 왜:** YOLO만 쓰면 COCO 밖 물체가 엉뚱한 이름이 되는 "문제"를 먼저 눈으로 확인한다. 이게 있어야 8-2·8-3이 무엇을 고쳤는지 비교할 수 있다.

tmux 창 두 개를 쓴다. 왼쪽은 앱, 오른쪽은 메모리 기록.

```bash
tmux new -s eg2-8
cd ~/eg2-재웅 && source ~/eg2/bin/activate
tegrastats --interval 1000 --logfile ~/eg2-재웅/media/tegra-8-yolo.log &   # RAM 기록 (media/는 커밋 안 됨)
~/practice.sh 재웅 python app.py --mode yolo
# 기대: 시작 전 available ... MB → [mode=yolo] conf=0.5 size=1280x720 batch=2 → 브라우저에서 http://100.113.228.46:8001
```

1. 노트북 브라우저 `http://100.113.228.46:8001`에서 `[mode: yolo]` 확인.
2. 물체를 하나씩 5초 이상 보여 주고, **영상 박스의 이름**을 5번 표 yolo 열에 적는다. 박스가 없으면 "미검출".
3. 검색창에 "세탁기", "건전지"를 넣고 결과 카드의 YOLO 라벨·사진을 본다(라벨은 틀려도 검색은 맞을 수 있다).
4. 화면 위쪽 숫자(fps, 임베딩 ms/개, available, 최저)를 적는다.
5. Ctrl+C로 끈다. `crops/yolo 지움`, `도는 중 available 최저 ... MB`가 찍힌다. 이어서 `kill %1`(tegrastats 끄기).

### 4-4. 8-2 개선 1 (`relabel`)

```bash
tegrastats --interval 1000 --logfile ~/eg2-재웅/media/tegra-8-relabel.log &
~/practice.sh 재웅 python app.py --mode relabel
```

- 같은 물체를 같은 순서로. **주황색 박스**가 이름이 바뀐 것이다. 웹 카드의 "이름(점수)"과 원래 YOLO 이름을 적는다.
- 맞는 이름의 점수를 보고 `LABEL_MIN`(0.20)을 조정할지 정한다(원문 8-2 3번). 바꾸려면 노트북에서 고쳐 PR → `git pull`. 바꾼 값과 이유를 보고서에 적는다.
- 끝나면 Ctrl+C, `kill %1`.

### 4-5. 8-3 개선 2 (`yoloe`)

```bash
tegrastats --interval 1000 --logfile ~/eg2-재웅/media/tegra-8-yoloe.log &
~/practice.sh 재웅 python app.py --mode yoloe
# 오검출이 많으면: ~/practice.sh 재웅 python app.py --mode yoloe --conf 0.35
```

- 8-1·8-2에서 "미검출"이던 물체에 박스가 생기는지, fps가 8-2보다 얼마나 떨어지는지 본다.
- 첫 실행은 가중치·텍스트 인코더를 받느라 오래 걸린다. 오류가 나면 화면을 그대로 보낸다(원문: ultralytics 버전에 따라 파일명·API가 다름).
- 끝나면 Ctrl+C, `kill %1`.

### 4-6. 끝나고

```bash
for m in yolo relabel yoloe; do echo "== $m"; grep -o 'RAM [0-9]*/[0-9]*MB' ~/eg2-재웅/media/tegra-8-$m.log | sort -t' ' -k2 -n | tail -1; done   # 모드별 RAM 최대
curl -s localhost:8080/health; echo
free -h
ls ~/eg2-재웅/crops 2>/dev/null   # 비어 있어야 한다 (남았으면 rm -r ~/eg2-재웅/crops)
```

모드별 화면 숫자, 표에 적은 이름, 위 결과를 보내 주면 5번을 채운다.

## 5. 자주 나는 오류

| 화면에 나온 것 | 원인 | 할 일 |
| --- | --- | --- |
| `available ... < 3000 MB: 켜지 않는다` | 메모리 부족 | `cat /tmp/practice.who.*`, 다른 계정 프로그램이 끝나길 기다린다 |
| `/dev/video0을 열 수 없다` | `live.py`·다른 팀원 앱이 카메라를 잡음 | 4-1을 다시 본다. 그래도 안 되면 `CAM=1 ~/practice.sh 재웅 python app.py ...` |
| `address already in use` | 8001을 다른 프로그램이 씀 | `live.py`를 끈다 (4-1) |
| `available ... < 1000 MB: 바로 끝낸다` (종료 코드 4) | 도는 중 메모리 부족 | 화면을 보낸다. 2차는 `SIZE=640x480`, `BATCH=1` 순으로 줄인다 |
| fps가 10 미만 | 1280x720 검출이 무거움 | 원문대로 `SIZE=640x480 ~/practice.sh 재웅 python app.py --mode relabel` (2차로 기록) |
| `No module named ...` | `--no-deps`로 빠진 패키지 | 4-2의 안내대로 그것만 설치 |
| `get_text_pe` 관련 오류 | ultralytics 버전 차이 | 코드가 이름만 넘기는 방식으로 다시 시도한다. 그래도 나면 화면을 보낸다 |
| 브라우저가 안 열림 | 노트북이 Tailscale에 없음 | 노트북 Tailscale을 켠다 |

## 6. 1차 결과 (8-4, 2026-10-23 16:29 KST, 보드 A, main `aa97889`)

실행은 보드 A 담당이 했고, 물체를 보여 주고 라벨을 눈으로 확인한 건 재웅이다. 세 모드 모두 끝난 뒤 `crops/`는 지워졌고 센서 ok, available 4.6GB.

| 물체 | yolo 라벨 | relabel 라벨 (점수) | yoloe 라벨 | 비고 |
| --- | --- | --- | --- | --- |
| 컵 | 맞음 | 맞음 (점수 못 받음) | 맞음 | |
| 노트북 | 맞음 | 맞음 (점수 못 받음) | 맞음 | |
| 휴대폰 | 맞음 | 맞음 (점수 못 받음) | 맞음 | |
| 드럼 세탁기 (사진) | 미검출 | 드럼 세탁기 (점수 못 받음) | 박스는 생김, 이름 휴대폰·키보드 | 사진을 **휴대폰·모니터 화면에** 띄웠다(3번 준비물과 다름). yoloe는 세탁기보다 그 화면을 잡았다 |
| 원통형 전지 포장 | 미검출 | 미검출 | 미검출 | |
| 건전지 | 미검출 | 건전지 (점수 못 받음) | 미검출 | |

"맞음"은 재웅이 영상 박스 이름을 보고 확인한 것이다. relabel 카드의 점수는 기록하지 못했다.

| 모드 | 검출 fps | 임베딩 ms/개 (배치 2) | RAM (MB, tegrastats 최대) | 도는 중 available 최저 |
| --- | --- | --- | --- | --- |
| yolo | 못 받음 | 못 받음 | 5785 | 1639 MB |
| relabel | 못 받음 | 못 받음 | 5227 | 2086 MB |
| yoloe | 12.8 | 699 | 5297 | 2027 MB |

| # | 기준 | 결과 | |
| --- | --- | --- | --- |
| 1 | 세 모드 실행·검색 동작 | 세 모드 다 켜지고 영상·카드는 나왔다. "세탁기"·"건전지" 검색 결과는 기록이 없다 | 일부 확인 |
| 2 | 표 빈칸 없음 | fps·ms 4칸, relabel 점수 6칸이 비었다 | **미충족** |
| 3 | COCO 물체 relabel에서도 정답 | 컵·노트북·휴대폰 맞음 | 충족 |
| 4 | COCO 밖 물체 1개 이상 개선 | 건전지·세탁기: yolo 미검출 → relabel 정답 | 충족 (세탁기는 화면에 띄운 사진이라 참고만, 건전지만으로도 충족) |
| 5 | relabel fps ≥ 10 | 숫자를 못 받았다 | **판정 못 함** |
| 6 | 메모리 정지 없음, 센서 ok | 종료 코드 4 없음, 센서 ok. 다만 **끌 때마다(세 번 모두) 멈춰서 강제 종료**했다(메모리 문제 아님, 이후 카메라·자물쇠 정상) | 충족 |
| | **판정** | | **미통과** (기준 2 미충족, 5 판정 못 함) |

**왜 이렇게 나왔는지:**

- **숫자를 못 받은 건 측정 방법 문제다.** fps·ms는 브라우저 화면 위쪽에만 나왔고 터미널에는 안 찍혔다. 화면을 보던 사람이 모드마다 숫자를 적어야 했는데, 물체를 보여 주느라 yoloe 때만 적었다. relabel 점수도 웹 카드에만 있었다.
- **yolo에서 미검출이던 건전지·세탁기가 relabel에서 맞은 이유.** 두 모드는 검출기(yolo11n)가 같지만 relabel은 conf를 0.5 → 0.35로 낮춘다(원문 설정). 그래서 애매한 박스가 생겼고, 그 크롭에 EmbeddingGemma 2가 맞는 이름을 붙였다. 원문 8-2가 말한 "박스를 더 확보 + 이름 바로잡기"가 그대로 일어났다.
- **전지 포장은 세 모드 모두 미검출.** 박스가 아예 없어서 재라벨할 크롭도 없었다. yoloe도 못 잡은 건 포장이 작거나 "battery pack"이라는 말과 겉모습이 달랐을 수 있다(추정).
- **yoloe의 세탁기가 휴대폰·키보드가 된 이유.** 세탁기 사진을 휴대폰·모니터 화면에 띄웠다. yoloe는 `LABELS`에 있는 "cell phone"·"keyboard"·"monitor"도 같이 찾으므로, 사진 속 세탁기보다 그걸 담은 기기를 먼저 잡았다. 3번 준비물에 "화면에 띄우면 그 자체가 노트북·휴대폰으로 잡혀 헷갈린다"고 적어 둔 경우다.
- **끌 때 멈춘 이유(추정).** `/stream`이 끝없이 영상을 보내는 연결이라, Ctrl+C 뒤 웹 서버(uvicorn)가 "연결이 끝나길" 계속 기다렸다. 카메라 스레드도 멈추라는 신호 없이 `cap.read()`를 계속 돌았다.
- 메모리: 도는 중 available 최저는 yolo 1.6GB, relabel 2.1GB, yoloe 2.0GB로 6단계 비전 모양(2.2GB)과 비슷하거나 조금 낮았다. 1GB 정지선에는 안 갔다. yolo 모드가 가장 낮았던 건 그때 다른 계정 프로그램이 같이 돌았을 수 있다(확인 안 함).

## 7. 2차 재측정 (실행 전에 적음, 2026-10-23)

### 7-1. 1차와 똑같이 두는 것

성공 기준(1번), `LABELS`, `LABEL_MIN` 0.20, 모드별 conf(0.5/0.35/0.25), 해상도 1280x720, 배치 2, 물체 6개와 보여 주는 순서, 검출·재라벨·검색 코드. 1차 기록(6번)은 지우지 않는다.

### 7-2. 바꾸는 것 (1차 원인 분석에 따라)

| 바꾸는 것 | 이유 |
| --- | --- |
| `app.py`가 10초마다 `[stats] fps=… embed_ms=…` 줄, 새 크롭마다 `[item] 검출기=… → 라벨=… (점수)` 줄, 끝날 때 `[summary] fps 평균 … 임베딩 평균 …` 줄을 **터미널에** 찍는다 | 1차에 화면 숫자를 못 받았다. 터미널 기록이면 보드 담당이 그대로 보낼 수 있다. 사진은 안 남기고 글자만 찍는다 |
| Ctrl+C 때 스트림·카메라 루프를 먼저 멈추고, 웹 서버는 2초만 기다린 뒤 끝낸다 | 1차에 끌 때마다 멈췄다 |
| 드럼 세탁기는 **종이에 뽑은 사진**으로 보여 준다(없으면 태블릿, 휴대폰·노트북 화면은 안 씀) | 3번 준비물대로. 화면 기기가 같이 잡히지 않게 |
| 모드마다 검색창에 "세탁기", "건전지"를 넣고 1위 카드의 이름·점수를 적는다 | 기준 1의 검색 확인 기록이 없었다 |

### 7-3. 순서

4-1(시작 전 확인) → `git pull` → 4-3·4-4·4-5를 같은 방법으로 한다. 실행 명령은 같다. 모드마다 터미널 출력 전체(특히 `[item]`, `[stats]`, `[summary]` 줄)와 검색 두 개의 1위 카드를 보낸다.

```bash
cd ~/eg2-재웅 && git pull && git log -1 --oneline
source ~/eg2/bin/activate
for m in yolo relabel yoloe; do
  echo "=== $m: 물체 6개를 순서대로 보여 주고, 검색 '세탁기'·'건전지'를 한 뒤 Ctrl+C"
  tegrastats --interval 1000 --logfile ~/eg2-재웅/media/tegra-8b-$m.log &
  ~/practice.sh 재웅 python app.py --mode $m 2>&1 | tee ~/eg2-재웅/media/app-8b-$m.txt
  kill %1
  grep -o 'RAM [0-9]*/[0-9]*MB' ~/eg2-재웅/media/tegra-8b-$m.log | sort -t' ' -k2 -n | tail -1
done
grep -h '^\[summary\]' ~/eg2-재웅/media/app-8b-*.txt
curl -s localhost:8080/health; echo; free -h; ls ~/eg2-재웅/crops 2>/dev/null
```

- Ctrl+C 뒤 2~3초 안에 `[summary]`가 찍히고 다음 모드로 넘어가야 한다. 10초 넘게 멈추면 화면을 보내고, 그때만 다른 창에서 `pkill -u $USER -f 'app.py'`.
- `media/app-8b-*.txt`는 글자 기록뿐이고 `media/`라 커밋되지 않는다. 보고서를 다 쓰면 지운다.
- 2차도 미통과면 원인과 바꿀 것을 여기에 먼저 적고 3차로 간다. 예: relabel fps가 10 미만이면 원문 확인 포인트대로 `SIZE=640x480`.
