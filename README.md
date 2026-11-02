# Jetson 카메라 응용 실습 — AIGJ_01_012 이재웅 (팀5)

Jetson Orin Nano Super 8GB와 USB 웹캠(Logitech C920)으로 한 **물체 인식(과제 2) · 행동 패턴 인식(과제 4) · 카메라 2대 거리 측정(과제 5)** 의 코드, 기능 시험 기록, 결과 보고서, 확인 문제 답안입니다.

> ### 출처와 범위를 먼저 밝힙니다
> - 이 저장소의 **측정값은 모두 EmbeddingGemma 2 실습 8~10단계**(웹캠 물체 검색, 사람 동작 인식, USB 카메라 2대 접근 감지)에서 2026-10-23~10-28에 잰 것입니다. 이 과제를 위해 새로 잰 값은 없고, 숫자·판정·날짜는 원래 보고서 그대로 옮겼습니다.
> - 원래 저장소: [AIP-1-5/AIGJ_01_012_embedding_gemma2](https://github.com/AIP-1-5/AIGJ_01_012_embedding_gemma2), 제출본: [AIP-1/embedding_gemma2-AIGJ_01_012](https://github.com/AIP-1/embedding_gemma2-AIGJ_01_012)
> - 그 실습은 이 과제의 제공 키트(T0~T4 기능 시험, TensorRT 벤치마크, 재실 인원, 정렬형 스테레오, 12시간 서비스)와 절차가 다릅니다. 그래서 **과제 1·3·6과 각 과제의 일부 항목은 하지 않았고, 표에 "미수행"으로 적었습니다.** 하지 않은 측정을 했다고 쓰거나 숫자를 채워 넣지 않았습니다.
> - 확인 문제 답안(과제 1~6 전부)은 측정이 아니라 이해를 묻는 문항이라 모두 썼습니다. 우리 실험에서 근거가 있는 문항은 그 숫자를 함께 적었습니다.

## 과제별 상태

| 과제 | 상태 | 한 것 (원래 단계) | 확인 기준 | 보고서 |
| :---: | --- | --- | --- | --- |
| 1. 카메라 연동과 기준선 | **미수행** | 헤드리스 보드를 노트북 브라우저로 보기(5번)만 함 | 판정 안 함 | [report §1](docs/report.md#과제-1-카메라-연동과-기준선) |
| 2. 물체 인식 | **일부 수행** (8단계) | 해결할 것 4번: COCO 밖 물체를 ① YOLO 그대로 ② 임베딩 재라벨 ③ YOLOE로 비교, 세 모드 12~13fps | 4개 중 0개 충족, 3개 미수행 | [report §2](docs/report.md#과제-2-물체-인식) |
| 3. 재실 인원 카운터 | **미수행** | — | — | [report §3](docs/report.md#과제-3-재실-인원-카운터) |
| 4. 행동 패턴 인식 | **수행** (9단계, 심화 B 포함) | raw vs 다수결, 행동 패턴 12회 시험, GRU 동작·행동 모델 | 4개 중 1개 충족 | [report §4](docs/report.md#과제-4-행동-패턴-인식) |
| 5. 거리 측정 | **심화 B만 수행** (10단계) | 방 좌표계 보정(`prox` 방식)으로 사람 바닥 좌표, 이동 타겟 TTC 알람. 기본(정렬형 스테레오 SGBM)은 안 함 | 기본 기준은 미수행 | [report §5](docs/report.md#과제-5-거리-측정-웹캠-2대) |
| 6. 응용 서비스 | **미수행** | 접근 알람 웹 화면까지만, MQTT·systemd·12시간 가동 없음 | — | [report §6](docs/report.md#과제-6-응용-서비스) |

**설계 변수 하나를 바꿔 가장 크게 달라진 것 (발표용):** 사람 위치를 **발목(카메라 1대)** 에서 **어깨 삼각측량(카메라 2대)** 으로 바꾸자, 책상에 앉은 사람(실제 0.81m)의 거리가 1.45m(오차 0.64m, ID 끊김)에서 0.67m(오차 0.14m)로 바뀌었습니다. 물체 인식에서는 재라벨 규칙에 "검출기 이름과 0.03 넘게 다를 때만 고친다"를 넣자 2차의 휴대폰→키보드 오교정이 사라지고 COCO 밖 물체 2개를 맞혔습니다.

## 제출물

| 제출물 (과제 3장) | 파일 | 상태 |
| --- | --- | --- |
| 기능 시험 기록 | [runs/test_log.md](runs/test_log.md) | T0~T4는 미수행. 대신 실제로 한 시험을 같은 형식(항목·조건·결과·판정)으로 적음 |
| 결과 보고서 | [docs/report.md](docs/report.md) | 과제 2·4·5 결과, 1·3·6 미수행 표시 |
| 확인 문제 답안 | [docs/answers.md](docs/answers.md) | 1-1 ~ 6-5 전부 |
| 응용 서비스 | — | **미수행** (가동 중인 서비스 없음) |
| 결과 형식 문서 | [docs/result_format.md](docs/result_format.md) | 실제 있는 HTTP 조회 주소만 문서화. MQTT 토픽은 없음 |
| 발표 | [docs/presentation/](docs/presentation/) | EmbeddingGemma 2 실습 1~10단계 전체 발표 자료(8~10단계는 뒷부분) |
| 원래 단계 보고서 | [docs/reports/](docs/reports/) | 08·09·10 원본 (성공 기준, 1·2·3차 기록, 판정, 원인) |

## 측정 조건 (모든 결과표 공통)

| 항목 | 과제 2 (8단계) | 과제 4 (9단계) | 과제 5 (10단계) |
| --- | --- | --- | --- |
| 보드 | Jetson Orin Nano Super 8GB, 보드 A | 보드 A | 보드 B (팀원과 함께) |
| 실행 환경 | 호스트 venv(`~/eg2`), 컨테이너 아님, JetPack 7.2.1, Python 3.12 | 같음 | 같음 |
| 전력 모드 | MAXN SUPER | MAXN SUPER | MAXN SUPER |
| 카메라 | C920 1대, 1280x720 | C920 1대, 1280x720 | C920 2대, 640x480, 자동 초점 끔 |
| 모델·엔진 | `yolo11n.pt` / `yoloe-11s-seg.pt` (PyTorch, TensorRT 아님) + EmbeddingGemma 2 | `yolo11n-pose.pt` (PyTorch) + GRU | `yolo11n-pose.pt`, `yoloe-11s-seg.pt` + EmbeddingGemma 2 |
| 조명 | 교실 실내 조명 (밝기 측정 안 함) | 같음 | 같음 |
| 메모리 제한 | 같은 보드의 공기 측정 서비스 보호를 위해 `MemoryMax=2500M` | 같음 | 같음 |

## 폴더

```
08_webcam_object_search/   과제 2 — app.py(--mode yolo/relabel/yoloe), eg2.py(EmbeddingGemma 2 로더)
09_pose_action/            과제 4 — pose_app.py, pose_model.py(53차원 특징·GRU), train_pose.py, wave_check.py, piece_check.py
10_two_camera_proximity/   과제 5 심화 B — stereo.py, calib_intrinsics.py, calib_extrinsics.py, proximity_app.py, calib/world_points.example.json
runs/test_log.md           기능 시험 기록
docs/report.md             결과 보고서
docs/answers.md            확인 문제 답안
docs/result_format.md      결과 형식 문서
docs/reports/              원래 8·9·10단계 보고서
docs/presentation/         발표 자료 PDF
```

- `10_two_camera_proximity/`의 코드는 실습 원문 README 10단계 코드를 옮긴 것입니다(10단계는 이 원문 코드로 실행). 바꾼 곳은 `half=True` 삭제 한 곳이고 파일 맨 위에 적었습니다.
- 원래 보고서 안의 명령·경로(`~/eg2-재웅`, `python app.py`)는 원래 저장소 기준이라 이 저장소의 폴더와 다를 수 있습니다.

## 실행 예

```bash
source ~/eg2/bin/activate && free -h        # available 3GB 이상일 때만
cd 08_webcam_object_search && ~/practice.sh 재웅 python app.py --mode relabel
cd 09_pose_action && ~/practice.sh 재웅 python pose_app.py --behavior --model models/action.pt
cd 10_two_camera_proximity && python calib_intrinsics.py --cam 0 && python calib_intrinsics.py --cam 2 \
  && python calib_extrinsics.py --cams 0 2 && python proximity_app.py --cams 0 2 --target "brown cardboard box"
```

패키지: 시스템 GPU용 torch 고정, `sentence-transformers`, `ultralytics`(`--no-deps`, 자동 설치 끔), `lap`, `fastapi`, `uvicorn`, `numpy==1.26.4`, `opencv-python-headless==4.11.0.86`. 설치 순서는 [08 보고서 4-2](docs/reports/08-webcam-search.md).

## 개인정보와 넣지 않은 것

- 비밀값(`.env`, HF 토큰), 모델(`*.pt`, `*.onnx`, `*.engine`), 녹화 데이터(`data/`), 사진·영상·소리는 넣지 않았습니다.
- 사람이 찍힌 사진·영상은 실습이 끝난 뒤 지웠습니다. 그래서 과제 2의 "오인식 사례 사진 5장"과 과제 5의 "촬영 분포 사진"은 사진 대신 표로 적었습니다.
- 9단계 앱은 사진을 저장하지 않고 관절 좌표 CSV만 녹화했으며, 그 CSV도 보고서 merge 뒤 지웠습니다.
