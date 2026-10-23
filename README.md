# AIGJ_01_012_embedding_gemma2

팀5 재웅의 EmbeddingGemma 2 실습 코드와 단계별 보고서. 실행은 보드 A(`~/eg2-재웅`에 clone, `~/eg2` venv)에서 `~/practice.sh`로 한다.

| 파일 | 내용 |
| --- | --- |
| `check.py` | 2단계 설치 확인 (GPU용 torch가 그대로인지) |
| `eg2.py` | 공통 로더 `load(vision=False, audio=False)`. 3~5단계가 같이 쓴다 |
| `test_text.py` | 3단계 한국어 텍스트 검색, 768/256/128차원 비교 (프롬프트 없는 변형 과제도 같이) |
| `cam.py` | 4단계 자료 찍기: 웹캠 사진(`snap 이름`), 동영상+2초 프레임(`clip 초`). OpenCV 사용 |
| `live.py` | 웹캠 실시간 보기(브라우저, 포트 8001)와 버튼으로 사진 저장 |
| `test_vision.py` | 4단계 텍스트로 사진·동영상 검색 (사진 이름 `empty`·`class`·`board`로 1위 자동 판정, 동영상 실패 시 프레임 대체) |
| `make_audio.py` | 5단계 소리 만들기: 경보음·잡음(numpy), 합성 음성(espeak-ng 있으면), `raw_*`를 16 kHz 모노로 변환 |
| `test_audio.py` | 5단계 텍스트로 소리 검색 (파일 이름 `speech`·`beep`·`noise`로 1위 자동 판정, `all`이면 사진+소리 한 인덱스) |
| `bench.py` | 6단계 성능 측정: 모양별(`text`·`vision`·`audio`·`all`) 로드 후 RAM·지연·GPU 사용률, `table`로 원문 표 출력 |
| `docs/reports/NN-*.md` | 단계별 안내와 결과 보고서 |
