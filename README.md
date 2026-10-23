# AIGJ_01_012_embedding_gemma2

팀5 재웅의 EmbeddingGemma 2 실습 코드와 단계별 보고서. 실행은 보드 A(`~/eg2-재웅`에 clone, `~/eg2` venv)에서 `~/practice.sh`로 한다.

| 파일 | 내용 |
| --- | --- |
| `check.py` | 2단계 설치 확인 (GPU용 torch가 그대로인지) |
| `eg2.py` | 공통 로더 `load(vision=False, audio=False)`. 3~5단계가 같이 쓴다 |
| `test_text.py` | 3단계 한국어 텍스트 검색, 768/256/128차원 비교 (프롬프트 없는 변형 과제도 같이) |
| `docs/reports/NN-*.md` | 단계별 안내와 결과 보고서 |
