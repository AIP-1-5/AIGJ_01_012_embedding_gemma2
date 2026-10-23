# 7단계 데이터 점검: 줄 수, 문서·질의 수, 학습/평가가 겹치지 않는지(같은 문장, 비슷한 문장)를 본다.
# 비슷함은 모델 없이 글자 3개씩 묶음(3-gram)의 겹침 비율(Jaccard)로 잰다. eval 점수는 보지 않는다.
# 사용: python check_data.py [train.jsonl] [eval.jsonl]
import hashlib
import json
import sys
from collections import Counter

TRAIN = sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl"
EVAL = sys.argv[2] if len(sys.argv) > 2 else "data/eval.jsonl"
LIMIT = 0.35   # 이 값 이상이면 "비슷한 문장"으로 표시한다 (데이터를 보기 전에 정함)

# 3단계 test_text.py의 문서·질의. 학습 데이터에 그대로 들어가면 3단계 재확인이 의미 없어진다.
STEP3 = ["환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
         "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
         "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
         "수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def grams(s):
    s = "".join(s.split())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def sim(a, b):
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(1, len(ga | gb))


tr, ev = rows(TRAIN), rows(EVAL)
ok = True
for name, rs, path in [("train", tr, TRAIN), ("eval", ev, EVAL)]:
    docs = Counter(r["positive"] for r in rs)
    qs = Counter(r["anchor"] for r in rs)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"{name}: 쌍 {len(rs)}, 문서 {len(docs)}, 질의 {len(qs)} (중복 질의 {sum(c > 1 for c in qs.values())}), "
          f"문서당 질의 {sorted(set(docs.values()))}, sha256 {sha[:16]}")
    ok &= all(c == 1 for c in qs.values())

tr_text = {r["positive"] for r in tr} | {r["anchor"] for r in tr}
ev_text = {r["positive"] for r in ev} | {r["anchor"] for r in ev}
same = tr_text & ev_text
print(f"train·eval 똑같은 문장: {len(same)}개")
ok &= not same


def worst(a_list, b_list, label):
    pairs = sorted(((sim(a, b), a, b) for a in a_list for b in b_list), reverse=True)
    hits = [p for p in pairs if p[0] >= LIMIT]
    print(f"{label}: 가장 비슷한 값 {pairs[0][0]:.2f}, 기준 {LIMIT} 이상 {len(hits)}개")
    for s, a, b in pairs[:3]:
        print(f"   {s:.2f}  {a[:40]} | {b[:40]}")
    return not hits


ok &= worst({r["positive"] for r in tr}, {r["positive"] for r in ev}, "train 문서 vs eval 문서")
ok &= worst({r["anchor"] for r in tr}, {r["anchor"] for r in ev}, "train 질의 vs eval 질의")
ok &= worst(tr_text, STEP3, "train vs 3단계 문장")
print("결과:", "겹침 없음 (사용 가능)" if ok else "겹침 있음 (고쳐야 함)")
