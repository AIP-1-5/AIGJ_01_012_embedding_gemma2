# 7단계 2차: train.jsonl을 문서 단위로 나눠 연습 시험(dev)을 만든다. 설정은 이 dev로만 고른다(eval.jsonl은 마지막에 한 번).
# 같은 문서의 질의 3개는 한쪽에만 들어간다. 무작위 씨앗은 0으로 고정.
# 사용: python split_dev.py  → data/train2.jsonl (74문서), data/dev.jsonl (20문서)
import json
import random

rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8") if l.strip()]
docs = sorted({r["positive"] for r in rows})
dev_docs = set(random.Random(0).sample(docs, 20))
for name, keep in [("train2", lambda r: r["positive"] not in dev_docs), ("dev", lambda r: r["positive"] in dev_docs)]:
    part = [r for r in rows if keep(r)]
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(f"data/{name}.jsonl: 쌍 {len(part)}, 문서 {len({r['positive'] for r in part})}")
