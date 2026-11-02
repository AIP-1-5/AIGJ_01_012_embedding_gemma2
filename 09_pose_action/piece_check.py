"""9-6 3차: 행동 녹화 CSV가 학습에 쓸 만큼 길게 잡혔는지 녹화 직후에 센다 (학습 아님, torch 안 씀, 사진 없음).
train_pose.py와 같은 방식으로 자른다: 파일 앞뒤 --trim초 버림 → 추적 ID별 → 0.5초 넘게 끊기면 다른 조각.
사용: ~/practice.sh 재웅 /home/jaeung/eg2/bin/python piece_check.py data/behavior_v3/passing.csv data/behavior_v3/glance.csv
"""
import argparse, csv, collections

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--trim", type=float, default=5.0)
args = ap.parse_args()

for path in args.files:
    by_tid = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) == 5 + 51:
                by_tid[r[1]].append(float(r[0]))
    t_all = [t for ts in by_tid.values() for t in ts]
    if not t_all:
        print(f"{path}: 0행")
        continue
    lo, hi = min(t_all) + args.trim, max(t_all) - args.trim
    pieces = []
    for ts in by_tid.values():
        ts = sorted(t for t in ts if lo <= t <= hi)
        if not ts:
            continue
        n = 1
        for t0, t1 in zip(ts, ts[1:]):
            if t1 - t0 > 0.5:
                pieces.append(n); n = 0
            n += 1
        pieces.append(n)
    pieces.sort(reverse=True)
    print(f"{path}: {len(t_all)}행, 추적 ID {len(by_tid)}개, 앞뒤 {args.trim:g}초 버린 뒤 조각 {len(pieces)}개, "
          f"40줄 이상 {sum(p >= 40 for p in pieces)}개, 80줄 이상 {sum(p >= 80 for p in pieces)}개, 긴 순 {pieces[:8]}")
