"""9-2 2차 진단: --record로 녹화한 손 흔들기 CSV를 다시 읽어, pose_app.py의 is_wave가 왜 WAVE를 못 잡았는지 숫자로 본다.
학습 아님(파이썬 csv만 씀, torch 안 씀). 사진 없음.
사용: ~/practice.sh 재웅 /home/jaeung/eg2/bin/python wave_check.py data/action/wave.csv
보는 것:
  - fps, 손목이 어깨 기준선 위(한팔들기/만세)로 잡힌 프레임 비율, 올린 손목의 관절 신뢰도
  - 1.5초 창마다 손목 x(어깨 너비 단위)의 좌우 폭(최대-최소)
  - 흔드는 폭 기준(원문 0.15)을 0.15/0.12/0.10/0.08로 바꿨을 때 WAVE로 판정되는 창의 비율
"""
import csv, sys, collections

KP_CONF, LW, RW = 0.5, 9, 10
WAVE_S = 1.5
XY, CF = 5, 5 + 34          # 행: 시각, ID, 규칙 라벨, 정면, 속도, 정규화 좌표 34, 신뢰도 17


def is_wave(xs, amp):
    """pose_app.py is_wave와 같은 로직, 폭 기준만 바꿀 수 있게"""
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > amp:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2


rows = collections.defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.reader(f):
        if len(r) != 5 + 51:
            continue
        rows[r[1]].append([float(r[0]), r[2]] + [float(v) for v in r[5:]])

amps = (0.15, 0.12, 0.10, 0.08)
n_all = n_raised = 0
confs, spans, wins = [], [], collections.Counter()
n_win = 0
for tid, rs in rows.items():
    rs.sort(key=lambda r: r[0])
    dur = rs[-1][0] - rs[0][0]
    print(f"ID {tid}: {len(rs)}프레임, {dur:.1f}초, {len(rs) / max(dur, 1e-3):.1f} fps")
    hist = []                                     # (시각, 올린 손목 x) — pose_app의 st["wave"]와 같음
    for r in rs:
        t, raw, v = r[0], r[1], r[2:]
        n_all += 1
        if raw not in ("한팔들기", "만세"):
            continue
        n_raised += 1
        w = LW if v[2 * LW + 1] < v[2 * RW + 1] else RW
        confs.append(v[34 + w])
        hist.append((t, v[2 * w]))
    # 1.5초 창을 0.5초씩 밀며 검사
    if not hist:
        continue
    t = hist[0][0] + WAVE_S
    while t <= hist[-1][0]:
        xs = [x for s, x in hist if t - WAVE_S < s <= t]
        if xs:
            n_win += 1
            spans.append(max(xs) - min(xs))
            for a in amps:
                wins[a] += is_wave(xs, a)
        t += 0.5


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


print(f"\n전체 {n_all}프레임 중 손을 든 프레임(한팔들기/만세) {n_raised} ({n_raised / max(n_all, 1):.0%})")
print(f"올린 손목 신뢰도: 중앙값 {q(confs, .5):.2f}, 하위 10% {q(confs, .1):.2f} (기준 {KP_CONF})")
print(f"1.5초 창 {n_win}개, 손목 좌우 폭(어깨 너비 단위): 중앙값 {q(spans, .5):.2f}, 하위 25% {q(spans, .25):.2f}, 상위 25% {q(spans, .75):.2f}")
for a in amps:
    print(f"  폭 기준 {a:.2f}{' (원문)' if a == 0.15 else ''}: WAVE 판정 창 {wins[a]}/{n_win} ({wins[a] / max(n_win, 1):.0%})")
