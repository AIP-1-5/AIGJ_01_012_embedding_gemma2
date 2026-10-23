# 9단계 사람 동작 인식 (만세·한팔들기·손 흔들기·악수, 행동 패턴, GRU 모델). 원문 README 9단계 pose_app.py를 우리 보드 규칙에 맞게 고친 것.
# 사용: ~/practice.sh 재웅 python pose_app.py [--raw | --behavior | --record data/action/wave.csv | --model models/action.pt]
#       → 브라우저에서 http://100.113.228.46:8001
# 원문과 다른 점 (docs/reports/09-pose.md 2번):
#   - 포트 8001, Tailscale 주소에만 연다. live.py·app.py(8단계)와 같은 포트·같은 카메라라 셋 중 하나만 켠다
#   - 메모리 감시: 시작 전 available 3GB 미만이면 안 켬, 도는 중 1GB 미만이면 바로 끝냄, GPU 몫 0.3
#   - ultralytics 자동 pip 설치 끔 (torch가 바뀌지 않게). 해상도는 SIZE=640x480으로 바꿀 수 있게
#   - 끝날 때 기록 CSV를 닫고 몇 줄 썼는지 알려 준다. 사진·영상은 원문처럼 저장하지 않는다
import os
os.environ["YOLO_AUTOINSTALL"] = "False"          # ultralytics가 몰래 pip install 하지 않게 (torch 고정 지키기)
import time, threading, collections, argparse, csv
import cv2, numpy as np, torch, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# ---------------- 실행 옵션 ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=int(os.environ.get("CAM", "0")))   # /dev/video0
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))   # 팀 문서 2번 표: 보드 A 재웅
ap.add_argument("--host", default=os.environ.get("HOST", "100.113.228.46"))        # Tailscale 주소에만
args = ap.parse_args()
MODE = "raw" if args.raw else "smooth"

W, H = (int(v) for v in os.environ.get("SIZE", "1280x720").split("x"))   # fps 10 미만이면 SIZE=640x480
START_MIN_MB, KILL_MIN_MB = 3000, 1000   # 팀 메모리 원칙: 시작 전 3GB, 도는 중 1GB
SMOOTH_S = 0.6        # 다수결 창(초)
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정 창(초)
FACE_TH = 0.6         # 정면으로 볼 facing 점수
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초)
ML_TH = 0.7           # 모델 확률 기준
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "..."}
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}

def mem_available_mb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return info["MemAvailable"] // 1024


avail0 = mem_available_mb()
print(f"시작 전 available {avail0} MB", flush=True)
if avail0 < START_MIN_MB:
    raise SystemExit(f"available {avail0} MB < {START_MIN_MB} MB: 켜지 않는다 (팀 메모리 원칙)")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)

det = YOLO("yolo11n-pose.pt")         # 첫 실행 시 자동 다운로드
models = [load_model(p) for p in args.model]
SEQ_LEN = max([m["window"] for m in models], default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} "
      f"models={[(m['task'], m['labels']) for m in models]} size={W}x{H}", flush=True)

frame_lock, ev_lock = threading.Lock(), threading.Lock()
latest_jpeg = None
tracks = {}
pairs = {}            # (tid_a, tid_b) -> deque[(t, bool)]
pair_on = set()
events = collections.deque(maxlen=50)
counts = collections.Counter()
beh_counts = collections.Counter()   # 사람(추적 ID) 단위 최종 행동
stats = {"fps": 0.0, "people": 0, "avail_mb": avail0, "min_avail_mb": avail0, "rec_rows": 0}
rec = None
if args.record:
    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    rec_f = open(args.record, "a", newline="")   # 같은 라벨을 여러 번 녹화하면 이어 붙는다 (원문 그대로)
    rec = csv.writer(rec_f)
    print(f"기록: {args.record} (관절 좌표만, 사진 없음)", flush=True)

def new_track(now):
    return {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
            "beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=SEQ_LEN),
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def is_wave(hist):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 2번 이상 방향을 바꾸면 손 흔들기"""
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.15:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2

def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴 판정"""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝에서부터 정면 연속 시간
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def log_event(now, who, label):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": label})
        counts[label] += 1

def close_track(tid, st, now):
    """추적이 끝난 사람의 요약을 남긴다"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1

def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            sw = (ba[1] + bb[1]) / 2
            best = 1e9
            for wa in (LW, RW):
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                            best = min(best, np.linalg.norm(a[wa, :2] - b[wb, :2]))
            key = tuple(sorted((ids[i], ids[j])))
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, best < 0.5 * sw))
            found.add(key)
    return found

def update_person(tid, k, now):
    st = tracks.setdefault(tid, new_track(now))
    dt = now - st["seen"]
    st["seen"] = now
    b = body(k)
    if b is None:
        return st
    raw = pose_label(k, b)
    fa = facing_score(k)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt
    st["n"] += 1
    if raw in ("한팔들기", "만세"):           # 손 흔들기용: 올라간 손목의 x (어깨 중심 기준, 어깨 너비 단위)
        w = LW if k[LW, 1] < k[RW, 1] else RW
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec:
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())
        stats["rec_rows"] += 1

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        if label == "한팔들기" and is_wave(recent):
            label = "손흔들기"

    # ---- 학습 모델 (5프레임마다) ----
    if models and st["n"] % 5 == 0:
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:
                lab, p = predict(m, seq[-m["window"]:])
                st["ml"][m["task"]] = (lab, p) if p >= ML_TH else None
    ml_act = st["ml"].get("action")
    if ml_act:
        label = "기본" if ml_act[0] == "idle" else ml_act[0]

    if label != st["label"]:
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
            st["best"] = bs
    return st

def capture_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print(f"/dev/video{args.cam}을 열 수 없다 (live.py·app.py 등 다른 프로그램이 쓰는 중인지 확인, 원문 Q11)", flush=True)
        os._exit(2)
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = det.track(frame, persist=True, conf=args.conf, half=True, verbose=False)[0]
        now = time.time()
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()
        stats["people"] = len(ids)

        for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
            st = update_person(tid, k, now)
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- 2인 동작 (악수) ----
        active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
        for key in list(pairs):
            if key not in active:
                pairs.pop(key); pair_on.discard(key); continue
            if MODE == "raw":
                on = pairs[key][-1][1]
            else:
                v, frac = majority(pairs[key], now, SMOOTH_S)
                on = bool(v) and frac >= 0.6
            if on and key not in pair_on:
                log_event(now, f"#{key[0]}+#{key[1]}", "악수")
            (pair_on.add if on else pair_on.discard)(key)
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # 3초 이상 안 보인 사람은 추적 종료 + 요약
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            close_track(tid, tracks.pop(tid), now)

        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

def mem_watch():
    # 도는 중 available이 1GB 아래로 내려가면 센서 서비스를 지키려고 바로 끝낸다 (6·8단계와 같은 기준)
    while True:
        a = mem_available_mb()
        stats["avail_mb"], stats["min_avail_mb"] = a, min(stats["min_avail_mb"], a)
        if a < KILL_MIN_MB:
            print(f"available {a} MB < {KILL_MIN_MB} MB: 바로 끝낸다", flush=True)
            cleanup()
            os._exit(4)
        time.sleep(1)


def cleanup():
    print(f"도는 중 available 최저 {stats['min_avail_mb']} MB", flush=True)
    global rec
    if rec:
        rec = None                    # 촬영 스레드가 더 쓰지 않게 먼저 끊고 닫는다
        rec_f.close()
        print(f"{args.record}에 이번에 {stats['rec_rows']}줄 기록", flush=True)


app = FastAPI()

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in tracks.items()}}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명, available ${r.stats.avail_mb} MB (최저 ${r.stats.min_avail_mb})`+(r.stats.rec_rows?`, 기록 ${r.stats.rec_rows}줄`:'');
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=mem_watch, daemon=True).start()
    print(f"브라우저에서 http://{args.host}:{args.port} (끝내려면 Ctrl+C)", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        cleanup()
