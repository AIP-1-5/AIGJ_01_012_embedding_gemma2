# 8단계 웹캠 실시간 물체 추출과 웹 검색. 원문 README 8단계 app.py를 우리 보드 규칙에 맞게 고친 것.
# 사용: ~/practice.sh 재웅 python app.py --mode yolo|relabel|yoloe   → 브라우저에서 http://100.113.228.46:8001
# 원문과 다른 점 (docs/reports/08-webcam-search.md 2번):
#   - 포트 8001, Tailscale 주소에만 연다 (학교망 전체에 웹캠을 열지 않음). live.py와 같은 포트라 둘 중 하나만 켠다
#   - 모델은 항상 원본 (7단계 모델은 텍스트 전용). HF 토큰은 eg2.py가 ~/eg2/.env에서 읽는다
#   - 메모리 감시: 시작 전 available 3GB 미만이면 안 켬, 도는 중 1GB 미만이면 바로 끝냄, GPU 몫 0.3
#   - 임베딩 배치 기본 2 (원문 8), ultralytics 자동 pip 설치 끔 (torch가 바뀌지 않게)
#   - 끝나면 crops/<mode>를 지운다(--keep-crops로 남김). 사람이 찍힌 사진은 남기지 않는 팀 원칙
import os
os.environ["YOLO_AUTOINSTALL"] = "False"          # ultralytics가 몰래 pip install 하지 않게 (torch 고정 지키기)
os.environ["EG2_MODEL"] = "google/embeddinggemma-2"   # 7단계 미세조정 모델이 아니라 원본 (eg2.py가 읽음)
import time, threading, collections, argparse, shutil
import cv2, numpy as np, torch, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from eg2 import load

# ---------------- 실행 모드 ----------------
# yolo    : YOLO 클래스 그대로 사용 (문제 확인용 기준선)
# relabel : YOLO 박스 + EmbeddingGemma 2 재라벨 (개선 1)
# yoloe   : 텍스트로 클래스를 지정하는 YOLOE 검출기 (개선 2)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["yolo", "relabel", "yoloe"], default="relabel")
ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 (기본: 모드별 값)")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))   # 팀 문서 2번 표: 보드 A 재웅
ap.add_argument("--host", default=os.environ.get("HOST", "100.113.228.46"))        # Tailscale 주소에만
ap.add_argument("--keep-crops", action="store_true", help="끝날 때 crops/<mode>를 지우지 않음")
args = ap.parse_args()
MODE = args.mode

# ---------------- 설정 ----------------
CAM = int(os.environ.get("CAM", "0"))                  # /dev/video0
W, H = (int(v) for v in os.environ.get("SIZE", "1280x720").split("x"))   # fps 10 미만이면 SIZE=640x480
CONF = args.conf if args.conf is not None else {"yolo": 0.5, "relabel": 0.35, "yoloe": 0.25}[MODE]
MIN_SIDE      = 48                    # 크롭 최소 크기(px)
REFRESH_S     = int(os.environ.get("REFRESH_S", "30"))   # 같은 추적 ID 재임베딩 간격(초)
SKIP_CLASSES  = {"person"}            # 크롭·저장하지 않을 클래스
DIM, MAX_ITEMS = 512, 1000
BATCH         = int(os.environ.get("BATCH", "2"))   # 한 번에 임베딩할 크롭 수 (원문 8, 메모리 때문에 2)
CROP_DIR      = f"crops/{MODE}"       # 모드별로 크롭을 따로 저장해 나중에 비교
START_MIN_MB, KILL_MIN_MB = 3000, 1000   # 팀 메모리 원칙: 시작 전 3GB, 도는 중 1GB

# YOLO(COCO 80종)에 없는 물체는 비슷한 클래스로 오인식된다 (세탁기→oven, 전지 포장→cell phone).
# relabel: 크롭 임베딩과 아래 라벨 문구의 유사도로 라벨을 다시 붙인다.
# yoloe  : 아래 영문 라벨을 검출 클래스 자체로 사용한다.  (영문 = 영상 표시, 한글 = 웹 표시)
LABELS = [
    ("drum washing machine", "드럼 세탁기"), ("top-load washing machine", "통돌이 세탁기"),
    ("oven", "오븐"), ("microwave", "전자레인지"), ("refrigerator", "냉장고"),
    ("cell phone", "휴대폰"), ("battery pack", "원통형 전지 포장"), ("AA battery", "건전지"),
    ("laptop", "노트북"), ("monitor", "모니터"), ("keyboard", "키보드"), ("mouse", "마우스"),
    ("cup", "컵"), ("water bottle", "물병"), ("book", "책"), ("backpack", "가방"), ("chair", "의자"),
]
LABEL_MIN     = 0.20                  # 이 점수 미만이면 YOLO 라벨 유지 (웹의 점수를 보고 조정)


def mem_available_mb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return info["MemAvailable"] // 1024


avail0 = mem_available_mb()
print(f"시작 전 available {avail0} MB", flush=True)
if avail0 < START_MIN_MB:
    raise SystemExit(f"available {avail0} MB < {START_MIN_MB} MB: 켜지 않는다 (팀 메모리 원칙)")
torch.cuda.set_per_process_memory_fraction(0.3)   # 공유 메모리 8GB 중 GPU 몫 상한 (팀 문서 11번)
os.makedirs(CROP_DIR, exist_ok=True)

if MODE == "yoloe":
    from ultralytics import YOLOE
    names = [en for en, _ in LABELS]
    det = YOLOE("yoloe-11s-seg.pt")     # 첫 실행 시 자동 다운로드
    try:
        det.set_classes(names, det.get_text_pe(names))
    except (AttributeError, TypeError):  # 새 ultralytics는 이름만 받는다 (원문: 버전에 따라 API가 다름)
        det.set_classes(names)
else:
    det = YOLO("yolo11n.pt")          # 첫 실행 시 자동 다운로드
KO = dict(LABELS)                     # 영문 -> 한글
print(f"[mode={MODE}] conf={CONF} size={W}x{H} batch={BATCH}", flush=True)
emb = load(vision=True)               # 텍스트+비전(440M), 오디오는 안 올림

with torch.inference_mode():
    label_vecs = np.asarray(emb.encode([f"a photo of a {en}" for en, _ in LABELS],
        prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True), dtype=np.float32)

frame_lock, emb_lock, idx_lock = threading.Lock(), threading.Lock(), threading.Lock()
latest_jpeg = None
crop_q = collections.deque(maxlen=64)
items = []          # {id, cls, conf, ts, img, vec}
seen = {}           # track_id -> 마지막 임베딩 시각
track_label = {}    # track_id -> (영문 라벨, 한글 라벨, 점수)
stats = {"fps": 0.0, "embed_ms": 0.0, "avail_mb": avail0, "min_avail_mb": avail0}


def capture_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print(f"/dev/video{CAM}을 열 수 없다 (live.py 등 다른 프로그램이 쓰는 중인지 확인, 원문 Q11)", flush=True)
        os._exit(2)
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = det.track(frame, persist=True, conf=CONF, half=True, verbose=False)[0]
        now = time.time()
        boxes = r.boxes.xyxy.int().tolist()
        clss = r.boxes.cls.int().tolist()
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(boxes, r.boxes.id.int().tolist(), clss, r.boxes.conf.tolist()):
                name = det.names[c]
                if name in SKIP_CLASSES or min(x2 - x1, y2 - y1) < MIN_SIDE:
                    continue
                if now - seen.get(tid, 0) < REFRESH_S:
                    continue
                seen[tid] = now
                crop_q.append((tid, name, cf, now, frame[y1:y2, x1:x2].copy()))
        vis = frame.copy()
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for (x1, y1, x2, y2), tid, c, cf in zip(boxes, ids, clss, r.boxes.conf.tolist()):
            yolo = det.names[c]
            en = track_label[tid][0] if (MODE == "relabel" and tid in track_label) else yolo
            color = (0, 200, 255) if en != yolo else (0, 255, 0)   # 재라벨된 박스는 주황색
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{en} {cf:.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
        cv2.putText(vis, f"{stats['fps']:.1f} fps | indexed {len(items)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()


def embed_loop():
    seq = 0
    while True:
        if not crop_q:
            time.sleep(0.05); continue
        batch = [crop_q.popleft() for _ in range(min(BATCH, len(crop_q)))]
        paths = []
        for tid, name, cf, ts, crop in batch:
            seq += 1
            p = f"{CROP_DIR}/{seq:06d}_{name.replace(' ', '_')}.jpg"
            cv2.imwrite(p, crop)
            paths.append(p)
        t = time.perf_counter()
        with emb_lock, torch.inference_mode():
            vecs = emb.encode([{"image": p} for p in paths],
                              truncate_dim=DIM, normalize_embeddings=True)
        stats["embed_ms"] = (time.perf_counter() - t) * 1000 / len(paths)
        vecs = np.asarray(vecs, dtype=np.float32)
        sims = vecs @ label_vecs.T
        with idx_lock:
            for (tid, name, cf, ts, _), p, v, sim in zip(batch, paths, vecs, sims):
                j = int(sim.argmax())
                if MODE != "relabel":            # yolo / yoloe: 검출기 라벨 그대로
                    en, ko, ls = name, KO.get(name, name), float(sim[j])
                elif sim[j] >= LABEL_MIN:
                    en, ko, ls = LABELS[j][0], LABELS[j][1], float(sim[j])
                else:
                    en, ko, ls = name, name, float(sim[j])
                track_label[tid] = (en, ko, ls)
                items.append({"id": tid, "cls": name, "label": ko, "lscore": round(ls, 3),
                              "conf": round(cf, 2), "ts": ts, "img": "/" + p, "vec": v})
            while len(items) > MAX_ITEMS:
                old = items.pop(0)
                try: os.remove(old["img"][1:])
                except OSError: pass


def mem_watch():
    # 도는 중 available이 1GB 아래로 내려가면 센서 서비스를 지키려고 바로 끝낸다 (6단계 bench.py와 같은 기준)
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
    if not args.keep_crops:
        shutil.rmtree(CROP_DIR, ignore_errors=True)
        print(f"{CROP_DIR} 지움", flush=True)


app = FastAPI()
app.mount("/crops", StaticFiles(directory="crops"), name="crops")   # /crops/<mode>/<파일>


def public(it, score=None):
    d = {k: it[k] for k in ("id", "cls", "label", "lscore", "conf", "ts", "img")}
    if score is not None: d["score"] = round(float(score), 3)
    return d


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


@app.get("/search")
def search(q: str, k: int = 12):
    with idx_lock:
        snap = list(items)
    if not snap:
        return []
    with emb_lock, torch.inference_mode():
        qv = emb.encode(q, prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True)
    s = np.stack([it["vec"] for it in snap]) @ np.asarray(qv, dtype=np.float32)
    return [public(snap[i], s[i]) for i in s.argsort()[::-1][:k]]


@app.get("/recent")
def recent(n: int = 12):
    with idx_lock:
        snap = items[-n:][::-1]
    return {"mode": MODE, "stats": stats, "total": len(items), "items": [public(it) for it in snap]}


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 물체 검색</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img.live{width:100%;max-width:960px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.card{background:#222;padding:6px;border-radius:6px;font-size:12px}
.card img{width:100%;height:110px;object-fit:contain;background:#000}
input{font-size:16px;padding:6px;width:60%}button{font-size:16px;padding:6px 12px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img class=live src="/stream">
<h2>텍스트로 물체 검색</h2>
<form onsubmit="go();return false"><input id=q placeholder="예: 빨간 컵, 열려 있는 노트북"><button>검색</button></form>
<div id=res class=grid></div>
<h2>최근 추출 <span id=st></span></h2><div id=rec class=grid></div>
<script>
const card=it=>`<div class=card><img src="${it.img}"><div><b>${it.label}</b> (${it.lscore})<br>YOLO: ${it.cls} ${it.conf}`+
 (it.score!==undefined?`<br>검색 점수 <b>${it.score}</b>`:'')+`<br>${new Date(it.ts*1000).toLocaleTimeString()}</div></div>`;
async function go(){const q=document.getElementById('q').value;if(!q)return;
 const r=await (await fetch('/search?q='+encodeURIComponent(q))).json();
 document.getElementById('res').innerHTML=r.map(card).join('')||'결과 없음';}
async function poll(){const r=await (await fetch('/recent')).json();
 document.getElementById('mode').textContent=`[mode: ${r.mode}]`;
 document.getElementById('st').textContent=`(총 ${r.total}개, 검출 ${r.stats.fps.toFixed(1)} fps, 임베딩 ${r.stats.embed_ms.toFixed(0)} ms/개, available ${r.stats.avail_mb} MB, 최저 ${r.stats.min_avail_mb} MB)`;
 document.getElementById('rec').innerHTML=r.items.map(card).join('');}
setInterval(poll,3000);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=embed_loop, daemon=True).start()
    threading.Thread(target=mem_watch, daemon=True).start()
    print(f"브라우저에서 http://{args.host}:{args.port} (끝내려면 Ctrl+C)", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        cleanup()
