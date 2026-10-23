# 웹캠 실시간 보기 + 버튼으로 사진 저장 (4단계 자료 찍기용). 보드 모니터가 없어서 노트북 브라우저로 본다.
# 사용: ~/practice.sh 재웅 python live.py   → 브라우저에서 http://100.113.228.46:8001
# 끝낼 때는 Ctrl+C. 켜 둔 동안은 카메라와 자물쇠를 잡고 있다. 화면만 보여 주고, 저장은 버튼을 누를 때만 한다.
# 메모리 아끼기: 640x480, 초당 약 10장만 보내고, 마지막 한 장만 메모리에 둔다.
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

PORT = int(os.environ.get("PORT", "8001"))   # 팀 문서 2번 표: 보드 A 재웅 포트
HOST = os.environ.get("HOST", "100.113.228.46")   # Tailscale 주소에만 연다 (학교망 전체에 웹캠을 열지 않음)
CAM = int(os.environ.get("CAM", "0"))
os.makedirs("media", exist_ok=True)

cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
assert cap.isOpened(), f"/dev/video{CAM}을 열 수 없다 (다른 프로그램이 쓰는 중인지 확인)"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
latest = None   # 가장 최근 프레임 한 장
lock = threading.Lock()
running = True


def grab():
    global latest
    while running:
        ok, frame = cap.read()
        if ok:
            with lock:
                latest = frame


threading.Thread(target=grab, daemon=True).start()

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>웹캠 보기</title>
<style>body{font-family:sans-serif;margin:16px}img{max-width:100%;border:1px solid #ccc}
button{font-size:16px;margin:4px;padding:8px 12px}#msg{margin-top:8px;font-weight:bold}</style>
<h3>웹캠 실시간</h3><img src="/stream"><div>
<button onclick="snap('empty')">empty 저장</button><button onclick="snap('class')">class 저장</button>
<button onclick="snap('board')">board 저장</button><button onclick="snap('monitor')">monitor 저장</button></div>
<div id=msg></div>
<script>function snap(n){fetch('/snap?name='+n).then(r=>r.text()).then(t=>document.getElementById('msg').textContent=t)}</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while running:
                    with lock:
                        frame = latest
                    if frame is not None:
                        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif url.path == "/snap":
            name = parse_qs(url.query).get("name", ["snap"])[0]
            if not re.fullmatch(r"[a-z0-9_]+", name):
                self.send_error(400)
                return
            with lock:
                frame = latest
            if frame is None:
                msg = "아직 프레임이 없다"
            else:
                cv2.imwrite(f"media/{name}.jpg", frame)
                msg = f"저장: media/{name}.jpg ({time.strftime('%H:%M:%S')})"
                print(msg, flush=True)
            body = msg.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


print(f"브라우저에서 http://{HOST}:{PORT} (끝내려면 Ctrl+C)", flush=True)
try:
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
except KeyboardInterrupt:
    pass
finally:
    running = False
    cap.release()
