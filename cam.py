# 4단계 자료 찍기: USB 웹캠에서 사진 한 장 또는 짧은 동영상을 media/에 저장한다.
# 보드의 ffmpeg(NVIDIA 판)에는 웹캠(v4l2) 입력이 없어서 OpenCV로 찍는다.
# 사용: python cam.py snap empty   → media/empty.jpg
#       python cam.py clip 10      → media/clip.mp4 (10초) + media/frames/frame_NN.jpg (2초마다)
# 카메라 번호를 바꾸려면 CAM=2 python cam.py ...
import os
import sys
import time

import cv2

mode = sys.argv[1] if len(sys.argv) > 1 else ""
assert mode in ("snap", "clip"), "사용: python cam.py snap 이름 | python cam.py clip 초"
cam = int(os.environ.get("CAM", "0"))
os.makedirs("media/frames", exist_ok=True)

cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
assert cap.isOpened(), f"/dev/video{cam}을 열 수 없다 (다른 프로그램이 쓰는 중인지 확인)"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(15):   # 켜자마자 나오는 어두운 장면을 버린다 (자동 노출이 맞춰질 때까지)
    cap.read()

try:
    if mode == "snap":
        name = sys.argv[2] if len(sys.argv) > 2 else "snap"
        ok, frame = cap.read()
        assert ok, "프레임을 못 읽었다"
        path = f"media/{name}.jpg"
        cv2.imwrite(path, frame)
        print(f"저장: {path} {frame.shape[1]}x{frame.shape[0]}")
    else:
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        ok, frame = cap.read()
        assert ok, "프레임을 못 읽었다"
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("media/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n, saved, t0 = 0, 0, time.perf_counter()
        while (el := time.perf_counter() - t0) < secs:
            out.write(frame)   # 한 장씩 바로 파일에 쓴다 (메모리에 모으지 않음)
            n += 1
            if el >= saved * 2:   # 2초마다 한 장: 동영상이 안 될 때 대신 쓸 프레임
                saved += 1
                cv2.imwrite(f"media/frames/frame_{saved:02d}.jpg", frame)
            ok, frame = cap.read()
            if not ok:
                break
        out.release()
        print(f"저장: media/clip.mp4 {w}x{h}, {n}장 / {el:.1f}초 (실제 {n / el:.1f} fps, 파일 {fps:.0f} fps)")
        print(f"프레임 {saved}장: media/frames/")
finally:
    cap.release()
