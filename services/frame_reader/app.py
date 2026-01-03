# services/frame_reader/app.py
import os
import cv2
import time
import json
import base64
import pika
from pathlib import Path
# --- START: cleanup code (paste at the top of services/frame_reader/app.py) ---
import os
import shutil
from pathlib import Path

def safe_remove_file(p: Path):
    try:
        if p.is_file():
            p.unlink()
            print(f"[cleanup] removed file: {p}")
    except Exception as e:
        print(f"[cleanup] failed to remove file {p}: {e}")

def safe_remove_dir(p: Path):
    try:
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            print(f"[cleanup] removed dir: {p}")
    except Exception as e:
        print(f"[cleanup] failed to remove dir {p}: {e}")

# Project known data names
names = {
    "debug_detections": "debug_detections",
    "violations_dir": "violations",
    "violations_db": "violations.db"
}

# Candidate base paths to check (order matters: common host path, container mount, relative)
candidates = []

# 1) If DATA_DIR or VIDEO_DIR env var exists, prefer relative to that
data_env = os.getenv("DATA_DIR") or os.getenv("VIDEO_DIR")  # sometimes apps use VIDEO_DIR
if data_env:
    # if VIDEO_DIR points to ./data/videos, transform to candidate ./data
    p = Path(data_env)
    if p.name == "videos":
        candidates.append(p.parent)         # e.g. .../data
    candidates.append(p)

# 2) Common explicit host path on Windows (your repo root)
candidates.append(Path.cwd() / "data")   # e.g., C:\...\<repo>\data

# 3) Common container mount path
candidates.append(Path("/data"))

# Deduplicate preserving order
seen = set()
bases = []
for c in candidates:
    try:
        rp = str(c.resolve())
    except Exception:
        rp = str(c)
    if rp not in seen:
        seen.add(rp)
        bases.append(Path(rp))

# Attempt removal for each base until at least one removal succeeded
removed_any = False
for base in bases:
    # build specific paths
    d_debug = base / names["debug_detections"]
    d_viol = base / names["violations_dir"]
    f_db = base / names["violations_db"]
    # remove items if they exist
    if d_debug.exists():
        safe_remove_dir(d_debug)
        removed_any = True
    if d_viol.exists():
        safe_remove_dir(d_viol)
        removed_any = True
    if f_db.exists():
        safe_remove_file(f_db)
        removed_any = True

# Ensure directories exist (fresh)
try:
    (bases[0] / names["debug_detections"]).mkdir(parents=True, exist_ok=True)
    (bases[0] / names["violations_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"[cleanup] ensured fresh dirs at {bases[0]}")
except Exception as e:
    print(f"[cleanup] failed to create fresh dirs: {e}")

# --- END: cleanup code ---




VIDEO_DIR = os.getenv("VIDEO_DIR", "./data/videos")
BROKER_URL = os.getenv("BROKER_URL", "amqp://guest:guest@localhost:5672/")
FPS_PUBLISH = int(os.getenv("FPS_PUBLISH", 10))

print("Frame Reader starting")
print("VIDEO_DIR=", VIDEO_DIR)
print("BROKER_URL=", BROKER_URL)
print("FPS_PUBLISH=", FPS_PUBLISH)

# Connect to RabbitMQ
params = pika.URLParameters(BROKER_URL)
connection = None
try:
    connection = pika.BlockingConnection(params)
except Exception as e:
    print("ERROR: Could not connect to RabbitMQ at", BROKER_URL)
    print("Exception:", e)
    print("Make sure RabbitMQ is running and BROKER_URL is correct.")
    raise SystemExit(1)

channel = connection.channel()
# durable queue so messages survive broker restart (demo)
channel.queue_declare(queue="frames", durable=True)

def publish_frame(source_id, frame_id, frame):
    # encode to JPEG bytes
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        print("Warning: failed to encode frame", frame_id)
        return
    payload = {
        "source_id": source_id,
        "frame_id": int(frame_id),
        "timestamp": time.time(),
        "image": base64.b64encode(jpeg.tobytes()).decode("utf-8")
    }
    channel.basic_publish(
        exchange="",
        routing_key="frames",
        body=json.dumps(payload),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )

# iterate mp4 files
for video_path in Path(VIDEO_DIR).glob("*"):
    if video_path.suffix.lower() not in [".mp4", ".mkv", ".avi"]:
        continue

    print("Processing video:", video_path.name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: cannot open video:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # We'll publish at FPS_PUBLISH frames per second (downsample)
    downsample_factor = max(1, int(round(fps / FPS_PUBLISH))) if FPS_PUBLISH > 0 else 1

    frame_id = 0
    published = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % downsample_factor == 0:
            publish_frame(video_path.name, frame_id, frame)
            published += 1
            # Debug print every N frames so logs aren't huge
            if published % 10 == 0:
                print(f"Published {published} frames from {video_path.name} (last frame_id={frame_id})")

            # small sleep to avoid flooding broker, approximate rate control
            time.sleep(1.0 / FPS_PUBLISH)

        frame_id += 1

    cap.release()
    print(f"Finished {video_path.name} — published {published} frames")

print("Frame Reader finished")
connection.close()
