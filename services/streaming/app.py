# services/streaming/app.py
import os
import json
import sqlite3
import threading
import asyncio
import time
from typing import Set, Dict, Any, Tuple
import pika
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import traceback
from pathlib import Path
from urllib.parse import unquote
import base64

# Config (env overrides)
BROKER_URL = os.getenv("BROKER_URL", "amqp://guest:guest@localhost:5672/")
DB_PATH = os.getenv("DB_PATH", "./data/violations.db")
WS_PORT = int(os.getenv("STREAMING_PORT", 8003))
VIOLATION_DIR_PATH = os.getenv("VIOLATION_DIR", "./data/violations")
ROIS_PATH = os.getenv("ROI_CONFIG", "./configs/rois.json")

import cv2
from fastapi import Query

VIDEO_DIR = os.getenv("VIDEO_DIR", "./data/videos")  # ensure this matches where videos are stored

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# allow requests from frontend (change origin if you host elsewhere)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8003",
    "http://127.0.0.1:8003",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # or restrict to origins list above
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# in-memory structures
# violation_lookup maps (source_id, frame_id) -> violation_event (dict)
violation_lookup: Dict[Tuple[str,int], Dict[str,Any]] = {}
# asyncio queue for detections to broadcast
broadcast_queue: asyncio.Queue = asyncio.Queue()

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: dict):
        to_remove = []
        text = json.dumps(message)
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)

manager = ConnectionManager()

# helper: load recent violations from DB into violation_lookup
def load_violations_from_db(limit=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        if limit:
            cur.execute("SELECT id, source_id, frame_id, ts, frame_path, details FROM violations ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cur.execute("SELECT id, source_id, frame_id, ts, frame_path, details FROM violations")
        rows = cur.fetchall()
        conn.close()
        for r in rows:
            try:
                source = r[1]
                frame_id = int(r[2])
            except Exception:
                continue
            try:
                details = json.loads(r[5]) if r[5] else {}
            except Exception:
                details = {}
            key = (source, frame_id)
            violation_lookup[key] = {"details": details, "frame_path": r[4], "ts": r[3], "id": r[0]}
        print(f"[streaming] loaded {len(rows)} violations into memory")
    except Exception as e:
        print("[streaming] load_violations_from_db error:", e)

# RabbitMQ consumer threads
def start_rabbit_consumers(loop):
    """Start two blocking pika consumers in separate threads and push to asyncio queue via loop."""
    params = pika.URLParameters(BROKER_URL)

    # detections consumer
    def consume_detections():
        while True:
            try:
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="detections", durable=True)

                def on_msg(ch_, method, props, body):
                    try:
                        msg = json.loads(body)
                    except Exception:
                        ch_.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    # annotate with violation if exists
                    key = (msg.get("source_id"), int(msg.get("frame_id", -1)))
                    vio = violation_lookup.get(key)
                    if vio:
                        msg["violation"] = True
                        msg["violation_details"] = vio.get("details")
                        # include frame_path if available
                        msg["violation_details"]["frame_path"] = vio.get("frame_path")
                    else:
                        msg["violation"] = False

                    # push to asyncio queue for broadcasting
                    try:
                        loop.call_soon_threadsafe(broadcast_queue.put_nowait, {"type":"detection", "payload": msg})
                    except Exception:
                        pass

                    ch_.basic_ack(delivery_tag=method.delivery_tag)

                ch.basic_consume(queue="detections", on_message_callback=on_msg)
                ch.start_consuming()
            except Exception as e:
                print("[streaming] detections consumer error, retrying in 2s:", e)
                traceback.print_exc()
                time.sleep(2)

    # violations consumer
    def consume_violations():
        while True:
            try:
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="violations", durable=True)

                def on_msg(ch_, method, props, body):
                    try:
                        ev = json.loads(body)
                    except Exception:
                        ch_.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    # store in lookup and also broadcast immediately
                    key = (ev.get("source_id"), int(ev.get("frame_id", -1)))
                    violation_lookup[key] = ev

                    # also broadcast a violation notice so frontends can react immediately
                    try:
                        loop.call_soon_threadsafe(broadcast_queue.put_nowait, {"type":"violation", "payload": ev})
                    except Exception:
                        pass

                    ch_.basic_ack(delivery_tag=method.delivery_tag)

                ch.basic_consume(queue="violations", on_message_callback=on_msg)
                ch.start_consuming()
            except Exception as e:
                print("[streaming] violations consumer error, retrying in 2s:", e)
                traceback.print_exc()
                time.sleep(2)

    # run each consumer in its own daemon thread
    t1 = threading.Thread(target=consume_detections, daemon=True)
    t2 = threading.Thread(target=consume_violations, daemon=True)
    t1.start()
    t2.start()

# Background broadcaster task
async def broadcaster_task():
    while True:
        msg = await broadcast_queue.get()
        try:
            await manager.broadcast(msg)
        except Exception:
            pass

@app.get("/violation-context")
def violation_context(source_id: str, frame_id: int, pre: int = 7, post: int = 7, stride: int = 3):
    """
    Return a short sequence of frames around a violation.
    Example: frame_id=100, pre=7, post=7, stride=3 -> frames [100-3*7 .. 100+3*7] sampled every `stride` frames.
    Response: {"source_id":..., "frames":[{"frame":n, "image":"data:image/jpeg;base64,..."}, ...]}
    """
    try:
        # compute requested frame indices
        total_pre = pre * stride
        total_post = post * stride
        start = max(0, frame_id - total_pre)
        end = frame_id + total_post

        video_path = Path(VIDEO_DIR) / source_id
        if not video_path.exists():
            return JSONResponse(status_code=404, content={"error": "video not found", "video_path": str(video_path)})

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return JSONResponse(status_code=500, content={"error": "cannot open video", "video_path": str(video_path)})

        # clamp end to frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        end = min(end, frame_count - 1)

        frames_out = []
        # iterate sampled frames
        for f in range(start, end + 1, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, img = cap.read()
            if not ret or img is None:
                continue
            ok, jpg = cv2.imencode(".jpg", img)
            if not ok:
                continue
            b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
            frames_out.append({"frame": f, "image": "data:image/jpeg;base64," + b64})

        cap.release()
        return {"source_id": source_id, "requested_frame": frame_id, "frames": frames_out}
    except Exception as e:
        print("[streaming] violation-context error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# REST endpoint: stats (counts in DB)
@app.get("/stats")
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM violations")
        total = cur.fetchone()[0] or 0

        cur.execute("SELECT source_id, COUNT(*) FROM violations GROUP BY source_id")
        per_video = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
    except Exception:
        total = 0
        per_video = {}
    return {"total_violations": total, "per_video": per_video, "uptime": int(time.time())}

# Serve ROIs JSON
@app.get("/rois")
def get_rois():
    try:
        with open(ROIS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Robust file serving for violation images
@app.get("/violation-file/{fname}")
def get_violation_file(fname: str):
    """
    Serve a saved violation image. Tolerant to basenames or some path forms.
    """
    try:
        raw = unquote(fname)
        candidates = []
        base = os.path.basename(raw)
        candidates.append(Path(VIOLATION_DIR_PATH) / base)
        candidates.append(Path(raw))
        cleaned = raw.replace("\\", "/")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        candidates.append(Path(cleaned))
        candidates.append(Path(".") / cleaned)

        tried = []
        for p in candidates:
            p_str = str(p)
            tried.append(p_str)
            if p.exists() and p.is_file():
                return FileResponse(path=str(p), filename=os.path.basename(str(p)), media_type="image/jpeg")

        print(f"[streaming] violation-file not found. fname='{fname}'. tried: {tried}. VIOLATION_DIR_PATH={VIOLATION_DIR_PATH}")
        return JSONResponse(status_code=404, content={"error":"not found", "tried": tried})
    except Exception as e:
        print("[streaming] get_violation_file error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# List violations for a specific source
@app.get("/violations/{source_id}")
def get_violations_for_source(source_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, source_id, frame_id, ts, frame_path, details FROM violations WHERE source_id = ? ORDER BY id DESC", (source_id,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            try:
                details = json.loads(r[5]) if r[5] else None
            except Exception:
                details = None
            out.append({
                "id": r[0],
                "source_id": r[1],
                "frame_id": r[2],
                "ts": r[3],
                "frame_path": r[4],
                "details": details
            })
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # send initial snapshot: few recent violations
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT id, source_id, frame_id, ts, frame_path, details FROM violations ORDER BY id DESC LIMIT 20")
            recent = []
            for r in cur.fetchall():
                recent.append({"id": r[0], "source_id": r[1], "frame_id": r[2], "ts": r[3], "frame_path": r[4], "details": json.loads(r[5]) if r[5] else None})
            conn.close()
            await ws.send_text(json.dumps({"type":"snapshot","payload": {"recent_violations": recent}}))
        except Exception:
            pass

        while True:
            # keep the socket open; we don't expect client messages
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        manager.disconnect(ws)

# Start-up event
@app.on_event("startup")
async def startup():
    load_violations_from_db(limit=1000)
    loop = asyncio.get_running_loop()
    start_rabbit_consumers(loop)
    asyncio.create_task(broadcaster_task())
    print("Streaming service started: WebSocket /ws and REST /stats")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT, log_level="info")
