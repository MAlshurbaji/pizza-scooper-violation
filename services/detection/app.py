# services/detection/app.py
import os, json, base64, time
import numpy as np
import cv2
import pika
from ultralytics import YOLO
import torch

BROKER_URL = os.getenv("BROKER_URL", "amqp://guest:guest@localhost:5672/")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/yolo12m-v2.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.20))
SAVE_DEBUG = os.getenv("SAVE_DEBUG", "1") == "1"
DEBUG_DIR = os.getenv("DEBUG_DIR", "./data/debug_detections")
DEBUG_EVERY_N = int(os.getenv("DEBUG_EVERY_N", "1"))  # save every N frames (per source)

os.makedirs(DEBUG_DIR, exist_ok=True)

# color map: label -> BGR tuple (OpenCV uses BGR)
COLOR_MAP = {
    "hand":    (180,  30, 200),   # purple
    "person":  (0,    0, 255),    # red
    "pizza":   (0,  255, 204),    # lime / yellow-green
    "scooper": (255, 255,   0),   # cyan-ish
    "_default": (200, 200, 200)   # default color for unknown labels
}

# track counters per source to control saving frequency
save_counters = {}

def get_color_for_label(label):
    return COLOR_MAP.get(label, COLOR_MAP["_default"])

def callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        img_b64 = msg.get("image")
        if not img_b64:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        img_bytes = base64.b64decode(img_b64)
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # run inference
        results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            name = model.names.get(cls, str(cls))
            xywh = box.xywh.tolist()[0]
            bbox_xyxy = xyxy_from_xywh(xywh)
            detections.append({
                "label": name,
                "confidence": conf,
                "bbox_xyxy": [float(x) for x in bbox_xyxy]
            })

        ok, jpeg = cv2.imencode(".jpg", img)
        img_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        out_msg = {
            "source_id": msg.get("source_id"),
            "frame_id": int(msg.get("frame_id", -1)),
            "timestamp": msg.get("timestamp", time.time()),
            "detections": detections,
            "image": img_b64
        }

        # publish detections
        chan.basic_publish(
            exchange="",
            routing_key="detections",
            body=json.dumps(out_msg),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        # Save debug image every DEBUG_EVERY_N frames per source (optional)
        if SAVE_DEBUG:
            src = out_msg["source_id"] or "unknown"
            fid = out_msg["frame_id"]
            # initialize counter for new source
            if src not in save_counters:
                save_counters[src] = 0
            save_counters[src] += 1

            # Save when frame_id % DEBUG_EVERY_N == 0 OR based on counter
            # (use frame_id modulus ensures predictable spacing even if frames skipped)
            try:
                should_save = (fid % DEBUG_EVERY_N == 0)
            except Exception:
                should_save = (save_counters[src] % DEBUG_EVERY_N == 0)

            if should_save:
                vis = img.copy()
                for d in detections:
                    x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
                    color = get_color_for_label(d["label"])
                    label_text = f'{d["label"]} {d["confidence"]:.2f}'
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    # draw filled rect for text background for readability
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (x1, y1 - int(th + 6)), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                # create a sanitized source name without extension
                src_name = src
                if "." in src_name:
                    src_name = ".".join(src_name.split(".")[:-1])  # remove last extension part
                # optionally remove path elements (shouldn't be present) and keep base name
                src_name = src_name.replace("/", "_").replace("\\", "_")
                
                fn = f"{DEBUG_DIR}/det_{src_name}_{fid}.jpg"
                cv2.imwrite(fn, vis)
                print("Saved debug:", fn)

        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("Detection callback error:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

os.makedirs(DEBUG_DIR, exist_ok=True)

print("Detection service starting")
print("BROKER_URL=", BROKER_URL)
print("MODEL_PATH=", MODEL_PATH)
print("CONF_THRESHOLD=", CONF_THRESHOLD)

model = YOLO(MODEL_PATH)
if torch.cuda.is_available():
    try:
        model.to('cuda:0')
        print("Moved model to cuda:0")
    except Exception as e:
        print("Failed to move model to cuda:", e)
print("Model loaded. device:", model.device)

# RabbitMQ setup
params = pika.URLParameters(BROKER_URL)
conn = pika.BlockingConnection(params)
chan = conn.channel()
chan.queue_declare(queue="frames", durable=True)
chan.queue_declare(queue="detections", durable=True)

def xyxy_from_xywh(xywh):
    # xywh = [x_center, y_center, w, h]
    x, y, w, h = xywh
    x1 = float(x - w/2)
    y1 = float(y - h/2)
    x2 = float(x + w/2)
    y2 = float(y + h/2)
    return [x1, y1, x2, y2]

seen_debug = set()

chan.basic_consume(queue="frames", on_message_callback=callback)
print("Waiting for frames from queue 'frames' ...")
chan.start_consuming()
