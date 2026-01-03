# tools/consume_frames.py
import pika
import json
import base64
import os
from pathlib import Path

BROKER_URL = os.getenv("BROKER_URL", "amqp://guest:guest@localhost:5672/")
SAVE_DIR = os.getenv("SAVE_DIR", "./data/consumed_frames")
DEBUG_EVERY_N = int(os.getenv("DEBUG_EVERY_N", "10"))  # save every N frames per source

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Track how many frames we've seen per source
frame_counters = {}

params = pika.URLParameters(BROKER_URL)
conn = pika.BlockingConnection(params)
ch = conn.channel()
ch.queue_declare(queue="frames", durable=True)

def callback(ch, method, properties, body):
    msg = json.loads(body)
    source_id = msg.get("source_id", "unknown")
    frame_id = msg.get("frame_id", -1)

    # update counter per source
    frame_counters.setdefault(source_id, 0)
    frame_counters[source_id] += 1

    print(f"Got frame: {source_id} frame_id={frame_id}")

    # Save only every N frames
    if frame_counters[source_id] % DEBUG_EVERY_N == 0:
        try:
            img_bytes = base64.b64decode(msg["image"])
            out_path = (
                Path(SAVE_DIR)
                / f"{source_id.replace('.', '_')}_frame{frame_id}.jpg"
            )
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"Saved debug frame → {out_path}")
        except Exception as e:
            print("Failed to decode/save image:", e)

    ch.basic_ack(delivery_tag=method.delivery_tag)

ch.basic_consume(queue="frames", on_message_callback=callback)
print(
    f"Consumer started — saving every {DEBUG_EVERY_N} frames per source. "
    "Press Ctrl+C to exit."
)
ch.start_consuming()
