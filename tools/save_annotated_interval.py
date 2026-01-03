# tools/save_annotated_interval.py
import os, json, base64, cv2, numpy as np, pika
BROKER_URL = os.getenv("BROKER_URL","amqp://guest:guest@localhost:5672/")
OUTDIR = "./data/debug_interval"
SRC_NAME = "Sah b3dha ghalt.mp4"   # change if different
FRAME_MIN = 560
FRAME_MAX = 620

os.makedirs(OUTDIR, exist_ok=True)
params = pika.URLParameters(BROKER_URL)
conn = pika.BlockingConnection(params)
ch = conn.channel()
ch.queue_declare(queue="detections", durable=True)

def draw_box(img, box, label, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

saved = 0
def cb(ch, method, prop, body):
    global saved
    msg = json.loads(body)
    if msg.get("source_id") != SRC_NAME:
        ch.basic_ack(delivery_tag=method.delivery_tag); return
    fid = int(msg.get("frame_id", -1))
    if fid < FRAME_MIN or fid > FRAME_MAX:
        ch.basic_ack(delivery_tag=method.delivery_tag); return

    img_b64 = msg.get("image")
    if not img_b64:
        ch.basic_ack(delivery_tag=method.delivery_tag); return
    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    for d in msg.get("detections", []):
        lbl = d.get("label","")
        box = d.get("bbox_xyxy") or d.get("bbox")
        if box:
            draw_box(img, box, f"{lbl} {d.get('confidence',0):.2f}", (0,255,0) if 'scoop' not in lbl.lower() else (0,128,255))
    # draw ROI if present
    try:
        roi_conf = json.load(open("configs/rois.json"))
        if SRC_NAME in roi_conf:
            for polyname, poly in roi_conf[SRC_NAME].items():
                pts = np.array(poly, np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(255,0,0), thickness=2)
                cv2.putText(img, polyname, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    except Exception:
        pass

    fn = f"{OUTDIR}/{SRC_NAME.replace(' ','_')}_f{fid}.jpg"
    cv2.imwrite(fn, img)
    print("Saved", fn)
    saved += 1
    ch.basic_ack(delivery_tag=method.delivery_tag)
    if saved >= (FRAME_MAX-FRAME_MIN+1):
        print("Done saving interval.")
        ch.stop_consuming()

ch.basic_consume(queue="detections", on_message_callback=cb)
print("Listening for detection messages and saving interval images...")
ch.start_consuming()
