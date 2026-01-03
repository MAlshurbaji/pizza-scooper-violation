# services/tracker/app.py
import os
import json
import time
import math
import base64
import sqlite3
from collections import defaultdict
import numpy as np
import cv2
import pika

# ---------------- Config (env overrides) ----------------
BROKER_URL = os.getenv("BROKER_URL", "amqp://guest:guest@localhost:5672/")
DB_PATH = os.getenv("DB_PATH", "./data/violations.db")
VIOLATION_DIR = os.getenv("VIOLATION_DIR", "./data/violations")
ROI_CONFIG = os.getenv("ROI_CONFIG", "./configs/rois.json")

# Detection/tracking thresholds (tune via env)
IOU_THRESH = float(os.getenv("IOU_THRESH", 0.15))            # iou to relink by bbox when visible
RELINK_DIST_PX = float(os.getenv("RELINK_DIST_PX", 120.0))  # max distance from last anchor to relink new hand detection
TEMP_WINDOW_FRAMES = int(os.getenv("TEMP_WINDOW_FRAMES", 8)) # how many frames back to look for scooper use
SCOOPER_STRICT_DIST = float(os.getenv("SCOOPER_STRICT_DIST", 30.0))  # distance to consider scooper 'near' hand
SCOOPER_MIN_IOU = float(os.getenv("SCOOPER_MIN_IOU", 0.08))  # small overlap also counts
MIN_PIZZA_IOU = float(os.getenv("MIN_PIZZA_IOU", 0.12))     # hand/pizza overlap threshold
MAX_TRACK_FRAMES = int(os.getenv("MAX_TRACK_FRAMES", 120))  # maximum lifetime of a track (frames)
MIN_ROI_FRAC = float(os.getenv("MIN_ROI_FRAC", 0.33))       # fraction of hand bbox inside ROI to start track
MIN_ROI_CORNERS = int(os.getenv("MIN_ROI_CORNERS", 2))      # corners inside ROI to start track
VERBOSE = os.getenv("TRACKER_VERBOSE", "1") == "1"

os.makedirs(VIOLATION_DIR, exist_ok=True)

# ---------------- Load ROI config ----------------
def normalize_rois(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for src, rois in raw.items():
        out[src] = {}
        if not isinstance(rois, dict):
            continue
        for name, poly in rois.items():
            pts = poly
            # handle common nesting patterns
            if isinstance(pts, list) and len(pts) > 0 and isinstance(pts[0], list) and len(pts[0])>0 and isinstance(pts[0][0], list):
                pts = pts[0]
            cleaned = []
            for p in pts:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    cleaned.append([int(p["x"]), int(p["y"])])
                elif isinstance(p, (list,tuple)) and len(p) >= 2:
                    cleaned.append([int(p[0]), int(p[1])])
            out[src][name] = cleaned
    return out

try:
    raw = json.load(open(ROI_CONFIG))
except Exception:
    raw = {}
ROI_DICT = normalize_rois(raw)

# ---------------- Database ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT,
    frame_id INTEGER,
    ts REAL,
    frame_path TEXT,
    details TEXT
)
""")
conn.commit()

# ---------------- RabbitMQ ----------------
params = pika.URLParameters(BROKER_URL)
connection = pika.BlockingConnection(params)
chan = connection.channel()
chan.queue_declare(queue="detections", durable=True)
chan.queue_declare(queue="violations", durable=True)

def publish_violation_event(ev):
    chan.basic_publish(exchange="", routing_key="violations", body=json.dumps(ev),
                       properties=pika.BasicProperties(delivery_mode=2))

def save_violation_record(source_id, frame_id, ts, frame_path, details):
    cur.execute("INSERT INTO violations (source_id, frame_id, ts, frame_path, details) VALUES (?, ?, ?, ?, ?)",
                (source_id, frame_id, ts, frame_path, json.dumps(details)))
    conn.commit()

# ---------------- Utilities ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    return (interArea / denom) if denom > 0 else 0.0

def bbox_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def decode_image_from_msg(msg):
    try:
        img_b64 = msg.get("image")
        if not img_b64:
            return None
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        if VERBOSE: print("decode image error", e)
        return None

def extract_bbox(d):
    if "bbox_xyxy" in d:
        return [float(x) for x in d["bbox_xyxy"]]
    if "bbox" in d:
        b = d["bbox"]
        if len(b) == 4:
            return [float(x) for x in b]
    return None

def area_fraction_points_in_poly(box, poly):
    # grid test (3x3) fraction of sample points inside poly
    xs = np.linspace(box[0], box[2], 3)
    ys = np.linspace(box[1], box[3], 3)
    pts = [(int(x),int(y)) for x in xs for y in ys]
    inside = sum(1 for p in pts if cv2.pointPolygonTest(np.array(poly, np.int32), p, False) >= 0)
    return inside / 9.0

def corners_in_poly(box, poly):
    x1,y1,x2,y2 = map(int, box)
    corners = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
    return sum(1 for c in corners if cv2.pointPolygonTest(np.array(poly, np.int32), c, False) >= 0)

# ---------------- Single active track per source ----------------
# active_track_by_source[src] -> dict or None
active_track_by_source = defaultdict(lambda: None)
next_track_id = 1

# track schema:
# {
#   id, bbox, last_frame, last_center, history[(fid,bbox,label)], scooper_frames:set(),
#   roi_name, roi_entry_frame, frames_alive (increment), anchor (last center)
# }

def log(*args):
    if VERBOSE:
        print(*args)

# ---------------- Core processing ----------------
def process_frame(msg):
    global next_track_id
    src = msg.get("source_id", "unknown")
    fid = int(msg.get("frame_id", -1))
    ts = msg.get("timestamp", time.time())
    dets = msg.get("detections", [])

    # parse detections
    hands = []
    pizzas = []
    scoopers = []
    for d in dets:
        lab = d.get("label","").lower()
        box = extract_bbox(d)
        if box is None:
            continue
        if lab.startswith("hand"):
            hands.append((lab, box))
        elif "pizza" in lab:
            pizzas.append((lab, box))
        elif "scoop" in lab or "scooper" in lab:
            scoopers.append((lab, box))

    log(f"[TRACKER] src={src} fid={fid} hands={len(hands)} pizzas={len(pizzas)} scoopers={len(scoopers)}")

    track = active_track_by_source[src]

    # STEP 1: detect majority-hand-in-ROI to start/reset a track
    started = False
    for h_lab, h_box in hands:
        h_center = bbox_center(h_box)
        if src in ROI_DICT:
            for roi_name, poly in ROI_DICT[src].items():
                if not poly:
                    continue
                # check center and majority condition
                center_inside = cv2.pointPolygonTest(np.array(poly, np.int32), (int(h_center[0]), int(h_center[1])), False) >= 0
                frac = area_fraction_points_in_poly(h_box, poly)
                corners = corners_in_poly(h_box, poly)
                if center_inside and (frac >= MIN_ROI_FRAC or corners >= MIN_ROI_CORNERS):
                    # start or reset
                    tid = next_track_id; next_track_id += 1
                    new_track = {
                        "id": tid,
                        "bbox": h_box,
                        "last_frame": fid,
                        "last_center": bbox_center(h_box),
                        "history": [(fid, h_box, h_lab)],
                        "scooper_seen_frames": set(),
                        "roi_name": roi_name,
                        "roi_entry_frame": fid,
                        "frames_alive": 0,
                    }
                    active_track_by_source[src] = new_track
                    log(f"[TRACKER] NEW active track {tid} ENTERED ROI '{roi_name}' (src={src} fid={fid})")
                    started = True
                    break
        if started: break

    # STEP 2: if there's an active track, try to update from current detections or relink by proximity
    track = active_track_by_source[src]
    if track is not None:
        track["frames_alive"] += 1

        # try to find best hand by IOU first
        best_h = None; best_i = 0.0
        for h_lab, h_box in hands:
            i = iou(h_box, track["bbox"])
            if i > best_i:
                best_i = i; best_h = (h_lab, h_box)

        if best_h and best_i >= IOU_THRESH:
            h_lab, h_box = best_h
            track["bbox"] = h_box
            track["last_frame"] = fid
            track["last_center"] = bbox_center(h_box)
            track["history"].append((fid, h_box, h_lab))
            log(f"[TRACKER] updated track {track['id']} by IOU (iou={best_i:.3f}) fid={fid}")
        else:
            # no good IOU candidate -> try relink by proximity to last center
            relinked = False
            for h_lab, h_box in hands:
                center = bbox_center(h_box)
                dist = math.hypot(center[0]-track["last_center"][0], center[1]-track["last_center"][1])
                if dist <= RELINK_DIST_PX:
                    # relink: treat as same hand
                    track["bbox"] = h_box
                    track["last_frame"] = fid
                    track["last_center"] = center
                    track["history"].append((fid, h_box, h_lab))
                    relinked = True
                    log(f"[TRACKER] relinked track {track['id']} by proximity dist={dist:.1f} fid={fid}")
                    break
            if not relinked:
                # no detection linked this frame; keep anchor (last_center) and continue
                log(f"[TRACKER] track {track['id']} not seen this frame (anchor kept) fid={fid}")

        # STEP 3: check scoopers near the track's anchor center and record frames
        anchor = track["last_center"]
        for s_lab, s_box in scoopers:
            s_center = bbox_center(s_box)
            dist = math.hypot(s_center[0]-anchor[0], s_center[1]-anchor[1])
            s_i = iou(track["bbox"], s_box)
            # count scooper only if close or small overlap
            if (dist <= SCOOPER_STRICT_DIST) or (s_i >= SCOOPER_MIN_IOU):
                track["scooper_seen_frames"].add(fid)
                log(f"[TRACKER] scooper near track {track['id']} (dist={dist:.1f} iou={s_i:.3f}) fid={fid}")

        # STEP 4: check pizza contact -> evaluate violation
        for p_lab, p_box in pizzas:
            p_i = iou(track["bbox"], p_box)
            if p_i >= MIN_PIZZA_IOU:
                # ensure the track is recent/alive (not extremely stale)
                if track["frames_alive"] > MAX_TRACK_FRAMES:
                    log(f"[TRACKER] track {track['id']} expired by lifetime (frames_alive={track['frames_alive']})")
                    active_track_by_source[src] = None
                    return
                # check recent scooper observations after roi entry or within temporal window
                lower_bound = max(track["roi_entry_frame"], fid - TEMP_WINDOW_FRAMES)
                recent_scooper = any(f >= lower_bound for f in track["scooper_seen_frames"])
                log(f"[TRACKER] candidate contact track={track['id']} p_i={p_i:.3f} scooper_recent={recent_scooper} fid={fid}")
                if not recent_scooper:
                    # VIOLATION
                    img = decode_image_from_msg(msg)
                    if img is not None:
                        fname = f"{src.replace('.','_')}_frame{fid}_violation.jpg"
                        fpath = os.path.join(VIOLATION_DIR, fname)
                        cv2.imwrite(fpath, img)
                    else:
                        fpath = None
                    details = {
                        "track_id": track["id"],
                        "hand_bbox": track["bbox"],
                        "pizza_bbox": p_box,
                        "scooper_seen_frames": sorted(list(track["scooper_seen_frames"])),
                        "frame_id": fid,
                        "source_id": src
                    }
                    save_violation_record(src, fid, ts, fpath, details)
                    ev = {"type":"violation","source_id":src,"frame_id":fid,"timestamp":ts,"frame_path":fpath,"details":details}
                    publish_violation_event(ev)
                    log(f"[TRACKER] VIOLATION recorded (src={src} fid={fid}) saved:{fpath}")
                    # reset track after violation
                    active_track_by_source[src] = None
                    return
                else:
                    # scooper present -> clear and reset track lifecycle (no violation)
                    log(f"[TRACKER] cleared (scooper present) track {track['id']} src={src} fid={fid}")
                    active_track_by_source[src] = None
                    return

        # STEP 5: expire if too old
        if track["frames_alive"] >= MAX_TRACK_FRAMES:
            log(f"[TRACKER] track {track['id']} auto-expired after {track['frames_alive']} frames")
            active_track_by_source[src] = None
            return

    # nothing else to do this frame
    return

# RabbitMQ consumer
def callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        process_frame(msg)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("Tracker callback error:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

chan.basic_consume(queue="detections", on_message_callback=callback)
print("Tracker service started — waiting for detections...")
chan.start_consuming()
