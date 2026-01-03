"""
tools/select_roi_and_save.py

Interactive polygon selector:
- Click to add polygon vertices (left click)
- Press 'r' to reset current polygon
- Press ENTER to finish and save polygon
- Press ESC or 'q' to quit without saving

Saves:
- configs/rois.json  (adds/updates: source_id -> roi_name -> list of polygons (if multiple))
- tools/roi/<safe_source>_<roi_name>_roi_preview.jpg (image with overlay)
"""
# copy and paste this in prompt:
# python tools/select_roi.py --image "data/debug_detections/det_Right (2)_0.jpg" --source "Right (2).mp4" --name "all_containers"
import os, json, argparse, cv2, numpy as np
from pathlib import Path

def ensure_dirs():
    Path("configs").mkdir(parents=True, exist_ok=True)
    Path("tools/roi").mkdir(parents=True, exist_ok=True)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"ERROR: cannot read image {path}")
    return img

def draw_polygon_overlay(img, polygons, color=(0,255,0), alpha=0.35):
    overlay = img.copy()
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def save_roi_to_config(source_id, roi_name, polygon):
    cfg_path = Path("configs/rois.json")
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        except Exception:
            cfg = {}

    # Ensure structure: source_id -> roi_name -> list_of_polygons
    if source_id not in cfg:
        cfg[source_id] = {}
    if roi_name not in cfg[source_id]:
        cfg[source_id][roi_name] = []

    # Append polygon (list of [x,y])
    cfg[source_id][roi_name].append([[int(x), int(y)] for (x,y) in polygon])

    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2)
    print(f"Saved ROI under configs/rois.json -> {source_id} -> {roi_name} (total polygons: {len(cfg[source_id][roi_name])})")

def sanitize_filename(s):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to representative frame (jpg/png)")
    parser.add_argument("--source", "-s", required=True, help="Source id / video filename (e.g. Video.mp4)")
    parser.add_argument("--name", "-n", default="protein_container", help="ROI name (e.g. protein_container)")
    args = parser.parse_args()

    ensure_dirs()
    img = load_image(args.image)
    display = img.copy()
    polygons = []     # saved polygons for preview overlay
    current = []      # current polygon points

    win = "Select ROI - left click to add, 'r' reset, Enter finish/save, q/ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, display)

    def on_mouse(event, x, y, flags, param):
        nonlocal display, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            # draw small dot
            cv2.circle(display, (x,y), 3, (0,255,0), -1)
            # draw lines for current polygon
            if len(current) >= 2:
                cv2.line(display, current[-2], current[-1], (0,255,0), 1)
            cv2.imshow(win, display)

    cv2.setMouseCallback(win, on_mouse)

    print("Instructions: left click to add vertices. Press ENTER to finish/save polygon. Press 'r' to reset current polygon. Press 'q' or ESC to quit without saving.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter: finish current polygon
            if len(current) < 3:
                print("Need at least 3 points for a polygon. Keep clicking.")
                continue
            polygons.append(list(current))
            # redraw display with overlay of all polygons
            display = draw_polygon_overlay(img, polygons, color=(0,255,0), alpha=0.35)
            cv2.imshow(win, display)
            # save polygon to config and write preview image
            save_roi_to_config(args.source, args.name, current)
            safe_src = sanitize_filename(args.source)
            preview_fname = f"tools/roi/{safe_src}_{args.name}_roi_preview.jpg"
            preview_img = draw_polygon_overlay(img, polygons)
            cv2.imwrite(preview_fname, preview_img)
            print("Saved preview image to", preview_fname)
            # reset current polygon to allow adding another
            current = []
            # continue to allow drawing more polygons if desired
        elif key == ord('r'):
            # reset current polygon (not saved)
            print("Resetting current polygon.")
            current = []
            # redraw base overlay (existing polygons)
            display = draw_polygon_overlay(img, polygons) if polygons else img.copy()
            cv2.imshow(win, display)
        elif key == ord('q') or key == 27:
            print("Quitting without adding new polygon.")
            break
        else:
            # ignore other keys; user may click more
            pass

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
