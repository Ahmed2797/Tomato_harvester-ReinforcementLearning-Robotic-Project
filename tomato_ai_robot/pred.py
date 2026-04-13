import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from trackers import ByteTrackTracker
from ros_nodes.moveit_control import RobotArm

from src.depth import *


# -----------------------------
# INIT MODEL + TRACKER
# -----------------------------
print("[INFO] Loading YOLO model...")
model = YOLO("models/best(150).pt")
tracker = ByteTrackTracker()
robot = RobotArm()

# -----------------------------
# VIDEO SOURCE
# -----------------------------
SOURCE_VIDEO_PATH = "video/tomato.mp4"

# -----------------------------
# CLUSTER FUNCTION (optional later)
# -----------------------------

def cluster_tomatoes(tomatoes):
    """
    Robust clustering for robotics pipeline.
    Works with:
    - pos (3D)
    - center (2D fallback)
    """

    if len(tomatoes) == 0:
        return tomatoes

    coords = []

    for t in tomatoes:
        print(t)
        # SAFE extraction (no crash)
        if "pos" in t and t["pos"] is not None:
            coords.append(t["pos"][:2])
        elif "center" in t:
            coords.append(t["center"])
        else:
            coords.append((0, 0))  # fallback safety

    coords = np.array(coords)

    # DBSCAN clustering
    labels = DBSCAN(eps=80, min_samples=2).fit(coords).labels_

    # attach cluster id
    for i, t in enumerate(tomatoes):
        t["cluster_id"] = int(labels[i])

    return tomatoes


# -----------------------------
# DRAW PICKING ORDER
# -----------------------------
def draw_picking_order(image, tomatoes, order):
    img = image.copy()

    for i, idx in enumerate(order):
        cx, cy = tomatoes[idx]["center"]
        cx, cy = int(cx), int(cy)

        cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

        cv2.putText(
            img,
            str(i + 1),
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return img

# -----------------------------
# VIDEO GENERATOR
# -----------------------------
generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(
    text_scale=0.4,
    text_thickness=1,
    text_position=sv.Position.TOP_LEFT,
    smart_position=True,
    text_padding=5
)

allowed = {"l_fully_ripened", "l_half_ripened"}

# -----------------------------
# MAIN LOOP
# -----------------------------
for frame in generator:
    frame = cv2.resize(frame, (640, 480))

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    result = model.predict(frame, conf=0.45)[0]
    detections = sv.Detections.from_ultralytics(result)

    # -----------------------------
    # BYTE TRACKING
    # -----------------------------
    detections_tracker = tracker.update(detections)

    # -----------------------------
    # LABELS
    # -----------------------------
    names = model.model.names
    labels = []

    if detections_tracker.class_id is not None:
        for cls_id, tracker_id in zip(
            detections_tracker.class_id,
            detections_tracker.tracker_id
        ):
            labels.append(f"ID:{tracker_id} {names[cls_id]}")

    # -----------------------------
    # BUILD TOMATO LIST (FIXED)
    # -----------------------------
    tomatoes = []

    if detections_tracker.xyxy is not None:

        for i, (box, cls_id, tracker_id) in enumerate(
            zip(
                detections_tracker.xyxy,
                detections_tracker.class_id,
                detections_tracker.tracker_id
            )
        ):

            label = names[cls_id]

            # ✅ FILTER ONLY RIPE TOMATOES
            if label in allowed:

                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                tomatoes.append({
                    "id": int(tracker_id) if tracker_id != -1 else i,
                    "label": label,
                    "center": (cx, cy),
                    "cluster_id": -1
                })

    # -----------------------------
    # SIMPLE PICK ORDER
    # -----------------------------
    # -----------------------------
    # 5. CLUSTERING
    # -----------------------------
    tomatoes = cluster_tomatoes(tomatoes)



    # -----------------------------
    # 6. PICKING ORDER (simple heuristic)
    # -----------------------------
    tomatoes = sorted(tomatoes, key=lambda t: t["pos"][2])  # nearest first


    # -----------------------------
    # 7. ROBOT EXECUTION (SIMULATION)
    # -----------------------------
    for t in tomatoes:
        x, y, z = t["pos"]

        robot.move_to(x, y, z)
        robot.pick()

    order = list(range(len(tomatoes)))

    # -----------------------------
    # VISUALIZATION BASE
    # -----------------------------
    annotated = frame.copy()
    # annotated = result.plot()

    annotated = box_annotator.annotate(
        scene=annotated,
        detections=detections_tracker
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections_tracker,
        labels=labels
    )

    # -----------------------------
    # DRAW PICK ORDER
    # -----------------------------
    annotated = draw_picking_order(annotated, tomatoes, order)

    # -----------------------------
    # SHOW
    # -----------------------------
    cv2.imshow("System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()