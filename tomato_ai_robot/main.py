import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from src.depth import pixel_to_3d
from src.cluster import cluster_tomatoes
from trackers import ByteTrackTracker
from ros_nodes.moveit_control import RobotArm


# -----------------------------
# INIT
# -----------------------------
print("[INFO] Loading system...")

model = YOLO("models/best(150).pt")  # segmentation model
tracker = ByteTrackTracker()
robot = RobotArm()

cap = cv2.VideoCapture("video/tomato.mp4")

intrinsics = {
    "fx": 600,
    "fy": 600,
    "cx": 320,
    "cy": 240
}


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (720, 480))

    # -----------------------------
    # YOLO SEGMENTATION
    # -----------------------------
    result = model.predict(frame, conf=0.3)[0]

    detections = sv.Detections.from_ultralytics(result)

    # -----------------------------
    # TRACKING (stable IDs)
    # -----------------------------
    tracked = tracker.update(detections)

    names = model.model.names
    print(names)

    # -----------------------------
    # DEPTH MAP (simple approximation)
    # -----------------------------
    depth_map = np.zeros((480, 640), dtype=np.float32)
    tomato_diameter_mm = 70.0
    focal_px = 0.5 * (intrinsics["fx"] + intrinsics["fy"])

    tomatoes = []

    if tracked.xyxy is not None and tracked.class_id is not None:

        # build depth first
        for box in tracked.xyxy:
            x1, y1, x2, y2 = box.astype(int)

            box_size = max(1, (x2 - x1 + y2 - y1) / 2)
            depth = (focal_px * tomato_diameter_mm) / box_size

            depth_map[y1:y2, x1:x2] = depth

        # build tomato objects
        for i, box in enumerate(tracked.xyxy):

            x1, y1, x2, y2 = box.astype(int)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cls_id = tracked.class_id[i]
            label = names[cls_id]

            # filter only ripe tomatoes
            if label not in ["l_fully_ripened", "l_half_ripened"]:
                continue

            pos = pixel_to_3d(cx, cy, depth_map, intrinsics)

            tomatoes.append({
                "id": int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i,
                "label": label,
                "center": (cx, cy),
                "pos": pos
            })

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    tomatoes = cluster_tomatoes(tomatoes)

    # IMPORTANT FIX: safety check
    tomatoes = [t for t in tomatoes if "pos" in t and t["pos"] is not None]

    # -----------------------------
    # PICK ORDER (nearest first)
    # -----------------------------
    tomatoes = sorted(tomatoes, key=lambda t: t["pos"][2])
    order = list(range(len(tomatoes)))

    # -----------------------------
    # ROBOT (DO NOT BLOCK LOOP in real system)
    # -----------------------------
    for t in tomatoes:
        x, y, z = t["pos"]
        robot.move_to(x, y, z)
        robot.pick()

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    annotated = result.plot(conf=False,font_size=0.1,labels=True,boxes=True,line_width=1)


    for t in tomatoes:
        cx, cy = t["center"]

        cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), 1)

        # cv2.putText(
        #     annotated,
        #     f"{t['id']} {t['label']}",
        #     (cx + 5, cy - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     2
        # )

    cv2.imshow("Tomato Robotics System", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()