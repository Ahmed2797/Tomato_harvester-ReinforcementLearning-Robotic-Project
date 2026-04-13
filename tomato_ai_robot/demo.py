import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("models/best(150).pt")

# -----------------------------
# LOAD IMAGE
# -----------------------------
frame = cv2.imread("video/demo-image.png")

# -----------------------------
# PREDICT
# -----------------------------
result = model.predict(frame, conf=0.5)[0]

names = model.model.names  # ✅ FIX

# -----------------------------
# FILTER (ONLY RIPE)
# -----------------------------
allowed = {"l_fully_ripened", "l_half_ripened"}

boxes = result.boxes.xyxy.cpu().numpy()
classes = result.boxes.cls.cpu().numpy()

# copy image for drawing
annotated = frame.copy()

for box, cls_id in zip(boxes, classes):
    label = names[int(cls_id)]

    if label not in allowed:
        continue

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        annotated,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# -----------------------------
# SHOW RESULT
# -----------------------------
cv2.imshow("Only Ripe Tomatoes", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()