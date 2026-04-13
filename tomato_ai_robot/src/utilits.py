import cv2

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
