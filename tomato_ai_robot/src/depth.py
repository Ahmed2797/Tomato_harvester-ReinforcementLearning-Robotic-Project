# # src/depth.py
import numpy as np


def pixel_to_3d(x, y, depth, intrinsics):

    x, y = int(x), int(y)

    h, w = depth.shape

    # ✅ boundary check
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.array([0, 0, 0])

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    Z = depth[y, x]

    # ✅ handle missing depth
    if Z == 0:
        window = depth[max(0,y-2):y+3, max(0,x-2):x+3]
        valid = window[window > 0]

        if len(valid) == 0:
            return np.array([0, 0, 0])

        Z = np.mean(valid)

    Z = Z * 0.001  # mm → meter

    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return np.array([X, Y, Z])