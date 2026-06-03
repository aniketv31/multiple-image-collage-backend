"""Feature-based overlap detection between image pairs."""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from app.models.pipeline import OverlapEdge, ProcessedImage

MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
MAX_ROTATION_DEG = 35.0
SCALE_MIN = 0.5
SCALE_MAX = 2.0


def _homography_is_coplanar(H: np.ndarray) -> bool:
    """Reject side-view / extreme transforms unsuitable for panorama stitch."""
    if H is None or H.shape != (3, 3):
        return False
    h33 = float(H[2, 2])
    if abs(h33) < 1e-8:
        return False
    Hn = H / h33
    scale_x = math.sqrt(Hn[0, 0] ** 2 + Hn[1, 0] ** 2)
    scale_y = math.sqrt(Hn[0, 1] ** 2 + Hn[1, 1] ** 2)
    if not (SCALE_MIN <= scale_x <= SCALE_MAX and SCALE_MIN <= scale_y <= SCALE_MAX):
        return False
    angle_deg = abs(math.degrees(math.atan2(Hn[1, 0], Hn[0, 0])))
    return angle_deg <= MAX_ROTATION_DEG


def _match_pair(i: int, j: int, img_a: np.ndarray, img_b: np.ndarray) -> OverlapEdge | None:
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < MIN_INLIERS:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None or not _homography_is_coplanar(H):
        return None

    inliers = int(mask.sum())
    ratio = inliers / len(good)
    if inliers >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO:
        return OverlapEdge(i=i, j=j, inlier_count=inliers, inlier_ratio=ratio)
    return None


def build_overlap_graph(images: list[ProcessedImage]) -> list[OverlapEdge]:
    edges: list[OverlapEdge] = []
    n = len(images)
    if n < 2:
        return edges

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as executor:
        futures = {
            executor.submit(
                _match_pair,
                i,
                j,
                images[i].cv_image,
                images[j].cv_image,
            ): (i, j)
            for i, j in pairs
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                edges.append(result)

    return edges


def find_connected_components(n: int, edges: list[OverlapEdge]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(edge.i, edge.j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return sorted(clusters.values(), key=len, reverse=True)
