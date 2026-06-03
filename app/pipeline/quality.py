"""Image quality scoring."""

import cv2
import numpy as np


def score_blur(cv_image: np.ndarray) -> float:
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_image_quality(blur_scores: list[float]) -> float:
    if not blur_scores:
        return 0.0
    normalized = [min(1.0, score / 200.0) for score in blur_scores]
    return round(sum(normalized) / len(normalized), 3)
