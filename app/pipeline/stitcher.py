"""OpenCV panorama stitcher wrapper."""

import cv2
import numpy as np

from app.models.pipeline import ProcessedImage, StitchClusterResult


def stitch_cluster(indices: list[int], images: list[ProcessedImage]) -> StitchClusterResult:
    if len(indices) < 2:
        idx = indices[0]
        return StitchClusterResult(
            indices=indices,
            stitched=images[idx].cv_image.copy(),
            confidence=0.0,
            success=False,
        )

    cluster_images = [images[i].cv_image for i in indices]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(cluster_images)

    if status != cv2.Stitcher_OK or pano is None:
        return StitchClusterResult(
            indices=indices,
            stitched=None,
            confidence=0.0,
            success=False,
        )

    confidence = min(1.0, 0.55 + 0.08 * len(indices))
    return StitchClusterResult(
        indices=indices,
        stitched=pano,
        confidence=round(confidence, 3),
        success=True,
    )
