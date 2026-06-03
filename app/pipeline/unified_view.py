"""Orchestrates collage, stitch, or grid unified view generation."""



from app.config import Settings

from app.models.pipeline import ProcessedImage, UnifiedViewResult

from app.models.responses import UnifiedViewMethod

from app.pipeline.angle_groups import filter_stitch_eligible, is_stitch_eligible

from app.pipeline.collage_composer import compose_collage_for_panorama

from app.pipeline.grid_composer import compose_hybrid_canvas, compose_labeled_grid, cv2_to_pil

from app.pipeline.overlap import build_overlap_graph, find_connected_components

from app.pipeline.stitcher import stitch_cluster





def build_unified_view(

    images: list[ProcessedImage],

    settings: Settings,

    quality_warnings: list[str],

    layout: str | None = None,

) -> UnifiedViewResult:

    layout_mode = (layout or settings.panorama_default_layout).strip().lower()



    if len(images) == 1:

        single = _cap_width(images[0].pil_image.copy(), settings.max_output_width_px)

        return UnifiedViewResult(

            image=single,

            method=UnifiedViewMethod.LABELED_GRID.value,

            stitching_confidence=0.0,

            quality_warnings=list(quality_warnings),

        )



    if layout_mode == "collage":

        canvas = compose_collage_for_panorama(images, settings)

        warnings = list(quality_warnings)

        if any(not is_stitch_eligible(img.label) for img in images):

            warnings.append("side_view_not_stitched")

        return UnifiedViewResult(

            image=canvas,

            method=UnifiedViewMethod.COLLAGE_CONTACT_SHEET.value,

            stitching_confidence=1.0,

            quality_warnings=warnings,

        )



    if layout_mode == "grid":

        grid = compose_labeled_grid(images, settings)

        return UnifiedViewResult(

            image=grid,

            method=UnifiedViewMethod.LABELED_GRID.value,

            stitching_confidence=0.0,

            quality_warnings=list(quality_warnings),

        )



    return _build_stitch_or_hybrid_view(images, settings, quality_warnings)





def _build_stitch_or_hybrid_view(

    images: list[ProcessedImage],

    settings: Settings,

    quality_warnings: list[str],

) -> UnifiedViewResult:

    eligible = filter_stitch_eligible(images)

    warnings = list(quality_warnings)

    if len(eligible) < len(images):

        warnings.append("side_view_not_stitched")



    if len(eligible) < 2:

        if eligible:

            canvas = _cap_width(eligible[0].pil_image.copy(), settings.max_output_width_px)

        else:

            canvas = compose_collage_for_panorama(images, settings)

            return UnifiedViewResult(

                image=canvas,

                method=UnifiedViewMethod.COLLAGE_CONTACT_SHEET.value,

                stitching_confidence=0.0,

                quality_warnings=warnings,

            )

        return UnifiedViewResult(

            image=canvas,

            method=UnifiedViewMethod.LABELED_GRID.value,

            stitching_confidence=0.0,

            quality_warnings=warnings,

        )



    reverse_map = {i: img.index for i, img in enumerate(eligible)}



    edges = build_overlap_graph(eligible)

    clusters = find_connected_components(len(eligible), edges)

    original_clusters = [

        [reverse_map[i] for i in cluster] for cluster in clusters

    ]



    stitchable = [c for c in original_clusters if len(c) >= 2]

    stitched_results = [

        stitch_cluster(c, images) for c in stitchable

    ]

    successful = [r for r in stitched_results if r.success and r.stitched is not None]



    stitched_indices: set[int] = set()

    best_stitch = None

    best_confidence = 0.0



    for result in successful:

        if result.confidence >= best_confidence:

            best_confidence = result.confidence

            best_stitch = result

        stitched_indices.update(result.indices)



    orphan_indices = [i for i in range(len(images)) if i not in stitched_indices]

    orphans = [images[i] for i in orphan_indices]



    if best_stitch and best_stitch.stitched is not None:

        max_edge = max(settings.panorama_tag_thumb_max_edge, 160)

        if orphans:

            canvas = compose_hybrid_canvas(

                best_stitch.stitched,

                orphans,

                main_label="Stitched Panorama",

                settings=settings,

                inset_max_edge=max_edge,

            )

            method = UnifiedViewMethod.HYBRID.value

        else:

            canvas = cv2_to_pil(best_stitch.stitched)

            canvas = _cap_width(canvas, settings.max_output_width_px)

            method = UnifiedViewMethod.STITCHED_PANORAMA.value



        return UnifiedViewResult(

            image=canvas,

            method=method,

            stitching_confidence=best_confidence,

            quality_warnings=warnings,

        )



    grid = compose_labeled_grid(images, settings)

    return UnifiedViewResult(

        image=grid,

        method=UnifiedViewMethod.LABELED_GRID.value,

        stitching_confidence=0.0,

        quality_warnings=warnings,

    )





def _cap_width(image, max_width: int):

    from PIL import Image



    if not isinstance(image, Image.Image):

        raise TypeError("Expected PIL Image")

    if image.width <= max_width:

        return image

    scale = max_width / image.width

    return image.resize((max_width, int(image.height * scale)), Image.Resampling.LANCZOS)


