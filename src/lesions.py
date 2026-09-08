"""Which retinal features coincide with the model's attention.

Grad-CAM answers *where* the network looked. It does not say *what* it found
there. This module detects the lesion types a clinician grades on, using
classical image processing, and measures how strongly each one is enriched
inside the attended region.

An important limit, stated plainly: APTOS 2019 carries image-level grades only,
with no lesion annotations, so nothing here is a trained lesion detector and
none of it proves causation. These are hand-written detectors, and the output
is a *correlation* between independently detected lesions and where the model
looked. It is evidence about the model's behaviour, not a readout of its
reasoning.

The four features are the ones the international grading scale is built on:

  red lesions  microaneurysms and haemorrhages -- the earliest sign of DR and
               the primary criterion separating grades 1 to 3
  exudates     hard exudates, the yellow lipid deposits of leaking vessels
  vessels      the retinal vasculature, used to exclude vessels from the red
               lesion mask rather than as a finding in itself
  optic disc   the bright nerve head; naturally bright and yellow, so it is
               masked out or it dominates every exudate detector
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocess import crop_black_borders

WORK_SIZE = 512

FEATURES = {
    "red_lesions": {
        "label": "Microaneurysms / haemorrhages",
        "short": "Red lesions",
        "colour": (214, 59, 59),
        "meaning": "Earliest and most decisive sign of diabetic retinopathy. "
                   "Their number and spread drive grades 1 to 3.",
    },
    "exudates": {
        "label": "Hard exudates",
        "short": "Exudates",
        "colour": (250, 178, 25),
        "meaning": "Yellow lipid deposits left by leaking capillaries. "
                   "Indicate vascular permeability and, near the macula, "
                   "threaten vision directly.",
    },
    "vessels": {
        "label": "Vascular network",
        "short": "Vessels",
        "colour": (42, 120, 214),
        "meaning": "Shown for orientation. Not ranked as a finding: the "
                   "vasculature covers a large, roughly uniform share of every "
                   "retina, so it scores near 1.0 whatever the model attended "
                   "to and would crowd out the real signal.",
    },
}

# Ranked as findings. Vessels are detected (to keep them out of the red-lesion
# mask) and drawn for orientation, but never ranked -- see the note above.
RANKED = ("red_lesions", "exudates")


# --------------------------------------------------------------------------
# Anatomy
# --------------------------------------------------------------------------
def to_working(image_rgb: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    """Crop the black frame and resize, without the Ben Graham step.

    Lesion detection needs the true colours: exudates are separated from
    haemorrhages by hue. The training-time enhancement deliberately destroys
    that by recentring every image on mid-grey.
    """
    img = crop_black_borders(image_rgb)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def field_of_view(img: np.ndarray) -> np.ndarray:
    """Mask of the circular retinal area, eroded to drop the rim."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = ((grey > 12).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.erode(mask, np.ones((9, 9), np.uint8))


def optic_disc(img: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Locate the nerve head as the brightest blurred region inside the FOV."""
    bright = img[:, :, 0].astype(np.float32) + img[:, :, 1].astype(np.float32)
    bright = cv2.GaussianBlur(bright, (0, 0), 12)
    bright[fov == 0] = 0
    _, _, _, centre = cv2.minMaxLoc(bright)
    disc = np.zeros(fov.shape, np.uint8)
    cv2.circle(disc, centre, int(img.shape[0] * 0.11), 255, -1)
    return disc, centre


# --------------------------------------------------------------------------
# Lesion detectors
# --------------------------------------------------------------------------
def detect_vessels(img: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Black top-hat at many orientations picks out dark elongated structures."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(3.0, (8, 8)).apply(grey)
    response = np.zeros_like(grey, np.float32)
    for angle in range(0, 180, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        rot = cv2.getRotationMatrix2D((7, 0), angle, 1)
        kernel = cv2.warpAffine(kernel.astype(np.float32), rot, (15, 15)).astype(np.uint8)
        if kernel.sum() == 0:
            continue
        response = np.maximum(
            response, cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        )
    inside = fov > 0
    if inside.sum() < 100:
        return np.zeros(fov.shape, np.uint8)
    threshold = max(12.0, float(np.percentile(response[inside], 96)))
    mask = ((response > threshold).astype(np.uint8)) * 255
    mask = cv2.medianBlur(mask, 3)
    mask[fov == 0] = 0
    return mask


def detect_exudates(img: np.ndarray, fov: np.ndarray, disc: np.ndarray,
                    vessels: np.ndarray) -> np.ndarray:
    """Bright, yellow, non-vascular spots outside the optic disc.

    Both an absolute floor and a percentile are required. A purely relative
    threshold always selects the brightest few percent of pixels, so a healthy
    retina would always appear to have exudates.
    """
    value = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    blue = img[:, :, 2].astype(np.float32)
    red_green = (img[:, :, 0].astype(np.float32) + img[:, :, 1].astype(np.float32)) / 2
    yellowness = red_green - blue

    valid = (fov > 0) & (disc == 0)
    if valid.sum() < 100:
        return np.zeros(fov.shape, np.uint8)

    v_thr = max(float(np.percentile(value[valid], 99.0)),
                float(np.median(value[valid])) + 45.0, 150.0)
    y_thr = max(float(np.percentile(yellowness[valid], 97.0)),
                float(np.median(yellowness[valid])) + 18.0)

    mask = (((value > v_thr) & (yellowness > y_thr) & valid).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[disc > 0] = 0
    mask[vessels > 0] = 0
    return mask


def detect_red_lesions(img: np.ndarray, fov: np.ndarray, disc: np.ndarray,
                       vessels: np.ndarray) -> np.ndarray:
    """Small dark round spots that are not part of the vessel tree.

    Haemorrhages and microaneurysms absorb green strongly, so they appear as
    local minima in the green channel. Subtracting a median-filtered copy
    removes slow illumination gradients.

    Candidates are then accepted or rejected **per blob** rather than per
    pixel. Two earlier pixel-wise steps were actively harmful:

    * a 3x3 morphological opening erased roughly 40% of candidates, because a
      microaneurysm is only 3-8 px across and an opening of that size erodes
      it away entirely;
    * deleting vessel pixels from the mask *fragmented* shapes, turning one
      blob into several and inflating the count instead of cleaning it.

    Judging whole components also allows the property that actually separates
    these lesions from vessels: they are compact and roughly round, whereas
    vessel fragments are elongated.
    """
    green = cv2.GaussianBlur(img[:, :, 1].astype(np.float32), (0, 0), 1.2)
    background = cv2.medianBlur(green.astype(np.uint8), 41).astype(np.float32)
    darkness = background - green

    valid = (fov > 0) & (disc == 0)
    if valid.sum() < 100:
        return np.zeros(fov.shape, np.uint8)

    # Lower than a pure high percentile: on a low-contrast retina the adaptive
    # threshold otherwise climbs so high that nothing survives.
    threshold = max(7.0, float(np.percentile(darkness[valid], 98.5)))
    candidates = ((darkness > threshold) & valid).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE,
                                  np.ones((2, 2), np.uint8))

    scale = img.shape[0] / 512.0
    min_area = max(3, int(3 * scale * scale))
    max_area = int(900 * scale * scale)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    keep = np.zeros(fov.shape, np.uint8)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Elongation: vessel fragments are long and thin, lesions are compact.
        if max(w, h) / max(min(w, h), 1) > 3.0:
            continue

        # Fill: a lesion fills most of its bounding box; a curved vessel
        # segment leaves most of it empty.
        if area / float(max(w * h, 1)) < 0.35:
            continue

        blob = labels == i
        # Reject only if the component sits mostly on the vessel tree. Judged
        # per blob, so a lesion touching a vessel survives intact.
        if (blob & (vessels > 0)).sum() / float(area) > 0.5:
            continue

        keep[blob] = 255

    keep[disc > 0] = 0
    return keep


def _count_blobs(mask: np.ndarray, min_area: int = 4) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)


def detect_all(image_rgb: np.ndarray, size: int = WORK_SIZE) -> dict:
    """Run every detector and return masks plus per-feature statistics."""
    img = to_working(image_rgb, size)
    fov = field_of_view(img)
    disc, disc_centre = optic_disc(img, fov)
    vessels = detect_vessels(img, fov)
    exudates = detect_exudates(img, fov, disc, vessels)
    red = detect_red_lesions(img, fov, disc, vessels)

    area = max(int((fov > 0).sum()), 1)
    masks = {"red_lesions": red, "exudates": exudates, "vessels": vessels}
    stats = {
        key: {
            "area_pct": 100.0 * float((mask > 0).sum()) / area,
            "count": _count_blobs(mask),
        }
        for key, mask in masks.items()
    }
    return {"image": img, "fov": fov, "disc": disc, "disc_centre": disc_centre,
            "masks": masks, "stats": stats}


# --------------------------------------------------------------------------
# Attention cross-reference
# --------------------------------------------------------------------------
def attention_enrichment(masks: dict, fov: np.ndarray, cam: np.ndarray,
                         top_fraction: float = 0.25) -> dict:
    """How concentrated each lesion type is inside the attended region.

    The attended region is the top `top_fraction` of Grad-CAM values within the
    field of view. Enrichment is the lesion density there divided by the
    density over the whole retina, so 1.0 means "no more than you would expect
    by area" and 3.0 means three times the average concentration.

    A ratio, not a raw overlap, because a heatmap covering a quarter of the
    image would otherwise "explain" everything simply by being large.
    """
    cam_resized = cv2.resize(cam.astype(np.float32), fov.shape[::-1],
                             interpolation=cv2.INTER_LINEAR)
    inside = fov > 0
    if inside.sum() < 100:
        return {}

    cutoff = float(np.quantile(cam_resized[inside], 1.0 - top_fraction))
    attended = inside & (cam_resized >= cutoff)
    if attended.sum() < 50:
        return {}

    out = {}
    for key, mask in masks.items():
        present = (mask > 0) & inside
        overall = present.sum() / inside.sum()
        local = (present & attended).sum() / attended.sum()
        out[key] = {
            "enrichment": float(local / overall) if overall > 0 else 0.0,
            "pct_of_lesion_in_focus": (
                100.0 * float((present & attended).sum()) / float(present.sum())
                if present.sum() else 0.0
            ),
            "density_pct": 100.0 * float(local),
        }
    return out


def explain(image_rgb: np.ndarray, cam: np.ndarray | None = None) -> dict:
    """Detect lesions and, when a Grad-CAM map is supplied, rank them.

    Findings are ordered by enrichment: the feature most concentrated where
    the model looked comes first.
    """
    result = detect_all(image_rgb)
    findings = []

    enrichment = (attention_enrichment(result["masks"], result["fov"], cam)
                  if cam is not None else {})

    for key in RANKED:
        meta, stat = FEATURES[key], result["stats"][key]
        findings.append({
            "key": key, "label": meta["label"], "short": meta["short"],
            "colour": meta["colour"], "meaning": meta["meaning"],
            "count": stat["count"], "area_pct": stat["area_pct"],
            **enrichment.get(key, {}),
        })

    if enrichment:
        findings.sort(key=lambda f: f.get("enrichment", 0.0), reverse=True)

    # A retina with almost nothing detected is itself the explanation for a
    # low grade, so say so rather than reporting a meaningless top feature.
    total = sum(f["count"] for f in findings)
    result["findings"] = findings
    result["sparse"] = total < 6
    result["total_lesions"] = total
    return result


def overlay_lesions(img: np.ndarray, masks: dict) -> np.ndarray:
    """Mark each detected lesion type on the retina.

    Red lesions are drawn as rings rather than filled blobs. A dark red dot on
    an orange retina is nearly invisible, whereas an outline reads against any
    background, and a ring leaves the lesion itself visible underneath so the
    viewer can judge the detection instead of taking it on trust.
    """
    out = img.copy()

    vessels = masks.get("vessels")
    if vessels is not None:                                 # faint, underneath
        colour = np.array(FEATURES["vessels"]["colour"], np.float32)
        sel = vessels > 0
        out[sel] = (0.30 * colour + 0.70 * out[sel]).astype(np.uint8)

    exudates = masks.get("exudates")
    if exudates is not None:
        colour = np.array(FEATURES["exudates"]["colour"], np.float32)
        sel = cv2.dilate(exudates, np.ones((2, 2), np.uint8)) > 0
        out[sel] = (0.85 * colour + 0.15 * out[sel]).astype(np.uint8)

    red = masks.get("red_lesions")
    if red is not None:
        colour = tuple(int(c) for c in FEATURES["red_lesions"]["colour"])
        contours, _ = cv2.findContours((red > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 3:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            centre = (int(cx), int(cy))
            r = max(int(radius) + 3, 5)
            cv2.circle(out, centre, r + 1, (20, 20, 20), 2, cv2.LINE_AA)  # halo
            cv2.circle(out, centre, r, colour, 2, cv2.LINE_AA)
    return out
