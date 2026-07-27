"""Technical quality assessment (CLAUDE.md §4.1). Pure CV — no model weights.

The mistake this module exists to avoid: measuring sharpness over the whole
frame. Long-lens wildlife work is mostly intentional background blur, so a
whole-frame variance-of-Laplacian scores excellent photographs badly and the
metric ends up inversely correlated with how good the lens is.

The insight that makes subject detection cheap here is that the same shallow
depth of field which breaks global sharpness also *identifies* the subject: for
this kind of photograph the in-focus region essentially is the subject. So the
detector uses a contrast-normalised focus map rather than an object detector,
which means no weights to download and nothing extra to freeze into a binary.

Circularity is worth addressing head on: finding the sharpest region and then
reporting that it is sharp sounds like cheating. It is not, because the detector
only chooses *where* to measure. On a genuinely soft frame it still finds a
region, but the absolute sharpness there is low, so the score is low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MelampusConfig


@dataclass
class QualityResult:
    """Every sub-metric individually, plus the composite. Nothing hidden.

    §4.1 is explicit that the sub-metrics must stay visible: the weights need
    tuning, and nobody trusts an opaque score.
    """

    file: str = ""
    composite: float = 0.0

    subject_detected: bool = False
    subject_box: tuple[int, int, int, int] | None = None
    subject_size_fraction: float = 0.0
    subject_confidence: float = 0.0

    subject_sharpness: float = 0.0
    eye_sharpness: float = 0.0
    eye_detected: bool = False
    frame_sharpness: float = 0.0

    motion_blur: float = 0.0
    defocus_blur: float = 0.0
    blur_angle_degrees: float = 0.0

    clipped_highlights_pct: float = 0.0
    clipped_shadows_pct: float = 0.0

    subject_edge_clipped: bool = False
    clipped_edges: list[str] = field(default_factory=list)

    width: int = 0
    height: int = 0
    error: str | None = None


def _focus_map(gray: np.ndarray, window: int) -> np.ndarray:
    """Contrast-normalised local high-frequency energy.

    Dividing Laplacian energy by local variance is what separates a sharp edge
    from a blurred but high-contrast one. Out-of-focus highlights have enormous
    contrast and very little high-frequency content, which is exactly the case a
    raw Laplacian gets wrong on bokeh.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.boxFilter(lap * lap, -1, (window, window), normalize=True)
    mean = cv2.boxFilter(gray, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, -1, (window, window), normalize=True)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    # The floor keeps featureless sky from producing a huge ratio out of noise.
    return energy / (variance + 16.0)


def _saliency(gray: np.ndarray, size: int, blur_sigma: float) -> np.ndarray:
    """Spectral-residual saliency, in a dozen lines of numpy.

    Implemented here rather than pulling in opencv-contrib for one function:
    smaller dependency surface and a smaller frozen binary.
    """
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    smoothed = cv2.blur(log_amplitude.astype(np.float32), (3, 3))
    residual = log_amplitude - smoothed
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    magnitude = np.abs(reconstructed) ** 2
    magnitude = cv2.GaussianBlur(magnitude.astype(np.float32), (0, 0), blur_sigma)
    span = float(magnitude.max() - magnitude.min())
    if span <= 0:
        return np.zeros_like(magnitude)
    return (magnitude - magnitude.min()) / span


def _curve(raw: float, low: float, high: float) -> float:
    """Map a raw focus value onto 0-100 through a log curve."""
    if raw <= 0:
        return 0.0
    lo, hi = np.log(max(low, 1e-9)), np.log(max(high, 1e-8))
    if hi <= lo:
        return 0.0
    return float(np.clip((np.log(raw) - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def _detect_subject(focus: np.ndarray, saliency_small: np.ndarray, cfg) -> tuple:
    """Pick the primary subject region from the focus map, arbitrated by saliency."""
    height, width = focus.shape
    threshold = float(np.percentile(focus, cfg.focus_percentile))
    mask = (focus >= threshold).astype(np.uint8)

    kernel = np.ones((cfg.merge_dilate, cfg.merge_dilate), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, 0.0

    saliency = cv2.resize(saliency_small, (width, height), interpolation=cv2.INTER_LINEAR)
    frame_area = float(width * height)
    best, best_score = None, -1.0

    for index in range(1, count):
        x, y, w, h, area = stats[index]
        area_fraction = area / frame_area
        if area_fraction < cfg.min_blob_area_frac:
            continue
        region = labels[y:y + h, x:x + w] == index
        mean_focus = float(focus[y:y + h, x:x + w][region].mean())
        mean_saliency = float(saliency[y:y + h, x:x + w][region].mean())

        cx, cy = centroids[index]
        offset = np.hypot((cx - width / 2) / width, (cy - height / 2) / height)
        centrality = 1.0 - cfg.centrality_strength * float(offset)

        score = (mean_focus * (area_fraction ** cfg.area_exponent)
                 * centrality * (0.5 + 0.5 * mean_saliency))
        if score > best_score:
            best_score, best = score, (int(x), int(y), int(w), int(h))

    if best is None:
        return None, 0.0

    x, y, w, h = best
    pad_x, pad_y = int(w * cfg.box_pad_frac), int(h * cfg.box_pad_frac)
    x = max(0, x - pad_x); y = max(0, y - pad_y)
    w = min(width - x, w + 2 * pad_x); h = min(height - y, h + 2 * pad_y)

    inside = focus[y:y + h, x:x + w].mean()
    outside_mask = np.ones(focus.shape, bool)
    outside_mask[y:y + h, x:x + w] = False
    outside = float(focus[outside_mask].mean()) if outside_mask.any() else 1e-9
    confidence = float(np.clip((inside / (outside + 1e-9)) / 6.0, 0.0, 1.0))

    return (x, y, w, h), confidence


def _blur_split(gray: np.ndarray, box, total_blur: float, cfg) -> tuple[float, float, float]:
    """Separate directional blur from isotropic defocus (§4.1).

    An aggregate structure tensor over the subject: motion blur suppresses
    gradients along the direction of travel, so the summed tensor becomes
    anisotropic. Defocus attenuates all directions equally.
    """
    x, y, w, h = box
    patch = gray[y:y + h, x:x + w]
    if patch.size < 64:
        return 0.0, total_blur, 0.0

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float((gx * gx).sum())
    jyy = float((gy * gy).sum())
    jxy = float((gx * gy).sum())
    trace = jxx + jyy
    if trace <= 0:
        return 0.0, total_blur, 0.0

    anisotropy = float(np.hypot(jxx - jyy, 2 * jxy) / trace)
    scaled = float(np.clip(
        (anisotropy - cfg.anisotropy_floor) / max(cfg.anisotropy_ceiling - cfg.anisotropy_floor, 1e-6),
        0.0, 1.0))

    # Dominant gradient direction; motion runs perpendicular to it.
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    angle = float((np.degrees(theta) + 90.0) % 180.0)

    return total_blur * scaled, total_blur * (1.0 - scaled), angle


def _find_eye(gray: np.ndarray, box, cfg) -> tuple[bool, tuple[int, int, int] | None, float]:
    """Catchlight detection: a small bright specular blob on a darker iris.

    Pure CV, so it is deliberately conservative — water droplets and bright
    feather edges are the obvious false positives, which is why confidence is
    reported and used to gate how much the eye metric counts.
    """
    x, y, w, h = box
    patch = gray[y:y + h, x:x + w]
    if patch.size < 256:
        return False, None, 0.0

    cutoff = max(float(np.percentile(patch, cfg.search_percentile)),
                 float(cfg.min_absolute_brightness))
    bright = (patch >= cutoff).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(bright, 8)
    if count <= 1:
        return False, None, 0.0

    subject_area = float(w * h)
    best, best_score = None, 0.0

    for index in range(1, count):
        bx, by, bw, bh, area = stats[index]
        fraction = area / subject_area
        if not (cfg.min_area_frac_of_subject <= fraction <= cfg.max_area_frac_of_subject):
            continue
        # Near-circular: a catchlight is a point source, not a streak.
        extent = area / float(max(bw * bh, 1))
        aspect = min(bw, bh) / float(max(bw, bh, 1))
        if extent < cfg.min_circularity or aspect < 0.5:
            continue

        radius = max(2, int(np.sqrt(area / np.pi)))
        cx, cy = int(centroids[index][0]), int(centroids[index][1])
        ring_outer = radius + cfg.ring_dilate
        rx0, ry0 = max(0, cx - ring_outer), max(0, cy - ring_outer)
        rx1, ry1 = min(patch.shape[1], cx + ring_outer), min(patch.shape[0], cy + ring_outer)
        ring = patch[ry0:ry1, rx0:rx1]
        if ring.size == 0:
            continue
        contrast = float(patch[by:by + bh, bx:bx + bw].mean() - ring.mean())
        if contrast < cfg.min_ring_contrast:
            continue

        score = contrast * extent
        if score > best_score:
            best_score = score
            best = (x + cx, y + cy, radius)

    if best is None:
        return False, None, 0.0
    confidence = float(np.clip(best_score / 120.0, 0.0, 1.0))
    return True, best, confidence


def analyze_quality(path: str | Path, config: MelampusConfig) -> QualityResult:
    """Score one photograph. Never raises — a bad file degrades to an error field."""
    path = Path(path)
    cfg = config.quality
    result = QualityResult(file=path.name)

    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            rgb_full = np.asarray(oriented.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.height, result.width = rgb_full.shape[:2]

    # Clipping is measured on the ORIGINAL pixels. Downsampling averages clipped
    # pixels together with their neighbours and hides blown highlights.
    highs = (rgb_full >= cfg.highlight_threshold).any(axis=2)
    lows = (rgb_full <= cfg.shadow_threshold).all(axis=2)
    total_px = float(rgb_full.shape[0] * rgb_full.shape[1])
    result.clipped_highlights_pct = float(highs.sum() / total_px * 100.0)
    result.clipped_shadows_pct = float(lows.sum() / total_px * 100.0)

    # Sharpness is scale dependent, so normalise to a working size first or the
    # same photo scores differently at different export resolutions.
    scale = cfg.working_long_edge / max(rgb_full.shape[:2])
    if scale < 1.0:
        working = cv2.resize(rgb_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        working = rgb_full
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY).astype(np.float32)

    focus = _focus_map(gray, cfg.focus_window)
    saliency_small = _saliency(gray, cfg.saliency_resize, cfg.saliency_blur_sigma)

    # The naive whole-frame number, kept so the difference is visible rather than
    # asserted. On a bokeh frame this is low while the subject is high.
    result.frame_sharpness = _curve(float(focus.mean()), cfg.knee_low, cfg.knee_high)

    box, subject_confidence = _detect_subject(focus, saliency_small, cfg)
    if box is None:
        h, w = gray.shape
        box = (w // 4, h // 4, w // 2, h // 2)
        result.subject_detected = False
    else:
        result.subject_detected = True
    result.subject_confidence = subject_confidence

    x, y, w, h = box
    frame_h, frame_w = gray.shape
    result.subject_box = (x, y, w, h)
    result.subject_size_fraction = float(w * h) / float(frame_w * frame_h)

    # High percentile inside the box: how sharp the sharpest structure on the
    # subject is, which is what a photographer actually judges.
    region = focus[y:y + h, x:x + w]
    raw_subject = float(np.percentile(region, cfg.region_percentile)) if region.size else 0.0

    # Small subjects give noisy estimates from few pixels; a gentle gain stops a
    # distant-but-sharp bird being punished for being distant.
    gain = 1.0
    if cfg.size_gain_strength > 0 and result.subject_size_fraction > 0:
        deficit = max(0.0, float(np.log(cfg.size_reference_frac /
                                        max(result.subject_size_fraction, 1e-6))))
        gain = min(cfg.size_gain_max, 1.0 + cfg.size_gain_strength * deficit)
    result.subject_sharpness = float(np.clip(
        _curve(raw_subject, cfg.knee_low, cfg.knee_high) * gain, 0.0, 100.0))

    total_blur = max(0.0, 100.0 - result.subject_sharpness)
    result.motion_blur, result.defocus_blur, result.blur_angle_degrees = _blur_split(
        gray, box, total_blur, cfg)

    found, eye, eye_conf = _find_eye(gray, box, cfg)
    result.eye_detected = found and eye_conf >= cfg.min_eye_confidence
    if result.eye_detected and eye is not None:
        ex, ey, radius = eye
        span = int(radius * cfg.patch_radius_mult)
        px0, py0 = max(0, ex - span), max(0, ey - span)
        px1, py1 = min(frame_w, ex + span), min(frame_h, ey + span)
        patch = focus[py0:py1, px0:px1]
        if patch.size:
            result.eye_sharpness = _curve(
                float(np.percentile(patch, cfg.region_percentile)), cfg.knee_low, cfg.knee_high)

    margin_x = max(1, int(frame_w * cfg.edge_margin_frac))
    margin_y = max(1, int(frame_h * cfg.edge_margin_frac))
    edges = []
    if x <= margin_x: edges.append("left")
    if y <= margin_y: edges.append("top")
    if x + w >= frame_w - margin_x: edges.append("right")
    if y + h >= frame_h - margin_y: edges.append("bottom")
    result.clipped_edges = edges
    result.subject_edge_clipped = bool(edges)

    result.composite = _composite(result, cfg)
    return result


def _composite(r: QualityResult, cfg) -> float:
    """Weighted mean of goodness terms. Weights come from config, never code."""
    def penalty(pct: float, tolerance: float, full: float) -> float:
        excess = max(0.0, pct - tolerance)
        return float(np.clip(100.0 - (excess / max(full, 1e-6)) * 100.0, 0.0, 100.0))

    exposure = min(
        penalty(r.clipped_highlights_pct, cfg.highlight_tolerance_pct, cfg.highlight_full_penalty_pct),
        penalty(r.clipped_shadows_pct, cfg.shadow_tolerance_pct, cfg.shadow_full_penalty_pct),
    )

    terms: list[tuple[float, float]] = [
        (cfg.weight_subject_sharpness, r.subject_sharpness),
        (cfg.weight_focus, 100.0 - r.defocus_blur),
        (cfg.weight_exposure, exposure),
        (cfg.weight_motion, 100.0 - r.motion_blur),
    ]
    # Eye sharpness only counts when an eye was actually found; otherwise its
    # weight flows to subject sharpness rather than scoring a zero.
    if r.eye_detected:
        terms.append((cfg.weight_eye_sharpness, r.eye_sharpness))
    else:
        terms.append((cfg.weight_eye_sharpness, r.subject_sharpness))

    total_weight = sum(weight for weight, _ in terms)
    if total_weight <= 0:
        return 0.0
    return float(np.clip(sum(w * v for w, v in terms) / total_weight, 0.0, 100.0))
