
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    label as component_labels,
    maximum_filter,
)

# Pillow compatibility
try:
    PIL_BOX = Image.Resampling.BOX
except AttributeError:
    PIL_BOX = Image.BOX


# ============================================================
# SETTINGS
# ============================================================

@dataclass(frozen=True)
class Settings:
    # Morphology decision
    min_morphology_score: float = 0.42
    min_morphology_margin: float = 0.10

    # Handedness / spiral reliability
    min_handedness_margin: float = 0.35
    min_stability: float = 0.90
    min_coherence: float = 0.45
    min_explained: float = 0.20
    min_vs_straight: float = 1.30
    min_arm_to_sky: float = 0.45
    min_axis_ratio_for_handedness: float = 0.45

    # Source quality
    min_source_pixels: int = 12
    min_snr_proxy: float = 12.0
    min_peak_snr: float = 5.0
    min_r90_resolved: float = 5.0
    min_r50_resolved: float = 1.5

    # Robustness
    noise_trials: int = 12
    random_seed: int = 7


# ============================================================
# BASIC UTILITIES
# ============================================================

def mad(a):
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def robust_scale(gray):
    lo, hi = np.percentile(gray, [1, 99.5])
    scale = hi - lo
    if scale <= np.finfo(float).eps * max(1.0, abs(hi)):
        scale = max(float(np.ptp(gray)), 1.0)
    return (gray - lo) / scale


def load_image(source, crop=None, max_side=None):
    """
    Load PNG/JPEG/TIFF/BMP (and arrays).

    crop = (left, top, right, bottom) is applied BEFORE any optional resize.
    max_side=None keeps native crop resolution.
    """

    title = "array"

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path.resolve()}")

        title = path.name
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)

        if crop is not None:
            im = im.crop(tuple(map(int, crop)))

        native_shape = (im.height, im.width)

        if im.mode in ("I", "F", "I;16", "L"):
            raw_gray = np.asarray(im, dtype=np.float32)
            raw_rgb = np.repeat(raw_gray[..., None], 3, axis=2)
        else:
            rgb_im = im.convert("RGB")
            raw_rgb = np.asarray(rgb_im, dtype=np.float32)
            raw_gray = (
                0.2126 * raw_rgb[..., 0]
                + 0.7152 * raw_rgb[..., 1]
                + 0.0722 * raw_rgb[..., 2]
            )

    else:
        arr = np.asarray(source, dtype=np.float32)

        if arr.ndim == 2:
            raw_gray = arr
            raw_rgb = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            raw_rgb = arr[..., :3]
            raw_gray = (
                0.2126 * raw_rgb[..., 0]
                + 0.7152 * raw_rgb[..., 1]
                + 0.0722 * raw_rgb[..., 2]
            )
        else:
            raise ValueError("Expected a 2-D grayscale or RGB/RGBA image.")

        if crop is not None:
            l, t, r, b = map(int, crop)
            raw_gray = raw_gray[t:b, l:r]
            raw_rgb = raw_rgb[t:b, l:r]

        native_shape = raw_gray.shape[:2]

    if min(native_shape) < 24:
        raise ValueError("Crop is too small. Use at least ~24x24 pixels.")

    # Optional computational resize.
    if max_side is not None and max(native_shape) > max_side:
        scale = max_side / max(native_shape)
        target = (
            max(1, int(round(native_shape[1] * scale))),
            max(1, int(round(native_shape[0] * scale))),
        )

        gray_img = Image.fromarray(raw_gray.astype(np.float32))
        raw_gray = np.asarray(
            gray_img.resize(target, PIL_BOX),
            dtype=np.float32
        )

        channels = []
        for i in range(3):
            ch = Image.fromarray(raw_rgb[..., i].astype(np.float32))
            channels.append(
                np.asarray(ch.resize(target, PIL_BOX), dtype=np.float32)
            )
        raw_rgb = np.stack(channels, axis=-1)

    # Scientific gray array: robust linear scaling only.
    gray = robust_scale(raw_gray).astype(np.float32)

    # Display RGB only.
    rgb = raw_rgb.astype(np.float32)
    rgb_lo = np.percentile(rgb, 1)
    rgb_hi = np.percentile(rgb, 99.5)
    if rgb_hi > rgb_lo:
        rgb = np.clip((rgb - rgb_lo) / (rgb_hi - rgb_lo), 0, 1)
    else:
        rgb = np.zeros_like(rgb)

    return {
        "filename": title,
        "gray": gray,
        "rgb": rgb,
        "native_shape": native_shape,
        "analysis_shape": gray.shape,
    }


# ============================================================
# SKY / CENTER / SOURCE DETECTION
# ============================================================

def estimate_sky(gray):
    h, w = gray.shape
    edge = max(2, round(min(h, w) * 0.09))

    border = np.zeros(gray.shape, dtype=bool)
    border[:edge] = True
    border[-edge:] = True
    border[:, :edge] = True
    border[:, -edge:] = True

    values = np.asarray(gray[border], dtype=float)

    # Robust sigma clipping on border pixels.
    for _ in range(4):
        location = np.median(values)
        sigma = mad(values)
        if sigma < 1e-12:
            break
        keep = np.abs(values - location) < 3.5 * sigma
        if keep.all() or keep.sum() < 20:
            break
        values = values[keep]

    background = float(np.median(values))
    noise = max(mad(values), 1e-10)

    return background, noise


def estimate_center(gray, background):
    h, w = gray.shape
    n = min(h, w)
    y, x = np.indices(gray.shape)

    smooth = gaussian_filter(gray, max(1.0, n * 0.018))

    prior = np.exp(
        -(
            (x - (w - 1) / 2) ** 2
            + (y - (h - 1) / 2) ** 2
        )
        / (2 * (0.22 * n) ** 2)
    )

    signal = np.maximum(smooth - background, 0)
    py, px = np.unravel_index(np.argmax(signal * prior), gray.shape)

    local_prior = np.exp(
        -(
            (x - px) ** 2
            + (y - py) ** 2
        )
        / (2 * (0.07 * n) ** 2)
    )

    weights = signal * local_prior
    total = float(weights.sum())

    if total <= 1e-12:
        return (w - 1) / 2, (h - 1) / 2

    cx = float((weights * x).sum() / total)
    cy = float((weights * y).sum() / total)
    return cx, cy


def flux_radius(radii, weights, fraction):
    radii = np.asarray(radii, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(radii) == 0 or weights.sum() <= 0:
        return 0.0

    order = np.argsort(radii)
    rr = radii[order]
    ww = weights[order]
    cumulative = np.cumsum(ww)

    return float(
        np.interp(
            fraction * cumulative[-1],
            cumulative,
            rr
        )
    )


def morphology_features(gray, center=None, settings=None):
    """
    Robust descriptive morphology features.

    These are diagnostics / proxies, not calibrated class probabilities.
    """
    settings = settings or Settings()

    h, w = gray.shape
    n = min(h, w)
    background, noise = estimate_sky(gray)

    if center is None:
        center = estimate_center(gray, background)

    cx, cy = center
    y, x = np.indices(gray.shape, dtype=float)
    r = np.hypot(x - cx, y - cy)

    smooth = gaussian_filter(gray, max(0.8, 0.015 * n))
    smooth_bg, smooth_noise = estimate_sky(smooth)

    detection = (
        (smooth > smooth_bg + 3.0 * smooth_noise)
        & (r < 0.43 * n)
    )

    labels, _ = component_labels(detection)

    # Pick the connected source containing the brightest point near center.
    central_zone = r < 0.12 * n
    if np.any(central_zone):
        py, px = np.unravel_index(
            np.argmax(np.where(central_zone, smooth, -np.inf)),
            gray.shape
        )
    else:
        py, px = int(round(cy)), int(round(cx))

    component_id = labels[py, px]
    component = (labels == component_id) & (labels > 0)

    pixels = int(component.sum())
    flux = gray - background
    source_flux = float(flux[component].sum())

    snr_proxy = source_flux / (
        noise * np.sqrt(max(pixels, 1))
    )

    peak_snr = float(
        np.max(smooth - smooth_bg)
        / max(smooth_noise, 1e-12)
    )

    weights = np.maximum(smooth - smooth_bg, 0) * component
    total = float(weights.sum())

    q = 1.0
    radii = {
        "r20": 0.0,
        "r50": 0.0,
        "r80": 0.0,
        "r90": 0.0,
    }

    if total > 1e-12:
        dx = x - cx
        dy = y - cy

        cov = np.array([
            [
                (weights * dx * dx).sum(),
                (weights * dx * dy).sum(),
            ],
            [
                (weights * dx * dy).sum(),
                (weights * dy * dy).sum(),
            ],
        ]) / total

        ev = np.maximum(
            np.linalg.eigvalsh(cov),
            1e-12
        )

        q = float(
            np.sqrt(ev[0] / ev[1])
        )

        for name, frac in (
            ("r20", 0.20),
            ("r50", 0.50),
            ("r80", 0.80),
            ("r90", 0.90),
        ):
            radii[name] = flux_radius(
                r[component],
                weights[component],
                frac
            )

    detected = bool(
        pixels >= settings.min_source_pixels
        and snr_proxy >= settings.min_snr_proxy
        and peak_snr >= settings.min_peak_snr
    )

    resolved = bool(
        detected
        and radii["r90"] >= settings.min_r90_resolved
        and radii["r50"] >= settings.min_r50_resolved
    )

    boundary = (
        min(
            cx,
            cy,
            w - 1 - cx,
            h - 1 - cy
        )
        - 1
    )

    aperture_radius = max(
        1.0,
        min(
            0.40 * n,
            boundary,
            max(
                8.0,
                1.15 * radii["r90"]
            )
        )
    )

    aperture = r < aperture_radius

    # 180-degree asymmetry around fractional-pixel center.
    xr = 2 * cx - x
    yr = 2 * cy - y
    paired = (
        (xr >= 0)
        & (xr <= w - 1)
        & (yr >= 0)
        & (yr <= h - 1)
    )

    rotated = map_coordinates(
        flux,
        [yr, xr],
        order=1,
        mode="constant",
        cval=0.0
    )

    delta = np.abs(flux - rotated)

    sky_mask = (
        paired
        & (r > max(aperture_radius * 1.1, 0.40 * n))
    )

    sky_delta = delta[
        sky_mask
        & (np.abs(flux) < 3.5 * noise)
    ]

    sky_bias = (
        float(np.mean(sky_delta))
        if len(sky_delta) >= 20
        else 2 * noise / np.sqrt(np.pi)
    )

    use = aperture & paired
    denom = max(
        float(flux[use].sum()),
        1e-12
    )

    asym_raw = float(
        delta[use].sum()
        / (2 * denom)
    )

    asymmetry = max(
        0.0,
        float(
            (
                delta[use].sum()
                - use.sum() * sky_bias
            )
            / (2 * denom)
        )
    )

    # Clumpiness / high-frequency residual.
    fine_residual = (
        flux
        - gaussian_filter(
            flux,
            max(1.0, n * 0.025)
        )
    )

    outer = (
        use
        & (r > max(2.0, 0.12 * aperture_radius))
    )

    sky_fine = fine_residual[
        sky_mask
        & (np.abs(flux) < 3.5 * noise)
    ]

    fine_bias = (
        float(
            np.maximum(
                sky_fine,
                0
            ).mean()
        )
        if len(sky_fine)
        else noise / np.sqrt(2 * np.pi)
    )

    clumpiness = max(
        0.0,
        float(
            (
                np.maximum(
                    fine_residual[outer],
                    0
                ).sum()
                - outer.sum() * fine_bias
            )
            / denom
        )
    )

    # Bright-component proxy.
    maxima = (
        smooth
        == maximum_filter(
            smooth,
            size=max(3, int(n * 0.05))
        )
    )

    maxima &= (
        component
        & (smooth > smooth_bg + 5 * smooth_noise)
    )

    _, bright_peaks = component_labels(maxima)

    concentration = None
    if detected and resolved:
        concentration = float(
            5
            * np.log10(
                max(radii["r80"], 1e-9)
                / max(radii["r20"], 1e-9)
            )
        )

    return {
        "center": tuple(center),
        "background": background,
        "noise": noise,
        "detected": detected,
        "resolved": resolved,
        "source_pixels": pixels,
        "source_mask": component,
        "snr_proxy": float(snr_proxy),
        "peak_snr_proxy": peak_snr,
        "axis_ratio": q,
        "concentration": concentration,
        "asymmetry": asymmetry if detected and resolved else None,
        "asymmetry_raw": asym_raw if detected and resolved else None,
        "clumpiness": clumpiness if detected and resolved else None,
        "bright_peaks": int(bright_peaks),
        "aperture_radius": aperture_radius,
        **radii,
    }


# ============================================================
# LOG-POLAR / FOURIER SPIRAL ANALYSIS
# ============================================================

def polar_residual(gray, center, rmin, rmax, q=1.0, pa_deg=0.0):
    nr = int(
        np.clip(
            round(rmax - rmin) * 2,
            48,
            128
        )
    )

    nt = int(
        np.clip(
            round(2 * np.pi * rmax),
            128,
            512
        )
    )

    if nt % 2:
        nt += 1

    u = np.linspace(
        np.log(rmin),
        np.log(rmax),
        nr
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        nt,
        endpoint=False
    )

    r = np.exp(u)[:, None]

    xx = r * np.cos(theta)[None, :]
    yy = q * r * np.sin(theta)[None, :]

    pa = np.deg2rad(pa_deg)

    x = (
        center[0]
        + xx * np.cos(pa)
        - yy * np.sin(pa)
    )

    y = (
        center[1]
        + xx * np.sin(pa)
        + yy * np.cos(pa)
    )

    if (
        x.min() < 0
        or y.min() < 0
        or x.max() > gray.shape[1] - 1
        or y.max() > gray.shape[0] - 1
    ):
        raise ValueError(
            "Analysis annulus leaves the image."
        )

    polar = map_coordinates(
        gray,
        [y, x],
        order=1,
        mode="constant",
        cval=np.nan
    )

    # Remove axisymmetric component; keep signed residuals.
    residual = (
        polar
        - np.mean(
            polar,
            axis=1,
            keepdims=True
        )
    )

    return residual, u, theta


def spiral_measurement(
    gray,
    center,
    rmin,
    rmax,
    q=1.0,
    pa_deg=0.0
):
    residual, u, theta = polar_residual(
        gray,
        center,
        rmin,
        rmax,
        q,
        pa_deg
    )

    weights = np.hanning(len(u))
    weights /= (
        weights.sum()
        + 1e-20
    )

    positive = np.linspace(
        0.65,
        10.0,
        96
    )

    slopes = np.r_[
        -positive[::-1],
        0.0,
        positive
    ]

    modes = np.array([
        2,
        3,
        4
    ])

    fm = (
        np.fft.rfft(
            residual,
            axis=1
        )[:, modes]
        / len(theta)
    )

    template = np.exp(
        1j
        * modes[:, None, None]
        * slopes[None, :, None]
        * (u - u[0])[None, None, :]
    )

    amplitude = np.einsum(
        "msr,rm,r->ms",
        template,
        fm,
        weights
    )

    power = np.abs(amplitude) ** 2

    cw = float(
        power[
            :,
            slopes > 0
        ].max()
    )

    ccw = float(
        power[
            :,
            slopes < 0
        ].max()
    )

    score = (
        cw - ccw
    ) / (
        cw + ccw
        + 1e-20
    )

    nonzero = power.copy()
    nonzero[
        :,
        slopes == 0
    ] = -1.0

    mi, ki = np.unravel_index(
        np.argmax(nonzero),
        nonzero.shape
    )

    peak = float(
        power[mi, ki]
    )

    incoherent = float(
        np.sum(
            weights
            * np.abs(
                fm[:, mi]
            ) ** 2
        )
    )

    residual_power = float(
        np.sum(
            weights
            * np.mean(
                residual ** 2,
                axis=1
            )
        )
    )

    straight = float(
        power[
            :,
            slopes == 0
        ].max()
    )

    return {
        "score": float(score),
        "coherence": peak / (
            incoherent + 1e-20
        ),
        "explained": 2 * peak / (
            residual_power + 1e-20
        ),
        "vs_straight": peak / (
            straight + 1e-20
        ),
        "arm_amplitude": 2 * np.sqrt(peak),
        "mode": int(modes[mi]),
        "slope": float(slopes[ki]),
        "polar": residual,
        "log_r": u,
        "theta": theta,
        "power": power,
        "slopes": slopes,
        "modes": modes,
    }


def robust_spiral_analysis(
    gray,
    center,
    axis_ratio,
    settings=None,
    q=None,
    pa_deg=0.0
):
    settings = settings or Settings()

    h, w = gray.shape
    n = min(h, w)
    cx, cy = center

    jitter = max(
        0.75,
        n * 0.008
    )

    boundary = (
        min(
            cx,
            cy,
            w - 1 - cx,
            h - 1 - cy
        )
        - jitter
        - 1
    )

    if boundary < 10:
        return {
            "label": "UNCERTAIN",
            "candidate": "NONE",
            "reasons": [
                "Object is too close to image boundary."
            ],
            "base": None,
        }

    radius = min(
        0.42 * n,
        boundary
    )

    use_q = 1.0 if q is None else float(q)

    shifts = [
        (0, 0),
        (-jitter, 0),
        (jitter, 0),
        (0, -jitter),
        (0, jitter),
    ]

    blur_scales = [
        max(0.5, n * 0.0025),        max(1.0, n * 0.005),
    ]

    annuli = [
        (0.16, 1.00),
        (0.24, 0.92),
        (0.32, 1.00),
    ]

    summaries = []
    scores = []

    for sigma in blur_scales:
        filtered = gaussian_filter(
            gray,
            sigma
        )

        for dx, dy in shifts:
            for inner, outer in annuli:
                d = spiral_measurement(
                    filtered,
                    (cx + dx, cy + dy),
                    max(3.0, inner * radius),
                    outer * radius,
                    use_q,
                    pa_deg
                )

                summaries.append({
                    k: d[k]
                    for k in (
                        "score",
                        "coherence",
                        "explained",
                        "vs_straight",
                        "arm_amplitude",
                    )
                })

                scores.append(
                    d["score"]
                )

    scores = np.asarray(scores)
    score = float(
        np.median(scores)
    )

    sign = (
        1
        if score > 0
        else -1
        if score < 0
        else 0
    )

    stability = float(
        np.mean(
            sign * scores > 0.10
        )
    )

    med = {
        k: float(
            np.median(
                [s[k] for s in summaries]
            )
        )
        for k in summaries[0]
    }

    sigma = blur_scales[0]
    rmin = max(
        3.0,
        0.24 * radius
    )
    rmax = 0.92 * radius

    base = spiral_measurement(
        gaussian_filter(
            gray,
            sigma
        ),
        center,
        rmin,
        rmax,
        use_q,
        pa_deg
    )

    # Noise stress test.
    _, sky_sigma = estimate_sky(gray)

    rng = np.random.default_rng(
        settings.random_seed
    )

    noise_scores = []

    for _ in range(
        settings.noise_trials
    ):
        noisy = (
            gray
            + rng.normal(
                0,
                0.5 * sky_sigma,
                gray.shape
            )
        )

        d = spiral_measurement(
            gaussian_filter(
                noisy,
                sigma
            ),
            center,
            rmin,
            rmax,
            use_q,
            pa_deg
        )

        noise_scores.append(
            d["score"]
        )

    noise_scores = np.asarray(
        noise_scores
    )

    noise_stability = float(
        np.mean(
            sign * noise_scores > 0.10
        )
    )

    # Mirror consistency.
    mirrored = spiral_measurement(
        gaussian_filter(
            gray[:, ::-1],
            sigma
        ),
        (
            gray.shape[1] - 1 - cx,
            cy
        ),
        rmin,
        rmax,
        use_q,
        -pa_deg
    )

    mirror_error = abs(
        base["score"]
        + mirrored["score"]
    )

    arm_to_sky = (
        med["arm_amplitude"]
        / max(sky_sigma, 1e-20)
    )

    reasons = []

    checks = [
        (
            abs(score)
            < settings.min_handedness_margin,
            "CW and CCW responses are too similar."
        ),
        (
            stability
            < settings.min_stability,
            "Result changes across center/annulus/smoothing tests."
        ),
        (
            noise_stability
            < settings.min_stability,
            "Result changes after adding small noise."
        ),
        (
            med["coherence"]
            < settings.min_coherence,
            "Spiral phase is not coherent enough across radius."
        ),
        (
            med["explained"]
            < settings.min_explained,
            "Spiral template explains too little structure."
        ),
        (
            med["vs_straight"]
            < settings.min_vs_straight,
            "Straight/bar structure fits almost as well as a spiral."
        ),
        (
            arm_to_sky
            < settings.min_arm_to_sky,
            "Spiral-arm signal is weak relative to sky noise."
        ),
        (
            axis_ratio
            < settings.min_axis_ratio_for_handedness,
            "Galaxy is too elongated for reliable handedness."
        ),
        (
            mirror_error > 0.03,
            "Mirror-consistency check failed."
        ),
        (
            base["score"] * sign <= 0.10,
            "Base annulus disagrees with the global sign."
        ),
    ]

    for failed, reason in checks:
        if failed:
            reasons.append(reason)

    candidate = (
        "CW"
        if sign > 0
        else "CCW"
        if sign < 0
        else "NONE"
    )

    label = (
        "UNCERTAIN"
        if reasons
        else candidate
    )

    return {
        "label": label,
        "candidate": candidate,
        "score": score,
        "stability": stability,
        "noise_stability": noise_stability,
        "coherence": med["coherence"],
        "explained": med["explained"],
        "vs_straight": med["vs_straight"],
        "arm_to_sky": arm_to_sky,
        "mirror_error": float(mirror_error),
        "reasons": reasons,
        "radius": float(radius),
        "q_used": use_q,
        "pa_deg": pa_deg,
        "base": base,
        "trial_scores": scores,
        "noise_scores": noise_scores,
    }


# ============================================================
# MORPHOLOGY SCORING
# ============================================================

def clip01(x):
    return float(
        np.clip(
            x,
            0,
            1
        )
    )


def morphology_scores(
    features,
    spiral,
    settings=None
):
    """
    Five-class interpretable heuristic classifier.

    Scores are NOT calibrated probabilities.
    """
    settings = settings or Settings()

    if not features["detected"]:
        return {
            "label": "NO CLEAR SOURCE",
            "candidate": "NONE",
            "margin": 0.0,
            "scores": {},
            "note": "No reliable extended source was detected.",
        }

    if not features["resolved"]:
        return {
            "label": "UNRESOLVED",
            "candidate": "NONE",
            "margin": 0.0,
            "scores": {},
            "note": "The source is too small for reliable morphology.",
        }

    q = features["axis_ratio"]
    asym = features["asymmetry"]
    clump = features["clumpiness"]
    conc = features["concentration"]
    peaks = features["bright_peaks"]

    # Structural evidence derived from robust spiral analysis.
    if spiral["base"] is None:
        spiral_evidence = 0.0
        robust_sign_support = 0.0
    else:
        spiral_evidence = (
            0.30
            * clip01(
                (
                    spiral["vs_straight"]
                    - 1.0
                ) / 1.3
            )
            + 0.25
            * clip01(
                spiral["explained"]
                / 0.45
            )
            + 0.20
            * clip01(
                spiral["coherence"]
            )
            + 0.15
            * clip01(
                spiral["stability"]
            )
            + 0.10
            * clip01(
                spiral["noise_stability"]
            )
        )

        robust_sign_support = (
            0.5
            * clip01(
                abs(
                    spiral["score"]
                )
                / 0.55
            )
            + 0.5
            * clip01(
                spiral["stability"]
            )
        )

    roundness = clip01(
        (q - 0.45) / 0.45
    )

    elongation = clip01(
        (0.60 - q) / 0.35
    )

    smoothness = (
        1.0
        - clip01(
            clump / 0.18
        )
    )

    symmetry = (
        1.0
        - clip01(
            asym / 0.40
        )
    )

    disturbed = clip01(
        (asym - 0.18) / 0.32
    )

    clumpy = clip01(
        (clump - 0.04) / 0.20
    )

    concentration_high = clip01(
        (conc - 2.2) / 2.8
    )

    multiple_peaks = clip01(
        (peaks - 1) / 4
    )

    # Moderate clumpiness can be consistent with spiral arms.
    moderate_clump = clip01(
        1.0
        - abs(
            clump - 0.10
        ) / 0.15
    )

    scores = {}

    scores["Spiral Galaxy"] = (
        0.20 * roundness
        + 0.45 * spiral_evidence
        + 0.10 * moderate_clump
        + 0.10 * symmetry
        + 0.15 * robust_sign_support
    )

    scores["Elliptical Galaxy"] = (
        0.25 * roundness
        + 0.30 * smoothness
        + 0.20 * symmetry
        + 0.25 * concentration_high
    )

    scores["Edge-on Disk"] = (
        0.80 * elongation
        + 0.20 * symmetry
    )

    scores["Irregular Galaxy"] = (
        0.55 * disturbed
        + 0.45 * clumpy
    )

    scores["Merger / Disturbed"] = (
        0.50 * disturbed
        + 0.30 * multiple_peaks
        + 0.20 * clumpy
    )

    for key in scores:
        scores[key] = max(
            0.0,
            float(scores[key])
        )

    ordered = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    candidate = ordered[0][0]
    top = ordered[0][1]
    second = ordered[1][1]
    margin = top - second

    uncertain = (
        top < settings.min_morphology_score
        or margin < settings.min_morphology_margin
    )

    label = (
        "UNCERTAIN"
        if uncertain
        else candidate
    )

    note = (
        "Morphology scores are interpretable heuristic evidence, "
        "not calibrated probabilities."
    )

    return {
        "label": label,
        "candidate": candidate,
        "margin": float(margin),
        "scores": scores,
        "note": note,
    }


# ============================================================
# FULL HYBRID PIPELINE
# ============================================================

def analyze_galaxy(
    source,
    crop=None,
    max_side=None,
    center=None,
    q=None,
    pa_deg=0.0,
    settings=None
):
    settings = settings or Settings()

    loaded = load_image(
        source,
        crop=crop,
        max_side=max_side
    )

    gray = loaded["gray"]

    if center is None:
        background, _ = estimate_sky(gray)
        center = estimate_center(
            gray,
            background
        )

    features = morphology_features(
        gray,
        center=center,
        settings=settings
    )

    if (
        features["detected"]
        and features["resolved"]
        and min(gray.shape) >= 32
    ):
        spiral = robust_spiral_analysis(
            gray,
            features["center"],
            features["axis_ratio"],
            settings=settings,
            q=q,
            pa_deg=pa_deg
        )
    else:
        spiral = {
            "label": "UNCERTAIN",
            "candidate": "NONE",
            "score": 0.0,
            "stability": 0.0,
            "noise_stability": 0.0,
            "coherence": 0.0,
            "explained": 0.0,
            "vs_straight": 0.0,
            "arm_to_sky": 0.0,
            "mirror_error": np.nan,
            "reasons": [
                "Source is not sufficiently detected/resolved for spiral analysis."
            ],
            "radius": None,
            "q_used": 1.0 if q is None else float(q),
            "pa_deg": pa_deg,
            "base": None,
            "trial_scores": np.array([]),
            "noise_scores": np.array([]),
        }

    morphology = morphology_scores(
        features,
        spiral,
        settings=settings
    )

    return {
        **loaded,
        "features": features,
        "spiral": spiral,
        "morphology": morphology,
        "methodology": (
            "hybrid interpretable morphology + robust log-polar Fourier baseline; "
            "not a trained neural network; thresholds need validation on labeled data"
        ),
    }


# ============================================================
# HUMAN-READABLE OUTPUT
# ============================================================

def print_result(result):
    f = result["features"]
    m = result["morphology"]
    s = result["spiral"]

    print("=" * 72)
    print("HYBRID GALAXY ANALYSIS")
    print("=" * 72)

    print("File:", result["filename"])

    if result["native_shape"] != result["analysis_shape"]:
        print(
            "Resolution:",
            result["native_shape"],
            "->",
            result["analysis_shape"]
        )
    else:
        print(
            "Resolution:",
            result["analysis_shape"],
            "(native)"
        )

    print()
    print("MORPHOLOGY")
    print("-" * 72)
    print("Final:", m["label"])
    print("Best candidate:", m["candidate"])
    print(f"Decision margin: {m['margin']:.3f}")

    if m["scores"]:
        print("Scores (NOT probabilities):")
        for name, value in sorted(
            m["scores"].items(),
            key=lambda kv: kv[1],
            reverse=True
        ):
            print(
                f"  {name:22s} {value:.3f}"
            )

    print()
    print("MEASURED FEATURES")
    print("-" * 72)
    print(
        f"center             = ({f['center'][0]:.2f}, {f['center'][1]:.2f})"
    )
    print(
        f"detected/resolved  = {f['detected']} / {f['resolved']}"
    )
    print(
        f"axis ratio b/a     = {f['axis_ratio']:.3f}"
    )
    print(
        f"r50 / r90          = {f['r50']:.2f} / {f['r90']:.2f} px"
    )
    print(
        f"SNR proxy          = {f['snr_proxy']:.2f}"
    )

    if f["concentration"] is not None:
        print(
            f"concentration      = {f['concentration']:.3f}"
        )

    if f["asymmetry"] is not None:
        print(
            f"asymmetry          = {f['asymmetry']:.3f}"
        )

    if f["clumpiness"] is not None:
        print(
            f"clumpiness         = {f['clumpiness']:.3f}"
        )

    print(
        f"bright components  = {f['bright_peaks']}"
    )

    print()
    print("SPIRAL / HANDEDNESS")
    print("-" * 72)
    print("Result:", s["label"])
    print("Candidate:", s["candidate"])
    print(
        f"signed margin      = {s['score']:+.3f}"
    )
    print(
        f"parameter stability= {s['stability']:.0%}"
    )
    print(
        f"noise stability    = {s['noise_stability']:.0%}"
    )
    print(
        f"coherence          = {s['coherence']:.3f}"
    )
    print(
        f"explained          = {s['explained']:.3f}"
    )
    print(
        f"spiral/straight    = {s['vs_straight']:.3f}"
    )
    print(
        f"arm/sky            = {s['arm_to_sky']:.3f}"
    )

    if np.isfinite(s["mirror_error"]):
        print(
            f"mirror error       = {s['mirror_error']:.5f}"
        )

    if s["reasons"]:
        print("Why not fully trusted:")
        for reason in s["reasons"]:
            print(" -", reason)

    print()
    print(
        "CW/CCW here means apparent outward winding on the image, "
        "not physical 3-D rotation."
    )
    print(
        "Morphology scores and stability values are diagnostics, "
        "not probabilities."
    )


# ============================================================
# VISUALIZATION
# ============================================================

def plot_result(result):
    f = result["features"]
    m = result["morphology"]
    s = result["spiral"]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        constrained_layout=True
    )

    # Input + center
    ax = axes[0, 0]
    ax.imshow(
        result["rgb"],
        origin="upper"
    )
    ax.scatter(
        [f["center"][0]],
        [f["center"][1]],
        marker="+",
        s=130
    )
    ax.set_title(
        f"Input\nMorphology: {m['label']}"
    )
    ax.axis("off")

    # Source mask
    ax = axes[0, 1]
    ax.imshow(
        result["rgb"],
        origin="upper"
    )
    mask = np.ma.masked_where(
        ~f["source_mask"],
        np.ones(
            f["source_mask"].shape
        )
    )
    ax.imshow(
        mask,
        cmap="autumn",
        alpha=0.35,
        origin="upper",
        vmin=0,
        vmax=1
    )
    ax.set_title(
        "Detected source mask"
    )
    ax.axis("off")

    # Morphology scores
    ax = axes[0, 2]

    if m["scores"]:
        items = sorted(
            m["scores"].items(),
            key=lambda kv: kv[1]
        )
        names = [
            x[0]
            for x in items
        ]
        values = [
            x[1]
            for x in items
        ]
        ax.barh(
            names,
            values
        )
        ax.set_xlabel(
            "heuristic score"
        )
        ax.set_title(
            f"Morphology scores\ncandidate={m['candidate']}"
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            m["label"],
            ha="center",
            va="center",
            fontsize=16
        )

    # Spiral plots
    base = s["base"]

    if base is not None:
        ax = axes[1, 0]
        limit = max(
            float(
                np.percentile(
                    np.abs(
                        base["polar"]
                    ),
                    98
                )
            ),
            1e-8
        )
        ax.imshow(
            base["polar"],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(
                0,
                360,
                base["log_r"][0],
                base["log_r"][-1]
            )
        )
        ax.set_title(
            "Signed log-polar residual"
        )
        ax.set_xlabel(
            "theta [deg]"
        )
        ax.set_ylabel(
            "ln(r)"
        )

        ax = axes[1, 1]
        norm = max(
            float(
                base["power"].max()
            ),
            1e-20
        )
        for mode, power in zip(
            base["modes"],
            base["power"]
        ):
            ax.plot(
                base["slopes"],
                power / norm,
                label=f"m={mode}"
            )
        ax.axvline(
            0,
            linewidth=1        )
        ax.set_title(
            "Spiral template response"
        )
        ax.set_xlabel(
            "k: CCW < 0 < CW"
        )
        ax.set_ylabel(
            "relative power"
        )
        ax.legend()

        ax = axes[1, 2]
        ax.plot(
            s["trial_scores"],
            ".",
            label="parameters"
        )
        start = len(
            s["trial_scores"]
        )
        ax.plot(
            np.arange(
                start,
                start + len(
                    s["noise_scores"]
                )
            ),
            s["noise_scores"],
            ".",
            label="added noise"
        )
        ax.axhline(
            0,
            linewidth=1
        )
        ax.set_ylim(
            -1.05,
            1.05
        )
        ax.set_title(
            f"Sensitivity\nhandedness={s['label']}"
        )
        ax.set_xlabel(
            "trial"
        )
        ax.set_ylabel(
            "signed CW/CCW margin"
        )
        ax.legend(
            fontsize=8
        )

    else:
        for ax in axes[1]:
            ax.axis("off")

        axes[1, 1].text(
            0.5,
            0.5,
            "Spiral analysis not supported\nfor this source/crop",
            ha="center",
            va="center"
        )

    fig.suptitle(
        f"{result['filename']} | "
        f"{m['label']} | "
        f"spiral winding: {s['label']}",
        fontsize=14
    )

    plt.show()
    return fig


# ============================================================
# OPTIONAL SINGLE-CALL HELPER
# ============================================================

def run_galaxy(
    filename,
    crop=None,
    max_side=None,
    settings=None
):
    result = analyze_galaxy(
        filename,
        crop=crop,
        max_side=max_side,
        settings=settings
    )

    print_result(
        result
    )

    plot_result(
        result
    )

    return result


# ============================================================
# BATCH FIELD DETECTION + AUTO CROPPING + CATALOG
# ============================================================

import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    find_objects,
)


@dataclass(frozen=True)
class BatchSettings:
    # Detection is allowed to use a resized COPY for speed.
    # Crops are ALWAYS cut from the native original.
    detection_max_side: int | None = 2600

    # Detection map parameters
    detection_sigma: float = 2.8
    detection_blur_sigma: float = 1.1
    background_sigma_fraction: float = 0.025

    # Connected-component filtering on the detection image
    min_component_area: int = 10
    min_component_width: int = 3
    min_component_height: int = 3

    # Crop geometry in native pixels after mapping from detection image
    crop_scale: float = 3.2
    min_crop_side: int = 48
    max_crop_side: int = 500

    # Extra source merging
    closing_iterations: int = 1
    dilation_iterations: int = 1

    # If not None, analyze only the largest N detected candidates.
    # Use None to analyze every candidate.
    max_objects: int | None = 150

    # A conservative flag only; compact objects are NOT automatically deleted.
    star_like_r90_max: float = 4.5
    star_like_concentration_min: float = 3.8

    # Run the expensive robust CW/CCW stage only when quick morphology
    # says Spiral Galaxy is the best candidate.
    robust_handedness_only_for_spiral_candidate: bool = True


def _field_luminance(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)

    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _resize_detection_copy(rgb, max_side):
    h, w = rgb.shape[:2]

    if max_side is None or max(h, w) <= max_side:
        return rgb, 1.0, 1.0

    scale = max_side / max(h, w)

    target = (
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )

    im = Image.fromarray(
        np.clip(rgb * 255, 0, 255).astype(np.uint8)
    )

    small = np.asarray(
        im.resize(target, PIL_BOX),
        dtype=np.float32
    ) / 255.0

    # detection-pixel -> native-pixel multipliers
    sx = w / small.shape[1]
    sy = h / small.shape[0]

    return small, sx, sy


def detect_source_candidates(field_image, batch_settings=None):
    """
    Detect extended/bright source candidates in a full astronomical field.

    Important:
    - This is a SOURCE detector, not a perfect galaxy-vs-star segmenter.
    - It may include stars, arcs, artifacts, and blended sources.
    - Every final crop is taken from the native original image.
    """
    batch_settings = batch_settings or BatchSettings()

    if isinstance(field_image, (str, Path)):
        path = Path(field_image).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"Field image not found: {path.resolve()}"
            )

        native_im = ImageOps.exif_transpose(
            Image.open(path)
        ).convert("RGB")

    else:
        native_im = Image.fromarray(
            np.asarray(field_image).astype(np.uint8)
        ).convert("RGB")

    native_rgb = (
        np.asarray(native_im, dtype=np.float32)
        / 255.0
    )

    det_rgb, sx, sy = _resize_detection_copy(
        native_rgb,
        batch_settings.detection_max_side
    )

    gray = robust_scale(
        _field_luminance(det_rgb)
    )

    n = min(gray.shape)

    small = gaussian_filter(
        gray,
        batch_settings.detection_blur_sigma
    )

    bg_sigma = max(
        8.0,
        n * batch_settings.background_sigma_fraction
    )

    local_background = gaussian_filter(
        small,
        bg_sigma
    )

    residual = small - local_background

    # Robust global residual noise.
    residual_median = float(
        np.median(residual)
    )

    residual_noise = max(
        mad(residual),
        1e-8
    )

    threshold = (
        residual_median
        + batch_settings.detection_sigma
        * residual_noise
    )

    mask = residual > threshold

    if batch_settings.closing_iterations > 0:
        mask = binary_closing(
            mask,
            iterations=batch_settings.closing_iterations
        )

    if batch_settings.dilation_iterations > 0:
        mask = binary_dilation(
            mask,
            iterations=batch_settings.dilation_iterations
        )

    labels, count = component_labels(mask)
    slices = find_objects(labels)

    candidates = []

    native_w, native_h = native_im.size

    for label_id, slc in enumerate(
        slices,
        start=1
    ):
        if slc is None:
            continue

        ys, xs = slc

        y1d, y2d = ys.start, ys.stop
        x1d, x2d = xs.start, xs.stop

        component = (
            labels[
                y1d:y2d,
                x1d:x2d
            ]
            == label_id
        )

        area = int(
            component.sum()
        )

        bw = x2d - x1d
        bh = y2d - y1d

        if area < batch_settings.min_component_area:
            continue

        if bw < batch_settings.min_component_width:
            continue

        if bh < batch_settings.min_component_height:
            continue

        yy, xx = np.nonzero(component)

        if len(xx) == 0:
            continue

        cxd = x1d + float(xx.mean())
        cyd = y1d + float(yy.mean())

        # Convert detection coordinates back to native coordinates.
        cx = cxd * sx
        cy = cyd * sy

        native_bw = max(
            1.0,
            bw * sx
        )

        native_bh = max(
            1.0,
            bh * sy
        )

        side = int(
            round(
                max(
                    batch_settings.min_crop_side,
                    min(
                        batch_settings.max_crop_side,
                        max(
                            native_bw,
                            native_bh
                        )
                        * batch_settings.crop_scale
                    )
                )
            )
        )

        half = side / 2

        left = int(
            max(
                0,
                np.floor(cx - half)
            )
        )

        top = int(
            max(
                0,
                np.floor(cy - half)
            )
        )

        right = int(
            min(
                native_w,
                np.ceil(cx + half)
            )
        )

        bottom = int(
            min(
                native_h,
                np.ceil(cy + half)
            )
        )

        # Require enough pixels after boundary clipping.
        if right - left < 24 or bottom - top < 24:
            continue

        candidates.append({
            "component_id": label_id,
            "det_area": area,
            "det_width": bw,
            "det_height": bh,
            "native_center_x": cx,
            "native_center_y": cy,
            "crop_left": left,
            "crop_top": top,
            "crop_right": right,
            "crop_bottom": bottom,
            "crop_width": right - left,
            "crop_height": bottom - top,
        })

    # Largest candidates first.
    candidates.sort(
        key=lambda d: d["det_area"],
        reverse=True
    )

    if batch_settings.max_objects is not None:
        candidates = candidates[
            :batch_settings.max_objects
        ]

    diagnostics = {
        "native_image": native_im,
        "detection_rgb": det_rgb,
        "detection_gray": gray,
        "residual": residual,
        "mask": mask,
        "sx": sx,
        "sy": sy,
        "raw_component_count": count,
        "threshold": threshold,
        "noise": residual_noise,
    }

    return candidates, diagnostics


# ============================================================
# QUICK PRE-CLASSIFIER
# ============================================================

def _quick_spiral_measurement(gray, features):
    """
    One inexpensive spiral measurement used before deciding whether
    to run the full robustness tests.
    """
    if (
        not features["detected"]
        or not features["resolved"]
    ):
        return None

    h, w = gray.shape
    n = min(h, w)
    cx, cy = features["center"]

    boundary = min(
        cx,
        cy,
        w - 1 - cx,
        h - 1 - cy
    )

    rmax = min(
        0.38 * n,
        boundary - 1
    )

    rmin = max(
        3.0,
        0.18 * n
    )

    if rmax <= rmin + 3:
        return None

    try:
        return spiral_measurement(
            gaussian_filter(
                gray,
                max(
                    0.6,
                    n * 0.004
                )
            ),
            features["center"],
            rmin,
            rmax,
            1.0,
            0.0
        )
    except Exception:
        return None


def _quick_morphology_scores(features, quick_spiral, settings=None):
    """
    Fast five-class heuristic used to avoid running the expensive
    robust handedness stage on obviously non-spiral objects.
    """
    settings = settings or Settings()

    if not features["detected"]:
        return {
            "label": "NO CLEAR SOURCE",
            "candidate": "NONE",
            "margin": 0.0,
            "scores": {},
        }

    if not features["resolved"]:
        return {
            "label": "UNRESOLVED",
            "candidate": "NONE",
            "margin": 0.0,
            "scores": {},
        }

    q = features["axis_ratio"]
    asym = features["asymmetry"]
    clump = features["clumpiness"]
    conc = features["concentration"]
    peaks = features["bright_peaks"]

    if quick_spiral is None:
        spiral_evidence = 0.0
    else:
        spiral_evidence = (
            0.55
            * clip01(
                (
                    quick_spiral["vs_straight"]
                    - 1.0
                )
                / 1.2
            )
            + 0.45
            * clip01(
                quick_spiral["explained"]
                / 0.45
            )
        )

    roundness = clip01(
        (q - 0.45) / 0.45
    )

    elongation = clip01(
        (0.60 - q) / 0.35
    )

    smoothness = (
        1.0
        - clip01(
            clump / 0.18
        )
    )

    symmetry = (
        1.0
        - clip01(
            asym / 0.40
        )
    )

    disturbed = clip01(
        (asym - 0.18) / 0.32
    )

    clumpy = clip01(
        (clump - 0.04) / 0.20
    )

    concentration_high = clip01(
        (conc - 2.2) / 2.8
    )

    multiple_peaks = clip01(
        (peaks - 1) / 4
    )

    moderate_clump = clip01(
        1.0
        - abs(
            clump - 0.10
        )
        / 0.15
    )

    scores = {
        "Spiral Galaxy": (
            0.25 * roundness
            + 0.45 * spiral_evidence
            + 0.15 * moderate_clump
            + 0.15 * symmetry
        ),
        "Elliptical Galaxy": (
            0.25 * roundness
            + 0.30 * smoothness
            + 0.20 * symmetry
            + 0.25 * concentration_high
        ),
        "Edge-on Disk": (
            0.80 * elongation
            + 0.20 * symmetry
        ),
        "Irregular Galaxy": (
            0.55 * disturbed
            + 0.45 * clumpy
        ),
        "Merger / Disturbed": (
            0.50 * disturbed
            + 0.30 * multiple_peaks
            + 0.20 * clumpy
        ),
    }

    scores = {
        k: max(
            0.0,
            float(v)
        )
        for k, v in scores.items()
    }

    ordered = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    candidate = ordered[0][0]
    top = ordered[0][1]
    second = ordered[1][1]
    margin = top - second

    uncertain = (
        top < settings.min_morphology_score
        or margin < settings.min_morphology_margin
    )

    return {
        "label": (
            "UNCERTAIN"
            if uncertain
            else candidate
        ),
        "candidate": candidate,
        "margin": float(margin),
        "scores": scores,
    }


# ============================================================
# ONE-CROP BATCH ANALYSIS
# ============================================================

def analyze_catalog_crop(
    crop_path,
    settings=None,
    batch_settings=None
):
    settings = settings or Settings()
    batch_settings = batch_settings or BatchSettings()

    loaded = load_image(
        crop_path,
        max_side=None
    )

    gray = loaded["gray"]

    background, _ = estimate_sky(gray)
    center = estimate_center(
        gray,
        background
    )

    features = morphology_features(
        gray,
        center=center,
        settings=settings
    )

    quick_spiral = _quick_spiral_measurement(
        gray,
        features
    )

    quick_morph = _quick_morphology_scores(
        features,
        quick_spiral,
        settings=settings
    )

    run_robust = (
        features["detected"]
        and features["resolved"]
        and (
            not batch_settings.robust_handedness_only_for_spiral_candidate
            or quick_morph["candidate"] == "Spiral Galaxy"
        )
    )

    if run_robust:
        robust_spiral = robust_spiral_analysis(
            gray,
            features["center"],
            features["axis_ratio"],
            settings=settings
        )

        final_morph = morphology_scores(
            features,
            robust_spiral,
            settings=settings
        )

    else:
        robust_spiral = {
            "label": "NOT RUN",
            "candidate": (
                quick_spiral["score"] > 0
                and "CW"
                or quick_spiral is not None
                and quick_spiral["score"] < 0
                and "CCW"
                or "NONE"
            ),
            "score": (
                float(
                    quick_spiral["score"]
                )
                if quick_spiral is not None
                else 0.0
            ),
            "stability": np.nan,
            "noise_stability": np.nan,
            "coherence": np.nan,
            "explained": (
                float(
                    quick_spiral["explained"]
                )
                if quick_spiral is not None
                else np.nan
            ),
            "vs_straight": (
                float(
                    quick_spiral["vs_straight"]
                )
                if quick_spiral is not None
                else np.nan
            ),
            "arm_to_sky": np.nan,
            "mirror_error": np.nan,
            "reasons": [
                "Robust handedness stage skipped because Spiral Galaxy "
                "was not the best quick morphology candidate."
            ],
            "base": quick_spiral,
            "trial_scores": np.array([]),
            "noise_scores": np.array([]),
        }

        final_morph = quick_morph

    # Conservative compact/star-like FLAG, not a hard scientific claim.
    possible_star_like = bool(
        features["resolved"]
        and features["r90"]
        <= batch_settings.star_like_r90_max
        and features["concentration"] is not None
        and features["concentration"]
        >= batch_settings.star_like_concentration_min
        and features["asymmetry"] is not None
        and features["asymmetry"] < 0.08
    )

    include_stats = bool(
        features["detected"]
        and features["resolved"]
        and not possible_star_like
        and final_morph["label"]
        not in (
            "NO CLEAR SOURCE",
            "UNRESOLVED",
        )
    )

    return {
        "loaded": loaded,
        "features": features,
        "quick_morphology": quick_morph,
        "morphology": final_morph,
        "spiral": robust_spiral,
        "possible_star_like": possible_star_like,
        "include_stats": include_stats,
    }


# ============================================================
# BATCH PIPELINE
# ============================================================

def build_galaxy_catalog(
    field_file,
    output_dir="galaxy_batch_output",
    settings=None,
    batch_settings=None,
    save_crops=True,
    save_overlay=True,
):
    settings = settings or Settings()
    batch_settings = batch_settings or BatchSettings()

    field_path = Path(
        field_file
    ).expanduser()

    if not field_path.is_file():
        raise FileNotFoundError(
            f"Field image not found: {field_path.resolve()}"
        )

    output_dir = Path(
        output_dir
    )

    crops_dir = (
        output_dir
        / "crops"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    crops_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    candidates, detection = detect_source_candidates(
        field_path,
        batch_settings=batch_settings
    )

    native_im = detection[
        "native_image"
    ]

    print(
        f"Raw connected components: "
        f"{detection['raw_component_count']}"
    )

    print(
        f"Candidates after filtering: "
        f"{len(candidates)}"
    )

    rows = []

    for i, candidate in enumerate(
        candidates,
        start=1
    ):
        box = (
            candidate["crop_left"],
            candidate["crop_top"],
            candidate["crop_right"],
            candidate["crop_bottom"],
        )

        crop = native_im.crop(
            box
        )

        crop_name = (
            f"object_{i:04d}.png"
        )

        crop_path = (
            crops_dir
            / crop_name
        )

        if save_crops:
            # PNG = lossless; no resize/interpolation.
            crop.save(
                crop_path,
                format="PNG"
            )
        else:
            # analyzer still needs a path; save temporarily in output.
            crop.save(
                crop_path,
                format="PNG"
            )

        try:
            analysis = analyze_catalog_crop(
                crop_path,
                settings=settings,
                batch_settings=batch_settings
            )

            f = analysis[
                "features"
            ]

            m = analysis[
                "morphology"
            ]

            s = analysis[
                "spiral"
            ]

            scores = m.get(
                "scores",
                {}
            )

            row = {
                "object_id": i,
                "crop_file": str(crop_path),
                "native_center_x": candidate["native_center_x"],
                "native_center_y": candidate["native_center_y"],
                "crop_left": candidate["crop_left"],
                "crop_top": candidate["crop_top"],
                "crop_right": candidate["crop_right"],
                "crop_bottom": candidate["crop_bottom"],
                "det_area": candidate["det_area"],

                "detected": f["detected"],
                "resolved": f["resolved"],
                "possible_star_like": analysis["possible_star_like"],
                "include_stats": analysis["include_stats"],

                "morphology": m["label"],
                "morphology_candidate": m["candidate"],
                "morphology_margin": m["margin"],

                "score_spiral": scores.get(
                    "Spiral Galaxy",
                    np.nan
                ),
                "score_elliptical": scores.get(
                    "Elliptical Galaxy",
                    np.nan
                ),
                "score_edge_on": scores.get(
                    "Edge-on Disk",
                    np.nan
                ),
                "score_irregular": scores.get(
                    "Irregular Galaxy",
                    np.nan
                ),
                "score_merger": scores.get(
                    "Merger / Disturbed",
                    np.nan
                ),

                "axis_ratio": f["axis_ratio"],
                "r50_px": f["r50"],
                "r90_px": f["r90"],
                "snr_proxy": f["snr_proxy"],
                "concentration": (
                    f["concentration"]
                    if f["concentration"] is not None
                    else np.nan
                ),
                "asymmetry": (
                    f["asymmetry"]
                    if f["asymmetry"] is not None
                    else np.nan
                ),
                "clumpiness": (
                    f["clumpiness"]
                    if f["clumpiness"] is not None
                    else np.nan
                ),
                "bright_peaks": f["bright_peaks"],

                "winding": s["label"],
                "winding_candidate": s["candidate"],
                "winding_score": s["score"],
                "parameter_stability": s["stability"],
                "noise_stability": s["noise_stability"],
                "coherence": s["coherence"],
                "spiral_explained": s["explained"],
                "spiral_vs_straight": s["vs_straight"],
                "arm_to_sky": s["arm_to_sky"],
                "mirror_error": s["mirror_error"],
            }

        except Exception as exc:
            row = {
                "object_id": i,
                "crop_file": str(crop_path),
                "native_center_x": candidate["native_center_x"],
                "native_center_y": candidate["native_center_y"],
                "crop_left": candidate["crop_left"],
                "crop_top": candidate["crop_top"],
                "crop_right": candidate["crop_right"],
                "crop_bottom": candidate["crop_bottom"],
                "det_area": candidate["det_area"],
                "detected": False,
                "resolved": False,
                "possible_star_like": False,
                "include_stats": False,
                "morphology": "ERROR",
                "morphology_candidate": "ERROR",
                "morphology_margin": np.nan,
                "winding": "ERROR",
                "winding_candidate": "ERROR",
                "winding_score": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            }

        rows.append(
            row
        )

        if i % 10 == 0 or i == len(candidates):
            print(
                f"Analyzed {i}/{len(candidates)}"
            )

    df = pd.DataFrame(
        rows
    )

    csv_path = (
        output_dir
        / "galaxy_catalog.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # Overlay with native crop boxes.
    overlay_path = (
        output_dir
        / "detections_overlay.png"
    )

    if save_overlay:
        fig, ax = plt.subplots(
            figsize=(18, 10)
        )

        ax.imshow(
            native_im
        )

        for candidate in candidates:
            left = candidate["crop_left"]
            top = candidate["crop_top"]
            width = (
                candidate["crop_right"]
                - left
            )
            height = (
                candidate["crop_bottom"]
                - top
            )

            rect = plt.Rectangle(
                (left, top),
                width,
                height,
                fill=False,
                linewidth=0.6
            )

            ax.add_patch(
                rect
            )

        ax.set_title(
            f"Auto-detected source candidates: {len(candidates)}"
        )

        ax.axis(
            "off"
        )

        fig.savefig(
            overlay_path,
            dpi=160,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

    print()
    print(
        "Saved catalog:",
        csv_path
    )

    print(
        "Saved crops:",
        crops_dir
    )

    if save_overlay:
        print(
            "Saved overlay:",
            overlay_path
        )

    return df, candidates, detection


# ============================================================
# SUMMARY TABLES + PLOTS
# ============================================================

def summarize_catalog(
    df,
    output_dir="galaxy_batch_output",
    show_plots=True,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    valid = df[
        df["include_stats"] == True
    ].copy()

    print("=" * 76)
    print("CATALOG SUMMARY")
    print("=" * 76)
    print(
        "Detected candidates:",
        len(df)
    )
    print(
        "Included in morphology statistics:",
        len(valid)
    )
    print(
        "Flagged possible star-like:",
        int(
            df["possible_star_like"].fillna(False).sum()
        )
    )
    print()

    if len(valid) == 0:
        print(
            "No valid objects remain for summary statistics."
        )
        return {}

    # Accepted final morphology counts.
    morphology_counts = (
        valid["morphology"]
        .value_counts()
        .rename_axis("morphology")
        .reset_index(name="count")
    )

    print("Morphology counts:")
    print(
        morphology_counts.to_string(
            index=False
        )
    )
    print()

    # Candidate counts are useful because many scientific cases are UNCERTAIN.
    candidate_counts = (
        valid["morphology_candidate"]
        .value_counts()
        .rename_axis("morphology_candidate")
        .reset_index(name="count")
    )

    print("Best-candidate morphology counts:")
    print(
        candidate_counts.to_string(
            index=False
        )
    )
    print()

    # Strict handedness only.
    strict_winding = valid[
        valid["winding"].isin(
            ["CW", "CCW"]
        )
    ]

    strict_counts = (
        strict_winding["winding"]
        .value_counts()
        .reindex(
            ["CW", "CCW"],
            fill_value=0
        )
    )

    print("Strict accepted winding:")
    print(
        strict_counts.to_string()
    )

    accepted_total = int(
        strict_counts.sum()
    )

    if accepted_total > 0:
        cw_fraction = (
            strict_counts["CW"]
            / accepted_total
        )

        print(
            f"CW fraction among accepted CW/CCW = "
            f"{cw_fraction:.3f}"
        )
    else:
        cw_fraction = np.nan
        print(
            "No strict CW/CCW classifications yet."
        )

    print()

    # Diagnostic candidate counts, including uncertain winding.
    candidate_winding = valid[
        valid["winding_candidate"].isin(
            ["CW", "CCW"]
        )
    ]

    candidate_winding_counts = (
        candidate_winding[
            "winding_candidate"
        ]
        .value_counts()
        .reindex(
            ["CW", "CCW"],
            fill_value=0
        )
    )

    print(
        "Diagnostic CW/CCW candidates "
        "(includes UNCERTAIN cases):"
    )
    print(
        candidate_winding_counts.to_string()
    )
    print()

    # Cross-table.
    cross = pd.crosstab(
        valid["morphology_candidate"],
        valid["winding_candidate"],
        dropna=False
    )

    print(
        "Morphology candidate x winding candidate:"
    )

    print(
        cross.to_string()
    )

    # Save summary tables.
    morphology_counts.to_csv(
        output_dir
        / "morphology_counts.csv",
        index=False
    )

    candidate_counts.to_csv(
        output_dir
        / "morphology_candidate_counts.csv",
        index=False
    )

    strict_counts.rename(
        "count"
    ).to_csv(
        output_dir
        / "strict_winding_counts.csv"
    )

    candidate_winding_counts.rename(
        "count"
    ).to_csv(
        output_dir
        / "candidate_winding_counts.csv"
    )

    cross.to_csv(
        output_dir
        / "morphology_vs_winding.csv"
    )

    # Plots: one figure per chart.
    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    morphology_counts.plot.bar(
        x="morphology",
        y="count",
        legend=False,
        ax=ax1
    )

    ax1.set_title(
        "Accepted morphology counts"
    )

    ax1.set_xlabel(
        "Morphology"
    )

    ax1.set_ylabel(
        "Count"
    )

    ax1.tick_params(
        axis="x",
        rotation=35
    )

    fig1.tight_layout()

    fig1.savefig(
        output_dir
        / "morphology_counts.png",
        dpi=170
    )

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(
        figsize=(6, 4)
    )

    strict_counts.plot.bar(
        ax=ax2
    )

    ax2.set_title(
        "Strict accepted spiral winding"
    )

    ax2.set_xlabel(
        "Winding"
    )

    ax2.set_ylabel(
        "Count"
    )

    ax2.tick_params(
        axis="x",
        rotation=0
    )

    fig2.tight_layout()

    fig2.savefig(
        output_dir
        / "strict_winding_counts.png",
        dpi=170
    )

    if show_plots:
        plt.show()
    else:
        plt.close(fig2)

    fig3, ax3 = plt.subplots(
        figsize=(6, 4)
    )

    candidate_winding_counts.plot.bar(
        ax=ax3
    )

    ax3.set_title(
        "Diagnostic winding candidates"
    )

    ax3.set_xlabel(
        "Candidate winding"
    )

    ax3.set_ylabel(
        "Count"
    )

    ax3.tick_params(
        axis="x",
        rotation=0
    )

    fig3.tight_layout()

    fig3.savefig(
        output_dir
        / "candidate_winding_counts.png",
        dpi=170
    )

    if show_plots:
        plt.show()
    else:
        plt.close(fig3)

    return {
        "valid": valid,
        "morphology_counts": morphology_counts,
        "candidate_counts": candidate_counts,
        "strict_winding_counts": strict_counts,
        "candidate_winding_counts": candidate_winding_counts,
        "cross_table": cross,
        "cw_fraction_strict": cw_fraction,
    }


# ============================================================
# OPTIONAL CONTACT SHEET
# ============================================================

def make_contact_sheet(
    df,
    output_file="galaxy_batch_output/contact_sheet.png",
    max_items=64,
    cols=8,
):
    subset = df.head(
        max_items
    )

    if len(subset) == 0:
        return None

    rows = int(
        np.ceil(
            len(subset) / cols
        )
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            2.3 * cols,
            2.5 * rows
        )
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for ax in axes:
        ax.axis(
            "off"
        )

    for ax, (_, row) in zip(
        axes,
        subset.iterrows()
    ):
        path = Path(
            row["crop_file"]
        )

        if path.is_file():
            ax.imshow(
                Image.open(path)
            )

        title = (
            f"#{int(row['object_id'])}\n"
            f"{row.get('morphology_candidate', '')}\n"
            f"{row.get('winding_candidate', '')}"
        )

        ax.set_title(
            title,
            fontsize=7
        )

        ax.axis(
            "off"
        )

    fig.tight_layout()

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_file,
        dpi=160
    )

    plt.show()

    return output_file



# ============================================================
# CHANGE THESE SETTINGS
# ============================================================

FIELD_FILE = "potm2606a(1).jpg"

OUTPUT_DIR = "galaxy_batch_output"

batch_settings = BatchSettings(
    # For the first run keep ~100-150 largest candidates.
    # Set None when you are ready to analyze every detected source.
    max_objects=120,

    # Lower = more sensitive, but more false detections.
    detection_sigma=2.8,

    # Crop each detected source with surrounding sky.
    crop_scale=3.2,

    # Crops are taken from the ORIGINAL native image.
    min_crop_side=48,
    max_crop_side=500,

    # Expensive CW/CCW robustness analysis is run only when
    # Spiral Galaxy is the best quick morphology candidate.
    robust_handedness_only_for_spiral_candidate=True,
)

settings = Settings(
    noise_trials=8
)



# ============================================================
# 1. DETECT -> CROP -> CLASSIFY -> CW/CCW -> CSV
# ============================================================

catalog, candidates, detection = build_galaxy_catalog(
    FIELD_FILE,
    output_dir=OUTPUT_DIR,
    settings=settings,
    batch_settings=batch_settings,
)

# Show the main catalog table.
display(
    catalog.head(30)
)


# ============================================================
# 2. SUMMARY TABLES + "WHO IS MORE COMMON?"
# ============================================================

summary = summarize_catalog(
    catalog,
    output_dir=OUTPUT_DIR,
    show_plots=True,
)


# ============================================================
# 3. VISUAL CHECK OF FIRST DETECTED OBJECTS
# ============================================================

make_contact_sheet(
    catalog,
    output_file=f"{OUTPUT_DIR}/contact_sheet.png",
    max_items=64,
    cols=8,
)