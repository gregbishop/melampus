"""Quality scoring tests (CLAUDE.md §4.1).

The headline requirement, and the reason this module exists at all: sharpness
must be measured on the detected subject, never on the whole frame. Long-lens
wildlife work means most of the frame is intentionally out of focus, so a
whole-frame variance-of-Laplacian scores excellent photographs badly.

These run against real frames from the corpus rather than synthetic ones,
because the failure mode is specifically about real bokeh.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from melampus.config import load_config
from melampus.quality import analyze_quality

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures"

# Tricolored Heron wading: sharp, patterned, ~36% of frame, water blurred behind.
# Deliberately not the white egret portrait — smooth pale plumage carries less
# high-frequency detail, so it under-scores for reasons unrelated to focus.
SHARP_ON_BOKEH = FIXTURES / "0A1A2829.jpg"
# Completely defocused frame — no subject at all.
FULLY_SOFT = FIXTURES / "0A1A4157.jpg"
# Submerged alligator, heavy blur, tiny in frame.
BLURRED_DISTANT = FIXTURES / "0A1A4117.jpg"
# Green Heron in mangrove: sharp, but busy and smaller in frame.
SHARP_BUSY = FIXTURES / "0A1A3141.jpg"

pytestmark = pytest.mark.skipif(
    not SHARP_ON_BOKEH.is_file(), reason="corpus fixtures not present"
)


@pytest.fixture(scope="module")
def config():
    return load_config()


# --------------------------------------------------------------------------- #
# the regression this module exists for
# --------------------------------------------------------------------------- #
def test_sharp_subject_against_heavy_bokeh_scores_high(config):
    """A sharp bird on a blurred background must not be punished for the blur."""
    result = analyze_quality(SHARP_ON_BOKEH, config)
    assert result.composite >= 70, (
        f"sharp subject on bokeh scored {result.composite:.1f}; "
        "this is the whole-frame-sharpness mistake"
    )


def test_subject_sharpness_exceeds_whole_frame_sharpness(config):
    """The mechanism, asserted directly.

    If these are close, sharpness is being measured over the whole frame even if
    the composite happens to pass.
    """
    result = analyze_quality(SHARP_ON_BOKEH, config)
    assert result.subject_sharpness > result.frame_sharpness * 1.5, (
        f"subject {result.subject_sharpness:.1f} vs frame {result.frame_sharpness:.1f} "
        "— measurement is not subject-localised"
    )


def test_subject_is_localised_not_the_whole_frame(config):
    """A detector returning the entire frame would trivially pass the above."""
    result = analyze_quality(SHARP_ON_BOKEH, config)
    assert result.subject_detected
    assert 0.01 < result.subject_size_fraction < 0.9, (
        f"subject covers {result.subject_size_fraction:.2f} of frame"
    )


def test_soft_frame_scores_below_sharp_frame(config):
    sharp = analyze_quality(SHARP_ON_BOKEH, config)
    soft = analyze_quality(FULLY_SOFT, config)
    assert sharp.composite > soft.composite + 15, (
        f"sharp {sharp.composite:.1f} vs fully defocused {soft.composite:.1f}"
    )


def test_blurred_distant_subject_scores_low(config):
    result = analyze_quality(BLURRED_DISTANT, config)
    assert result.composite < 55, f"heavily blurred frame scored {result.composite:.1f}"


# --------------------------------------------------------------------------- #
# sub-metrics must all be exposed individually (§4.1)
# --------------------------------------------------------------------------- #
def test_every_submetric_is_exposed(config):
    result = analyze_quality(SHARP_ON_BOKEH, config)
    for field in (
        "subject_sharpness", "eye_sharpness", "frame_sharpness",
        "motion_blur", "defocus_blur", "blur_angle_degrees",
        "subject_size_fraction", "clipped_highlights_pct", "clipped_shadows_pct",
        "subject_edge_clipped", "composite",
    ):
        assert hasattr(result, field), f"sub-metric {field} is not exposed"


def test_motion_and_defocus_are_separate_values(config):
    """§4.1: directional blur is often desirable, isotropic defocus is not."""
    result = analyze_quality(SHARP_ON_BOKEH, config)
    assert result.motion_blur != result.defocus_blur or result.motion_blur == 0
    assert 0 <= result.motion_blur <= 100
    assert 0 <= result.defocus_blur <= 100


def test_composite_weights_come_from_config(config):
    """Weights must be config-driven, not hardcoded.

    Uses a blurred frame deliberately: on a tack-sharp one every term is already
    100, so reweighting cannot change the result and the test proves nothing.
    """
    base = analyze_quality(BLURRED_DISTANT, config)

    tweaked = config.model_copy(update={
        "quality": config.quality.model_copy(update={
            "weight_subject_sharpness": 0.0,
            "weight_eye_sharpness": 0.0,
            "weight_focus": 0.0,
            "weight_exposure": 1.0,
        })
    })
    altered = analyze_quality(BLURRED_DISTANT, tweaked)
    assert altered.composite != base.composite, "weights had no effect"


def test_exposure_percentages_are_real_numbers(config):
    result = analyze_quality(SHARP_ON_BOKEH, config)
    assert 0.0 <= result.clipped_highlights_pct <= 100.0
    assert 0.0 <= result.clipped_shadows_pct <= 100.0


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_unreadable_file_degrades_rather_than_raising(tmp_path: Path, config):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg at all")
    result = analyze_quality(broken, config)
    assert result.error is not None
    assert result.composite == 0.0


def test_scoring_is_deterministic(config):
    a = analyze_quality(SHARP_BUSY, config)
    b = analyze_quality(SHARP_BUSY, config)
    assert a.composite == b.composite
