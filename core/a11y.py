"""Accessibility (WCAG 2.1 AA) helpers for the Unhooked UI.

Pure functions and constants only — no Streamlit imports — so every rule
(contrast math, generated CSS) is unit-testable without a running app.
Success criteria implemented here are annotated inline (e.g. WCAG 2.4.7).
"""

from __future__ import annotations

from typing import Final

#: User-selectable text sizes (WCAG 1.4.4 Resize Text — up to 200 % via browser zoom on top of this).
TEXT_SCALES: Final[dict[str, float]] = {"Default": 1.0, "Large": 1.15, "Extra large": 1.3}

#: Skip-navigation link injected at the top of the page (WCAG 2.4.1 Bypass Blocks).
SKIP_LINK_HTML: Final[str] = '<a class="skip-link" href="#main-content">Skip to main content</a>'

#: Focusable landmark the skip link jumps to, placed before the page body.
MAIN_ANCHOR_HTML: Final[str] = '<div id="main-content" tabindex="-1"></div>'

_BASE_FONT_PX: Final[int] = 16

#: Length of an ``rrggbb`` hex color string.
_HEX_DIGITS: Final[int] = 6

#: sRGB linearization cutoff from the WCAG 2.x relative-luminance definition.
_SRGB_LINEAR_CUTOFF: Final[float] = 0.04045

#: Minimum WCAG AA contrast for normal text (1.4.3) and for UI components (1.4.11).
AA_TEXT_CONTRAST: Final[float] = 4.5
AA_COMPONENT_CONTRAST: Final[float] = 3.0

BASE_CSS: Final[str] = """
/* WCAG 2.4.1 Bypass Blocks — skip link, visible only on keyboard focus */
a.skip-link {
    position: absolute; left: -999px; top: 0; z-index: 10000;
    background: #34d399; color: #0b1120; padding: 0.6rem 1.2rem;
    font-weight: 700; text-decoration: none; border-radius: 0 0 8px 0;
}
a.skip-link:focus { left: 0; outline: 3px solid #e6edf7; }

/* WCAG 2.4.7 Focus Visible — high-visibility ring on every focusable element */
*:focus-visible {
    outline: 3px solid #34d399 !important;
    outline-offset: 2px !important;
}

/* WCAG 2.5.8 Target Size — comfortable minimum hit area for controls */
button, [role="button"], [role="radio"], [role="slider"] { min-height: 44px; }

/* WCAG 2.3.3 Animation from Interactions — honor the user's reduced-motion preference */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
"""


def relative_luminance(hex_color: str) -> float:
    """Compute the WCAG relative luminance (0.0–1.0) of a ``#rrggbb`` color.

    Args:
        hex_color: Color in ``#rrggbb`` (leading ``#`` optional).

    Returns:
        Relative luminance per the WCAG 2.x definition.

    Raises:
        ValueError: If the string is not a 6-digit hex color.
    """
    value = hex_color.lstrip("#")
    if len(value) != _HEX_DIGITS:
        raise ValueError(f"Expected #rrggbb color, got {hex_color!r}")
    linear: list[float] = []
    for i in (0, 2, 4):
        channel = int(value[i : i + 2], 16) / 255
        linear.append(channel / 12.92 if channel <= _SRGB_LINEAR_CUTOFF else ((channel + 0.055) / 1.055) ** 2.4)
    red, green, blue = linear
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio (1.0–21.0) between two ``#rrggbb`` colors.

    Args:
        foreground: Text / component color.
        background: Color it sits on.

    Returns:
        Contrast ratio; ≥ 4.5 passes AA for normal text (WCAG 1.4.3).
    """
    lighter, darker = sorted((relative_luminance(foreground), relative_luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def text_scale_css(scale: float) -> str:
    """Build CSS applying the user's chosen text size (WCAG 1.4.4 Resize Text).

    Args:
        scale: Multiplier from :data:`TEXT_SCALES` (1.0 = browser default).

    Returns:
        A CSS rule setting the root font size; empty string for the default scale.
    """
    if scale == 1.0:
        return ""
    return f"html {{ font-size: {round(_BASE_FONT_PX * scale)}px; }}"


def high_contrast_css() -> str:
    """Build opt-in high-contrast overrides (~21:1, beyond WCAG AAA 1.4.6).

    Returns:
        CSS forcing pure white text on pure black surfaces.
    """
    return (
        ".stApp { background-color: #000000 !important; }\n"
        '[data-testid="stSidebar"] { background-color: #000000 !important; '
        "border-right: 1px solid #ffffff55; }\n"
        ".stApp p, .stApp label, .stApp li, .stApp span, .stApp div[data-testid='stMarkdownContainer'] "
        "{ color: #ffffff !important; }"
    )


def build_css(*, text_scale: float = 1.0, high_contrast: bool = False) -> str:
    """Assemble the full accessibility stylesheet for the current user preferences.

    Args:
        text_scale: Multiplier from :data:`TEXT_SCALES`.
        high_contrast: Whether the user enabled high-contrast mode.

    Returns:
        Combined CSS: base WCAG rules plus any user-preference overrides.
    """
    parts = [BASE_CSS, text_scale_css(text_scale)]
    if high_contrast:
        parts.append(high_contrast_css())
    return "\n".join(part for part in parts if part)
