"""Automated accessibility (WCAG 2.1 AA) checks.

Two layers of enforcement:
1. Unit tests for ``core.a11y`` — contrast math and generated CSS.
2. Static regression gates — the theme config must keep AA contrast, and every
   interactive widget in ``app.py`` must ship an accessible ``help`` description.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from core import a11y

ROOT = Path(__file__).resolve().parents[1]

#: Widget factories that must always carry a ``help=`` description (WCAG 1.3.1 / 3.3.2).
WIDGETS_REQUIRING_HELP = frozenset(
    {
        "text_input",
        "text_area",
        "slider",
        "number_input",
        "selectbox",
        "radio",
        "toggle",
        "metric",
        "button",
        "form_submit_button",
    }
)


# ------------------------------------------------------------ contrast math


class TestContrastMath:
    def test_white_on_black_is_max_contrast(self) -> None:
        assert a11y.contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)

    def test_same_color_is_min_contrast(self) -> None:
        assert a11y.contrast_ratio("#34d399", "#34d399") == pytest.approx(1.0, abs=0.001)

    def test_order_does_not_matter(self) -> None:
        assert a11y.contrast_ratio("#e6edf7", "#0b1120") == pytest.approx(
            a11y.contrast_ratio("#0b1120", "#e6edf7")
        )

    def test_luminance_bounds(self) -> None:
        assert a11y.relative_luminance("#000000") == pytest.approx(0.0)
        assert a11y.relative_luminance("#ffffff") == pytest.approx(1.0)

    def test_accepts_bare_hex(self) -> None:
        assert a11y.relative_luminance("ffffff") == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", ["#fff", "", "#12345", "not-a-color1"])
    def test_rejects_malformed_colors(self, bad: str) -> None:
        with pytest.raises(ValueError):
            a11y.relative_luminance(bad)


# ------------------------------------------------------- theme conformance


class TestThemeContrast:
    """The deployed Streamlit theme must stay WCAG AA compliant."""

    @pytest.fixture(scope="class")
    def theme(self) -> dict[str, str]:
        config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
        return config["theme"]

    def test_body_text_meets_aa(self, theme: dict[str, str]) -> None:
        ratio = a11y.contrast_ratio(theme["textColor"], theme["backgroundColor"])
        assert ratio >= a11y.AA_TEXT_CONTRAST, f"Body text contrast {ratio:.2f} < 4.5:1 (WCAG 1.4.3)"

    def test_text_on_secondary_surface_meets_aa(self, theme: dict[str, str]) -> None:
        ratio = a11y.contrast_ratio(theme["textColor"], theme["secondaryBackgroundColor"])
        assert ratio >= a11y.AA_TEXT_CONTRAST, f"Card text contrast {ratio:.2f} < 4.5:1 (WCAG 1.4.3)"

    def test_primary_color_meets_component_aa(self, theme: dict[str, str]) -> None:
        ratio = a11y.contrast_ratio(theme["primaryColor"], theme["backgroundColor"])
        assert ratio >= a11y.AA_COMPONENT_CONTRAST, f"Accent contrast {ratio:.2f} < 3:1 (WCAG 1.4.11)"


# ------------------------------------------------------------ generated CSS


class TestGeneratedCss:
    def test_base_css_covers_required_criteria(self) -> None:
        css = a11y.build_css()
        assert ":focus-visible" in css, "Missing focus indicator (WCAG 2.4.7)"
        assert "prefers-reduced-motion" in css, "Missing reduced-motion support (WCAG 2.3.3)"
        assert "min-height: 44px" in css, "Missing minimum target size (WCAG 2.5.8)"
        assert ".skip-link" in css, "Missing skip-link styling (WCAG 2.4.1)"

    def test_default_scale_adds_no_font_override(self) -> None:
        assert "font-size" not in a11y.text_scale_css(1.0)

    @pytest.mark.parametrize(("label", "scale"), [("Large", 1.15), ("Extra large", 1.3)])
    def test_text_scales_resize_root_font(self, label: str, scale: float) -> None:
        assert a11y.TEXT_SCALES[label] == scale
        assert f"font-size: {round(16 * scale)}px" in a11y.text_scale_css(scale)

    def test_high_contrast_uses_pure_black_and_white(self) -> None:
        css = a11y.build_css(high_contrast=True)
        assert "#000000" in css and "#ffffff" in css
        assert a11y.contrast_ratio("#ffffff", "#000000") >= 7.0  # exceeds AAA (WCAG 1.4.6)

    def test_high_contrast_off_by_default(self) -> None:
        assert "#000000" not in a11y.build_css()

    def test_skip_link_targets_main_anchor(self) -> None:
        assert 'href="#main-content"' in a11y.SKIP_LINK_HTML
        assert 'id="main-content"' in a11y.MAIN_ANCHOR_HTML
        assert 'tabindex="-1"' in a11y.MAIN_ANCHOR_HTML


# ----------------------------------------------------- app-wide static gates


def _widget_calls_missing_help(source: str) -> list[str]:
    """Return descriptions of widget calls in *source* lacking a ``help`` keyword."""
    tree = ast.parse(source)
    missing: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in WIDGETS_REQUIRING_HELP
            and not any(kw.arg == "help" for kw in node.keywords)
        ):
            missing.append(f"{node.func.attr} at line {node.lineno}")
    return missing


class TestAppAccessibilityGates:
    """Static regression gates so accessibility cannot silently erode."""

    @pytest.fixture(scope="class")
    def app_source(self) -> str:
        return (ROOT / "app.py").read_text()

    def test_every_widget_has_help_text(self, app_source: str) -> None:
        missing = _widget_calls_missing_help(app_source)
        assert not missing, f"Widgets missing accessible help text (WCAG 3.3.2): {missing}"

    def test_charts_have_text_alternatives(self, app_source: str) -> None:
        assert app_source.count("st.line_chart") == app_source.count("Chart summary"), (
            "Every chart needs a plain-text summary (WCAG 1.1.1)"
        )

    def test_app_injects_accessibility_css_and_skip_link(self, app_source: str) -> None:
        assert "_inject_accessibility()" in app_source
        assert "a11y.SKIP_LINK_HTML" in app_source
        assert "a11y.build_css" in app_source

    def test_user_facing_a11y_controls_exist(self, app_source: str) -> None:
        assert "a11y_text_scale" in app_source, "Text-size control missing (WCAG 1.4.4)"
        assert "a11y_high_contrast" in app_source, "High-contrast control missing"

    def test_risk_icons_paired_with_text_labels(self, app_source: str) -> None:
        # Emoji must never be the sole carrier of meaning (WCAG 1.4.1 Use of Color).
        for label in ("Low risk", "Watch zone", "High risk", "Not enough data"):
            assert label in app_source

    def test_conformance_statement_exists(self) -> None:
        doc = ROOT / "ACCESSIBILITY.md"
        assert doc.is_file(), "ACCESSIBILITY.md conformance statement is required"
        assert "WCAG 2.1" in doc.read_text()
