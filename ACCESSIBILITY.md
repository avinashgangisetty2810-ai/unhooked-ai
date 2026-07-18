# ♿ Accessibility Conformance Statement

**Unhooked** targets **WCAG 2.1 Level AA** conformance. Accessibility is engineered into the
code (`core/a11y.py`), enforced by automated tests (`tests/test_a11y.py`), and surfaced to
users through in-app controls in the sidebar's **♿ Accessibility** panel.

## User-facing accessibility features

| Feature | Where |
|---|---|
| **Text size control** (Default / Large / Extra large) | Sidebar → ♿ Accessibility |
| **High-contrast mode** (pure white on black, ~21:1) | Sidebar → ♿ Accessibility |
| **Skip to main content** link | Press `Tab` once on page load |
| **Help tooltips on every input, button, and metric** | Hover / focus the `?` icon |
| **Plain-text chart summaries** | Below every chart |
| **Full keyboard operation** | `Tab` to move, arrow keys for radios/sliders, `Enter` to activate |

## WCAG 2.1 success criteria addressed

| Criterion | Level | How it is met |
|---|---|---|
| 1.1.1 Non-text Content | A | Every chart has a plain-text data summary; emoji are decorative and always paired with text |
| 1.3.1 Info and Relationships | A | Native Streamlit widgets emit semantic HTML with programmatic labels |
| 1.4.1 Use of Color | A | Risk levels use icon **and** text label ("🔴 High risk"), never color alone |
| 1.4.3 Contrast (Minimum) | AA | Body text `#e6edf7` on `#0b1120` ≈ 15.9:1 (test-enforced ≥ 4.5:1) |
| 1.4.4 Resize Text | AA | In-app text-size control up to 130 %, compatible with 200 % browser zoom |
| 1.4.6 Contrast (Enhanced) | AAA | Opt-in high-contrast mode delivers ~21:1 |
| 1.4.11 Non-text Contrast | AA | Accent `#34d399` on background ≥ 3:1 (test-enforced) |
| 2.1.1 Keyboard | A | All functionality operable via keyboard (native widgets, no pointer-only interactions) |
| 2.3.3 Animation from Interactions | AAA | `prefers-reduced-motion` disables animations/transitions |
| 2.4.1 Bypass Blocks | A | "Skip to main content" link, visible on keyboard focus |
| 2.4.2 Page Titled | A | Descriptive page title ("Unhooked — AI Recovery Coach") |
| 2.4.7 Focus Visible | AA | 3 px high-contrast focus ring on every focusable element |
| 2.5.8 Target Size (Minimum) | AA (2.2) | 44 px minimum height on buttons and interactive controls |
| 3.2.3 Consistent Navigation | AA | Persistent sidebar navigation in fixed order across all pages |
| 3.3.1 Error Identification | A | Validation and AI-failure errors described in plain text |
| 3.3.2 Labels or Instructions | A | Every input carries a visible label **and** contextual help text |

## Automated enforcement

`tests/test_a11y.py` fails the build if accessibility regresses:

- **Contrast gates** — theme colors in `.streamlit/config.toml` are recomputed with the WCAG
  luminance formula on every test run; dropping below AA fails CI.
- **Widget gate** — the AST of `app.py` is scanned; any input, button, radio, slider, or metric
  added without `help=` text fails CI.
- **Chart gate** — every `st.line_chart` must have a matching plain-text summary.
- **Feature gates** — skip link, focus CSS, reduced motion, text-size and high-contrast
  controls must remain wired into the app.

## Known limitations

- Streamlit renders some internal DOM we cannot modify; screen-reader ordering of toast
  notifications is controlled by the framework.
- Chart tooltips are pointer-driven (framework limitation); the text summary below each chart
  is the accessible equivalent.

## Feedback

Found a barrier? Open an issue on the repository — accessibility bugs are triaged as `P1`.
