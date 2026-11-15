# Document QA — visual refresh

Implemented 15 November 2026 from all four supplied reference images. The supplementary hyve.systems URL could not be fetched. Existing Streamlit architecture, model selection/routing, QA, OCR, speech, document sessions, history and source rendering were retained.

## Appearance

Neutral graphite replaces the green cast. Dense neutral panels sit in front of one large silver ribbon on the right, with dotted/ASCII dissolution and broad dark gradients. The artwork is original SVG, not a screenshot or borrowed logo. Muted emerald replaces lime. Headings use regular-weight system Arial, a restrained size and tight spacing; body text and long explanations use Arial, and only small technical labels use system monospace. No remote fonts or raster downloads are required.

Controls use the native theme plus scoped styles for forms, upload, selectors and expanders. Semantic warning/error colours are retained. Empty-state and answer surfaces use explicit Streamlit container keys. No widget is hidden. The original image/source remains available below the answer.

## Animation and performance

One CSS pseudo-element supplies the background, using the local SVG as a data URI. Motion is a slow 8×9 px translation over 32 seconds. There is no canvas, 3D library, SVG filter, render loop, timer or animation-triggered Streamlit rerun. The own, trusted script only watches tab visibility and reduced-motion preference; a single window controller prevents duplicate listeners. Hidden tabs pause movement. Mobile and reduced-motion CSS disable it. The decoration cannot receive clicks and does not contain focusable elements.

## Files

- `document_app/assets/silver-wave.svg` — original silver ribbon, dots and ASCII marks.
- `document_app/assets/ambient.js` — visibility/reduced-motion controller.
- `document_app/assets/README.md` — origin and asset behaviour.
- `document_app/appearance.py` — embeds trusted local assets.
- `document_app/theme.css` — visual rules and responsive/reduced-motion variants.
- `app.py` — appearance injection and explicit presentation keys only in this redesign.
- `.streamlit/config.toml` — theme colours and system font only; local server/privacy settings unchanged.

No document-processing dependencies, QA/model weights or inference logic were changed for this redesign. No publication or GitHub actions were performed.

## Verification and limits

33 existing targeted tests passed: application upload/error/reset, sessions/history, resource upgrades and multimodal interface/routing/error scenarios. A Node execution of the actual motion script verified pause on hidden tab, reduced-motion pause, resume, and one listener after repeated execution. The SVG was rendered with rsvg-convert and visually inspected as an artwork preview (not an application screenshot). Source formatting passes `git diff --check`.

A separate Streamlit AppTest stress state checks a long Cyrillic filename, a long Russian/English answer and source, extraction diagnostics, uploader, question/submit and audio widgets. These are rendering/state checks; they do not prove pixel readability or responsive geometry.

The browser tool returned `The browser access host is unavailable`. Therefore desktop/mobile pixel inspection, horizontal-scroll/overlap checks, keyboard navigation/focus appearance, actual tab-switch animation observation and final application screenshots could not be completed. The CSS includes wrapping, mobile layout, visible focus and reduced-motion behaviour, but those browser-visible outcomes are not claimed as visually verified. No mockup is presented as a screenshot.

The running app remains at http://127.0.0.1:8502. Refresh the page for the updated appearance; no forced restart/session reset was performed.


## Root layout correction

The first live browser screenshot showed the decorative layer with the entire interface clipped. Inspection of installed Streamlit frontend styles confirmed that both the root app and app view depend on absolute positioning with top/right/bottom/left zero and hidden overflow. The custom `position: relative` on `.stApp` changed that sizing contract and collapsed the content container. Removed that declaration; the decorative pseudo-element remains fixed and does not alter native root geometry. This was a CSS layout defect, not a QA/model failure. AppTest checks did not detect it because they do not render browser layout.
