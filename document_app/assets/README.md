# Presentation assets

`silver-wave.svg` is original vector artwork created for this Document QA redesign: a metallic ribbon, dotted dissolution and small ASCII marks. No reference screenshots, logos, photographs or external textures are embedded. It has no SVG animation, filters or external resources.

`ambient.js` only updates the namespaced motion state on visibility/preference changes. A single window controller prevents duplicate listeners on Streamlit reruns. CSS provides one background pseudo-element; there are no timers, canvas loops, network calls, model requests or Streamlit reruns. Mobile and reduced-motion CSS disable movement. Without JavaScript the reduced-motion and mobile rules still apply.

The SVG is embedded as a local data URI by `appearance.py`; static hosting is not required. Typography uses system Arial (Cyrillic and Latin, generic sans-serif fallback) and system monospace for short labels. No font files are copied or downloaded.
