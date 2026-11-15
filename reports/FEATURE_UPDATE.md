# Local OCR, speech, vision QA and interface update

The existing Streamlit application now supports local image OCR, local speech transcription and a second answer path using Qwen3-VL 4B Instruct through loopback Ollama. Text documents retain the existing extractive QA checkpoint and retrieval pipeline; images and selected visual PDF pages use the vision model directly. Mixed PDF routing is page-level, OCR assists page search, and explicit page/mode selection is available. There is no hidden cloud or QA-failure fallback.

The interface uses neutral graphite, muted emerald and original local silver SVG artwork. Reduced-motion/mobile modes disable the decorative movement; hidden tabs pause it. Document upload/selection, editable voice input, history, citations and original-page previews remain available. A root-position CSS regression found in the live interface was corrected by retaining Streamlit's native root positioning.

Validation before publication: **190 tests passed** with the optional local timetable fixture enabled; Python syntax and Git whitespace checks passed. Earlier real local-model acceptance exercised 11 scenarios and actual Streamlit upload/submit widgets. Browser-based desktop/mobile visual acceptance remains unavailable. Vision output is generative and can hallucinate, including an incorrect game title in the acceptance run; it is labelled accordingly.

Private documents, screenshots, extracted content, detailed local diagnostic traces and model weights are excluded from publication. Real-model acceptance scripts require explicitly supplied local fixture paths. See README for dependencies, one-time model setup, local-only configuration and resource limits. No training or fine-tuning was performed.
