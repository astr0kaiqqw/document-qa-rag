"""Local, trusted presentation assets; no model or document processing imports."""
import base64
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def appearance_html():
    svg=(ROOT/'assets/silver-wave.svg').read_bytes()
    uri='data:image/svg+xml;base64,'+base64.b64encode(svg).decode('ascii')
    css=(ROOT/'theme.css').read_text().replace('__AMBIENT_SVG__',uri)
    script=(ROOT/'assets/ambient.js').read_text()
    return '<style>'+css+'</style><script>'+script+'</script>'
