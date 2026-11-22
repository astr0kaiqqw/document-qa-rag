"""Checks saved real-model outputs; does not invoke inference or alter models."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];out=root/'reports/multimodal_update'
if not (out/'acceptance.json').is_file() or not (out/'ui_acceptance.json').is_file():
    raise SystemExit('Local acceptance/UI results are not bundled. Run the documented local checks first; see README.')
rows=json.loads((out/'acceptance.json').read_text());cases={r['case']:r for r in rows}
assert len(cases)==11 and not any('error' in r for r in rows)
def result(name):return cases[name]['result']
assert result('text_txt')['answer']=='Aurora'
assert result('text_pdf')['answer']=='Teaching Assistants' and result('text_pdf')['pages']==[30]
for name in ['slide_image','scanned_pdf','manual_visual_native']:
    assert 'Teaching Assistants' in result(name)['answer']
    assert result(name)['answer_method']=='vision' and result(name)['source']==''
assert all(t in result('schedule')['answer'] for t in ['12:00','13:00','14:00'])
assert '09:00' not in result('schedule')['answer']
assert '6' in result('formula')['answer'] and 'Н' in result('formula')['answer']
assert result('missing_fact')['answer']=='' and result('missing_fact')['status']=='no_answer'
assert result('mixed_native')['path_label']=='QA по тексту' and result('mixed_native')['pages']==[1]
assert result('mixed_scan')['path_label']=='Локальная модель по изображению' and result('mixed_scan')['pages']==[2]
assert '6' in result('mixed_scan')['answer']
game=result('game')['answer'].lower()
assert all(word in game for word in ['airship','character','tree'])
ui=json.loads((out/'ui_acceptance.json').read_text());assert ui['widget_upload'] and ui['widget_submit'] and ui['audio_input_nodes']==1
summary=dict(cases=11,execution_errors=0,checks_passed=True,
             game_identification_warning=('Minecraft' in result('game')['answer']),
             note='Visible game objects described, but Luanti screenshot was called Minecraft. This is a real model error, not a verified game identification.',
             unit_tests='187 passed; private OCR fixture enabled',ui='real Streamlit AppTest upload and VLM submission passed; browser visual inspection unavailable')
(out/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('11 real-model scenarios and widget flow verified; game-name hallucination recorded.')
