"""Real local-model acceptance. Saves genuine outputs, never fabricated answers."""
import sys,os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from document_app.service import DocumentSession
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader,PdfWriter
import json,time
# These are local acceptance fixtures, intentionally excluded from Git.
def fixture(variable):
    value=os.environ.get(variable)
    if not value or not Path(value).is_file():
        raise SystemExit(f'Set {variable} to an existing local fixture. See README: local acceptance inputs. No private documents are bundled.')
    return Path(value)
lecture=fixture('DOCUMENT_QA_LECTURE_PDF')
slide=fixture('DOCUMENT_QA_SLIDE_PNG')
schedule=fixture('DOCUMENT_QA_SCHEDULE_PDF')
game=fixture('DOCUMENT_QA_GAME_IMAGE')
OUT=ROOT/'reports/multimodal_update';INPUT=OUT/'inputs';INPUT.mkdir(parents=True,exist_ok=True)
font_path=next((p for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).is_file()),None)
if not font_path:raise SystemExit('Install a Cyrillic/Latin TrueType font (Arial or DejaVu Sans) for generated fixtures.')
font=ImageFont.truetype(font_path,70)
formula=Image.new('RGB',(1200,450),'white');draw=ImageDraw.Draw(formula)
draw.text((70,70),'F = m × a',font=font,fill='black');draw.text((70,210),'m = 2 kg; a = 3 m/s²',font=font,fill='black');formula.save(INPUT/'formula.png')
with Image.open(slide) as image:image.convert('RGB').save(INPUT/'scanned_slide.pdf','PDF',resolution=150)
w=PdfWriter();w.add_page(PdfReader(lecture).pages[29]);formula.save(INPUT/'formula_scan.pdf','PDF');w.append(PdfReader(INPUT/'formula_scan.pdf'));w.write(INPUT/'mixed.pdf')
s=DocumentSession();results=[]
def run(name,filename,data,question,expected_path,**kwargs):
    started=time.monotonic()
    try:
        doc=s.add_upload(filename,data)
        result=s.ask(doc.key,question,**kwargs)
        assert result['path_label']==expected_path,(name,result)
        row=dict(case=name,seconds=round(time.monotonic()-started,2),routes=[p.__dict__ for p in doc.page_routes],result=result)
    except Exception as error:
        row=dict(case=name,seconds=round(time.monotonic()-started,2),error=type(error).__name__+': '+str(error))
    results.append(row);(OUT/'acceptance.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(name,json.dumps(row,ensure_ascii=False),flush=True)
    # Free only this acceptance session's documents, not user sessions.
    s.documents.clear()

try:
    run('text_txt','fact.txt',b'The research project is named Aurora.','What is the research project named?','QA по тексту')
    run('text_pdf',lecture.name,lecture.read_bytes(),'who will check the email?','QA по тексту')
    run('slide_image',slide.name,slide.read_bytes(),'who will check the email?','Локальная модель по изображению')
    run('schedule',schedule.name,schedule.read_bytes(),'Во сколько в понедельник основы кибербезопасности? Укажи все интервалы.','Локальная модель по изображению')
    run('formula','formula.png',(INPUT/'formula.png').read_bytes(),'Прочитай формулу, объясни её и вычисли силу по указанным значениям.','Локальная модель по изображению')
    run('game','game_luanti.jpg',game.read_bytes(),'Describe the visible objects in this game screenshot.','Локальная модель по изображению')
    run('scanned_pdf','scanned_slide.pdf',(INPUT/'scanned_slide.pdf').read_bytes(),'who will check the email?','Локальная модель по изображению')
    run('missing_fact',slide.name,slide.read_bytes(),"What is the lecturer's bank account number?",'Локальная модель по изображению')
    run('mixed_native','mixed.pdf',(INPUT/'mixed.pdf').read_bytes(),'who will check the email?','QA по тексту',page=1)
    run('mixed_scan','mixed.pdf',(INPUT/'mixed.pdf').read_bytes(),'Вычисли силу по изображению.','Локальная модель по изображению',page=2)
    run('manual_visual_native','mixed.pdf',(INPUT/'mixed.pdf').read_bytes(),'who will check the email?','Локальная модель по изображению',page=1,mode='visual')
finally:s.close()
