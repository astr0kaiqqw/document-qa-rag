"""Run: streamlit run app.py. Runtime inference and documents stay on this machine."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import streamlit as st
from document_app.service import DocumentSession, LocalModels, friendly_error, PROCESSING_VERSION, upgrade_cached_models

st.set_page_config(page_title='Document QA · Локальный поиск ответов',page_icon='📖',layout='wide')
from pathlib import Path
from document_app.speech import LocalSpeech, SpeechError
from document_app.sources import render_source
from document_app.vision import VisionError
from document_app.appearance import appearance_html
st.html(appearance_html(),unsafe_allow_javascript=True)


@st.cache_resource(show_spinner=False)
def _cached_models(version=4):
    return LocalModels()


def shared_models():
    # Outside the decorated function: its body does not run on a cache hit.
    return upgrade_cached_models(_cached_models())


@st.cache_resource(show_spinner=False)
def speech_model():
    return LocalSpeech()


if 'documents' not in st.session_state:
    st.session_state.documents=DocumentSession(shared_models())
if 'upload_generation' not in st.session_state:
    st.session_state.upload_generation=0
session=st.session_state.documents
# Keep existing document objects and their owned temporary files during hot
# reload, while upgrading methods to the current service implementation.
if not isinstance(session,DocumentSession):session.__class__=DocumentSession
session.models=upgrade_cached_models(session.models)

@st.cache_data(ttl=15,show_spinner=False)
def vision_readiness():
    try:return True,shared_models().vision().check()
    except (VisionError,ValueError) as error:return False,str(error)
vision_ready,vision_status=vision_readiness()

with st.sidebar:
    st.markdown('### 📖 Document QA')
    st.caption('Ваши документы. Ответы из источника.')
    st.caption('Визуальная модель: '+vision_status)
    if st.button('Проверить модель'):
        vision_readiness.clear()
        st.rerun()
    uploads=st.file_uploader('Загрузить документы',type=['pdf','docx','txt','png','jpg','jpeg','webp'],accept_multiple_files=True,
        key=f'uploads_{st.session_state.upload_generation}',help='PDF, DOCX, TXT, PNG, JPG, WEBP · 25 МБ · PDF до 100 страниц')
    if st.button('Обработать файлы',type='primary',use_container_width=True,disabled=not uploads):
        for upload in uploads:
            try:
                with st.spinner(f'Извлечение текста / OCR и индексация: {upload.name}…'):
                    added=session.add_upload(upload.name,upload.getvalue())
                st.session_state.selected_document=added.key
            except Exception as error:
                st.error(friendly_error(error))
    st.divider()
    if session.documents:
        keys=list(session.documents)
        if st.session_state.get('selected_document') not in keys:
            st.session_state.selected_document=keys[0]
        selected=st.selectbox('Выбранный документ',keys,
            format_func=lambda k:(session.documents[k].name+' · '+k[:6]) if sum(d.name==session.documents[k].name for d in session.documents.values())>1 else session.documents[k].name,key='selected_document')
        st.caption(f'Документов в сессии: {len(keys)} / 8')
    else:
        selected=None
        st.info('Загрузите документ, чтобы начать.')
    if st.button('Сбросить сессию',use_container_width=True):
        session.close()
        st.session_state.documents=DocumentSession(shared_models())
        st.session_state.upload_generation+=1
        for key in ['selected_document','question','voice_digest']:
            st.session_state.pop(key,None)
        st.rerun()
    with st.expander('О приложении'):
        st.write('Обработка локальная: extractive QA для текста, Qwen3-VL через Ollama для изображений. Визуальные ответы генерируются и требуют проверки.')
        st.caption('Модель: deepset/xlm-roberta-base-squad2. Поддержка EN / RU / KK зависит от текста и языка; проверяйте источник.')

st.markdown('<div class="workspace-header"><strong><i class="pixel"></i>Document QA</strong><span>ЛОКАЛЬНАЯ ОБРАБОТКА / RU + EN</span></div>',unsafe_allow_html=True)
st.caption('01 / DOCUMENT WORKSPACE')
st.title('Ваши документы.\nТочные фрагменты.')
st.write('Загрузите материал, задайте вопрос и проверьте ответ в его контексте.')

if not selected:
    st.markdown('<div class="empty-orbit"><span></span> ТЕКСТ → ПОИСК → ИСТОЧНИК</div>',unsafe_allow_html=True)
    with st.container(border=False,key='empty_state'):
        st.subheader('Добавьте первый источник')
        st.write('Перетащите документ или фотографию в панель загрузки. Сканированные страницы распознаются локально.')
        st.caption('PDF / DOCX / TXT / PNG / JPG / WEBP · до 25 МБ · до 8 документов')
        st.info('Откройте панель документов слева и нажмите «Обработать файлы».')
    st.stop()

document=session.documents[selected]
if getattr(document.pipeline, 'processing_version', 1) != PROCESSING_VERSION:
    st.warning('Для этого документа доступна новая маршрутизация текста и изображений. Нажмите «Обработать файлы» повторно; если файла нет в загрузке, загрузите его снова.')
st.subheader(document.name)
pages=[p for p in document.pipeline.document_pages if p.number is not None]
routes=getattr(document,'page_routes',())
needs_model=any(p.visual for p in routes)
st.caption((f'{len(pages)} стр. · ' if pages else '')+('источник подготовлен, визуальная модель недоступна' if needs_model and not vision_ready else 'обработка источника завершена'))
if needs_model and not vision_ready:st.warning(vision_status)
if document.pipeline.extraction_diagnostics:
    st.caption('Для источника доступны замечания об извлечении / OCR. Проверяйте распознанный текст по оригиналу.')
    with st.expander('Замечания к извлечению'):
        for diagnostic in document.pipeline.extraction_diagnostics:
            st.text(diagnostic)

mode='auto';page=None
if routes:
    mode_label=st.selectbox('Способ ответа',['Автоматически','По изображению страницы','QA по тексту'],help='Авто учитывает текстовый слой и крупные изображения. Для схем, таблиц и формул можно явно выбрать изображение. Отказ QA не переключает модель автоматически.')
    mode={'Автоматически':'auto','По изображению страницы':'visual','QA по тексту':'text'}[mode_label]
    if len(routes)>1:
        choice=st.selectbox('Страница PDF',['Найти по вопросу']+[str(p.number) for p in routes])
        page=None if choice=='Найти по вопросу' else int(choice)
    with st.expander('Обработка страниц'):
        for route in routes:st.caption(f'Стр. {route.number}: {route.reason}')
st.caption('02 / ВОПРОС К ВЫБРАННОМУ ИСТОЧНИКУ')
question_column, microphone_column=st.columns([3,2])
with microphone_column, st.expander('Микрофон · локальное распознавание', expanded=True):
    st.caption('Начните и остановите запись кнопкой микрофона · до 60 секунд. Текст можно исправить перед отправкой.')
    language=st.selectbox('Язык записи', ['Автоматически', 'Русский', 'English'])
    audio=st.audio_input('Записать вопрос', sample_rate=16000, key=f'audio_{st.session_state.upload_generation}')
    st.caption('Если микрофон недоступен: разрешите доступ в настройках браузера, проверьте устройство. Используйте localhost или HTTPS. Текстовый ввод доступен ниже.')
    if st.button('Распознать запись', disabled=audio is None):
        try:
            with st.spinner('Распознаём речь локально…'):
                text=speech_model().transcribe(audio.getvalue(), {'Русский':'ru','English':'en'}.get(language))
            st.session_state.question=text
            st.success('Текст вставлен. Проверьте его и нажмите «Найти ответ».')
        except SpeechError as error:
            st.error(str(error))
        except Exception:
            st.error('Распознавание недоступно. Проверьте установку модели и повторите попытку.')
with question_column, st.form('question_form',clear_on_submit=False):
    question=st.text_input('Ваш вопрос',placeholder='Например: В каком кабинете занятие?',key='question',max_chars=2000)
    submitted=st.form_submit_button('Найти ответ',type='primary')
if submitted:
    try:
        with st.status('Выбираем источник…',expanded=True) as work_status:
            session.ask(selected,question,mode=mode,page=page,progress=lambda message:work_status.update(label=message))
            document=session.documents[selected]
            work_status.update(label='Обработка ответа завершена',state='complete',expanded=False)
    except Exception as error:
        st.error(friendly_error(error))

if document.history:
    result=document.history[-1]
    st.divider()
    with st.container(border=False,key='answer_panel'):
        st.caption(result.get('path_label','QA по тексту'))
        st.text('Вопрос: '+result['question'])
        if result.get('model'):st.caption('Модель: '+result['model'])
        if result['answer']:
            st.text(result['answer'])
            st.caption('Сгенерированный ответ по изображению. Модель может ошибаться — проверьте оригинал.' if result.get('answer_method')=='vision' else 'Извлечённый кандидат. Модель может ошибаться — сверьте контекст ниже.')
        else:
            st.info('Надёжный ответ не найден в выбранном документе. Уточните вопрос или выберите другой файл.')
        if result.get('note') and not result['answer']:st.write(result['note'])
        if result['answer']:
            page_label=', '.join(map(str,result.get('pages',[])))
            st.text(f"Источник: {result['filename']}"+(f' · стр. {page_label}' if page_label else ' · номер страницы отсутствует'))
            if result.get('answer_method')!='vision':
                st.markdown('**Подтверждающий фрагмент**')
                st.text(result['supporting_paragraph'])
    if result.get('preview_pages') and getattr(document,'source_path',''):
        with st.expander('Оригинал изображения / страницы',expanded=True):
            for number in result['preview_pages']:
                try:
                    image=render_source(document.source_path,number)
                    st.image(image,caption=f'{document.name} · стр. {number}')
                    image.close()
                except Exception:
                    st.warning('Не удалось открыть предпросмотр. Скачайте исходный файл.')
            st.download_button('Скачать исходный файл',Path(document.source_path).read_bytes(),file_name=document.name)
    if len(document.history)>1:
        with st.expander(f'Предыдущие вопросы · {len(document.history)-1}'):
            for previous in reversed(document.history[:-1]):
                st.text('Вопрос: '+previous['question'])
                st.text('Ответ: '+(previous['answer'] or 'Надёжный ответ не найден.'))
                st.divider()
