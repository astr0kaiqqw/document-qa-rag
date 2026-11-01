"""Run: streamlit run app.py. Runtime inference and documents stay on this machine."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import streamlit as st
from document_app.service import DocumentSession, LocalModels, friendly_error

st.set_page_config(page_title='Document QA · Локальный поиск ответов',page_icon='📖',layout='wide')
st.markdown('''<style>
.block-container{max-width:1100px;padding-top:2.3rem}
[data-testid="stSidebar"]{background:#f2f5f7}
h1{letter-spacing:-.035em}
[data-testid="stText"]{white-space:pre-wrap;font-size:1.05rem;line-height:1.65}
.stButton>button{border-radius:9px}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def shared_models():
    return LocalModels()


if 'documents' not in st.session_state:
    st.session_state.documents=DocumentSession(shared_models())
if 'upload_generation' not in st.session_state:
    st.session_state.upload_generation=0
session=st.session_state.documents

with st.sidebar:
    st.markdown('### 📖 Document QA')
    st.caption('Ваши документы. Ответы из источника.')
    uploads=st.file_uploader('Загрузить документы',type=['pdf','docx','txt'],accept_multiple_files=True,
        key=f'uploads_{st.session_state.upload_generation}',help='PDF, DOCX или UTF-8 TXT · до 25 МБ на файл')
    if st.button('Обработать файлы',type='primary',use_container_width=True,disabled=not uploads):
        for upload in uploads:
            try:
                with st.spinner(f'Обрабатываем {upload.name}…'):
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
        for key in ['selected_document','question']:
            st.session_state.pop(key,None)
        st.rerun()
    with st.expander('О приложении'):
        st.write('Документы и ответы обрабатываются локально. Ответ извлекается из текста; генеративные модели не используются.')
        st.caption('Модель: deepset/xlm-roberta-base-squad2. Поддержка EN / RU / KK зависит от текста и языка; проверяйте источник.')

st.caption('LOCAL DOCUMENT WORKSPACE')
st.title('Ответ находится в документе')
st.write('Загрузите материал, задайте вопрос и проверьте ответ в его контексте.')

if not selected:
    st.container(border=True).info('Начните с PDF, DOCX или TXT в панели слева. Затем нажмите «Обработать файлы».')
    st.stop()

document=session.documents[selected]
st.subheader(document.name)
pages=[p for p in document.pipeline.document_pages if p.number is not None]
st.caption(f'{len(pages)} стр. · готов к вопросам' if pages else 'Текстовый документ · готов к вопросам')
if document.pipeline.extraction_diagnostics:
    st.warning('Часть текста могла извлечься некорректно. Проверяйте формулы и необычные символы по оригиналу.')
    with st.expander('Замечания к извлечению'):
        for diagnostic in document.pipeline.extraction_diagnostics:
            st.text(diagnostic)

with st.form('question_form',clear_on_submit=False):
    question=st.text_input('Ваш вопрос',placeholder='Например: What is Word2vec?',key='question',max_chars=2000)
    submitted=st.form_submit_button('Найти ответ',type='primary')
if submitted:
    try:
        with st.spinner('Ищем фрагмент и извлекаем ответ…'):
            session.ask(selected,question)
    except Exception as error:
        st.error(friendly_error(error))

if document.history:
    result=document.history[-1]
    st.divider()
    with st.container(border=True):
        st.caption('ОТВЕТ И ИСТОЧНИК')
        if result['answer']:
            st.text(result['answer'])
            st.caption('Извлечённый кандидат. Модель может ошибаться — сверьте контекст ниже.')
        else:
            st.info('Надёжный ответ не найден в выбранном документе. Уточните вопрос или выберите другой файл.')
        if result['answer']:
            page_label=', '.join(map(str,result.get('pages',[])))
            st.text(f"Источник: {result['filename']}"+(f' · стр. {page_label}' if page_label else ' · номер страницы отсутствует'))
            st.markdown('**Подтверждающий фрагмент**')
            st.text(result['supporting_paragraph'])
    if len(document.history)>1:
        with st.expander(f'Предыдущие вопросы · {len(document.history)-1}'):
            for previous in reversed(document.history[:-1]):
                st.text('Вопрос: '+previous['question'])
                st.text('Ответ: '+(previous['answer'] or 'Надёжный ответ не найден.'))
                st.divider()
