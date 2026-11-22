"""Conservative grid OCR and evidence-only timetable time lookup.

Day labels are read from actual column headers, never assigned by column index.
Only explicitly printed time ranges are returned; no timetable is invented.
"""
import re
import unicodedata
import numpy as np

DAYS = {'пн':'Понедельник', 'вт':'Вторник', 'ср':'Среда',
        'чт':'Четверг', 'пт':'Пятница', 'сб':'Суббота', 'вс':'Воскресенье'}
DAYS.update(zip(('mon','tue','wed','thu','fri','sat','sun'), tuple(DAYS.values())))
DAY_STEMS = {'понедельник':'Понедельник', 'вторник':'Вторник', 'сред':'Среда',
             'четверг':'Четверг', 'пятниц':'Пятница', 'суббот':'Суббота', 'воскресен':'Воскресенье'}
TIME = re.compile(r'(?<!\d)([0-2]?\d:[0-5]\d)\s*[-–—]\s*([0-2]?\d:[0-5]\d)(?!\d)')
PREFIX = 'Расписание | '
DETAIL_QUESTION = re.compile(
    r'\b(?:в\s+како[йм]\s+(?:кабинете?|аудитории|аудиторию)|где\s+проходит|'
    r'кто\s+(?:(?:его|её|их|это)\s+)?вед[её]т|какой\s+преподаватель)'
    r'\b',
    re.I,
)


def is_time_question(question):
    return bool(re.search(r'во\s+сколько|когда|время|what\s+time|when', question, re.I))


def is_day_list_question(question):
    words = _words(question)
    return bool(_question_days(words) and re.search(
        r'(?:\bкакие\b.*\b(?:уроки|пары|занятия|предметы)\b)', question, re.I))


def is_timetable_question(question):
    return (is_time_question(question) or is_day_list_question(question)
            or bool(DETAIL_QUESTION.search(question) and _question_days(_words(question))))


def _lines(scores, threshold):
    positions = np.flatnonzero(scores > threshold)
    groups = np.split(positions, np.where(np.diff(positions) > 1)[0] + 1)
    return [int(np.mean(group)) for group in groups if len(group)]


def read_timetable(image, read):
    """Recognize only a regular, clearly ruled grid with readable weekday headers.

    `read` is a bounded OCR callback (image, page segmentation mode).
    No detected timetable means the normal OCR path is used unchanged.
    """
    gray = np.asarray(image.convert('L'))
    xs = _lines((gray < 225).mean(axis=0), .60)
    ys = _lines((gray < 225).mean(axis=1), .65)
    if not (3 <= len(xs) <= 9 and 4 <= len(ys) <= 30):
        return None
    steps = np.diff(ys)
    height = int(np.median(steps))
    if height < 15 or np.max(np.abs(steps-height)) > height*.2:
        return None
    headers=[]
    for left,right in zip(xs,xs[1:]):
        crop=image.crop((left+3,max(0,ys[0]-height),right-3,ys[0]-3))
        label=read(crop,7).strip().lower()
        headers.append(label)
    if sum(label in DAYS for label in headers) < 2:
        return None
    records=[]
    for (left,right),label in zip(zip(xs,xs[1:]),headers):
        day = DAYS.get(label)
        if day is None:
            continue  # Never infer an unreadable/missing weekday from its neighbours.
        for top,bottom in zip(ys,ys[1:]):
            if np.count_nonzero(gray[top+4:bottom-4,left+4:right-4] < 140) < 12:
                continue
            crop=image.crop((left+4,top+4,right-4,bottom-4))
            # Small cell text needs a dedicated segmentation pass; full-page OCR
            # often treats the coloured grid as a graphic and discards it.
            if crop.height < 100:
                crop=crop.resize((crop.width*2,crop.height*2))
            text=' '.join(read(crop,6).split())
            if text:
                records.append(f'{PREFIX}{day} | {text}')
    return '\n\n'.join(records) if records else None


def _words(text):
    text=unicodedata.normalize('NFKC',text).lower().replace('ё','е')
    return re.findall(r'[^\W\d_]+',text)


def _question_days(words):
    days = {day for stem, day in DAY_STEMS.items() if any(w.startswith(stem) for w in words)}
    days.update(DAYS[w] for w in words if w in DAYS)
    return days


def _stem(word):
    # Explicit inflections, not arbitrary prefix similarity (which would match
    # unrelated longer course names or misspellings).
    return re.sub(r'(ами|ями|ого|ему|ому|ыми|ими|ах|ях|ов|ев|ей|ий|ый|ая|яя|ое|ее|ые|ие|ой|ую|юю|ом|ем|ых|их|а|я|ы|и|у|ю|е|о|ь)$', '', word) if len(word)>4 else word


def time_answer(question, text):
    """Return matching source spans, or an empty list for unsupported/absent facts.

    Records are exact slices of the indexed OCR text, including every matching
    lesson. Deliberately no arbitrary fuzzy course match or inferred time.
    """
    list_day = is_day_list_question(question)
    details = DETAIL_QUESTION.search(question) and _question_days(_words(question))
    if not list_day and not details and not is_time_question(question):
        return []
    if details:
        question = DETAIL_QUESTION.sub(' ', question)
    question=re.sub(r'\bкб\b','кибербезопасности',question,flags=re.I)
    question=re.sub(r'\bбд\b','базы данных',question,flags=re.I)
    words=_words(question)
    days = _question_days(words)
    stop={'во','сколько','когда','время','у','меня','в','на','по','будет','будут','начинается',
          'начинаются','занятие','занятия','пара','пары','урок','уроки','the','what','time','when','is','my',
          'on','и','его','ее',
          'этот','эта','это'}
    subject=[w for w in words if w not in stop and w not in DAYS and
             not any(w.startswith(stem) for stem in DAY_STEMS)]
    if not list_day and not details and not subject:
        return []
    matches=[]
    for record in re.finditer(r'^Расписание \| ([^|\n]+) \| ([^\n]+)',text,re.M):
        day,body=record.group(1).strip(),record.group(2)
        if days and day not in days:
            continue
        time=TIME.search(body)
        if not time or any(int(t.split(':')[0])>23 for t in time.groups()):
            continue
        minutes=[int(t.split(':')[0])*60+int(t.split(':')[1]) for t in time.groups()]
        if minutes[0] >= minutes[1]:
            continue
        # Course words must all occur before the printed time, rather than
        # elsewhere on the page. Conservative suffix normalization tolerates Russian case endings.
        course_words=_words(body[:time.start()])
        if not list_day and not all(
            any(_stem(candidate) == _stem(word) for candidate in course_words)
            for word in subject
        ):
            continue
        matches.append(record)
    return [(record.start(), record.end()) for record in matches]


def format_records(records, question):
    """Combine contiguous slots with identical OCR text outside the time range."""
    groups = []
    for record in records:
        day, body = record.removeprefix(PREFIX).split(' | ', 1)
        match = TIME.search(body)
        label = day, body[:match.start()], body[match.end():]
        start, end = match.groups()
        if groups and groups[-1][0] == label and groups[-1][2] == start:
            groups[-1] = label, groups[-1][1], end
        else:
            groups.append((label, start, end))
    return '\n\n'.join(
        f'{day} · {before}{start}–{end}{after}'
        for (day, before, after), start, end in groups
    )
