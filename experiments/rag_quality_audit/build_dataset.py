"""Build small authored QA fixtures and verbatim PDF-page evidence, without downloads."""
import json
from pathlib import Path
from pypdf import PdfReader
OUT=Path(__file__).resolve().parent
rows=[]
def add(q,c,a,kind='fact',split='eval',page=None):
    assert not a or a in c,(q,a)
    rows.append(dict(id=f'q{len(rows)+1:02}',question=q,context=c,answer=a,kind=kind,split=split,page=page))
add('What does Word2Vec learn?','Word2Vec is a neural network technique that learns word embeddings.','word embeddings','definition','dev')
add('What is the goal of learning word vectors?','The goal of learning word vectors is to adjust those word vectors to maximize the similarity of positive word pairs and minimize the similarity of negative word pairs.','to adjust those word vectors to maximize the similarity of positive word pairs and minimize the similarity of negative word pairs','long','dev')
add('What is the capital of France?','The capital of France is Paris. It is a major European city.','Paris',split='dev')
add('What does NLP stand for?','NLP stands for Natural Language Processing. It is a field of artificial intelligence.','Natural Language Processing','multiword','dev')
add('Who introduced the Transformer architecture?','The Transformer architecture was introduced by researchers at Google in 2017.','researchers at Google','multiword','dev')
add('Who created Python?','Python was created by Guido van Rossum. Its first release appeared in 1991.','Guido van Rossum','multiword')
add('When was the first release of Python?','Python was created by Guido van Rossum. Its first release appeared in 1991.','1991')
add('What is photosynthesis?','Photosynthesis is the process by which plants convert light energy into chemical energy.','the process by which plants convert light energy into chemical energy','definition')
add('Where is the observatory located?','The observatory is located on Mount Wilson in California. It houses several telescopes.','Mount Wilson in California','multiword')
add('What carries oxygen in the blood?','Hemoglobin carries oxygen in the blood. White blood cells help fight infections.','Hemoglobin','boundary_start')
add('What is the largest planet in the solar system?','Mercury is the smallest planet. The largest planet in the solar system is Jupiter.','Jupiter','boundary_end')
add('Как называется столица Казахстана?','Столица Казахстана называется Астана. Алматы является крупнейшим городом страны.','Астана','russian')
add('Қазақстанның астанасы қай қала?','Қазақстанның астанасы — Астана. Алматы — үлкен қала.','Астана','kazakh')
add('Which protocol provides reliable ordered delivery?','TCP provides reliable ordered delivery of bytes between applications. UDP sends independent datagrams.','TCP')
add('What does the backup policy require?','The backup policy requires copying all project documents to two independent encrypted drives every Friday and checking that the copied files can be restored before deleting any old backup versions.','copying all project documents to two independent encrypted drives every Friday and checking that the copied files can be restored before deleting any old backup versions','long')
add('What is the name of the experimental enzyme?','The experimental enzyme is named Zeta-Q7. Its activity is measured in the laboratory.','Zeta-Q7','punctuation')
add('What identifier was assigned to the sample?','The sample was assigned the identifier AB-123/X. The identifier appears on the label.','AB-123/X','punctuation')
add('What is stored in the archive?','The archive stores letters,  maps, and photographs. The catalog is searchable.','letters,  maps, and photographs','whitespace')
add('What is the access code?','The record contains routine weather measurements. '*140+'The access code is violet lantern.','violet lantern','beyond_512')
add('What material forms the outer shell?','The report lists routine equipment checks. '*60+'The outer shell is made of titanium alloy.','titanium alloy','near_limit')
add('Who invented this telescope?','This telescope has a mirror two meters wide. The observatory opened in 1980.','','unanswerable','dev')
add('What is the password?','The museum opens at nine and closes at five. Entry is free on Sundays.','','unanswerable')
pages=[(p.extract_text() or '').strip() for p in PdfReader('uploads/embedding.pdf').pages]
spec=[
 ('What is Word2vec?','Popular embedding method','Word2vec\nPopular','definition'),
 ('Which classifier is used to distinguish positive and negative examples?','logistic regression','Approach: predict','fact'),
 ('How is similarity computed?','dot product','Similarity is computed','fact'),
 ('Which function turns a dot product into a probability?','sigmoid','Turning dot products','fact'),
 ('How are the word weights learned?','Stochastic gradient descent','Learning the classifier','fact'),
 ('Which two sets of embeddings does SGNS learn?','Target embeddings matrix W\nContext embedding matrix C','Two sets of embeddings','multiword'),
]
for j,(q,a,needle,k) in enumerate(spec):
 p=next(i for i,t in enumerate(pages) if needle in t)
 add(q,pages[p],a,k,'dev' if j==0 else 'eval',p+1)
assert len(rows)==28
(OUT/'dataset.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(OUT/'pdf_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2))
print('Built',len(rows),'questions; answerable',sum(bool(r['answer']) for r in rows))
