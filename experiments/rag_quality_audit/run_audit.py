"""Offline, production-preserving diagnostic. Run from repository root."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,re,string,math,hashlib,platform,inspect
from pathlib import Path
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)
from rag.qa_model import QAModel
from rag.embeddings import EmbeddingModel
from rag.pipeline import RAGPipeline
from rag.semantic_chunker import semantic_chunk_text
from rag.chunker import chunk_text
from rag.document_loader import load_document
from rag.vector_store import VectorStore
from rag.retriever import Retriever

def save(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def norm(s):
 s=s.lower().translate(str.maketrans('','',string.punctuation));return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def metrics(p,a,c):
 x,y=norm(p).split(),norm(a).split(); n=sum((Counter(x)&Counter(y)).values())
 return dict(em=int(norm(p)==norm(a)),f1=(2*n/(len(x)+len(y)) if x and y else float(x==y)),valid_span=bool(p) and p in c,empty=not bool(p))
def summary(rows):
 return {k:float(np.mean([r[k] for r in rows])) for k in ['em','f1','valid_span','empty']}|{'n':len(rows)} if rows else None

def predict_variants(qa,q,c,windows=False):
 kw=dict(return_tensors='pt',truncation='only_second',max_length=512,return_offsets_mapping=True)
 if windows:kw.update(return_overflowing_tokens=True,stride=128,padding=True)
 enc=qa.tokenizer(q,c,**kw); offsets=enc.pop('offset_mapping');enc.pop('overflow_to_sample_mapping',None)
 with torch.inference_mode():o=qa.model(**enc)
 best={}
 for w in range(len(enc['input_ids'])):
  ids=[i for i,s in enumerate(enc.sequence_ids(w)) if s==1]; st=o.start_logits[w].numpy();en=o.end_logits[w].numpy()
  top_s=set(sorted(ids,key=lambda i:st[i],reverse=True)[:5]);top_e=set(sorted(ids,key=lambda i:en[i],reverse=True)[:5])
  candidates=[]
  for s in ids:
   for e in range(s,min(s+90,ids[-1]+1)):
    a,b=offsets[w,s,0].item(),offsets[w,e,1].item();score=float(st[s])+float(en[e]);length=e-s+1
    candidates.append((s,e,a,b,score,length))
  for mode in (['window_offset'] if windows else ['offset','max90','length_alpha2','top5','top5_length_alpha2']):
   lim=90 if mode=='max90' else 30
   pool=[v for v in candidates if v[5]<=lim and (not mode.startswith('top5') or (v[0] in top_s and v[1] in top_e))]
   if not pool:continue
   v=max(pool,key=lambda v:v[4]+(2*math.log1p(v[5]) if 'alpha2' in mode else 0))
   s,e,a,b,score,length=v
   ranked_score=score+(2*math.log1p(length) if 'alpha2' in mode else 0)
   if mode not in best or ranked_score>best[mode]['ranking_score']:
    best[mode]=dict(answer=c[a:b],start=a,end=b,score=score,ranking_score=ranked_score,tokens=length,window=w,cls_score=float(st[0]+en[0]),start_logit=float(st[s]),end_logit=float(en[e]))
 return best

def main():
 data=json.loads((OUT/'dataset.json').read_text());qa=QAModel();tokenizer=qa.tokenizer
 protected=list((ROOT/'rag').glob('*.py'))+list((ROOT/'models').rglob('*'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'evaluation').glob('*'))+[ROOT/'requirements.txt',ROOT/'README.md',ROOT/'uploads/embedding.pdf']
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
 save('protected_hashes_before.json',hashes)
 env=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,tokenizer_size=len(tokenizer),model_vocab=qa.model.config.vocab_size,tokenizer_fast=tokenizer.is_fast,tokens=tokenizer.tokenize('What is the capital of Kazakhstan?'))
 save('environment.json',env)
 # First major model experiment: oracle contexts only, actual production entry point.
 oracle=[]
 for i,r in enumerate(data):
  p=qa.answer(r['question'],r['context']);variants=predict_variants(qa,r['question'],r['context'])
  variants.update(predict_variants(qa,r['question'],r['context'],True))
  variants['baseline']={'answer':p}
  for v in variants.values():v.update(metrics(v['answer'],r['answer'],r['context']))
  oracle.append(r|{'predictions':variants});print('oracle',i+1,r['id'],repr(p),flush=True)
 save('oracle_results.json',oracle)
 # Standard HF implementation, same object/checkpoint and no-answer disabled.
 try:
  from transformers import pipeline
  hf=pipeline('question-answering',model=qa.model,tokenizer=tokenizer,device=-1)
  for row in oracle:
   v=hf(question=row['question'],context=row['context'],max_seq_len=512,doc_stride=128,max_answer_len=30,handle_impossible_answer=False)
   row['predictions']['hf_standard']=v|metrics(v['answer'],row['answer'],row['context'])
  save('oracle_results.json',oracle)
 except Exception as e:save('hf_limitation.json',{'error':repr(e)})
 agg={}
 for split in ['dev','eval','all']:
  subset=[r for r in oracle if r['answer'] and (split=='all' or r['split']==split)]
  agg[split]={m:summary([r['predictions'][m] for r in subset if m in r['predictions']]) for m in oracle[0]['predictions']}
 agg['unanswerable']={m:summary([r['predictions'][m] for r in oracle if not r['answer'] and m in r['predictions']]) for m in oracle[0]['predictions']}
 save('oracle_summary.json',agg)
 if '--oracle-only' in sys.argv:return
 # Synthetic TXT corpus exercises document-to-answer without PDF ambiguity.
 contexts=list(dict.fromkeys(r['context'] for r in data if r['page'] is None))
 txt=OUT/'fixture_corpus.txt';txt.write_text('\n\n'.join(contexts))
 embed=EmbeddingModel();env.update(embedding_max_seq_length=embed.model.max_seq_length);save('environment.json',env)
 configs={'original':(180,36,True),'smaller_chunks':(48,12,True),'fixed_word_chunks':(180,36,False)}
 retrieval=[];endtoend=[];chunk_stats={}
 for name,(size,overlap,semantic) in configs.items():
  for corpus,path in [('synthetic',txt),('pdf',ROOT/'uploads/embedding.pdf')]:
   text=load_document(str(path));chunks=(semantic_chunk_text if semantic else chunk_text)(text,chunk_size=size,overlap=overlap)
   vec=embed.encode(chunks);store=VectorStore(vec.shape[1]);store.add(vec,chunks);retriever=Retriever(embed,store)
   lens=[len(embed.model.tokenizer(c)['input_ids']) for c in chunks];qlens=[len(tokenizer(c)['input_ids']) for c in chunks]
   chunk_stats[name+'_'+corpus]=dict(chunks=len(chunks),words=[len(c.split()) for c in chunks],embedding_tokens=lens,qa_context_tokens=qlens,embedding_overlimit=sum(l>embed.model.max_seq_length for l in lens),embedding_norms=np.linalg.norm(vec,axis=1).tolist())
   pipe=RAGPipeline(chunk_size=size,overlap=overlap,use_semantic_chunking=semantic);pipe.embedding_model=embed;pipe.qa_model=qa;pipe.load_document(str(path))
   for r in [r for r in data if (r['page'] is not None)==(corpus=='pdf')]:
    hits=retriever.retrieve(r['question'],top_k=5)
    # Strict support labels use full normalized source context, plus separate answer containment.
    support=[j for j,c in enumerate(chunks) if norm(r['answer']) and norm(r['answer']) in norm(c)] if r['answer'] else []
    rank=next((j+1 for j,h in enumerate(hits) if h['index'] in support),None)
    retrieval.append(dict(id=r['id'],config=name,corpus=corpus,question=r['question'],expected=r['answer'],answerable=bool(r['answer']),support_chunk_indices=support,hits=hits,recall1=int(rank is not None and rank<=1),recall3=int(rank is not None and rank<=3),recall5=int(rank is not None and rank<=5),rr=1/rank if rank else 0))
    # B uses the production selection rule on retrieved text. C executes actual pipeline.
    if name=='original':
     answers=[qa.answer(r['question'],h['chunk']) for h in hits[:3]]
     chosen=next((a for a in answers if a.strip()),'')
     actual=pipe.ask(r['question'])
     endtoend.append(r|{'condition':'B','answer_pred':chosen,**metrics(chosen,r['answer'],' '.join(h['chunk'] for h in hits[:3])),'retrieval_candidates':answers})
     endtoend.append(r|{'condition':'C','answer_pred':actual['answer'],**metrics(actual['answer'],r['answer'],actual['source']),'response':actual})
    else:
     actual=pipe.ask(r['question']);endtoend.append(r|{'condition':name,'answer_pred':actual['answer'],**metrics(actual['answer'],r['answer'],actual['source'])})
    print('retrieval',name,corpus,r['id'],'rank',rank,flush=True)
 save('retrieval_results.json',retrieval);save('end_to_end_results.json',endtoend);save('chunk_statistics.json',chunk_stats)
 totals={}
 for name in configs:
  totals[name]={}
  for split in ['dev','eval','all']:
   ids={r['id'] for r in data if r['answer'] and (split=='all' or r['split']==split)}
   selected=[r for r in retrieval if r['config']==name and r['id'] in ids]
   totals[name][split]={k:float(np.mean([r[k] for r in selected])) for k in ['recall1','recall3','recall5','rr']}
 save('retrieval_summary.json',totals)
 save('end_to_end_summary.json',{split:{condition:summary([r for r in endtoend if r['answer'] and r['condition']==condition and (split=='all' or r['split']==split)]) for condition in ['B','C','smaller_chunks','fixed_word_chunks']} for split in ['dev','eval','all']})
 # Training-label reproduction: execute only the function definition from cell 18.
 notebook=json.loads((ROOT/'models/notebooks/rag_qa_training.ipynb').read_text());ns={'tokenizer':tokenizer};exec(''.join(notebook['cells'][18]['source']),ns)
 labels=[]
 for r in data:
  if not r['answer']:continue
  enc=ns['preprocess_function']({'question':[r['question']],'context':[r['context']],'answers':[{'text':[r['answer']],'answer_start':[r['context'].index(r['answer'])]}]})
  s,e=enc['start_positions'][0],enc['end_positions'][0]
  decoded=tokenizer.decode(enc['input_ids'][0][s:e+1],skip_special_tokens=True)
  labels.append(dict(id=r['id'],start=s,end=e,decoded=decoded,expected=r['answer'],truncated_label=(s==e==0),**metrics(decoded,r['answer'],r['context'])))
 save('training_label_checks.json',labels)
 after={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in hashes};save('integrity.json',{'changed':[p for p in hashes if hashes[p]!=after[p]],'files_checked':len(hashes)})
 print('FINISHED',flush=True)
if __name__=='__main__':main()
