"""Reproduce notebook mismatch, embedding blind spots and FAISS cosine math."""
from run_audit import *
def main():
 nb=json.loads((ROOT/'models/notebooks/rag_qa_training.ipynb').read_text())
 tok=__import__('transformers').AutoTokenizer.from_pretrained(str(ROOT/'models/final_qa_model'),local_files_only=True)
 # Recover the printed, full first SQuAD example from notebook output, not a network dataset.
 out=''.join(''.join(o.get('text',[])) for o in nb['cells'][4]['outputs'])
 c=out.split('CONTEXT:\n',1)[1].split('\nANSWER:',1)[0].strip();q='To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?';a='Saint Bernadette Soubirous'
 ns={'tokenizer':tok};exec(''.join(nb['cells'][18]['source']),ns)
 enc=ns['preprocess_function']({'question':[q],'context':[c],'answers':[{'text':[a],'answer_start':[c.index(a)]}]})
 s,e=enc['start_positions'][0],enc['end_positions'][0]
 evidence={'notebook_alignment':{'current_cell18_start':s,'current_cell18_end':e,'current_label':tok.decode(enc['input_ids'][0][s:e+1]),'stored_cell19_first_start':136,'stored_cell19_first_end':141,'cell17_start_minus_one':tok.decode(enc['input_ids'][0][s-1:e+1]),'interpretation':'Cell 18 current source and stored cell19 output disagree; actual training labels cannot be established from notebook.'}}
 embed=EmbeddingModel();retr=json.loads((OUT/'retrieval_results.json').read_text());blind=[]
 for corpus in ['synthetic','pdf']:
  text=load_document(str(OUT/'fixture_corpus.txt' if corpus=='synthetic' else ROOT/'uploads/embedding.pdf'))
  chunks=semantic_chunk_text(text,180,36);vectors=embed.encode(chunks);store=VectorStore(vectors.shape[1]);store.add(vectors,chunks)
  testq='What is Word2vec?';v=embed.encode([testq])[0];hits=store.search(v,5)
  expected=sorted([float(np.dot(v,x)) for x in vectors],reverse=True)[:5]
  evidence['cosine_'+corpus]={'max_abs_error':max(abs(h['score']-s) for h,s in zip(hits,expected)),'query_norm':float(np.linalg.norm(v))}
  for r in json.loads((OUT/'dataset.json').read_text()):
   if not r['answer'] or ((r['page'] is not None)!=(corpus=='pdf')):continue
   for i,chunk in enumerate(chunks):
    if r['answer'] not in chunk:continue
    encoding=embed.model.tokenizer(chunk,truncation=True,max_length=embed.model.max_seq_length,return_offsets_mapping=True)
    end=max(b for a,b in encoding['offset_mapping']);pos=chunk.index(r['answer'])
    blind.append(dict(id=r['id'],corpus=corpus,chunk_index=i,answer_start=pos,answer_end=pos+len(r['answer']),last_visible_char=end,answer_fully_visible=pos+len(r['answer'])<=end))
 evidence['embedding_answer_visibility']=blind
 evidence['kazakh_tokens']=tok.convert_ids_to_tokens(tok('Қазақстанның астанасы — Астана. Алматы — үлкен қала.')['input_ids'])
 evidence['renderer_available']=False
 save('additional_evidence.json',evidence)
 print(json.dumps(evidence['notebook_alignment'],ensure_ascii=False));print('FAISS',evidence['cosine_pdf']);print('Answer spans beyond embedding prefix',sum(not r['answer_fully_visible'] for r in blind),'/',len(blind))
if __name__=='__main__':main()
