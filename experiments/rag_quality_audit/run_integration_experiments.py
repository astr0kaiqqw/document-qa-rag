"""Ablations of offset extraction, window coverage and chunk size; no production edits."""
from run_audit import *
def main():
 data=json.loads((OUT/'dataset.json').read_text());retrieval=json.loads((OUT/'retrieval_results.json').read_text());qa=QAModel();rows=[]
 for r in data:
  for config in ['original','smaller_chunks']:
   hits=next(x['hits'] for x in retrieval if x['id']==r['id'] and x['config']==config)[:3]
   variants=[predict_variants(qa,r['question'],h['chunk']) for h in hits]
   # Window mode is exactly offset mode when context fits; verify token count with question.
   windows=[(predict_variants(qa,r['question'],h['chunk'],True)['window_offset'] if len(qa.tokenizer(r['question'],h['chunk'])['input_ids'])>512 else v['offset']) for h,v in zip(hits,variants)]
   options=({'offset_only':variants[0]['offset'],'window_only':windows[0], 'qa_score_ranking':max((v['offset'] for v in variants),key=lambda x:x['score'])} if config=='original' else {'combined_short_offset_window':windows[0]})
   for name,v in options.items():
    rows.append(r|{'condition':name,'prediction':v,'answer_pred':v['answer'],**metrics(v['answer'],r['answer'],' '.join(h['chunk'] for h in hits))})
  print('integration',r['id'],flush=True)
 save('integration_experiments.json',rows)
 save('integration_summary.json',{split:{name:summary([r for r in rows if r['condition']==name and r['answer'] and (split=='all' or r['split']==split)]) for name in ['offset_only','window_only','qa_score_ranking','combined_short_offset_window']} for split in ['dev','eval','all']})
if __name__=='__main__':main()
