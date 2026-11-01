"""Inspect score overlap; never install a rejection threshold from a tiny sample."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import json
from pathlib import Path
import torch
from rag.qa_model import QAModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);qa=QAModel()
    data=json.loads((Path(__file__).parent/'data/audit_questions.json').read_text())
    rows=[]
    for r in data:
        if r['split']!='dev':continue
        c=qa.candidates(r['question'],r['context'],top_n=1)[0]
        rows.append(dict(id=r['id'],answerable=bool(r['answer']),**c))
    # Unanswerable development probes: each supplied context demonstrably omits the fact.
    probes=[('Who founded the museum?','The museum opens at nine. Entry is free.'),
            ('What is the password?','The capital of France is Paris.'),
            ('When was the enzyme discovered?','The experimental enzyme is named Zeta-Q7.'),
            ('What is the server IP address?','TCP provides reliable ordered delivery.'),
            ('Кто основал город?','Столица Казахстана — Астана.'),
            ('Қазақстанның халқы қанша?','Қазақстанның астанасы — Астана.')]
    for i,(q,c) in enumerate(probes):
        rows.append(dict(id=f'noanswer_dev_{i}',answerable=False,question=q,context=c,
                         **qa.candidates(q,c,top_n=1)[0]))
    ranges={}
    for key in ['log_score','null_margin']:
        ranges[key]={str(answerable):[min(r[key] for r in rows if r['answerable']==answerable),
                                     max(r[key] for r in rows if r['answerable']==answerable)]
                     for answerable in [True,False]}
    output=dict(rows=rows,ranges=ranges,threshold_enabled=False,
        conclusion='Development probes only; log_score overlap and insufficient document-level calibration (null_margin separates this tiny dev sample only). No native no-answer support or calibrated threshold.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2));print(json.dumps(ranges,indent=2))

if __name__=='__main__':main()
