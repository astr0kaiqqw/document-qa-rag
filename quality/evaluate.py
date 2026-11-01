"""Offline QA/retrieval/pipeline evaluation; baseline loads code from a Git revision.

python -m quality.evaluate --revision f69c57f --output reports/local_results/baseline.json
python -m quality.evaluate --output reports/local_results/final.json
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import subprocess
import sys

import torch
from rag.qa_model import QAModel, DEFAULT_MODEL_PATH
from rag.pipeline import RAGPipeline
from rag.embeddings import EmbeddingModel

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score(answer, reference, context):
    a,b = normalize(answer).split(),normalize(reference).split()
    common = sum((Counter(a) & Counter(b)).values())
    return {'em':int(a == b), 'f1':2*common/(len(a)+len(b)) if a and b else float(a==b),
            'invalid':bool(answer) and answer not in context, 'empty':not bool(answer),
            'failure':bool(reference) and not common}


def aggregate(rows):
    answerable = [r for r in rows if r['reference']]
    missing = [r for r in rows if not r['reference']]
    result = {'n':len(answerable),'no_answer_n':len(missing)}
    for k in ['em','f1','invalid','empty','failure']:
        result[k] = sum(r[k] for r in answerable)/len(answerable) if answerable else None
    for k in ['recall1','recall3','recall5','rr']:
        if answerable and k in answerable[0]:
            result[k] = sum(r[k] for r in answerable)/len(answerable)
    result['false_answer_rate'] = sum(not r['empty'] for r in missing)/len(missing) if missing else None
    result['false_refusal_rate'] = result['empty']
    return result


def historical_classes(revision):
    def load(path):
        source = subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=ROOT,text=True)
        ns = {'__file__':str(ROOT/path),'__name__':'quality_historical'}
        exec(compile(source, str(ROOT/path), 'exec'),ns)
        return ns
    qa = load('rag/qa_model.py')['QAModel']
    embeddings = load('rag/embeddings.py')['EmbeddingModel']
    pipe = load('rag/pipeline.py')
    pipe.update(QAModel=qa,EmbeddingModel=embeddings,
                load_document=load('rag/document_loader.py')['load_document'],
                semantic_chunk_text=load('rag/semantic_chunker.py')['semantic_chunk_text'],
                chunk_text=load('rag/chunker.py')['chunk_text'])
    return qa,embeddings,pipe['RAGPipeline']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision')
    parser.add_argument('--independent',action='store_true',help='Fixed document set not used for parameter selection')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--dataset',type=Path,default=ROOT/'quality/data/audit_questions.json')
    parser.add_argument('--split',choices=['all','dev','eval'],default='all')
    parser.add_argument('--oracle-only',action='store_true')
    parser.add_argument('--token-budget',type=int)
    parser.add_argument('--selection',default=None)
    parser.add_argument('--no-whole-words',action='store_true')
    parser.add_argument('--stride',type=int)
    args=parser.parse_args();torch.set_num_threads(4)
    data=json.loads(args.dataset.read_text());original=data
    data=[r for r in data if args.split=='all' or r['split']==args.split]
    qc,ec,pc=historical_classes(args.revision) if args.revision else (QAModel,EmbeddingModel,RAGPipeline)
    opts={}
    if args.no_whole_words:opts['whole_words']=False
    if args.stride is not None:opts['stride']=args.stride
    qa=qc(DEFAULT_MODEL_PATH,**opts);rows=[];budgets={}
    for r in data:
        answer=qa.answer(r['question'],r['context'])
        rows.append(dict(id=r['id'],condition='oracle',split=r['split'],corpus='pdf' if r['page'] else 'synthetic',language='ru' if r['id']=='q12' else 'kk' if r['id']=='q13' else r.get('language','en'),reference=r['answer'],prediction=answer,**score(answer,r['answer'],r['context'])))
        print('oracle',r['id'],repr(answer),flush=True)
    if not args.oracle_only:
        embed=ec()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        synthetic=args.output.parent/'fixture_corpus.txt'
        synthetic.write_text('\n\n'.join(dict.fromkeys(r['context'] for r in original if r['page'] is None)))
        for corpus,path in [('synthetic',synthetic),('pdf',ROOT/'uploads/embedding.pdf')]:
            selected=[r for r in data if (r['page'] is not None)==(corpus=='pdf')]
            if not selected:continue
            kw={}
            if args.token_budget is not None:kw['embedding_token_limit']=args.token_budget
            if args.selection is not None:kw['answer_selection']=args.selection
            pipe=pc(**kw);pipe.embedding_model=embed;pipe.qa_model=qa;pipe.load_document(str(path))
            chunks=pipe.vector_store.chunks
            lengths=[len(embed.model.tokenizer(c,add_special_tokens=True)['input_ids']) for c in chunks]
            budgets[corpus]={'chunks':len(chunks),'max_tokens':max(lengths),'over_limit':sum(n>embed.model.max_seq_length for n in lengths)}
            for r in selected:
                hits=pipe.retriever.retrieve(r['question'],top_k=5)
                rank=next((i+1 for i,h in enumerate(hits) if r['answer'] and normalize(r['answer']) in normalize(h['chunk'])),None)
                result=pipe.ask(r['question'])
                rows.append(dict(id=r['id'],condition='pipeline',split=r['split'],corpus=corpus,language='ru' if r['id']=='q12' else 'kk' if r['id']=='q13' else r.get('language','en'),reference=r['answer'],prediction=result['answer'],response=result,hits=hits,recall1=int(rank is not None and rank<=1),recall3=int(rank is not None and rank<=3),recall5=int(rank is not None and rank<=5),rr=1/rank if rank else 0,**score(result['answer'],r['answer'],result['source'])))
                print('pipeline',r['id'],repr(result['answer']),flush=True)
    groups={}
    for cond in ['oracle','pipeline']:
        for split in ['all','dev','eval']:
            for subset in ['all','pdf','synthetic','en','ru','kk']:
                selected=[r for r in rows if r['condition']==cond and (split=='all' or r['split']==split) and (subset=='all' or r['corpus']==subset or r['language']==subset)]
                if selected:groups[f'{cond}/{split}/{subset}']=aggregate(selected)
    result={'revision':args.revision or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'working_tree':not bool(args.revision),'python':sys.version,'torch':torch.__version__,'dataset':str(args.dataset),'previously_exposed':not args.independent,'exposure_note':'Independent means not used for parameter selection; every evaluated set becomes exposed after this run','options':{'token_budget':args.token_budget,'selection':args.selection,'whole_words':not args.no_whole_words,'stride':args.stride},'budgets':budgets,'summary':groups,'rows':rows}
    result['production_source_hashes'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'rag').glob('*.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(groups.get('pipeline/eval/all',groups.get('oracle/eval/all')),indent=2))

if __name__=='__main__':main()
