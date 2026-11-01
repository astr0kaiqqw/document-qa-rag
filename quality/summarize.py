"""Create a compact reviewable snapshot; keep full context/prediction runs local."""
import argparse
import hashlib
import json
from pathlib import Path
from .evaluate import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'reports/local_results')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/quality_metrics.json')
    args=parser.parse_args()
    names={'baseline':'baseline_comparable','final':'final_verified',
           'independent_baseline':'independent_baseline','independent_final':'independent_final_bounded'}
    runs={key:json.loads((args.results/(name+'.json')).read_text()) for key,name in names.items()}
    comparisons={}
    for name,before,after in [('exposed_audit','baseline','final'),('new_documents','independent_baseline','independent_final')]:
        lookup={(r['condition'],r['id']):r for r in runs[before]['rows']}
        comparisons[name]=[dict(id=r['id'],condition=r['condition'],split=r['split'],language=r['language'],
                               reference=r['reference'],before=lookup[r['condition'],r['id']]['prediction'],
                               after=r['prediction'],f1_before=lookup[r['condition'],r['id']]['f1'],f1_after=r['f1'])
                           for r in runs[after]['rows']]
    output={'method':'SQuAD-style normalized token F1/EM; recall=gold span containment; MRR@5',
            'no_answer_threshold_enabled':False,
            'default_config':dict(embedding_token_limit=96,token_overlap=24,top_k=3,answer_selection='retrieval',max_answer_tokens=30),
            'runs':{key:{k:run[k] for k in ['revision','working_tree','python','torch','previously_exposed','budgets','summary']}
                    for key,run in runs.items()},'comparisons':comparisons,
            'model_and_tokenizer_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'models/final_qa_model').iterdir() if p.is_file()},
            'final_source_sha256':runs['final'].get('production_source_hashes',{})}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(f'Wrote compact snapshot: {args.output}')

if __name__=='__main__':main()
