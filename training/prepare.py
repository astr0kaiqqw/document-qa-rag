"""Prepare and validate local JSON QA data. Never trains or overwrites outputs.

Input: list of {id, question, context, answers: {text: [...], answer_start: [...]}}.
"""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from .preprocessing import prepare_qa_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--stride',type=int,default=128)
    args=p.parse_args()
    rows=json.loads(args.data.read_text())
    if not rows:raise ValueError('No examples')
    examples={k:[r[k] for r in rows] for k in ['id','question','context','answers']}
    tok=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    features,provenance=prepare_qa_features(examples,tok,max_length=args.max_length,stride=args.stride)
    output=dict(features=features,provenance=provenance,max_length=args.max_length,stride=args.stride,
                source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                tokenizer_sha256=hashlib.sha256((args.tokenizer/'tokenizer.json').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(f'Validated {len(rows)} examples, prepared {len(provenance)} windows. Training not started.')

if __name__=='__main__':main()
