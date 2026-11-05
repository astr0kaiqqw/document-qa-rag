"""One-time explicit download of local extractive QA and embedding assets.

Never called by the running app; original models/final_qa_model is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT=Path(__file__).resolve().parents[1]
QA_REPO='deepset/xlm-roberta-base-squad2'
QA_REVISION='a5fab9908c8d856e8c583fd41ba6d92444e46477'
EMBED_REPO='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
PATTERNS=['*.json','model.safetensors','sentencepiece.bpe.model','README.md','1_Pooling/*.json','2_Normalize/*.json','vocab.txt']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-only',action='store_true')
    args=parser.parse_args()
    repos=[(QA_REPO,'extractive_qa')]
    if not args.qa_only:repos.append((EMBED_REPO,'embedding_model'))
    for repo,directory in repos:
        target=ROOT/'models'/directory
        info=HfApi().model_info(repo,revision=QA_REVISION if repo==QA_REPO else None)
        print(f'Downloading {repo}@{info.sha} to {target}',flush=True)
        snapshot_download(repo_id=repo,revision=info.sha,local_dir=str(target),
                          cache_dir=str(ROOT/'models/.download_cache'),allow_patterns=PATTERNS,
                          max_workers=2)
        manifest={'repo_id':repo,'revision':info.sha,'files':{str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name!='local_manifest.json'}}
        (target/'local_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'Ready: {repo}',flush=True)

if __name__=='__main__':main()
