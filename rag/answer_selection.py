"""Inspectable ranking of evidence candidates; scores are not correctness probabilities."""
import math

STRATEGIES = ('retrieval', 'qa', 'rrf')


def rank_candidates(candidates, strategy='retrieval'):
    """Rank admissible source spans, deduplicating the same document coordinates.

    `qa` compares conditional start/end log-softmax concentration. RRF combines
    QA and retrieval ranks equally. Neither method is calibrated for correctness;
    selection is a heuristic evaluated on development data, without length bonus.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown answer selection strategy: {strategy}')
    unique = {}
    for candidate in candidates:
        c = dict(candidate)
        source = c['source']
        a,b = c['start'],c['end']
        if not (0 <= a < b <= len(source)) or source[a:b] != c['answer']:
            continue
        if not c['answer'].strip():
            continue
        value = c.get('log_score')
        if value is not None and not math.isfinite(value):
            continue
        key = (c['document_start'],c['document_end']) if c.get('document_start') is not None else (c['chunk_index'],a,b)
        if key not in unique or c['retrieval_rank'] < unique[key]['retrieval_rank']:
            unique[key] = c
    rows = list(unique.values())
    quality = sorted(rows,key=lambda c: (-(c['log_score'] if c.get('log_score') is not None else -float('inf')),c['retrieval_rank']))
    qa_ranks = {id(c):i+1 for i,c in enumerate(quality)}
    for c in rows:
        c['qa_rank'] = qa_ranks[id(c)]
        c['selection_score'] = (c['retrieval_score'] if strategy=='retrieval' else
                                c.get('log_score',-float('inf')) if strategy=='qa' and c.get('log_score') is not None else
                                c['retrieval_score'] if strategy=='qa' else
                                1/c['retrieval_rank'] + 1/c['qa_rank'])
    return sorted(rows,key=lambda c: (-c['selection_score'],c['retrieval_rank'],c['start']))
