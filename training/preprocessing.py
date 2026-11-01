"""Offset-aligned overlapping QA features; no model training or downloading."""


def prepare_qa_features(examples, tokenizer, *, max_length=384, stride=128):
    """Return model features and a separate provenance map for inspection/evaluation.

    Supports multiple gold spans and empty gold lists. Every original annotation
    is validated before tokenization. CLS labels are window-specific negatives,
    not evidence of a calibrated SQuAD-v2 no-answer model.
    """
    if not tokenizer.is_fast or max_length < 4 or stride < 0:
        raise ValueError('Invalid tokenizer or token budget')
    if 'id' not in examples or len(set(examples['id'])) != len(examples['id']):
        raise ValueError('Examples require unique stable ids')
    n=len(examples['id'])
    if any(len(examples[k])!=n for k in ['question','context','answers']):
        raise ValueError('Example column lengths differ')
    questions=[q.strip() for q in examples['question']]
    if any(not q for q in questions) or any(not c.strip() for c in examples['context']):
        raise ValueError('Questions and contexts must not be empty')
    for c,a in zip(examples['context'],examples['answers']):
        if len(a['text'])!=len(a['answer_start']):
            raise ValueError('Answer texts/starts length mismatch')
        for text,start in zip(a['text'],a['answer_start']):
            if not text or start<0 or c[start:start+len(text)]!=text:
                raise ValueError('Gold answer is not the annotated source span')
    capacity=min(max_length-len(tokenizer(q,add_special_tokens=False)['input_ids'])-
                 tokenizer.num_special_tokens_to_add(pair=True) for q in questions) if questions else 0
    if capacity<=0 or stride>=capacity:
        raise ValueError('Question leaves insufficient context budget for requested stride')
    encoded=tokenizer(questions,examples['context'],max_length=max_length,
                      truncation='only_second',stride=stride,return_overflowing_tokens=True,
                      return_offsets_mapping=True,padding='max_length')
    mappings=encoded.pop('overflow_to_sample_mapping');offsets=encoded.pop('offset_mapping')
    starts=[];ends=[];provenance=[]
    for window,(sample,span_offsets) in enumerate(zip(mappings,offsets)):
        ids=encoded['input_ids'][window];cls=ids.index(tokenizer.cls_token_id)
        context_tokens=[i for i,s in enumerate(encoded.sequence_ids(window)) if s==1]
        start=end=cls;selected=None
        answers=examples['answers'][sample]
        for answer,a in zip(answers['text'],answers['answer_start']):
            b=a+len(answer)
            if context_tokens and span_offsets[context_tokens[0]][0]<=a and span_offsets[context_tokens[-1]][1]>=b:
                start=next(i for i in context_tokens if span_offsets[i][1]>a)
                end=next(i for i in reversed(context_tokens) if span_offsets[i][0]<b)
                selected=answer
                break
        starts.append(start);ends.append(end)
        provenance.append(dict(example_id=examples['id'][sample],sample=sample,
            context_offsets=[offset if i in context_tokens else None for i,offset in enumerate(span_offsets)],
            gold_text=selected,cls_label=(start==end==cls)))
    features=dict(encoded,start_positions=starts,end_positions=ends)
    return features,provenance
