import numpy as np
from utils.vocabulary import Vocabulary

def collate_fn_one_s(enc_text, n_sents=1, max_tokens=12, verbose=False):
    assert type(enc_text) is list, f"expected type 'list', received {type(enc_text)}"
    if verbose: 
        print(f"collating_one_s, len {len(enc_text)}:", enc_text)
    
    if type(enc_text[0]) is list:
        enc_text = enc_text[0]
        
    bos = Vocabulary.BOS
    eos = Vocabulary.EOS
    pad = Vocabulary.PAD
    
    enc_text = [bos] + enc_text + [eos]
    l = len(enc_text)
    if verbose: print(f"len with bos and eos: {l}, max is {max_tokens}")
    if l > max_tokens:
        if verbose: print("truncating")
        enc_text[max_tokens-1] = eos
        enc_text = enc_text[:max_tokens]
    elif l < max_tokens:
        if verbose: print("padding", flush=True)
        enc_text += ([pad] * (max_tokens -l) )
    
    if verbose: print(f"returning collation ({len(enc_text)}):", enc_text)
    assert len(enc_text) == max_tokens
    return np.array(enc_text)

    
def collate_fn_n_sents(enc_text, n_sents, max_tokens, verbose=False):
    res = []
    for i, enc_sent in enumerate(enc_text):
        if i == n_sents:
            break
        v = collate_fn_one_s(enc_sent, n_sents=1, max_tokens=max_tokens, verbose=verbose)
        res.append(v)

    if len(res) > n_sents:
        res = res[:n_sents]
    elif len(res) < n_sents:
        padded = [Vocabulary.PAD] * max_tokens
        for i in range(n_sents - len(res)):
            res.append(np.array(padded))

    res = np.array(res)
    if verbose: print(f"collate_fn_n_sents, returning {res.shape}")
    return res



