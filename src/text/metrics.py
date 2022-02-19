# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from text.vocabulary import Vocabulary

def compute_bleu(predictions, text):
    def build_str(seqs_of_tokens):
        text = []
        for j in range(seqs_of_tokens.shape[0]):
            for k in range(seqs_of_tokens.shape[1]):
                token = seqs_of_tokens[j,k]  # sentence j, token k
                text.append(str(token))
                if token == Vocabulary.EOS_I:
                    break
        return text

    bleu = 0
    for i in range(predictions.shape[0]):
        pred = build_str(predictions[i, :, :].squeeze())
        txt = build_str(text[i, :, :].squeeze())
        bleu += sentence_bleu(pred, txt)
    return bleu / predictions.shape[0]
#< 
