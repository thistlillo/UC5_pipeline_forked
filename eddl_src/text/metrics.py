# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from text.vocabulary import Vocabulary

# _eddl: predictions and text from, respectively, edll text generation and edll dataset

def compute_bleu_edll(predictions, text):
    def build_str(seqs_of_tokens):
        text = []
        for wi in seqs_of_tokens:
            text.append(str(wi))
            if wi == Vocabulary.EOS_I:
                break
        return text
    #<

    bleu = 0
    #> 
    if type(predictions) is str and type(text) is str:
        txt = build_str(text.split(" "))
        pred = build_str(predictions.split(" "))
        return sentence_bleu([txt], pred)
    #<

    for i in range(predictions.shape[0]):
        pred = build_str(predictions[i, :])
        txt = build_str(text[i, :])
        bleu += sentence_bleu([txt], pred)
    return bleu / predictions.shape[0]
#< compute_bleu_edll
