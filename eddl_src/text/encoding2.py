# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from nltk.tokenize import word_tokenize
import pickle

from text.vocabulary import Vocabulary

# ***************************************************************
class BaseTextEncoder:
    def __init__(self, vocab=None, vocab_filename=None):
        if vocab is not None:
            self.vocab = vocab
        elif vocab_filename is not None:
            with open(vocab_filename, "rb") as fin:
                self.vocab = pickle.load(fin)
    #< init

    def encode(self, text, as_string=False, insert_dots=False):
        """
        Encode text using the vocabulary provided in the contructor.

        Params:
            text (string): text to encode.
            as_string (boolean): returns a string representation of the encoding. Default: False.
            insert_dots (boolean): separate sentences with dots in the output string encoding. Used only if as_string is True. Default: False.

        Returns: 
            - a list of list of int, where inner lists correspond to encoded sentences.
            - if as_string is True, a string encoding that can be saved to disk.
        """
        out = self.base_encoding_(text)
        if as_string:
            out = self.to_string(out, insert_dots)
        return out
    #< encode

    def base_encoding_(self, text):
        vocab = self.vocab
        out = []
        for sentence in text.split(".")[:-1]:  # last split is an empty string
            s = [Vocabulary.BOS_I]
            # s = []
            for word in word_tokenize(sentence):
                if word in vocab:
                    s.append(vocab.get_index(word))
                else:
                    print(f"{word} not in vocab")
                    s.append(Vocabulary.OOV_I)
                    pass
            s = s + [Vocabulary.EOS_I]
            out.append(s)
        return out
    #< base_encoding

    def to_string(self, enc_sentences, insert_dots=False):
        # values: list of list of word indexes (int)
        out = []
        for s in enc_sentences:  # values = [[ 1 1 2 3 ... 2], [1, 312, 213, ... 2]]
            s = " ".join([str(i) for i in s])
            out.append(s)
        
        if insert_dots:
            out = ". ".join(out)
            out = out + "."
        else:
            out = " ".join(out)
        return out
    #< to_string
#< class BaseTextEncoder

# ***************************************************************
class Collator:
    def __init__(self):
        pass
    #<        
    def split(self, enc_text):
        """
        Splits a text encoding (as ints: word indexes) in sentences
        """
        if "." in enc_text:
            # if there is a dot, it means it was encoded with insert_dots=True
            sentences = self.split_with_dots_(enc_text)
        else:
            sentences = self.split_flat_seq_(enc_text)
        return sentences
    #<

    def text2int_sents(self, sentences):
        """
        Input: sentences is a list of strings, each string element corresponds to a sentence encoding
        """
        out_sentences = []
        for sent in sentences:
            sent = sent.strip()
            out = [int(v.strip()) for v in sent.split(" ")]
            out_sentences.append(out)
        return out_sentences
    #<

    def split_with_dots_(self, text):
        sentences = text.split(".")[:-1]
        return self.text2int_sents(sentences)
    #<
         
    def split_flat_seq_of_int_(self, codes):
        """
        Used when the string representation of a sentence encoding does not contain dots separating sentences
        """
        eosi = int(Vocabulary.EOS_I)
        i = 0
        sentences = []
        while i < len(codes):
            sen = [codes[i]]   # corresponds to [Vocabulary.BOS_I]
            i += 1
            while codes[i] != eosi:
                assert codes[i] != eosi
                # print(f"i={i} / {len(codes)} : {codes[i]}  ({eosi})")
                sen.append(codes[i])
                i += 1
            sen.append(eosi)  # i not incremented here, where we consume eosi (line above)
            i += 1
            sentences.append(sen)
        return sentences
    #<

    def split_flat_seq_(self, text):
        codes = [int(t.strip()) for t in text.split(" ")]
        sentences = self.split_flat_seq_of_int_(codes)
        
        # for i, ss in enumerate(sentences):
        #     print(f"{i}: {ss}")
        return sentences
    #<
#< class Collator

# ***************************************************************
class SimpleCollator(Collator):
    """
    Parse a text encoding of a text and returns a single sentence (padded to a specified number of tokens)
    """
    def __init__(self):
        super(Collator, self).__init__()
    #<

    def parse_and_collate(self, e_text, n_tokens, n_sentences=1, pad=True):
        sentences = self.split(e_text)
        si = 0
        out = [Vocabulary.BOS_I]
        while si < len(sentences) and (len(out) < n_tokens -1):
            ii = sentences[si][1:-1]
            
            for idx in ii:
                out.append(idx)
                if len(out) == n_tokens -1:
                    break
            si+=1
        out.append(Vocabulary.EOS_I)
        
        if len(out) < n_tokens and pad:
            for _ in range(len(out), n_tokens):
                out.append(Vocabulary.PAD_I)
        
        if pad:
            assert len(out) == n_tokens
        else:
            assert out[-1] == Vocabulary.EOS_I and len(out) < n_tokens
       
        return out
    #<
#< class SimpleCollator

class SimpleCollator2(Collator):
    """
    Parse a text encoding of a text and returns a single sentence (padded to a specified number of tokens)
    """
    def __init__(self):
        super(Collator, self).__init__()
    #<

    def parse_and_collate(self, e_text, n_tokens, n_sentences=1, pad=True):
        sentences = self.split(e_text)
        si = 0
        out = [] # Vocabulary.BOS_I]
        while si < len(sentences) and (len(out) < n_tokens -1):
            ii = sentences[si][1:-1]
            for idx in ii:
                out.append(idx)
                if len(out) == n_tokens -1:
                    break
            si+=1
        out.append(Vocabulary.EOS_I)
        
        if len(out) < n_tokens and pad:
            for _ in range(len(out), n_tokens):
                out.append(Vocabulary.PAD_I)
        
        if pad:
            assert len(out) == n_tokens
        else:
            assert out[-1] == Vocabulary.EOS_I and len(out) < n_tokens
       
        return out



# ***************************************************************
class StandardCollator(Collator):
    """
    Parse a text encoding and returns a list of list of word indexes (one per sentence).
    Each decoded sentence can be padded to a specified length.
    The output list can be padded to a specified number of sentences, using empty sentences.
    """
    def __init__(self):
        super(Collator, self).__init__()
    #<
        
    def parse_and_collate(self, e_text, n_tokens, n_sentences, pad=True):
        sentences = self.split(e_text)
        
        si = 0
        out_sentences = []
        while si < len(sentences) and si < n_sentences:
            out = [Vocabulary.BOS_I]
            ii = sentences[si][1:-1]  # remove bos eos
            for idx in ii:
                out.append(idx)
                if len(out) == n_tokens -1:
                    break
            out.append(Vocabulary.EOS_I)
            if len(out) < n_tokens and pad:
                for _ in range(len(out), n_tokens):
                    out.append(Vocabulary.PAD_I)
            out_sentences.append(out)
            si +=1

        if len(out_sentences) < n_sentences and pad:
            for _ in range( len(out_sentences), n_sentences):
                padded = [Vocabulary.PAD_I for _ in range(n_tokens)]
                out_sentences.append(padded)

        if pad:
            assert len(out_sentences) == n_sentences
            for sentence in out_sentences:
                assert len(sentence) == n_tokens

        return out_sentences
    # parse_and_collate
#< class  StandardCollator

# ***************************************************************
def decode(e_sents, vocab=None, vocab_filename=None):
    if (vocab is None) and vocab_filename:
        with open(vocab_filename, "rb") as fin:
            vocab = pickle.load(fin)
    
    if vocab is None:
        print("FAILURE: provide a Vocabulary")
        exit(1)
    out = []

    def decode_sentence(e_sen):
        s = []
        for code in e_sen:
            code = int(code)
            if code not in Vocabulary.EXTRA_TOKENS_I:
                s.append(vocab.get_word(code))
        return " ".join(s)

    # list of indexes
    outer_list_ok = type(e_sents) is list and len(e_sents) > 0
    c1_ok = outer_list_ok and type(e_sents[0]) is int
    c2_ok = outer_list_ok and (type(e_sents[0]) is list) 
    if not (c1_ok or c2_ok):
        print("decode: WRONG INPUT PARAMETER.")
        print(f"decode:\t {type(e_sents)}, len = {len(e_sents)}")
        return ""

    # we assume that type(values)=list. check what we received
    if type(e_sents[0]) is int:
        return decode_sentence(e_sents)
    else:
        dec_s = []
        for s in e_sents:
            dec = decode_sentence(s)
            if len(dec) > 0:
                dec_s.append(dec)
        out = ". ".join(dec_s)
        out = out + "."
        return out
#< decode