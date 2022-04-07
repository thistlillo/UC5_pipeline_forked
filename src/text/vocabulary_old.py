# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from functools import reduce
from nltk.tokenize import sent_tokenize
import sys


class Vocabulary:
    VERBOSE_NONE = 0
    VERBOSE_INFO = 1
    VERBOSE_DEBUG = 2
    VERBOSE_TRACE = 3  # enabled only in source code and not via cli


    # static fields for using Vocabulary in transforming index sequences
    PAD = "_pad_"
    BOS = "_bos_"
    EOS = "_eos_"
    OOV = "_oov_"  # out of vocabular: not used, ignored in current implementation

    PAD_I = 0  # mask_zeros
    BOS_I = 1
    EOS_I = 2
    
    EXTRA_TOKENS = {PAD:PAD_I, BOS:BOS_I, EOS:EOS_I}
    EXTRA_TOKENS_I = EXTRA_TOKENS.values()
    
    def __init__(self, name="", verbose=False, debug=False):
        self.name = name

        self.verbose = Vocabulary.VERBOSE_NONE
        if debug:
            self.verbose = Vocabulary.VERBOSE_DEBUG
        elif verbose:
            self.verbose = Vocabulary.VERBOSE_INFO

        self.word2index = self.n_words = \
            self.index2word = self.word_count = None
        self.init_indexes()
        self.n_sents = 0
        self.longest_sent = 0
    #< init

    def init_indexes(self):
        self.word2index = {k:v for k,v in Vocabulary.EXTRA_TOKENS.items()}
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.n_words = len(self.word2index)
        self.word_count = {}
    #<

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
    #<

    def add_sentence(self, sentence):
        l = 0
        # [:-1] the last token is a dot, that separates sentences
        for word in sentence.split(" ")[:-1]:
            self.add_word(word)
            l += 1

        if l > self.longest_sent:
            self.longest_sent = l

        self.n_sents += 1
    #<

    def add_text(self, text):
        for s in sent_tokenize(text):
            if self.verbose > Vocabulary.VERBOSE_DEBUG:
                print(f"sentence: {s}")
            self.add_sentence(s)
    #<

    def __contains__(self, word):
        return word in self.word2index
    #<

    def get_word(self, index, default=None):
        if default:
            return self.index2word(index, default)
        else:
            return self.index2word[index]
    #<

    def get_index(self, word, default=None):
        if default:
            return self.word2index(word, default)
        else:
            return self.word2index[word]
    #<

    def print_stats(self, file=sys.stdout):
        print(f"|words| = {self.n_words} (check: {len(self.word2index)})", file=file)
        print(f"|total tokens| = {reduce(lambda x, y: x + y, [v for _, v in self.word_count.items()])}", file=file)
        print(f"|sentences| = {self.n_sents}", file=file)
        print(f"longest sentence: {self.longest_sent}", file=file)
        print(f"|special tokens| = {len(self.EXTRA_TOKENS)}", file=file)
        print(f"highest number of occurrencies: {max([c for _, c in self.word_count.items()])}", file=file)
        print(f"|words occurring x1| = {len(self.words_with_frequency(1))}", file=file)
        print(f"|words occurring x2| = {len(self.words_with_frequency(2))}", file=file)
        print(f"|words occurring x3| = {len(self.words_with_frequency(3))}", file=file)
        # print([w for w, c in self.word_count.items() if c > 1000])
    #<

    def words_with_frequency(self, freq):
        words = [w for w, c in self.word_count.items() if c == freq]
        return words
    #<

    def set_n_words(self, max_words=0, min_freq=0):
        n_original_words = self.n_words

        if max_words > 0:
            print(f"number of words set to {max_words}")
        if min_freq > 0:
            print(f"only words with a frequency >= {min_freq}")
    
        # keep the most frequent ones: order keys
        words = sorted(self.word_count.items(), key=lambda item: -int(item[1]))
        if self.verbose > Vocabulary.VERBOSE_DEBUG:
            for w, c in words:
                print(f"{w}: {c}")

        self.init_indexes()
        if max_words > 0:
            words = words[:max_words]

        if min_freq > 0:
            for w, c in words:
                if c > min_freq:
                    self.add_word(w)

        print(f"|vocabulary| = {self.n_words} (incl. special tokens), before it was {n_original_words} ")
    #< set_n_words

    # def encode_sentence(self, sentence, sentence_length=0):
    #     out = [self.word2index[Vocabulary.BOS]]
    #     counter = 1
    #     for word in sentence.split(" "):
    #         idx = self.word2index.get(word, None)
    #         if idx:
    #             out.append(idx)
    #             counter += 1
    #         if counter == sentence_length-1:
    #             break
    #     out.append(self.word2index["eos"])
    #     counter += 1
        
    #     if counter < sentence_length:
    #         for i in range(counter, sentence_length):
    #             out.append(self.word2index["pad"])

    #     return out
    # #< encode_sentence

    # def empty_padded_sentence(self, length):
    #     return [self.word2index["pad"] for _ in range(length + 2)]
    # #<

    # def simple_encode(self, text, max_tokens):
    #     counter = 0
    #     out = []
    #     for sent in sent_tokenize(text):
    #         for token in sent.split(" "):
    #             out.append(token)
    #             counter += 1
    #             if counter == (max_tokens - 2): # for bos, eos
    #                 break
    #         if counter == (max_tokens -2):
    #             break
    #     enc = self.encode_sentence(" ".join(out), sentence_length=max_tokens)
    #     return enc
    # #< simple_encode

    # def encode(self, text, n_sentences=0, sentence_length=0):
    #     """Encode a text with sentences delimited by dots using the indexes previously set.
    #     Args:
    #         text ([string]): text to encode
    #         n_sentences (int, optional): If greater than 0, the output encoding will have exactly n_sentences, possibly empty; if 0 the encoding will be as expected. Defaults to 0.
    #         sentence_length (int, optional): If greater than 0, the encoded sentence will be trimmed or padded to sentence_length words. If 0 sentences will contain the indexes of the input tokens and nothing else. Defaults to 0.

    #     Returns:
    #         [list of lists of int]: Each inner list corresponds to an encoded input sentence.
    #     """
    #     e_text = []
    #     counter = 0
    #     for sent in sent_tokenize(text):
    #         e_sent = self.encode_sentence(sent, sentence_length)
    #         if len(e_sent) > 2:  # bos and eos always included
    #             e_text.append(e_sent)
    #         counter += 1
    #         if counter == n_sentences:
    #             break

    #     # TODO: a little strange here if sentence_length == 0
    #     for _ in range(counter, n_sentences):
    #         e_text.append(self.empty_padded_sentence(sentence_length))
    #     return e_text
    # #< encode

    # used by text_generation_phi
    def decode_sentence(self, e_sent: str):
        dec = " ".join( [self.index2word[int(i)] for i in e_sent.split(" ") if (int(i) not in Vocabulary.EXTRA_TOKENS.values())] )
        return dec
    
    def decode(self, e_text):
        sents = []
        for e_sent in e_text:
            sents.append(self.decode_sentence(e_sent) )
        return ". ".join(sents) + "."
    #<
#< class