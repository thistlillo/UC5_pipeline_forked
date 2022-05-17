from nltk import sent_tokenize, word_tokenize

class Vocabulary:
    PAD = 0
    OOV = 1
    BOS = 2
    EOS = 3

    def __init__(self):
        self.initialize()

    def initialize(self):
        self.idx2word = {Vocabulary.PAD: "<pad>", Vocabulary.OOV: "<oov>", Vocabulary.BOS: "<bos>", Vocabulary.EOS: "<eos>"}
        self.word2idx = {w:i for i,w in self.idx2word.items()}
        assert self.word2idx["<pad>"] == 0

        self.word2count = {}
        self.idx = len(self.idx2word)
        self.word_count = 0

    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.idx
            self.word2count[word] = 1
            self.idx2word[self.idx] = word
            self.idx += 1
        else:
            self.word2count[word] += 1
        self.word_count += 1
    #<

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            if word != ".":
                # commas and other punctuation already removed
                self.add_word(word)
    #<

    def add_text(self, text):
        for sentence in sent_tokenize(text):
            self.add_sentence(sentence)
    #<
    
    def keep_n_words(self, n_words: int):
        n = self.word_count
        print("(vocabulary) initial word count (total):", self.word_count)
        print("(vocabulary) initial number of words:", len(self.word2count))

        wc = list(self.word2count.items())
        wc = sorted(wc, key=lambda elem: -elem[1])
        # wc does not contain special tokens
        keep = wc[:n_words]
        rem = wc[n_words:]
        self.initialize()
        for w, _ in keep:
            self.add_word(w)

        print("(vocabulary) after iterating with add_word number of words:", len(self.word2count))
        assert len(self.word2idx) == n_words+4, f"words: {len(self.word2count)}, requested: {n_words}+4"  # number of special tokens
        # reset self.word2count
        self.word_count = 0
        for w, c in keep:
            self.word_count += c
            self.word2count[w] = c
        print("(vocabulary) final word_count (total): ", self.word_count)
        print("(vocabulary) final number of words:", len(self.word2count))

        #print("diff:", n - self.word_count)
        #print("removed words:", len(rem))
        #print("coverage:", self.word_count / n)

    def decode(self, idxs):
        return [self.idx2word[i] for i in idxs]

def collate_fn_single_sentence(text, max_tokens=12):
    pass
