import re

class SmplTokenizer():
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
        }

        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.reverse_vocab = {idx: token for token, idx in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        self.vocab_size = len(self.special_tokens)

    def preprocess(self, text: str) -> list[str]:
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.lower().split()

        return words

    def fit(self, data: list[dict]) -> None:
        for sample in data:
            query = self.preprocess(sample['query'])
            pos_passage = self.preprocess(sample['pos_passage'])
            neg_passage = self.preprocess(sample['neg_passage'])

            for word in query + pos_passage + neg_passage:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.reverse_vocab[self.vocab_size] = word
                    self.vocab_size += 1
    
    def tokenize(self, text: str) -> list[int]:
        words = self.preprocess(text)
        return [self.vocab.get(word, self.special_tokens['<UNK>']) for word in words]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.reverse_vocab[id] for id in ids]