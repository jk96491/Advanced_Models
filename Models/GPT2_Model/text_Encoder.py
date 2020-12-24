from Models.GPT2_Model.Utils import convert_byte_to_unicode
from Models.GPT2_Model.Utils import get_pairs
import regex
import json


class Encoder:
    def __init__(self, encoder, byte_pair_encoding_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = convert_byte_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.byte_pair_encoding_ranks = dict(zip(byte_pair_encoding_merges, range(len(byte_pair_encoding_merges))))
        self.cache = {}

        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def byte_pair_encoding(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.byte_pair_encoding_ranks.get(pair, float('inf')))
            if bigram not in self.byte_pair_encoding_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.byte_pair_encoding(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder():
    with open('encoder.json', 'r') as f:
        encoder = json.load(f)
    with open('vocab.bpe', 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

    return Encoder(encoder=encoder, byte_pair_encoding_merges=bpe_merges)




