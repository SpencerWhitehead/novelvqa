import os
import copy
import pickle
import numpy as np

from collections import namedtuple


QTokenEmbed = namedtuple('QToken', ['text', 'vector'])


class StoredEmbeds(object):
    def __init__(self, embed_fname='./ckpts/glove_embeds.pkl'):
        self.embed_fname = embed_fname
        self._embeddings = []
        self._token_to_ix = {}
        if os.path.exists(self.embed_fname):
            print('Found embedding file: {}\n\tLoading...'.format(self.embed_fname))
            self._token_to_ix, self._embeddings = self.load()

    def get_embeds(self):
        return copy.deepcopy(self._token_to_ix), np.array(self._embeddings)

    def set_embeds(self, token2idx, embed_mtx):
        self._token_to_ix = token2idx
        self._embeddings = embed_mtx

    def has_embeds(self):
        return len(self._token_to_ix) and len(self._embeddings)

    def load(self):
        with open(self.embed_fname, 'rb+') as embedf:
            data_ = pickle.load(embedf)
        return data_

    def save(self):
        # Embeddings will not be overwritten if file already exists.
        if not os.path.exists(self.embed_fname):
            print('Embedding file does not exist. Saving to: {}'.format(self.embed_fname))
            with open(self.embed_fname, 'wb+') as outf:
                pickle.dump((self._token_to_ix, self._embeddings), outf, protocol=-1)
        else:
            print('Embedding file already exists... New embeddings are not saved.')

    def __call__(self, word):
        return QTokenEmbed(text=word, vector=self._embeddings[self._token_to_ix[word]])
