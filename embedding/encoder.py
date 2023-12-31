import numpy as np
from scipy.stats import ortho_group

from embedding.structure import Struct


class Encoder:
    def __init__(self, schema, dim, seed=123):
        self.schema = schema
        self.dim = dim
        np.random.seed(seed)
        self.token_emb = np.random.normal(size=(len(schema.labels), dim))
        self.token_emb /= np.linalg.norm(self.token_emb, axis=1, keepdims=True)

        self.attr_emb = np.zeros((len(schema.attributes), dim, dim))
        for i in range(len(schema.attributes)):
            self.attr_emb[i] = ortho_group.rvs(dim)

    def encode(self, struct, depth=0):
        if not isinstance(struct, Struct):
            struct = Struct.create(self.schema, struct)

        if struct.is_strings:
            struct = struct.to_indexes()

        v = np.zeros(self.dim)
        for k, val in struct.attributes.items():
            v += self.attr_emb[k] @ self.encode(val, depth=depth + 1)

        v += self.token_emb[struct.label]

        return v

    def decode(self, v, depth=0, max_depth=100, to_strings=False):
        try:
            if to_strings:
                return self.decode_impl(v, depth, max_depth).to_strings()
            else:
                return self.decode_impl(v, depth, max_depth)
        except ValueError:
            return None

    def decode_impl(self, v, depth, max_depth):
        if depth > max_depth:
            raise ValueError('Decode failed')

        if len(v) != self.dim:
            raise ValueError('Vector has wrong shape.')

        # Decode label
        dots = (v.reshape(1, -1) * self.token_emb).sum(axis=1)

        if dots.max() < 0.5:
            return None

        label_ind = np.argmax(dots)
        attrs = dict()

        for a in range(len(self.schema.attributes)):
            A = self.attr_emb[a]
            x = self.decode_impl(A.T @ v, depth + 1, max_depth)
            if x is not None:
                attrs[a] = x

        return Struct(self.schema, label_ind, attrs, is_strings=False)
