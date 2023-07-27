import numpy as np
from scipy.stats import ortho_group

from embedding.struct import Struct


class Encoder:
    def __init__(self, schema, dim, seed=123, multiplicity=2):
        self.schema = schema
        self.dim = dim
        np.random.seed(seed)
        self.multiplicity = multiplicity
        self.token_emb = np.random.normal(size=(len(schema.labels)*self.multiplicity, dim))
        # self.token_emb = np.random.choice([-np.sqrt(dim), np.sqrt(dim)], size=(len(schema.labels), dim), replace=True)
        self.token_emb /= np.linalg.norm(self.token_emb, axis=1, keepdims=True)

        self.attr_emb = np.zeros((len(schema.attributes), dim, dim))
        for i in range(len(schema.attributes)):
            self.attr_emb[i] = ortho_group.rvs(dim)

    def encode(self, struct, depth=0, select=None, it=0):
        if it > 250:
            raise('Encode failed')

        if not isinstance(struct, Struct):
            struct = Struct.create(self.schema, struct)

        if select is None:
            np.random.seed()
            select = np.random.choice(range(self.multiplicity), size=len(self.schema.labels), replace=True)

        if struct.is_strings:
            struct = struct.to_indexes()

        v = self.token_emb[self.multiplicity*struct.label + select[struct.label]].copy()
        for k, val in struct.attributes.items():
            v += self.attr_emb[k] @ self.encode(val, depth=depth + 1, select=select, it=it)

        if depth == 0:
            try:
                dec = self.decode(v)
            except Exception as e:
                return self.encode(struct, it=it + 1)
            if dec is None or dec != struct:
                return self.encode(struct, it=it + 1)

            print(f' {it}')
        return v

    def decode(self, v, depth=0, max_depth=100):
        if depth > max_depth:
            raise Exception('Decode failed')

        if len(v) != self.dim:
            raise Exception('Vector has wrong shape.')

        # Decode label
        dots = (v.reshape(1, -1) * self.token_emb).sum(axis=1)

        if dots.max() < 0.65:
            return None

        label_ind = np.argmax(dots) // self.multiplicity
        attrs = dict()

        for a in range(len(self.schema.attributes)):
            A = self.attr_emb[a]
            x = self.decode(A.T @ v, depth + 1)
            if x is not None:
                attrs[a] = x

        return Struct(self.schema, label_ind, attrs, is_strings=False)
