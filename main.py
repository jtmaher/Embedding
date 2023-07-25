import numpy as np
from scipy.stats import ortho_group


class Schema:
    def __init__(self, labels, attributes):
        self.labels = labels
        self.attributes = attributes

        self.ind_to_token = dict(enumerate(labels))
        self.token_to_ind = dict([(y, x) for x, y in enumerate(labels)])

        self.ind_to_attr = dict(enumerate(attributes))
        self.attr_to_ind = dict([(y, x) for x, y in enumerate(attributes)])

    def __repr__(self):
        return f"Schema(labels={self.labels}, attributes={self.attributes})"


class Struct:
    def __init__(self, schema, label, attributes=None, is_strings=True):
        if attributes is None:
            attributes = {}

        self.label = label
        self.attributes = attributes
        self.is_strings = is_strings
        self.schema = schema

    def __repr__(self):
        res = f"({self.label}"
        if isinstance(self.attributes, dict):
            for k, v in self.attributes.items():
                res += f" {k}: {str(v)}"
        res += ")"
        return res

    def to_indexes(self):
        if not self.is_strings:
            return self

        res_lab = self.schema.token_to_ind[self.label]

        res_attr = dict((self.schema.attr_to_ind[k], v.to_indexes()) for k, v in self.attributes.items())
        return Struct(self.schema, res_lab, res_attr, is_strings=False)

    def to_strings(self):
        if self.is_strings:
            return self

        res_lab = self.schema.ind_to_token[self.label]

        res_attr = dict((self.schema.ind_to_attr[k], v.to_strings()) for k, v in self.attributes.items())
        return Struct(self.schema, res_lab, res_attr, is_strings=True)

    def to_graph(self):
        pass

    def __eq__(self, other):
        return (self.label == other.label) and (self.attributes == other.attributes)

    @classmethod
    def cast(cls, sc, x):
        if isinstance(x, Struct):
            return x
        if (isinstance(x, tuple) or isinstance(x, list)) \
                and len(x) == 2 \
                and isinstance(x[1], dict):
            attrs = x[1]
            for k, v in attrs.items():
                attrs[k] = Struct.cast(sc, v)
            return Struct(sc, x[0], x[1])
        if isinstance(x, str):
            return Struct(sc, x)

        raise Exception('This is not a struct.')


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

    def encode(self, struct):
        if not isinstance(struct, Struct):
            struct = Struct.cast(self.schema, struct)

        v = self.token_emb[struct.label].copy()
        for k, val in struct.attributes.items():
            v += self.attr_emb[k] @ self.encode(val)
        return v

    def decode(self, v, depth=0, max_depth=100):
        if depth > max_depth:
            raise Exception('Decode failed')

        if len(v) != self.dim:
            raise Exception('Vector has wrong shape.')

        # Decode label
        dots = (v.reshape(1, -1) * self.token_emb).sum(axis=1)

        if dots.max() < 0.5:
            return None

        label_ind = np.argmax(dots)
        attrs = dict()

        for a in range(len(self.schema.attributes)):
            A = self.attr_emb[a]
            x = self.decode(A.T @ v, depth + 1)
            if x is not None:
                attrs[a] = x
        return Struct(self.schema, label_ind, attrs, is_strings=False)
