class Struct:
    def __init__(self, schema, label, attributes=None, is_strings=True):
        if attributes is None:
            attributes = {}

        self.schema = schema
        self.label = label
        self.attributes = attributes
        self.is_strings = is_strings

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
    def create(cls, sc, x):
        if isinstance(x, Struct):
            return x
        if (isinstance(x, tuple) or isinstance(x, list)) \
                and len(x) == 2 \
                and isinstance(x[1], dict):
            attrs = x[1]
            for k, v in attrs.items():
                attrs[k] = Struct.create(sc, v)
            return Struct(sc, x[0], x[1])
        if isinstance(x, str):
            return Struct(sc, x)

        raise Exception('This is not a struct.')