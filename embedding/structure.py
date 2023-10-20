class Struct:
    def __init__(self, schema, label, attributes=None, is_strings=True):
        if attributes is None:
            attributes = {}

        self.schema = schema
        self.label = label
        self.attributes = attributes
        self.is_strings = is_strings

    def __repr__(self):
        if isinstance(self.attributes, dict) and not self.attributes == {}:
            res = f'("{self.label}",'
            res += " {"
            for k, v in self.attributes.items():
                res += f' "{k}": {str(v)},'
            res += "} "
            res += ")"
        else:
            res = f'"{self.label}"'
        return res

    def to_array(self):
        if len(self.attributes) == 0:
            return self.label
        else:
            return (self.label,
                    dict((k, v.to_array()) for k, v in self.attributes.items()))

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

    def __eq__(self, other):
        if self.label is None:
            return False
        return (self.label == other.label) and (self.attributes == other.attributes)

    @classmethod
    def create(cls, sc, x):
        import copy
        x = copy.deepcopy(x)
        if isinstance(x, Struct):
            return x
        if (isinstance(x, tuple) or isinstance(x, list)) \
                and len(x) == 2 \
                and isinstance(x[1], dict):
            attrs = x[1]
            for k, v in attrs.items():
                if k not in sc.attr_to_ind:
                    raise Exception(f'Attribute {k} not in schema.')

                attrs[k] = Struct.create(sc, v)
            return Struct(sc, x[0], x[1])
        if isinstance(x, str):
            if x in sc.token_to_ind:
                return Struct(sc, x)
            else:
                raise Exception(f'Label {x} not in schema.')

        raise Exception('This is not a struct.')
