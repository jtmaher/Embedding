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
