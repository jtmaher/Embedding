from embedding.main import Encoder
from embedding.struct import Struct
from embedding.schema import Schema


class ArrayEncoder:
    def __init__(self, labels, dim=128, seed=42):
        self.schema = Schema(labels=labels, attributes=['next'])
        self.encoder = Encoder(schema=self.schema, dim=dim, seed=seed)

    def encode(self, array):
        st = None
        for a in array[::-1]:
            if st is not None:
                st = (a, {'next': st})
            else:
                st = a

        st = Struct.create(self.schema, st)
        return self.encoder.encode(st)

    def decode(self, vector):
        st = self.encoder.decode(vector).to_strings()

        s = [st.label]
        while 'next' in st.attributes:
            st = st.attributes['next']
            s.append(st.label)

        return s
