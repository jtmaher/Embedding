import unittest
import numpy as np
from embedding.encoder import Encoder
from embedding.structure import Struct
from embedding.schema import Schema


class TestMain(unittest.TestCase):
    def test_schema(self):
        sc = Schema([], [])

        self.assertEqual(sc.labels, [])

        sc = Schema(labels=['a', 'b', 'c'], attributes=['x', 'y', 'z'])

        self.assertEqual(sc.labels, ['a', 'b', 'c'])
        self.assertEqual(sc.attributes, ['x', 'y', 'z'])

    def test_struct(self):
        sc = Schema(labels=['a', 'b', 'c'], attributes=['x', 'y', 'z'])
        st = Struct(sc, 'a', {'x': 'b', 'y': 'c', 'z': 'd'})

        self.assertEqual(st.label, 'a')
        self.assertEqual(st.attributes, {'x': 'b', 'y': 'c', 'z': 'd'})
        self.assertEqual(st.is_strings, True)
        self.assertEqual(st.schema, sc)
        self.assertEqual(st.attributes['x'], 'b')

    def test_encoder(self):
        sc = Schema(labels=['a', 'b', 'c', 'd'], attributes=['x', 'y', 'z'])
        en = Encoder(schema=sc, dim=128, seed=42)

        s = Struct.create(sc, ('a', {'x': 'b', 'y': 'c', 'z': 'd'}))
        s = s.to_indexes()
        v = en.encode(s)
        self.assertAlmostEqual(np.linalg.norm(v), 2, places=0)

        d = en.decode(v)
        self.assertEqual(d, s)

    def test_encoder2(self):
        sc = Schema(labels=['a', 'b', 'c', 'd'], attributes=['x', 'y', 'z'])
        en = Encoder(schema=sc, dim=128, seed=42)

        s = Struct.create(sc, ('a', {'x': 'b', 'y': ('a', {'x': 'c'}), 'z': ('a', {'x': 'c'})}))
        s = s.to_indexes()
        v = en.encode(s)
        self.assertAlmostEqual(np.linalg.norm(v), np.sqrt(5), places=0)

        d = en.decode(v)
        self.assertEqual(d, s)

    def test_encoder3(self):
        sc = Schema(labels=['a', 'b', 'c', 'd'], attributes=['x', 'y', 'z'])
        en = Encoder(schema=sc, dim=256, seed=42)

        s = Struct.create(sc, ('a', {'x': 'b', 'y': ('a', {'x': ('a', {'x': 'c'})}), 'z': ('a', {'x': 'c'})}))
        s = s.to_indexes()
        v = en.encode(s)
        self.assertAlmostEqual(np.linalg.norm(v), np.sqrt(6), places=0)

        d = en.decode(v)
        self.assertEqual(d, s)


if __name__ == '__main__':
    unittest.main()
