import unittest
from embedding.array import ArrayEncoder


class TestArray(unittest.TestCase):
    def test_array(self):
        ae = ArrayEncoder(labels=['a', 'b', 'c', 'd'])
        a = ['a', 'b', 'a', 'b', 'c', 'd']
        v = ae.encode(a)
        s = ae.decode(v)
        self.assertEqual(a, s)
