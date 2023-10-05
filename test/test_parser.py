import unittest
from embedding.parser import Parser, Rule
from embedding.schema import Schema
from embedding.encoder import Encoder
from embedding.array import array_to_tree
from embedding.structure import Struct


class TestParser(unittest.TestCase):
    def check_parse(self, seq, exp):
        sc = Schema(labels=['L', 'R', 'E'],
                    attributes=['next', 'arg1', 'arg2', 'arg3'])

        en = Encoder(schema=sc, dim=512, seed=42)

        def enc(a):
            return en.encode(array_to_tree(a))

        e = enc(['E'])

        r1 = Rule(length=2,
                  pattern=enc(['L', 'R']),
                  replacement=e)

        r2 = Rule(length=3,
                  pattern=enc(['L', 'E', 'R']),
                  replacement=e)

        r3 = Rule(length=2,
                  pattern=enc(['E', 'E']),
                  replacement=e)

        p = Parser(schema=sc,
                   rules=[r1, r2, r3],
                   next_a=en.attr_emb[sc.attr_to_ind['next']],
                   args=[en.attr_emb[sc.attr_to_ind[x]] for x in ['arg1', 'arg2', 'arg3']])

        seq_vec = [x for x in map(enc, seq)]

        y = p.parse(seq_vec)
        exp = Struct.create(sc, exp)
        tr = en.decode(y[0]).to_strings()

        self.assertEqual(exp, tr)  # add assertion here

    def test_parse1(self):
        seq = ['L', 'L', 'R', 'R']
        exp = ("E", {"arg1": "L", "arg2": ("E", {"arg1": "L", "arg2": "R", }), "arg3": "R", })
        self.check_parse(seq, exp)

    def test_parse2(self):
        seq = ['L', 'R', 'L', 'R']
        exp = ("E", {"arg1": ("E", {"arg1": "L", "arg2": "R", }), "arg2": ("E", {"arg1": "L", "arg2": "R", }), })
        self.check_parse(seq, exp)

    def test_parse3(self):
        seq = ['L', 'R', 'L', 'R', 'L', 'R']
        exp = ("E", {
            "arg1": ("E", {
                "arg1": ("E", {
                    "arg1": "L", "arg2": "R", }),
                "arg2": ("E", {
                    "arg1": "L", "arg2": "R", }), }),
            "arg2": ("E", {
                "arg1": "L", "arg2": "R", }), })
        self.check_parse(seq, exp)


if __name__ == '__main__':
    unittest.main()
