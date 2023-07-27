import numpy as np
from embedding.array import ArrayEncoder

labs = [str(n) for n in range(100)]


def make_array(length):
    return list(np.random.choice(labs, size=length, replace=True))


def sweep():
    for dim in range(100, 5000, 50):
        ae = ArrayEncoder(labels=labs, dim=dim)

        for length in range(1, 100):
            a = make_array(length)
            v = ae.encode(a)

            try:
                b = ae.decode(v)
            except Exception as e:
                b = None

            success = np.all(a == b)
            if not success:
                print(dim, length, success)
                break


def stress(dim=500, length=10, n=100):
    ae = ArrayEncoder(labels=labs, dim=dim, seed=None, multiplicity=2)
    good = 0
    bad = 0
    for _ in range(n):
        a = make_array(length)
        try:
            v = ae.encode(a)
        except Exception as e:
            print(a)
            continue
        b = ae.decode(v)

        if b is not None:
            success = np.all(a == b)
        else:
            success = False
        good += success
        bad += not success
        print(good, bad, good / (good + bad))


stress(1028, 48, 1000)
