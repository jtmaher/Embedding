import numpy as np
from embedding.array import ArrayEncoder


def labs(k=2):
    return [str(n) for n in range(k)]


def make_array(length, k):
    return list(np.random.choice(labs(k), size=length, replace=True))


def sweep():
    for dim in range(100, 5000, 50):
        ae = ArrayEncoder(labels=labs(), dim=dim)

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


encoders = dict()


def stress(dim=500, length=10, k=2, n=1000):
    if (dim, k) in encoders:
        ae = encoders[(dim, k)]
    else:
        ae = ArrayEncoder(labels=labs(k), dim=dim, seed=None)
        encoders[(dim, k)] = ae
    good = 0
    bad = 0

    # T = ae.encoder.token_emb
    # A = ae.encoder.attr_emb[0]
    # M = T @ A @ T.T
    # M -= np.diag(np.diag(M))
    # N = T @ T.T
    # N -= np.diag(np.diag(N))
    # print(abs(M).max(), abs(N).max())
    for _ in range(n):
        a = make_array(length, k)
        v = ae.encode(a)

        try:
            b = ae.decode(v)

        except Exception as e:
            bad += 1
            continue
        if b is not None:
            success = np.all(a == b)
        else:
            success = False
        good += success
        bad += not success

    return good, bad, good / (good + bad)


Z = 1000
N = 100

with open('stress.txt', 'w') as f:
    for Z in range(200,1000,25):
        all_bad = 0
        for L in range(1,50):
            Z = round(Z)
            g, b, _ = stress(Z,L,2**8,100)
            print(Z, L, g, b)
            f.write(f'{Z},{L},{g},{b}\n')
            if g == 0:
                all_bad += 1
                if all_bad >5:
                    break
#stress(Z, 1, 2 ** 10, n=N)
#stress(Z, 5, 2 ** 10, n=N)
#stress(Z, 10, 10000, n=N)
#stress(Z, 15, 2 ** 10, n=N)
#stress(Z, 20, 2 ** 10, n=N)
#stress(Z, 30, 2 ** 10, n=N)
