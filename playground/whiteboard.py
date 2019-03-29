import numpy as np


EPS = 1e-10


def main():
    x = np.arange(1, 10)

    while True:
        np.random.shuffle(x)
        a = x[0]
        b = 10 * x[1] + x[2]

        c = x[3]
        d = 10 * x[4] + x[5]

        e = x[6]
        f = 10 * x[7] + x[8]

        if abs(a / b + c / d + e / f - 1) < EPS:
            print(a, b, c, d, e, f)


if __name__ == '__main__':
    main()
