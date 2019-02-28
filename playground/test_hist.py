import numpy as np
import sys

import matplotlib.pyplot as plt


def main():
    x = np.array([1, 2, 1, 3, 3, 1, 2])

    x = np.random.randint(3, 50, size=500000)
    x = np.random.normal(0, 10, size=50000)

    bins = np.arange(np.round(x.min()) - 1, np.round(x.max()) + 1, dtype=np.float32)
    print(bins)
    bins += 0.5
    print(bins)
    plt.hist(x, bins=bins)
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
