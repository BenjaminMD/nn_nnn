from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def cubic_2D_lattice(cell, basis, extent) -> Tuple[str, float, float]:
    xy = np.array(np.meshgrid(np.arange(extent[0]), np.arange(extent[1])))

    lattice = {}

    for x, y in xy.reshape(2, -1).T:
        for atom, coords in basis.items():
            lattice[f"{atom}_{x}_{y}"] = cell[:2] * (np.array([x, y]) + coords)
    return lattice


def main() -> None:

    cell = (5.59, 5.59, 5.59, 90, 90, 90)

    basis = {
        "Na": np.array([0, 0]),
        "Cl": np.array([0.5, 0.5]),
    }

    extent = np.array([4, 4])

    lattice = cubic_2D_lattice(cell, basis, extent)

    print(lattice)

    for atom, coords in lattice.items():
        plt.scatter(*coords, label=atom)


if __name__ == "__main__":
    main()
