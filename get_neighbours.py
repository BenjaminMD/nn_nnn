import numpy as np


def get_nn(pos, all_pos: tuple):
    distances = np.linalg.norm(all_pos - pos, axis=1)
    distances = distances[distances > 0]
    print(distances)
    distance_id = np.isclose(distances, np.min(distances))

    return np.argwhere(distance_id).flatten()


def main():
    pos = np.array([0, 0])
    all_pos = ([0, 1], [1, 0], [1, 1], [2, 2])

    print(get_nn(pos, all_pos))


if __name__ == '__main__':
    main()
