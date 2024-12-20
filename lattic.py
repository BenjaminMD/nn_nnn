import matplotlib.pyplot as plt
import numpy as np


class Lattice2D:
    def __init__(self, a1, a2, basis):
        """
        Initialize the primitive lattice.
        :param a1: Primitive lattice vector 1 (numpy array)
        :param a2: Primitive lattice vector 2 (numpy array)
        :param basis: List of tuples (atomic fractional coordinates, label)
                      Example: [([0, 0], 'A'), ([0.5, 0.5], 'B')]
        """
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.basis = [(np.array(atom[0]), atom[1]) for atom in basis]

    def transform_supercell(self, transformation_matrix):
        """
        Generate the supercell lattice vectors.
        :param transformation_matrix: 2x2 matrix defining the supercell transformation
        :return: New supercell lattice vectors A1 and A2
        """
        transformation_matrix = np.array(transformation_matrix)
        A1 = transformation_matrix[0, 0] * self.a1 + \
            transformation_matrix[0, 1] * self.a2
        A2 = transformation_matrix[1, 0] * self.a1 + \
            transformation_matrix[1, 1] * self.a2
        return A1, A2

    def generate_supercell_positions(self, A1, A2, repetitions):
        """
        Generate atomic positions for the supercell with labels.
        :param A1: Transformed lattice vector 1
        :param A2: Transformed lattice vector 2
        :param repetitions: Tuple of repetitions (n1, n2) along A1 and A2
        :return: List of tuples (position, label) in Cartesian coordinates
        """
        n1, n2 = repetitions
        positions_with_labels = []
        for i in range(n1):
            for j in range(n2):
                lattice_translation = i * A1 + j * A2  # Shift by lattice vector multiples
                for atom, label in self.basis:
                    # Use new supercell lattice vectors to compute Cartesian positions
                    position = lattice_translation + \
                        atom[0] * A1 + atom[1] * A2
                    positions_with_labels.append((position, label))
        return positions_with_labels

    def generate_supercell(self, transformation_matrix, repetitions):
        """
        Complete pipeline: Generate the 2D supercell lattice and positions.
        :param transformation_matrix: 2x2 matrix defining the supercell transformation
        :param repetitions: Tuple of repetitions (n1, n2) along the supercell axes
        :return: Supercell lattice vectors and atomic positions with labels
        """
        # Step 1: Transform primitive lattice to supercell lattice
        A1, A2 = self.transform_supercell(transformation_matrix)

        # Step 2: Generate atomic positions in the supercell
        positions_with_labels = self.generate_supercell_positions(
            A1, A2, repetitions)

        return A1, A2, positions_with_labels

    def generate_supercell_positions_with_indices(self, A1, A2, repetitions):
        """
        Generate atomic positions for the supercell with labels and supercell indices.
        :param A1: Transformed lattice vector 1
        :param A2: Transformed lattice vector 2
        :param repetitions: Tuple of repetitions (n1, n2) along A1 and A2
        :return: List of tuples (position, label, supercell_index) in Cartesian coordinates
        """
        n1, n2 = repetitions
        positions_with_labels_and_indices = []
        for i in range(n1):  # Loop over supercell indices along A1
            for j in range(n2):  # Loop over supercell indices along A2
                lattice_translation = i * A1 + j * A2  # Translation for the current supercell
                for atom, label in self.basis:
                    # Calculate the absolute position using the supercell lattice vectors
                    position = lattice_translation + \
                        atom[0] * A1 + atom[1] * A2
                    # Append position, label, and supercell index
                    positions_with_labels_and_indices.append(
                        (position, label, (i, j)))
        return positions_with_labels_and_indices


def plot_lattice(A1, A2, positions_with_labels, repetitions):
    """
    Plot the 2D supercell lattice and atomic positions with labels.
    :param A1: Supercell lattice vector 1
    :param A2: Supercell lattice vector 2
    :param positions_with_labels: List of tuples (position, label) in Cartesian coordinates
    :param repetitions: Tuple of repetitions (n1, n2) along A1 and A2
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot lattice vectors
    ax.quiver(0, 0, A1[0], A1[1], angles='xy',
              scale_units='xy', scale=1, color='r', label='A1')
    ax.quiver(0, 0, A2[0], A2[1], angles='xy',
              scale_units='xy', scale=1, color='g', label='A2')

    # Plot atomic positions with labels
    for pos, label in positions_with_labels:
        ax.scatter(pos[0], pos[1], c='blue', s=50)
        ax.annotate(label, (pos[0], pos[1]), textcoords="offset points", xytext=(5, 5),
                    ha='center', fontsize=12, color='black')

    # Draw lattice grid
    n1, n2 = repetitions
    for i in range(n1):
        for j in range(n2):
            origin = i * A1 + j * A2
            ax.quiver(origin[0], origin[1], A1[0], A1[1], angles='xy',
                      scale_units='xy', scale=1, color='r', alpha=0.3)
            ax.quiver(origin[0], origin[1], A2[0], A2[1], angles='xy',
                      scale_units='xy', scale=1, color='g', alpha=0.3)

    # Set plot limits
    ax.set_xlim(-0.5, n1 * np.linalg.norm(A1) + 0.5)
    ax.set_ylim(-0.5, n2 * np.linalg.norm(A2) + 0.5)

    # Add grid, legend, and labels
    ax.set_aspect('equal', 'box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Lattice with Atom Labels')
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Define primitive lattice vectors
    a1 = [1, 0]
    a2 = [0, 1]

    # Define atomic basis (fractional coordinates with labels)
    basis = [
        ([0, 0], 'A'),   # Atom 1
        ([0.5, 0.5], 'B')  # Atom 2
    ]

    # Create lattice
    lattice = Lattice2D(a1, a2, basis)

    # Define supercell transformation matrix
    transformation_matrix = [
        [1, 0],  # Stretch along a1
        [0, 1]   # Stretch along a2
    ]

    # Define repetitions along the supercell axes
    repetitions = (10, 10)

    # Generate supercell
    A1, A2, positions_with_labels = lattice.generate_supercell(
        transformation_matrix, repetitions)

    positions_with_labels_and_indices = lattice.generate_supercell_positions_with_indices(
        A1, A2, repetitions)

    print(positions_with_labels_and_indices)

    # Plot the lattice
    # plot_lattice(A1, A2, positions_with_labels, repetitions)
