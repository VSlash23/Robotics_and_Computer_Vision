import numpy as np

# Define the matrix
matrix = np.array([[-1.32, -0.18, 2.13],
                   [2.64, -4.68, 4.65],
                   [1.47, -4.75, 6.80]])

# Calculate the rank of the matrix
rank = np.linalg.matrix_rank(matrix)

# Print the result
print(f"The rank of the matrix is: {rank}")

# Check if there are only two independent columns
if rank == 2:
    print("There are only two independent columns.")
else:
    print("There are not exactly two independent columns.")
