import numpy as np
import matplotlib.pyplot as plt

# Function to perform ray intersection (triangulation)
def triangulate_point(P1, P2, x1, x2):
    A = np.array([
        (x1[0] * P1[2, :] - P1[0, :]),
        (x1[1] * P1[2, :] - P1[1, :]),
        (x2[0] * P2[2, :] - P2[0, :]),
        (x2[1] * P2[2, :] - P2[1, :])
    ])
    
    # Solve for the 3D point using SVD (Ax = 0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # The 3D point in homogeneous coordinates
    X /= X[3]   # Normalize the point
    
    return X[:3]  # Return the 3D point in Cartesian coordinates

# Example usage
P1 = np.array([[500, 0, 320, 0],
               [0, 500, 240, 0],
               [0, 0, 1, 0]])

P2 = np.array([[500, 0, 320, -100],
               [0, 500, 240, 0],
               [0, 0, 1, 0]])

u1, v1 = 350, 250  # Example 2D point in view 1
u2, v2 = 330, 260  # Example 2D point in view 2

x1 = np.array([u1, v1, 1])  # Homogeneous coordinates for image 1
x2 = np.array([u2, v2, 1])  # Homogeneous coordinates for image 2

# Recover the 3D point
X_3D = triangulate_point(P1, P2, x1, x2)
print("Recovered 3D point:", X_3D)

# Plotting the recovered 3D point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3D[0], X_3D[1], X_3D[2], color='red')

# Labels and formatting
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Recovered 3D Point from Ray Intersection')
plt.show()
