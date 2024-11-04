import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Camera Matrix from calibration (adjusted focal length)
K = np.array([[300, 0, 320],  # Reduced fx
              [0, 300, 240],  # Reduced fy
              [0, 0, 1]])

# Print out the camera matrix and its components
print("Camera Matrix (K):\n", K)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"Focal length (fx, fy): ({fx}, {fy})")
print(f"Principal point (cx, cy): ({cx}, {cy})\n")

# Define a larger 3D object: a pyramid (in world coordinates)
vertices_world = np.array([
    [0, 0, 0, 1],  # Bottom left
    [3, 0, 0, 1],  # Bottom right
    [3, 3, 0, 1],  # Top right
    [0, 3, 0, 1],  # Top left
    [1.5, 1.5, 5, 1]  # Apex (increased height)
]).T  # Transpose to get 4x5 shape

# Rotation matrix (R) and translation vector (t) from calibration
R = np.array([[0.707, -0.707, 0],
              [0.707, 0.707, 0],
              [0, 0, 1]])

# Adjust the translation vector to move the camera closer
t = np.array([[0], [0], [5]])  # Move the camera closer to the object

# Create the extrinsic matrix (3x4) by combining R and t
extrinsic_matrix = np.hstack((R, t))

# Projection matrix P = K * [R | t]
P = np.dot(K, extrinsic_matrix)

# Project the 3D vertices to 2D image coordinates
vertices_image = np.dot(P, vertices_world)

# Normalize the image coordinates by dividing by the third (z) coordinate
vertices_image /= vertices_image[2]

# Extract only the x and y coordinates for plotting
vertices_image_2d = vertices_image[:2]

# Define the edges of the pyramid (connect the vertices)
edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Base
         (0, 4), (1, 4), (2, 4), (3, 4)]  # Sides connecting to the apex

# Plot the 2D projection using Line2D
fig, ax = plt.subplots()

# Plot each edge
for edge in edges:
    point1 = vertices_image_2d[:, edge[0]]
    point2 = vertices_image_2d[:, edge[1]]
    
    line = Line2D([point1[0], point2[0]], [point1[1], point2[1]], color='blue')
    ax.add_line(line)

# Set plot limits and show the plot
ax.set_xlim([0, 640])
ax.set_ylim([480, 0])  # Reverse y-axis to match image coordinates
plt.gca().set_aspect('equal', adjustable='box')
plt.title('2D Projection of a Larger 3D Pyramid (Closer Camera)')
plt.show()
