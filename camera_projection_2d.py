import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Define the calibration matrices
K = np.array([[-100, 0, 200],
              [0, -100, 200],
              [0, 0, 1]])

R_cw = np.array([[0.707, 0.707, 0],
                 [-0.707, 0.707, 0],
                 [0, 0, 1]])

t_cw = np.array([[-3], [-0.5], [3]])

# Create extrinsic matrix [R | t]
T_cw = np.hstack((R_cw, t_cw))

# 2. Define the vertices of a simple 3D object (e.g., a cube)
vertices_world = np.array([[0, 0, 0, 1],  # Bottom face
                           [1, 0, 0, 1],
                           [1, 1, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1],  # Top face
                           [1, 0, 1, 1],
                           [1, 1, 1, 1],
                           [0, 1, 1, 1]]).T  # Transpose to get (4, 8) shape

# 3. Compute the camera projection matrix P = K * [R | t]
P = np.dot(K, T_cw)

# 4. Project the vertices from world to image coordinates
vertices_image = np.dot(P, vertices_world)

# Normalize the image coordinates by dividing by the third (z) coordinate
vertices_image /= vertices_image[2]

# Extract only the x and y coordinates for plotting
vertices_image_2d = vertices_image[:2]

# 5. Define the edges of the cube (connect the vertices)
edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
         (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
         (0, 4), (1, 5), (2, 6), (3, 7)]  # Vertical lines connecting top and bottom

# 6. Plot the 2D projection using Line2D
fig, ax = plt.subplots()

# Plot each edge
for edge in edges:
    point1 = vertices_image_2d[:, edge[0]]
    point2 = vertices_image_2d[:, edge[1]]
    
    line = Line2D([point1[0], point2[0]], [point1[1], point2[1]], color='blue')
    ax.add_line(line)

# Set plot limits and show the plot
ax.set_xlim([0, 400])
ax.set_ylim([400, 0])  # Reverse y-axis to match image coordinates
plt.gca().set_aspect('equal', adjustable='box')
plt.title('2D Projection of 3D Object')
plt.show()
