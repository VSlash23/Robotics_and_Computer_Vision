# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Describe the Object Model: A Simple House
def define_house():
    # Base of the house (a square in XY plane)
    base = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])  # Square base
    roof = np.array([[0.5, 0.5, 1]])  # Peak of the roof
    house = np.vstack((base, roof))  # Combine base and roof vertices
    return house

# Create the Camera Matrix: Choosing a Point to Look At
def create_camera_matrix(f, sx, sy, ox, oy, R, t):
    # Intrinsic camera matrix
    K = np.array([[f / sx, 0, ox],    # Focal length and scaling
                  [0, f / sy, oy],
                  [0, 0, 1]])
    
    # Extrinsic parameters matrix (R = rotation, t = translation)
    extrinsic = np.hstack((R, t.reshape(-1, 1)))  # Combine R and t into a 3x4 matrix
    camera_matrix = np.dot(K, extrinsic)
    return camera_matrix

# Project the 3D points to 2D using the camera matrix
def project_points(points_3d, camera_matrix):
    # Add a 1 to the points for homogeneous coordinates
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Project the 3D points to 2D using matrix multiplication
    points_2d_h = camera_matrix.dot(points_3d_h.T).T
    
    # Convert homogeneous 2D coordinates to Cartesian 2D by dividing by the third coordinate
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2].reshape(-1, 1)
    return points_2d

# Plot the projected 2D points using Line2D (no 3D plotting)
def plot_house_2d(points_2d):
    fig, ax = plt.subplots()
    
    # Define the edges of the house to connect vertices with lines
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Base edges
             (0, 4), (1, 4), (2, 4), (3, 4)]  # Roof edges
    
    # Plot each edge using Line2D
    for edge in edges:
        p1, p2 = points_2d[edge[0]], points_2d[edge[1]]
        line = Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='blue')
        ax.add_line(line)
    
    ax.set_aspect('equal')
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    plt.show()

# Main function: Define the house, set up the camera parameters, and project the house
def main():
    # Define the 3D house object
    house = define_house()
    
    # Camera Parameters
    # Intrinsic parameters:
    f = 800  # Focal length (in pixels)
    sx, sy = 1, 1  # Scaling factors (1 = no scaling)
    ox, oy = 0, 0  # Principal point shift (assumed to be at the origin)
    
    # Extrinsic parameters: Rotation and Translation
    R = np.eye(3)  # No rotation (identity matrix)
    t = np.array([0, 0, -5])  # Translate the camera back along the Z-axis (5 units back)
    
    # Create the camera matrix
    camera_matrix = create_camera_matrix(f, sx, sy, ox, oy, R, t)
    
    # Print the camera matrix to the terminal
    print("Camera Matrix:\n", camera_matrix)
    
    # Project the 3D house to 2D
    house_2d = project_points(house, camera_matrix)
    
    # Plot the 2D projection of the house
    plot_house_2d(house_2d)

# Run the main function
if __name__ == "__main__":
    main()
