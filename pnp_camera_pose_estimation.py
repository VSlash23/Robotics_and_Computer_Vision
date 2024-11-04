import cv2
import numpy as np

# Step 1: Define 3D world points (add at least one more point)
object_points = np.array([
    [0, 0, 0],    # Ground corner 1
    [1, 0, 0],    # Ground corner 2
    [1, 1, 0],    # Ground corner 3
    [0, 1, 0],    # Ground corner 4
    [0.5, 0.5, 1.5],  # Roof peak
    [1, 0.5, 1.0]  # Additional point on the roof
], dtype=np.float32)

# Step 2: Define corresponding 2D image points (add corresponding 2D point for the new 3D point)
image_points = np.array([
    [320, 240],  # Projection of corner 1
    [400, 240],  # Projection of corner 2
    [400, 320],  # Projection of corner 3
    [320, 320],  # Projection of corner 4
    [360, 200],  # Projection of roof peak
    [370, 220]   # Projection of new roof corner
], dtype=np.float32)

# Step 3: Define the camera intrinsic matrix (K)
K = np.array([
    [800, 0, 320],  # Focal length and principal point in X
    [0, 800, 240],  # Focal length and principal point in Y
    [0, 0, 1]
], dtype=np.float64)

# Assume no lens distortion (if you have it, replace this with actual distortion coefficients)
dist_coeffs = np.zeros((4, 1))  

# Step 4: Run the PnP algorithm to estimate camera pose
success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

# Convert the rotation vector to a rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Step 5: Display the result: Rotation matrix and Translation vector
print("Rotation Matrix:\n", rotation_matrix)
print("Translation Vector:\n", tvec)

# Step 6: Reproject the 3D points to verify the camera pose
reprojected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
print("Original 2D Image Points:\n", image_points)
print("Reprojected 2D Points:\n", reprojected_points.reshape(-1, 2))

# Step 7: Calculate the reprojection error
error = np.linalg.norm(image_points - reprojected_points.reshape(-1, 2), axis=1)
print("Reprojection Error: ", error)
print("Average Error: ", np.mean(error))
