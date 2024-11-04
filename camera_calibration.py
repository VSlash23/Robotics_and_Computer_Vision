import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set chessboard dimensions (9 internal corners per row, 6 internal corners per column)
chessboard_size = (9, 6)

# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Set criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load the images (left01.jpg to left23.jpg)
images = glob.glob('left*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)

        # Refine corner detection for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix (K matrix)
print("Camera Matrix (K): \n", camera_matrix)

# Print the first three R matrices (rotation) and t vectors (translation) for 3 views
for i in range(3):
    rmat, _ = cv2.Rodrigues(rvecs[i])  # Convert rotation vectors to rotation matrices
    print(f"\nRotation Matrix (R) for View {i+1}: \n", rmat)
    print(f"Translation Vector (t) for View {i+1}: \n", tvecs[i])

# Save the calibration results
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

# Plot the camera locations (t vectors) in world coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the t vectors (camera locations)
for i, tvec in enumerate(tvecs):
    ax.scatter(tvec[0][0], tvec[1][0], tvec[2][0], color='red', marker='o')
    ax.text(tvec[0][0], tvec[1][0], tvec[2][0], f'Cam {i+1}', size=8)

# Set labels for axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.title("Camera Locations in World Coordinates")
plt.show()

# Optional: undistort one of the images to verify the calibration
img = cv2.imread('left01.jpg')
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the undistorted image for verification
cv2.imwrite('undistorted_left01.jpg', undistorted_img)
