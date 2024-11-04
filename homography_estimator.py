import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the images
frontal_img = cv2.imread('frontal_view.jpg')
side_img = cv2.imread('side_view.jpg')

# Convert images to RGB (for displaying purposes)
frontal_img_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)
side_img_rgb = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)

# 2. Define corresponding points between the two images (fine-tuned points)
pts_frontal = np.array([
    [100, 100],   # Top-left corner of the window/door in the frontal view
    [800, 100],   # Top-right corner of the window/door in the frontal view
    [800, 600],   # Bottom-right corner of the window/door in the frontal view
    [100, 600]    # Bottom-left corner of the window/door in the frontal view
], dtype=np.float32)

pts_side = np.array([
    [120, 140],   # Corresponding top-left corner in the side view
    [820, 130],   # Corresponding top-right corner in the side view
    [820, 620],   # Corresponding bottom-right corner in the side view
    [120, 600]    # Corresponding bottom-left corner in the side view
], dtype=np.float32)

# 3. Estimate homography using OpenCV's findHomography
H, status = cv2.findHomography(pts_side, pts_frontal)

# 4. Warp the side view image using the estimated homography
warped_side = cv2.warpPerspective(side_img, H, (frontal_img.shape[1], frontal_img.shape[0]))

# Convert warped image to RGB for display
warped_side_rgb = cv2.cvtColor(warped_side, cv2.COLOR_BGR2RGB)

# 5. Display the original side view and warped view
plt.subplot(1, 2, 1)
plt.imshow(side_img_rgb)
plt.title('Original Side View')

plt.subplot(1, 2, 2)
plt.imshow(warped_side_rgb)
plt.title('Warped to Frontal View')

plt.show()
