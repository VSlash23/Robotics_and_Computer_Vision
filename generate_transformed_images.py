import cv2
import numpy as np
import glob
import os

# Function to apply random transformations to an image
def transform_image(img, angle, scale, tx, ty):
    rows, cols = img.shape[:2]
    
    # Rotation matrix
    M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    
    # Translation matrix
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply transformations
    rotated = cv2.warpAffine(img, M_rotation, (cols, rows))
    transformed = cv2.warpAffine(rotated, M_translation, (cols, rows))
    
    return transformed

# Load your existing images
image_files = sorted(glob.glob('left*.jpg'))

# Process the last available image (e.g., left14.jpg) and generate 10 more images
last_image = cv2.imread(image_files[-1])

for i in range(15, 25):
    # Apply random transformations: rotation between -30 and 30 degrees, scale between 0.9 and 1.1, translation
    angle = np.random.uniform(-30, 30)  # Random rotation between -30 and 30 degrees
    scale = np.random.uniform(0.9, 1.1)  # Random scaling between 0.9x and 1.1x
    tx = np.random.uniform(-50, 50)  # Random translation in the X direction
    ty = np.random.uniform(-50, 50)  # Random translation in the Y direction
    
    # Transform image
    transformed_img = transform_image(last_image, angle, scale, tx, ty)
    
    # Save the new image
    output_filename = f'left{i}.jpg'
    cv2.imwrite(output_filename, transformed_img)
    print(f'Generated {output_filename}')

# Show the first transformed image for verification
cv2.imshow('Transformed Image', transformed_img)
cv2.waitKey(500)
cv2.destroyAllWindows()
