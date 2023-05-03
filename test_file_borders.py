import cv2
import numpy as np

# Load the image
img = cv2.imread('/home/mario3/arinapin/python/git/deeplearning-apps/data-augmentation/test.jpeg')

# Define the translation matrix
tx, ty = 50, 100
M = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation
result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Get the amount of padding needed
top, bottom, left, right = abs(ty), abs(ty), abs(tx), abs(tx)

# Get the last pixel value of the image
last_pixel_value = result[img.shape[0]-1, img.shape[1]-1]

# Add the padding
result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

# # Fill the padding with the last pixel value of the image
# if top > 0:
#     result[0:top, left:left+result.shape[1]] = last_pixel_value
# if bottom > 0:
#     result[result.shape[0]-bottom:result.shape[0], left:left+result.shape[1]] = last_pixel_value
# if left > 0:
#     result[top:top+result.shape[0], 0:left] = last_pixel_value
# if right > 0:
#     result[top:top+result.shape[0], result.shape[1]-right:result.shape[1]] = last_pixel_value

# Show the result
cv2.imshow('Translated Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()