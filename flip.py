#Change flip 


import cv2 
import numpy as np


def flip_image(img, horizontal=0, vertical=0):
    """
    This functions flips the already read image in the X, Y axis. 
    Horizontal = Flip in the Y axis
    Vertical = Flip in the X axis"""

    if vertical==1:
        if horizontal==1:
            flip_code = -1
        else:
            flip_code = 0
    elif horizontal==1:
        flip_code = 1
    elif vertical==0 & horizontal==0:
        return
    
    image_flip = cv2.flip(img, flip_code)

    return image_flip

if __name__ == "__main__":
    image = cv2.imread("data-augmentation/13.jpg")
    image_v = flip_image(image, 1, 0)
    # display the original and the Translated image

    cv2.imshow('Original image', image)
    cv2.waitKey(0)

    cv2.imshow('Brightness image', image_v)
    cv2.waitKey(0)
    cv2.imwrite("test_brightness.jpg", image_v)

    