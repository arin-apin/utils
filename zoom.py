#Zoom 

import cv2 
import numpy as np


def zoom_image(img, zoomout=0, zoomin=0):
    """
    This functions zoom in or out in the image passed as img. 
    Out = Zoom out of the image 
    In = Zoom in the image """

    zoomin= zoomin+1
    
    zoom_factor = np.random.uniform(zoomout,zoomin)

    #Get the image heigh and widht
    h, w = img.shape[:2]

    
    #Calculate the new image size
    new_size = (int(w * zoom_factor), int(h * zoom_factor))

    #Resize the image
    image_flip = cv2.resize(img, new_size)

    return image_flip

import cv2 as cv

def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

if __name__ == "__main__":
    image = cv2.imread("data-augmentation/13.jpg")
    image_result = zoom_image(image, 0.5, 0)

    image_result2 = zoom_at(image, zoom=0.5, angle=0)
    # display the original and the Translated image

    cv2.imshow('Original image', image)
    cv2.waitKey(0)

    cv2.imshow('Zoomed image', image_result2)
    cv2.waitKey(0)
    cv2.imwrite("test_zoom.jpg", image_result2)

    